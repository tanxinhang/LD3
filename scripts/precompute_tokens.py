#!/usr/bin/env python3
"""Offline token preprocessing: DD detection + VP refinement + LS gains -> .npz.

This removes VP from the training DataLoader's critical path entirely.
Tokens are generated once per config and loaded from disk during training.

Usage:
  python scripts/precompute_tokens.py \
    --config configs/gate2_vp_optimize.yaml \
    --output-dir data/tokens_vp5r12p \
    --split train,val,test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ld3.channel import ChannelConfig, OFDMConfig, generate_path_set, synthesize_tf_channel
from ld3.dd_estimation import (
    build_dd_grid,
    detect_paths_nms,
    masked_matched_filter_map,
    refine_paths_variable_projection,
)
from ld3.interpolation import nearest_smooth_interpolation
from ld3.oracle import (
    _build_raw_dict,
    _col_norms,
    _ridge_ls,
    compute_path_quality,
    estimated_path_tokens_v2,
    oracle_path_tokens_v2,
)
from ld3.pilots import generate_noise_grid, make_pilot_mask, observe_pilots


def generate_sample(
    ofdm: OFDMConfig,
    channel: ChannelConfig,
    seed: int,
    sample_idx: int,
    snr_min: float,
    snr_max: float,
    pilot_density: float,
    pilot_pattern: str,
    max_paths: int,
    token_version: int,
    token_source: str,
    token_refine: str,
    vp_rounds: int,
    vp_probes: int,
    vp_search: str,
) -> dict[str, Any]:
    """Generate one sample with full DD pipeline + optional VP refinement."""
    rng = np.random.default_rng([seed, sample_idx])
    paths = generate_path_set(ofdm, channel, rng)
    truth = synthesize_tf_channel(ofdm, paths)
    snr_db = float(rng.uniform(snr_min, snr_max))
    mask = make_pilot_mask(
        ofdm.num_subcarriers, ofdm.num_symbols,
        pilot_density, rng, pilot_pattern,
    )
    signal_power = float(np.mean(np.abs(truth) ** 2))
    noise_grid, noise_var = generate_noise_grid(
        truth.shape, signal_power, snr_db, rng,
    )
    observed, _ = observe_pilots(
        truth, mask, snr_db, rng,
        noise_grid=noise_grid, noise_var=noise_var,
    )
    initial = nearest_smooth_interpolation(observed, mask)

    if token_source == "estimated":
        grid = build_dd_grid(
            ofdm.num_subcarriers, ofdm.num_symbols,
            channel.max_delay_bins, channel.max_abs_doppler_bins, 2, 4,
        )
        score_map, gain_map = masked_matched_filter_map(observed, mask, grid)
        est = detect_paths_nms(
            score_map, gain_map, grid, num_paths=channel.num_paths,
        )
        if len(est.delay_bins) > 0:
            if token_refine == "vp":
                est, _vp_diag = refine_paths_variable_projection(
                    est,
                    pilot_observations=observed,
                    pilot_mask=mask,
                    num_subcarriers=ofdm.num_subcarriers,
                    num_symbols=ofdm.num_symbols,
                    n_rounds=vp_rounds,
                    n_probes=vp_probes,
                    search_method=vp_search,
                )
            n_idx, m_idx = np.nonzero(mask)
            A_raw = _build_raw_dict(
                ofdm.num_subcarriers, ofdm.num_symbols,
                n_idx, m_idx, est.delay_bins, est.doppler_bins,
            )
            norms = _col_norms(A_raw)
            for j in range(A_raw.shape[1]):
                if norms[j] > 1e-15:
                    A_raw[:, j] /= norms[j]
            y = observed[mask]
            g_hat = _ridge_ls(A_raw, y)
            g_hat = g_hat / np.maximum(norms, np.finfo(float).eps)
            conf, sig_t, sig_n, rel = compute_path_quality(
                est, g_hat, observed, mask, score_map,
                ofdm.num_subcarriers, ofdm.num_symbols,
            )
            tokens, valid = estimated_path_tokens_v2(
                est, g_hat, max_paths,
                confidence=conf, sigma_tau=sig_t,
                sigma_nu=sig_n, relevance=rel,
            )
        else:
            dim = 9 if token_version >= 2 else 7
            tokens = np.zeros((max_paths, dim), dtype=np.float32)
            valid = np.zeros(max_paths, dtype=bool)
    elif token_version >= 2:
        tokens, valid = oracle_path_tokens_v2(paths, max_paths)
    else:
        tokens, valid = oracle_path_tokens(paths, max_paths)

    tf_input = np.stack([initial.real, initial.imag, mask.astype(np.float64)], axis=0)
    target = np.stack([truth.real, truth.imag], axis=0)

    return {
        "tf_input": tf_input.astype(np.float32),
        "target": target.astype(np.float32),
        "path_tokens": tokens.astype(np.float32),
        "path_valid": valid,
        "snr_db": np.float32(snr_db),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute DD tokens offline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ofdm = OFDMConfig(**config["ofdm"])
    channel = ChannelConfig(**config["channel"])
    data_cfg = config["dataset"]
    base_seed = config["seed"]
    token_ver = int(data_cfg.get("token_version", 2))
    token_src = str(data_cfg.get("token_source", "oracle"))
    token_ref = str(data_cfg.get("token_refine", ""))
    vp_r = int(data_cfg.get("token_vp_rounds", 3))
    vp_p = int(data_cfg.get("token_vp_probes", 8))
    vp_search = str(data_cfg.get("token_vp_search", "random"))

    splits = [s.strip() for s in args.splits.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_configs = {
        "train": {"size": int(data_cfg["train_size"]), "seed": base_seed},
        "val": {"size": int(data_cfg.get("val_size", 1024)), "seed": base_seed + 20000},
        "test": {"size": int(data_cfg["test_size"]), "seed": base_seed + 10000},
    }

    for split in splits:
        if split not in split_configs:
            continue
        sc = split_configs[split]
        print(f"Generating {split} ({sc['size']} samples)...")
        tokens_list, valid_list, tf_list, tgt_list, snr_list = [], [], [], [], []

        for idx in range(sc["size"]):
            sample = generate_sample(
                ofdm, channel, sc["seed"], idx,
                float(data_cfg["snr_min_db"]), float(data_cfg["snr_max_db"]),
                float(data_cfg["pilot_density"]), str(data_cfg["pilot_pattern"]),
                int(data_cfg["max_paths"]), token_ver, token_src, token_ref,
                vp_r, vp_p, vp_search,
            )
            tokens_list.append(sample["path_tokens"])
            valid_list.append(sample["path_valid"])
            tf_list.append(sample["tf_input"])
            tgt_list.append(sample["target"])
            snr_list.append(sample["snr_db"])
            if (idx + 1) % 500 == 0:
                print(f"  {idx + 1}/{sc['size']}")

        out_path = args.output_dir / f"{split}.npz"
        np.savez_compressed(
            out_path,
            tokens=np.array(tokens_list, dtype=np.float32),
            valid=np.array(valid_list, dtype=bool),
            tf_input=np.array(tf_list, dtype=np.float32),
            target=np.array(tgt_list, dtype=np.float32),
            snr_db=np.array(snr_list, dtype=np.float32),
        )
        print(f"  Saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Save metadata
    meta = {
        "config": config,
        "token_dim": tokens_list[0].shape[-1],
        "max_paths": int(data_cfg["max_paths"]),
        "splits": {s: split_configs[s]["size"] for s in splits if s in split_configs},
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata -> {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
