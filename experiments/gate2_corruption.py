#!/usr/bin/env python3
"""Gate 2-A: Failure-boundary audit — clean-trained model under corrupted tokens.

Design rules:
  - Model is FROZEN (no training, no fine-tuning).
  - Two evaluation chains: Oracle token + corruption, Estimated token + corruption.
  - Per-sample TF-only as safety baseline.
  - Reports: mean NMSE, harm rate, worst-10%, max degradation, gate statistics,
    gate-vs-NMSE_H_phys correlation.

Gate 2-A smoke (minimal first step):
  4 perturbation types × few levels × 200 samples × oracle + estimated chains.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ld3.channel import ChannelConfig, OFDMConfig, generate_path_set, synthesize_tf_channel
from ld3.config import OFDMConfig as OFDMConfig2
from ld3.dataset import DatasetConfig, SyntheticOFDMISACDataset
from ld3.dd_estimation import build_dd_grid, detect_paths_nms, masked_matched_filter_map
from ld3.metrics import nmse_numpy, nmse_torch
from ld3.models import PhysicalResidualEstimator, TFOnlyEstimator
from ld3.oracle import (
    corrupt_tokens,
    estimated_path_tokens_v2,
    oracle_path_tokens_v2,
    compute_path_quality,
)
from ld3.pilots import generate_noise_grid, make_pilot_mask, observe_pilots


# ===========================================================================
# Helpers
# ===========================================================================


def _nmse_db(linear: float) -> float:
    return float(10.0 * math.log10(max(linear, 1e-30)))


def _per_sample_nmse_torch(output: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """[B, 2, N, M] → [B] linear NMSE per sample."""
    err = (output - target).square().sum(dim=(1, 2, 3))
    pwr = target.square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return (err / pwr).cpu().numpy()


# ===========================================================================
# Token construction (must match dataset.py's token generation)
# ===========================================================================


def _build_oracle_tokens(paths, max_paths: int) -> tuple[np.ndarray, np.ndarray]:
    return oracle_path_tokens_v2(paths, max_paths)


def _build_estimated_tokens(
    ofdm: OFDMConfig,
    channel: ChannelConfig,
    observed: np.ndarray,
    mask: np.ndarray,
    max_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = build_dd_grid(
        ofdm.num_subcarriers, ofdm.num_symbols,
        channel.max_delay_bins, channel.max_abs_doppler_bins, 2, 4,
    )
    score_map, gain_map = masked_matched_filter_map(observed, mask, grid)
    est = detect_paths_nms(score_map, gain_map, grid, num_paths=channel.num_paths)
    if len(est.delay_bins) == 0:
        return np.zeros((max_paths, 9), dtype=np.float32), np.zeros(max_paths, dtype=bool)

    from ld3.oracle import _ridge_ls, _col_norms, _build_raw_dict
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
    return estimated_path_tokens_v2(
        est, g_hat, max_paths, confidence=conf, sigma_tau=sig_t,
        sigma_nu=sig_n, relevance=rel,
    )


# ===========================================================================
# Core evaluation: one model × one corruption level
# ===========================================================================


def evaluate_corrupted(
    model: torch.nn.Module,
    dataset: SyntheticOFDMISACDataset,
    device: torch.device,
    ofdm: OFDMConfig,
    channel: ChannelConfig,
    corruption: dict[str, Any],
    rng_seed: int,
    tf_only_model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate a PhysicalResidualEstimator with corrupted tokens, per-sample.

    Returns per-sample NMSE arrays + aggregate statistics.
    """
    model.eval()
    if tf_only_model is not None:
        tf_only_model.eval()

    n_samples = len(dataset)
    rng = np.random.default_rng(rng_seed)

    # Per-sample accumulators
    fusion_nmse: list[float] = []
    tf_only_nmse: list[float] = []
    gate_mean_vals: list[float] = []
    gate_p10_vals: list[float] = []
    gate_p50_vals: list[float] = []
    gate_p90_vals: list[float] = []
    h_phys_nmse: list[float] = []   # NMSE of H_phys alone
    # Null error decomposition accumulators (per-component NMSE vs truth)
    decomp_tf_nmse: list[float] = []       # internal H_Tf NMSE
    decomp_fused_nmse: list[float] = []    # H_fused NMSE (after gating)
    decomp_delta_nmse: list[float] = []    # delta residual NMSE
    decomp_phys_mix: list[float] = []      # |g * H_phys|² power fraction

    for idx in range(n_samples):
        sample = dataset[idx]
        tf_input = sample["tf_input"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        pt_clean = sample["path_tokens"].numpy()
        pv_clean = sample["path_valid"].numpy()

        # --- Reconstruct ground-truth paths for coherent false + H_phys NMSE ---
        sample_rng = np.random.default_rng([dataset.cfg.seed, idx])
        paths = generate_path_set(ofdm, channel, sample_rng)
        truth = synthesize_tf_channel(ofdm, paths)
        true_delays = paths.delay_bins
        true_dopplers = paths.doppler_bins
        true_gains = paths.gains

        # --- Build the right token source ---
        token_source = corruption.get("token_source", "oracle")
        if token_source == "oracle":
            pt, pv = _build_oracle_tokens(paths, dataset.cfg.max_paths)
        else:
            observed_np = None
            mask_np = None
            # Re-run pilot observation to get raw data for estimated tokens
            sample_rng2 = np.random.default_rng([dataset.cfg.seed, idx])
            snr_db = float(sample["snr_db"].item())
            mask_np = make_pilot_mask(
                ofdm.num_subcarriers, ofdm.num_symbols,
                dataset.cfg.pilot_density, sample_rng2, dataset.cfg.pilot_pattern,
            )
            signal_power = float(np.mean(np.abs(truth) ** 2))
            noise_grid, noise_var = generate_noise_grid(
                truth.shape, signal_power, snr_db, sample_rng2,
            )
            observed_np, _ = observe_pilots(
                truth, mask_np, snr_db, sample_rng2,
                noise_grid=noise_grid, noise_var=noise_var,
            )
            pt, pv = _build_estimated_tokens(
                ofdm, channel, observed_np, mask_np, dataset.cfg.max_paths,
            )

        # --- Apply corruption ---
        corrupt_kwargs = {k: v for k, v in corruption.items()
                          if k not in ("name", "token_source")}
        corrupt_kwargs.setdefault("max_delay_bins", channel.max_delay_bins)
        corrupt_kwargs.setdefault("max_abs_doppler_bins", channel.max_abs_doppler_bins)
        # Pass true path info for coherent false
        if corrupt_kwargs.get("coherent_false_paths", 0) > 0:
            corrupt_kwargs["true_delays"] = true_delays
            corrupt_kwargs["true_dopplers"] = true_dopplers

        pt_corrupted, pv_corrupted = corrupt_tokens(pt, pv, rng, **corrupt_kwargs)

        # --- Model forward with corrupted tokens ---
        pt_t = torch.tensor(pt_corrupted, dtype=torch.float32).unsqueeze(0).to(device)
        pv_t = torch.tensor(pv_corrupted, dtype=torch.bool).unsqueeze(0).to(device)

        with torch.no_grad():
            output, diagnostics = model(tf_input, pt_t, pv_t)

        nmse_val = float(nmse_torch(output, target).cpu())
        fusion_nmse.append(nmse_val)

        # Gate statistics (per-pixel gate, flattened)
        if "gate" in diagnostics:
            gate_map = diagnostics["gate"].cpu().numpy().ravel()  # [B*1*N*M]
            gate_mean_vals.append(float(np.mean(gate_map)))
            gate_p10_vals.append(float(np.percentile(gate_map, 10)))
            gate_p50_vals.append(float(np.percentile(gate_map, 50)))
            gate_p90_vals.append(float(np.percentile(gate_map, 90)))

        # Null error decomposition: collect component power diagnostics from model
        # (aggregated per-condition in result dict below)
        if "p_tf_mean" in diagnostics:
            decomp_tf_nmse.append(float(diagnostics["p_tf_mean"].cpu()))
            decomp_fused_nmse.append(float(diagnostics["p_fused_mean"].cpu()))
            decomp_delta_nmse.append(float(diagnostics["p_delta_mean"].cpu()))
            decomp_phys_mix.append(float(diagnostics["phys_mix_mean"].cpu()))

        # H_phys NMSE
        if "gate" in diagnostics:
            # H_phys is NOT directly returned; we synthesize it
            from ld3.channel import PathSet
            # Use corrupted token positions and gains (only valid ones)
            c_valid = pv_corrupted
            c_tau = pt_corrupted[c_valid, 0]
            c_nu = pt_corrupted[c_valid, 1]
            c_alpha = pt_corrupted[c_valid, 7] + 1j * pt_corrupted[c_valid, 8]
            if len(c_tau) > 0:
                phys_paths = PathSet(
                    delay_bins=c_tau, doppler_bins=c_nu, gains=c_alpha,
                )
                H_phys_np = synthesize_tf_channel(ofdm, phys_paths)
                h_phys_nmse_val = nmse_numpy(H_phys_np, truth)
            else:
                h_phys_nmse_val = 1.0  # no paths → worst case
            h_phys_nmse.append(h_phys_nmse_val)

        # TF-only baseline (same sample)
        if tf_only_model is not None:
            with torch.no_grad():
                tf_out = tf_only_model(tf_input)
            tf_only_nmse.append(float(nmse_torch(tf_out, target).cpu()))

    # --- Aggregate ---
    fusion_arr = np.array(fusion_nmse)
    tf_arr = np.array(tf_only_nmse) if tf_only_nmse else None

    result: dict[str, Any] = {
        "corruption": corruption.get("name", "unnamed"),
        "token_source": corruption.get("token_source", "unknown"),
        "n_samples": n_samples,
        "nmse_linear_mean": float(np.mean(fusion_arr)),
        "nmse_db": _nmse_db(float(np.mean(fusion_arr))),
        "nmse_linear_std": float(np.std(fusion_arr, ddof=1)),
    }

    if tf_arr is not None and len(tf_arr) > 0:
        R = fusion_arr - tf_arr  # positive = fusion WORSE than TF-only
        result["tf_only_nmse_db"] = _nmse_db(float(np.mean(tf_arr)))
        result["mean_regret_R"] = float(np.mean(R))
        result["harm_rate"] = float(np.mean(R > 0))
        result["worst10_nmse"] = float(np.mean(np.sort(fusion_arr)[-max(1, len(fusion_arr)//10):]))
        result["max_degradation"] = float(np.max(R))
        # AUC: trapezoid over sorted R
        r_sorted = np.sort(R)
        result["auc_degradation"] = float(np.trapezoid(r_sorted) / len(r_sorted))

    if gate_mean_vals:
        gm = np.array(gate_mean_vals)
        result["gate_mean"] = float(np.mean(gm))
        result["gate_p10"] = float(np.percentile(gm, 10))
        result["gate_p50"] = float(np.percentile(gm, 50))
        result["gate_p90"] = float(np.percentile(gm, 90))
        # Correlation: gate_mean vs -NMSE_H_phys
        if h_phys_nmse:
            hp = np.array(h_phys_nmse)
            neg_hp_nmse = -10.0 * np.log10(np.clip(hp, 1e-30, None))
            corr = np.corrcoef(gm, neg_hp_nmse)[0, 1]
            result["gate_vs_neg_nmse_hphys_corr"] = float(corr) if np.isfinite(corr) else float("nan")

    # Null error decomposition (component powers from model diagnostics)
    if decomp_tf_nmse:
        result["decomp_tf_power_mean"] = float(np.mean(decomp_tf_nmse))
        result["decomp_fused_power_mean"] = float(np.mean(decomp_fused_nmse))
        result["decomp_delta_power_mean"] = float(np.mean(decomp_delta_nmse))
        result["decomp_phys_mix_mean"] = float(np.mean(decomp_phys_mix))

    if h_phys_nmse:
        hp = np.array(h_phys_nmse)
        result["h_phys_nmse_db"] = _nmse_db(float(np.mean(hp)))

    return result


# ===========================================================================
# Sweep runner
# ===========================================================================


def run_corruption_sweep(
    model: torch.nn.Module,
    tf_only_model: torch.nn.Module | None,
    dataset: SyntheticOFDMISACDataset,
    device: torch.device,
    ofdm: OFDMConfig,
    channel: ChannelConfig,
    corruption_specs: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run a full sweep of corruption specifications."""
    results = []
    for i, spec in enumerate(corruption_specs):
        name = spec.get("name", f"corruption_{i}")
        print(f"  [{i+1}/{len(corruption_specs)}] {name} ...", end=" ", flush=True)
        result = evaluate_corrupted(
            model, dataset, device, ofdm, channel,
            corruption=spec,
            rng_seed=100 + i,
            tf_only_model=tf_only_model,
        )
        results.append(result)
        print(f"NMSE={result['nmse_db']:+.2f} dB", end="")
        if "harm_rate" in result:
            print(f", harm_rate={result['harm_rate']:.3f}", end="")
        if "gate_mean" in result:
            print(f", gate={result['gate_mean']:.3f}", end="")
        print()
    return results


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2-A: Corruption failure audit")
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Directory containing pre-trained .pt checkpoints",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gate2_corruption",
    )
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of test samples (default 200 for smoke)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run only the 4 smoke-test perturbation types")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- Load model checkpoint metadata ---
    results_json = args.model_dir / "gate1_results.json"
    if results_json.exists():
        with open(results_json) as f:
            prev_results = json.load(f)
        config = prev_results["config"]
    else:
        print(f"WARNING: {results_json} not found — using default K=4 config")
        config = {
            "ofdm": {"num_subcarriers": 64, "num_symbols": 14,
                     "subcarrier_spacing_hz": 120e3, "carrier_frequency_hz": 28e9,
                     "cp_ratio": 0.07},
            "channel": {"num_paths": 4, "max_delay_bins": 12.0,
                        "max_abs_doppler_bins": 3.0, "fractional_delay": True,
                        "fractional_doppler": True, "exponential_power_decay": 0.25},
            "dataset": {"pilot_density": 0.125, "pilot_pattern": "random",
                        "snr_min_db": 10.0, "snr_max_db": 10.0,
                        "max_paths": 8, "token_version": 2},
        }

    ofdm = OFDMConfig(**config["ofdm"])
    channel = ChannelConfig(**config["channel"])
    data_cfg = config["dataset"]
    training_cfg = config.get("training", {})
    hidden_dim = int(training_cfg.get("hidden_dim", 48))

    # --- Build test dataset ---
    test_cfg = DatasetConfig(
        size=args.samples,
        snr_min_db=float(data_cfg["snr_min_db"]),
        snr_max_db=float(data_cfg["snr_max_db"]),
        pilot_density=float(data_cfg["pilot_density"]),
        pilot_pattern=str(data_cfg["pilot_pattern"]),
        max_paths=int(data_cfg["max_paths"]),
        seed=config.get("seed", 2036) + 50000,
        token_version=int(data_cfg.get("token_version", 2)),
        token_source=str(data_cfg.get("token_source", "oracle")),
        token_refine=str(data_cfg.get("token_refine", "")),
        token_vp_rounds=int(data_cfg.get("token_vp_rounds", 3)),
        token_vp_probes=int(data_cfg.get("token_vp_probes", 8)),
        token_vp_fast=bool(data_cfg.get("token_vp_fast", False)),
        dd_oversample_delay=int(data_cfg.get("dd_oversample_delay", 2)),
        dd_oversample_doppler=int(data_cfg.get("dd_oversample_doppler", 4)),
        detector_method=str(data_cfg.get("detector_method", "nms")),
    )
    test_dataset = SyntheticOFDMISACDataset(ofdm, channel, test_cfg)

    # --- Load models ---
    # Detect quality-gate from training config
    use_quality_gate = bool(training_cfg.get("use_quality_gate", False))
    use_path_stats = bool(training_cfg.get("use_path_stats", False))
    gate_kernel_size = int(training_cfg.get("gate_kernel_size", 1))
    zero_init_residual = bool(training_cfg.get("zero_init_residual", True))
    # Try estimated_residual first, then physics_residual
    model = PhysicalResidualEstimator(
        hidden_dim=hidden_dim,
        num_subcarriers=ofdm.num_subcarriers,
        num_symbols=ofdm.num_symbols,
        use_quality_gate=use_quality_gate,
        use_path_stats=use_path_stats,
        gate_kernel_size=gate_kernel_size,
        zero_init_residual=zero_init_residual,
    ).to(device)

    model_pt = args.model_dir / "estimated_residual_seed0.pt"
    if not model_pt.exists():
        model_pt = args.model_dir / "physics_residual_seed0.pt"
    if not model_pt.exists():
        # Try without seed suffix
        candidates = sorted(args.model_dir.glob("*residual*.pt"))
        if candidates:
            model_pt = candidates[0]
        else:
            raise FileNotFoundError(f"No residual model found in {args.model_dir}")

    print(f"Loading model: {model_pt}")
    model.load_state_dict(torch.load(model_pt, map_location=device, weights_only=True))
    model.eval()

    # Load TF-only model
    tf_model = TFOnlyEstimator(hidden_dim=hidden_dim).to(device)
    tf_pt = args.model_dir / "tf_only_seed0.pt"
    if tf_pt.exists():
        print(f"Loading TF-only: {tf_pt}")
        tf_model.load_state_dict(torch.load(tf_pt, map_location=device, weights_only=True))
        tf_model.eval()
    else:
        print("WARNING: TF-only model not found — safety baseline unavailable")
        tf_model = None

    # ===================================================================
    # Corruption specifications
    # ===================================================================
    if args.smoke_only:
        corruption_specs = _build_smoke_specs(channel)
    else:
        corruption_specs = _build_full_specs(channel)

    print(f"\nRunning {len(corruption_specs)} corruption specifications "
          f"({len(test_dataset)} samples each)")
    print(f"{'='*60}")

    all_results = []
    # Split into oracle and estimated chains
    for chain_token_source in ["oracle", "estimated"]:
        print(f"\n--- Chain: {chain_token_source.upper()} tokens ---")
        specs = [{**s, "token_source": chain_token_source} for s in corruption_specs]
        chain_results = run_corruption_sweep(
            model, tf_model, test_dataset, device, ofdm, channel,
            specs, args.output_dir,
        )
        all_results.extend(chain_results)

    # --- Output ---
    output = {
        "device": str(device),
        "model_path": str(model_pt),
        "config": config,
        "n_samples": args.samples,
        "results": all_results,
    }

    out_path = args.output_dir / "gate2_corruption_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults → {out_path}")

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("Gate 2-A Corruption Summary")
    print(f"{'='*80}")
    print(f"{'Corruption':<35s} {'Chain':<10s} {'NMSE(dB)':>8s} {'Harm%':>7s} {'Gate':>7s} {'R_mean':>8s}")
    print("-" * 75)
    for r in all_results:
        name = r["corruption"][:34]
        chain = r["token_source"][:9]
        nmse = f"{r['nmse_db']:+.2f}"
        harm = f"{r.get('harm_rate', float('nan'))*100:.1f}%" if 'harm_rate' in r else "N/A"
        gate = f"{r['gate_mean']:.3f}" if 'gate_mean' in r else "N/A"
        r_mean = f"{r.get('mean_regret_R', float('nan')):+.4f}" if 'mean_regret_R' in r else "N/A"
        print(f"{name:<35s} {chain:<10s} {nmse:>8s} {harm:>7s} {gate:>7s} {r_mean:>8s}")


# ===========================================================================
# Corruption spec builders
# ===========================================================================


def _build_smoke_specs(channel: ChannelConfig) -> list[dict[str, Any]]:
    """4 perturbation types for the minimal smoke test."""
    return [
        # --- Baseline (no corruption) ---
        {"name": "clean", "null_all": False},

        # --- Token dropout ---
        {"name": "dropout_0.25", "drop_probability": 0.25},
        {"name": "dropout_0.50", "drop_probability": 0.50},
        {"name": "dropout_0.75", "drop_probability": 0.75},
        {"name": "dropout_1.00", "drop_probability": 1.00},

        # --- Coherent false paths ---
        {"name": "coherent_false_1", "coherent_false_paths": 1},
        {"name": "coherent_false_2", "coherent_false_paths": 2},
        {"name": "coherent_false_4", "coherent_false_paths": 4},

        # --- Phase error ---
        {"name": "phase_pi8", "gain_phase_rad": math.pi / 8},
        {"name": "phase_pi4", "gain_phase_rad": math.pi / 4},
        {"name": "phase_pi2", "gain_phase_rad": math.pi / 2},
        {"name": "phase_pi", "gain_phase_rad": math.pi},

        # --- Joint location jitter ---
        {"name": "jitter_0.1", "delay_jitter_std": 0.1, "doppler_jitter_std": 0.1},
        {"name": "jitter_0.5", "delay_jitter_std": 0.5, "doppler_jitter_std": 0.5},
        {"name": "jitter_1.0", "delay_jitter_std": 1.0, "doppler_jitter_std": 1.0},
        {"name": "jitter_2.0", "delay_jitter_std": 2.0, "doppler_jitter_std": 2.0},

        # --- Null all ---
        {"name": "null_all", "null_all": True},
    ]


def _build_full_specs(channel: ChannelConfig) -> list[dict[str, Any]]:
    """Full perturbation matrix for Gate 2-A audit."""
    specs = [{"name": "clean", "null_all": False}]

    # Token dropout
    for p in [0.25, 0.50, 0.75, 1.0]:
        specs.append({"name": f"dropout_{p}", "drop_probability": p})

    # Coherent false paths
    for n in [1, 2, 4]:
        specs.append({"name": f"coherent_false_{n}", "coherent_false_paths": n})

    # Random false paths
    for n in [1, 2, 4]:
        specs.append({"name": f"random_false_{n}", "random_false_paths": n})

    # Phase error
    for phi_name, phi_val in [("pi8", math.pi/8), ("pi4", math.pi/4),
                               ("pi2", math.pi/2), ("pi", math.pi)]:
        specs.append({"name": f"phase_{phi_name}", "gain_phase_rad": phi_val})

    # Magnitude error
    for a in [0.5, 0.75, 1.25, 1.5, 2.0]:
        specs.append({"name": f"mag_{a}", "gain_magnitude_factor": a})

    # Location jitter
    for s in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
        specs.append({"name": f"jitter_delay_{s}",
                       "delay_jitter_std": s, "doppler_jitter_std": 0.0})
        specs.append({"name": f"jitter_doppler_{s}",
                       "delay_jitter_std": 0.0, "doppler_jitter_std": s})
        specs.append({"name": f"jitter_joint_{s}",
                       "delay_jitter_std": s, "doppler_jitter_std": s})

    # Common bias
    for b in [0.5, 1.0, 2.0]:
        specs.append({"name": f"bias_delay_{b}", "common_bias_delay": b})
        specs.append({"name": f"bias_doppler_{b}", "common_bias_doppler": b})

    # Permutation
    specs.append({"name": "permute", "permute_paths": True})

    # Null all
    specs.append({"name": "null_all", "null_all": True})

    return specs


if __name__ == "__main__":
    main()
