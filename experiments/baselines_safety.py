#!/usr/bin/env python3
"""Evaluate mechanism-gradient safety baselines against the learned spatial gate.

Runs on a pre-trained PhysicalResidualEstimator model:
  1. Fixed blend           (λ sweep on val, no training)
  2. Hard discrepancy switch (θ sweep on val, no training)
  3. Logistic quality gate  (light training ~few sec, no GPU)
  4. Hold-out pilot selector (no training)

All baselines use the SAME H_phys and H_Tf from the frozen model.
Output: comparison table showing NMSE, safety, and complexity.
"""

from __future__ import annotations

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
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ld3.baselines import (
    LogisticQualityGate,
    _quality_features,
    evaluate_fixed_blend,
    evaluate_hard_switch,
    evaluate_holdout_pilot_selector,
    optimise_fixed_blend,
    optimise_hard_switch,
)
from ld3.channel import ChannelConfig, OFDMConfig, generate_path_set, synthesize_tf_channel
from ld3.dataset import DatasetConfig, SyntheticOFDMISACDataset
from ld3.metrics import nmse_numpy, nmse_torch
from ld3.models import PhysicalResidualEstimator, TFOnlyEstimator
from ld3.oracle import corrupt_tokens, estimated_path_tokens_v2, oracle_path_tokens_v2
from ld3.pilots import generate_noise_grid, make_pilot_mask, observe_pilots


def _nmse_db(linear: float) -> float:
    return float(10.0 * math.log10(max(linear, 1e-30)))


# ===========================================================================
# Data extraction: run frozen model, collect H_phys, H_tf, tokens
# ===========================================================================


def extract_components(
    model: PhysicalResidualEstimator,
    tf_only_model: TFOnlyEstimator | None,
    dataset: SyntheticOFDMISACDataset,
    device: torch.device,
    ofdm: OFDMConfig,
    channel: ChannelConfig,
) -> dict[str, np.ndarray]:
    """Run frozen model on dataset, return all component arrays for baselines.

    Returns dict with keys:
      H_true:    [N, N_sc, N_sym] complex128
      H_phys:    [N, N_sc, N_sym] complex128  (from model's physics branch)
      H_tf:      [N, N_sc, N_sym] complex128  (from model's TF branch)
      H_out:     [N, N_sc, N_sym] complex128  (final model output)
      H_tf_only: [N, N_sc, N_sym] complex128  (standalone TF-only)
      tokens:    [N, L, 9] float32
      valid:     [N, L] bool
      snr_db:    [N] float32
      pilot_observations: [N, N_sc, N_sym] complex128
      pilot_mask:         [N, N_sc, N_sym] bool
    """
    model.eval()
    if tf_only_model is not None:
        tf_only_model.eval()

    n = len(dataset)
    H_true_list, H_phys_list, H_tf_list, H_out_list = [], [], [], []
    H_tf_only_list, tokens_list, valid_list, snr_list = [], [], [], []
    obs_list, mask_list = [], []

    with torch.no_grad():
        for idx in range(n):
            sample = dataset[idx]
            tf_input = sample["tf_input"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            pt = sample["path_tokens"].unsqueeze(0).to(device)
            pv = sample["path_valid"].unsqueeze(0).to(device)

            # Model forward with components
            output, diag = model(tf_input, pt, pv, return_components=True)
            H_phys_t = diag["H_phys"]  # [1, 2, N_sc, N_sym]
            H_tf_t = diag["H_tf"]      # [1, 2, N_sc, N_sym]

            # Convert to complex numpy
            def _to_complex(t: torch.Tensor) -> np.ndarray:
                t = t.squeeze(0).cpu().numpy()  # [2, N_sc, N_sym]
                return (t[0] + 1j * t[1]).astype(np.complex128)

            H_true_list.append(_to_complex(target))
            H_phys_list.append(_to_complex(H_phys_t))
            H_tf_list.append(_to_complex(H_tf_t))
            H_out_list.append(_to_complex(output))
            tokens_list.append(sample["path_tokens"].numpy())
            valid_list.append(sample["path_valid"].numpy())
            snr_list.append(float(sample["snr_db"].item()))

            # Standalone TF-only
            if tf_only_model is not None:
                tf_out = tf_only_model(tf_input)
                H_tf_only_list.append(_to_complex(tf_out))

            # Pilot observations for hold-out selector
            sample_rng = np.random.default_rng([dataset.cfg.seed, idx])
            paths = generate_path_set(ofdm, channel, sample_rng)
            truth_c = synthesize_tf_channel(ofdm, paths)
            snr_db = float(sample["snr_db"].item())
            mask_np = make_pilot_mask(
                ofdm.num_subcarriers, ofdm.num_symbols,
                dataset.cfg.pilot_density, sample_rng, dataset.cfg.pilot_pattern,
            )
            signal_power = float(np.mean(np.abs(truth_c) ** 2))
            noise_grid, noise_var = generate_noise_grid(
                truth_c.shape, signal_power, snr_db, sample_rng,
            )
            observed_np, _ = observe_pilots(
                truth_c, mask_np, snr_db, sample_rng,
                noise_grid=noise_grid, noise_var=noise_var,
            )
            obs_list.append(observed_np)
            mask_list.append(mask_np)

    result = {
        "H_true": np.array(H_true_list),
        "H_phys": np.array(H_phys_list),
        "H_tf": np.array(H_tf_list),
        "H_out": np.array(H_out_list),
        "tokens": np.array(tokens_list), "valid": np.array(valid_list),
        "snr_db": np.array(snr_list),
        "pilot_observations": np.array(obs_list),
        "pilot_mask": np.array(mask_list),
    }
    if H_tf_only_list:
        result["H_tf_only"] = np.array(H_tf_only_list)
    return result


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate safety baselines")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "baselines_safety")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--val-frac", type=float, default=0.3,
                        help="Fraction of samples for validation (blend/switch optimisation)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- Load config ---
    results_json = args.model_dir / "gate1_results.json"
    with open(results_json) as f:
        prev_results = json.load(f)
    config = prev_results["config"]

    ofdm = OFDMConfig(**config["ofdm"])
    channel = ChannelConfig(**config["channel"])
    data_cfg = config["dataset"]
    training_cfg = config.get("training", {})
    hidden_dim = int(training_cfg.get("hidden_dim", 48))
    use_quality_gate = bool(training_cfg.get("use_quality_gate", False))

    # --- Dataset ---
    test_cfg = DatasetConfig(
        size=args.samples,
        snr_min_db=float(data_cfg["snr_min_db"]),
        snr_max_db=float(data_cfg["snr_max_db"]),
        pilot_density=float(data_cfg["pilot_density"]),
        pilot_pattern=str(data_cfg["pilot_pattern"]),
        max_paths=int(data_cfg["max_paths"]),
        seed=config.get("seed", 2036) + 60000,
        token_version=int(data_cfg.get("token_version", 2)),
        token_source=str(data_cfg.get("token_source", "oracle")),
    )
    test_dataset = SyntheticOFDMISACDataset(ofdm, channel, test_cfg)

    # --- Load models ---
    model = PhysicalResidualEstimator(
        hidden_dim=hidden_dim, num_subcarriers=ofdm.num_subcarriers,
        num_symbols=ofdm.num_symbols, use_quality_gate=use_quality_gate,
    ).to(device)
    model_pt = args.model_dir / "physics_residual_seed0.pt"
    model.load_state_dict(torch.load(model_pt, map_location=device, weights_only=True))
    model.eval()

    tf_model = TFOnlyEstimator(hidden_dim=hidden_dim).to(device)
    tf_pt = args.model_dir / "tf_only_seed0.pt"
    if tf_pt.exists():
        tf_model.load_state_dict(torch.load(tf_pt, map_location=device, weights_only=True))
        tf_model.eval()
    else:
        tf_model = None

    # --- Extract components ---
    print(f"Extracting components from {args.samples} samples...")
    comps = extract_components(model, tf_model, test_dataset, device, ofdm, channel)
    H_true, H_phys, H_tf, H_out = comps["H_true"], comps["H_phys"], comps["H_tf"], comps["H_out"]
    tokens, valid = comps["tokens"], comps["valid"]
    obs, mask = comps["pilot_observations"], comps["pilot_mask"]
    H_tf_only = comps.get("H_tf_only")

    n_total = len(H_true)
    n_val = int(n_total * args.val_frac)
    n_test = n_total - n_val

    # Split: first n_val for optimisation, rest for test
    val_sl = slice(0, n_val)
    test_sl = slice(n_val, n_total)

    # ===================================================================
    # Baselines
    # ===================================================================
    results: dict[str, Any] = {
        "config": config, "n_samples": args.samples,
        "n_val": n_val, "n_test": n_test,
    }

    print(f"\n=== Safety Baseline Comparison ({n_test} test samples) ===\n")

    # --- Model NMSEs (reference) ---
    model_nmse_test = np.mean([nmse_numpy(H_out[i], H_true[i]) for i in range(test_sl.start, test_sl.stop)])
    results["spatial_quality_gate_nmse_db"] = _nmse_db(model_nmse_test)

    tf_only_nmse_test = None
    if H_tf_only is not None:
        tf_only_nmse_test = np.mean([nmse_numpy(H_tf_only[i], H_true[i]) for i in range(test_sl.start, test_sl.stop)])
        results["tf_only_nmse_db"] = _nmse_db(tf_only_nmse_test)

    phys_only_nmse_test = np.mean([nmse_numpy(H_phys[i], H_true[i]) for i in range(test_sl.start, test_sl.stop)])
    results["phys_only_nmse_db"] = _nmse_db(phys_only_nmse_test)

    print(f"  Reference:  TF-only={results.get('tf_only_nmse_db', 'N/A'):>6s}  "
          f"H_phys-only={results['phys_only_nmse_db']:+.2f} dB  "
          f"SpatialGate={results['spatial_quality_gate_nmse_db']:+.2f} dB")
    print()

    # --- 1. Fixed blend ---
    print("1. Fixed blend (λ sweep on val)...")
    best_lam, fb_info = optimise_fixed_blend(
        H_phys[val_sl], H_tf[val_sl], H_true[val_sl],
    )
    fb_test = evaluate_fixed_blend(H_phys[test_sl], H_tf[test_sl], H_true[test_sl], best_lam)
    results["fixed_blend"] = {"best_lam": best_lam, "val": fb_info, "test": fb_test}
    print(f"   best λ={best_lam:.2f}  test NMSE={fb_test['nmse_db']:+.2f} dB")

    # --- 2. Hard discrepancy switch ---
    print("2. Hard discrepancy switch (θ sweep on val)...")
    best_theta, hs_info = optimise_hard_switch(
        H_phys[val_sl], H_tf[val_sl], H_true[val_sl],
    )
    hs_test = evaluate_hard_switch(H_phys[test_sl], H_tf[test_sl], H_true[test_sl], best_theta)
    results["hard_switch"] = {"best_theta": best_theta, "val": hs_info, "test": hs_test}
    print(f"   best θ={best_theta:.4f}  test NMSE={hs_test['nmse_db']:+.2f} dB  "
          f"phys_sel={hs_test['frac_phys_selected']:.2f}")

    # --- 3. Logistic quality gate ---
    print("3. Logistic quality gate (train on val)...")
    X_val = _quality_features(H_phys[val_sl], H_tf[val_sl], tokens[val_sl], valid[val_sl])
    lqg = LogisticQualityGate()
    lqg.fit(X_val, H_phys[val_sl], H_tf[val_sl], H_true[val_sl])
    X_test = _quality_features(H_phys[test_sl], H_tf[test_sl], tokens[test_sl], valid[test_sl])
    lqg_test = lqg.evaluate(X_test, H_phys[test_sl], H_tf[test_sl], H_true[test_sl])
    results["logistic_quality_gate"] = lqg_test
    print(f"   w={lqg.w.tolist()} b={lqg.b:.4f}  test NMSE={lqg_test['nmse_db']:+.2f} dB  "
          f"q_mean={lqg_test['q_mean']:.3f}")

    # --- 4. Hold-out pilot selector ---
    print("4. Hold-out pilot selector (no training)...")
    ho_test = evaluate_holdout_pilot_selector(
        obs[test_sl], mask[test_sl], H_phys[test_sl], H_tf[test_sl], H_true[test_sl],
    )
    results["holdout_pilot_selector"] = ho_test
    print(f"   test NMSE={ho_test['nmse_db']:+.2f} dB  "
          f"phys_sel={ho_test['frac_phys_selected']:.2f}")

    # ===================================================================
    # Summary table
    # ===================================================================
    print(f"\n{'='*75}")
    print(f"{'Method':<28s} {'NMSE(dB)':>8s} {'Complexity':>12s} {'Trained':>8s}")
    print(f"{'-'*75}")
    rows = [
        ("H_phys-only", _nmse_db(phys_only_nmse_test), "0 params", "No"),
        ("TF-only (standalone)", results.get("tf_only_nmse_db", "N/A"), "CNN", "Yes"),
        ("Fixed blend", fb_test["nmse_db"], f"λ={best_lam:.2f}", "No"),
        ("Hard switch", hs_test["nmse_db"], f"θ={best_theta:.4f}", "No"),
        ("Logistic quality gate", lqg_test["nmse_db"], "3 params", "Light"),
        ("Hold-out pilot select", ho_test["nmse_db"], "Pilot split", "No"),
        ("**Spatial quality gate**", results["spatial_quality_gate_nmse_db"], "CNN gate", "Yes"),
    ]
    for name, nmse, compl, trained in rows:
        nmse_str = f"{nmse:+.2f} dB" if isinstance(nmse, float) else str(nmse)
        print(f"{name:<28s} {nmse_str:>8s} {compl:>12s} {trained:>8s}")

    # --- Save ---
    out_path = args.output_dir / "baselines_safety.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
