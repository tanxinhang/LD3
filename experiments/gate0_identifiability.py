#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ld3.channel import generate_path_set, synthesize_tf_channel
from ld3.config import ChannelConfig, OFDMConfig
from ld3.dd_estimation import (
    build_dd_grid,
    detect_paths_nms,
    identifiability_metrics,
    masked_matched_filter_map,
)
from ld3.pilots import make_pilot_mask, observe_pilots


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mean_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else float("nan")


def run(config: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    trials = int(config["trials"])
    ofdm = OFDMConfig(**config["ofdm"])
    channel = ChannelConfig(**config["channel"])
    estimator = config["estimator"]
    grid = build_dd_grid(
        ofdm.num_subcarriers,
        ofdm.num_symbols,
        channel.max_delay_bins,
        channel.max_abs_doppler_bins,
        int(estimator["oversample_delay"]),
        int(estimator["oversample_doppler"]),
    )

    trial_rows: list[dict[str, Any]] = []
    for pattern_index, pattern in enumerate(config["sweep"]["pilot_pattern"]):
        for density_index, density in enumerate(config["sweep"]["pilot_density"]):
            for snr_index, snr_db in enumerate(config["sweep"]["snr_db"]):
                for trial in range(trials):
                    rng = np.random.default_rng(
                        [seed, pattern_index, density_index, snr_index, trial]
                    )
                    paths = generate_path_set(ofdm, channel, rng)
                    truth = synthesize_tf_channel(ofdm, paths)
                    mask = make_pilot_mask(
                        ofdm.num_subcarriers,
                        ofdm.num_symbols,
                        float(density),
                        rng,
                        str(pattern),
                    )
                    observed, _ = observe_pilots(truth, mask, float(snr_db), rng)
                    score_map, gain_map = masked_matched_filter_map(observed, mask, grid)
                    estimated = detect_paths_nms(
                        score_map,
                        gain_map,
                        grid,
                        num_paths=channel.num_paths,
                        delay_radius=int(estimator["nms_delay_radius"]),
                        doppler_radius=int(estimator["nms_doppler_radius"]),
                        relative_threshold=float(estimator["relative_threshold"]),
                    )
                    metrics = identifiability_metrics(
                        paths.delay_bins,
                        paths.doppler_bins,
                        paths.power,
                        estimated,
                        float(estimator["delay_tolerance_bins"]),
                        float(estimator["doppler_tolerance_bins"]),
                    )
                    trial_rows.append(
                        {
                            "pilot_pattern": pattern,
                            "pilot_density": float(density),
                            "snr_db": float(snr_db),
                            "trial": trial,
                            "num_pilots": int(mask.sum()),
                            **metrics,
                        }
                    )

    trial_path = output_dir / "gate0_trials.csv"
    with trial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)

    group_keys = sorted(
        {
            (row["pilot_pattern"], row["pilot_density"], row["snr_db"])
            for row in trial_rows
        }
    )
    metric_names = [
        "path_recall",
        "path_precision",
        "power_recovery",
        "delay_rmse_bins",
        "doppler_rmse_bins",
        "num_estimated",
    ]
    summary_rows: list[dict[str, Any]] = []
    for pattern, density, snr_db in group_keys:
        selected = [
            row
            for row in trial_rows
            if row["pilot_pattern"] == pattern
            and row["pilot_density"] == density
            and row["snr_db"] == snr_db
        ]
        summary = {
            "pilot_pattern": pattern,
            "pilot_density": density,
            "snr_db": snr_db,
            "trials": len(selected),
            "num_pilots": mean_or_nan([float(row["num_pilots"]) for row in selected]),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = mean_or_nan(values)
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            summary[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
        summary_rows.append(summary)

    summary_path = output_dir / "gate0_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config,
                "trial_csv": str(trial_path),
                "summary_csv": str(summary_path),
                "gate_pass_reference": {
                    "power_recovery": 0.8,
                    "note": "A research decision threshold, not a universal theorem.",
                },
            },
            handle,
            indent=2,
        )

    for pattern in config["sweep"]["pilot_pattern"]:
        plt.figure(figsize=(7.0, 4.5))
        for density in config["sweep"]["pilot_density"]:
            rows = sorted(
                [
                    row
                    for row in summary_rows
                    if row["pilot_pattern"] == pattern
                    and row["pilot_density"] == float(density)
                ],
                key=lambda item: item["snr_db"],
            )
            plt.plot(
                [row["snr_db"] for row in rows],
                [row["power_recovery_mean"] for row in rows],
                marker="o",
                label=f"pilot density={density}",
            )
        plt.axhline(0.8, linestyle="--", linewidth=1.0, label="Gate reference=0.8")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Recovered true-path power ratio")
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"gate0_power_recovery_{pattern}.png", dpi=180)
        plt.close()

    print(f"Gate 0 complete: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate 0 DD identifiability audit")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "gate0.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gate0",
    )
    parser.add_argument("--trials", type=int, default=None, help="Override config trials")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.trials is not None:
        config["trials"] = args.trials
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
