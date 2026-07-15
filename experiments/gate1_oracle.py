#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ld3.config import ChannelConfig, OFDMConfig
from ld3.dataset import DatasetConfig, SyntheticOFDMISACDataset
from ld3.metrics import nmse_loss, nmse_torch
from ld3.models import PhysicsGuidedCrossAttention, TFOnlyEstimator


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_cross: bool,
    token_mode: str = "oracle",
) -> dict[str, float]:
    model.eval()
    nmse_values: list[torch.Tensor] = []
    initial_values: list[torch.Tensor] = []
    null_values: list[torch.Tensor] = []
    gate_values: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if is_cross:
                path_tokens = batch["path_tokens"]
                path_valid = batch["path_valid"]
                if token_mode == "shuffled":
                    if path_tokens.shape[0] > 1:
                        permutation = torch.roll(
                            torch.arange(path_tokens.shape[0], device=device), shifts=1
                        )
                        path_tokens = path_tokens[permutation]
                        path_valid = path_valid[permutation]
                elif token_mode == "null":
                    path_valid = torch.zeros_like(path_valid)
                elif token_mode != "oracle":
                    raise ValueError(f"unknown token_mode: {token_mode}")
                output, diagnostics = model(
                    batch["tf_input"], path_tokens, path_valid
                )
                null_values.append(diagnostics["null_attention"].mean().cpu())
                gate_values.append(diagnostics["gate"].mean().cpu())
            else:
                output = model(batch["tf_input"])
            nmse_values.append(nmse_torch(output, batch["target"]).cpu())
            initial_values.append(
                nmse_torch(batch["tf_input"][:, :2], batch["target"]).cpu()
            )
    elapsed = time.perf_counter() - start
    nmse = torch.cat(nmse_values)
    initial = torch.cat(initial_values)
    result = {
        "token_mode": token_mode,
        "nmse_linear": float(nmse.mean()),
        "nmse_db": float(10.0 * torch.log10(nmse.mean())),
        "initial_nmse_db": float(10.0 * torch.log10(initial.mean())),
        "inference_seconds": elapsed,
        "samples": int(nmse.numel()),
    }
    if null_values:
        result["mean_null_attention"] = float(torch.stack(null_values).mean())
        result["mean_gate"] = float(torch.stack(gate_values).mean())
    return result


def train_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    is_cross: bool,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if is_cross:
                output, _ = model(
                    batch["tf_input"], batch["path_tokens"], batch["path_valid"]
                )
            else:
                output = model(batch["tf_input"])
            loss = nmse_loss(output, batch["target"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": float(epoch), "train_nmse": float(np.mean(losses))}
        history.append(row)
        print(f"epoch={epoch:03d} train_nmse={row['train_nmse']:.6f}")
    return history


def run(config: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(str(config["device"]))
    ofdm = OFDMConfig(**config["ofdm"])
    channel = ChannelConfig(**config["channel"])
    data_cfg = config["dataset"]
    training = config["training"]
    torch.set_num_threads(max(1, int(training.get("num_threads", 4))))

    train_cfg = DatasetConfig(
        size=int(data_cfg["train_size"]),
        snr_min_db=float(data_cfg["snr_min_db"]),
        snr_max_db=float(data_cfg["snr_max_db"]),
        pilot_density=float(data_cfg["pilot_density"]),
        pilot_pattern=str(data_cfg["pilot_pattern"]),
        max_paths=int(data_cfg["max_paths"]),
        seed=seed,
    )
    test_cfg = replace(train_cfg, size=int(data_cfg["test_size"]), seed=seed + 10000)
    train_set = SyntheticOFDMISACDataset(ofdm, channel, train_cfg)
    test_set = SyntheticOFDMISACDataset(ofdm, channel, test_cfg)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    hidden_dim = int(training["hidden_dim"])
    models: dict[str, tuple[torch.nn.Module, bool]] = {
        "tf_only": (TFOnlyEstimator(hidden_dim), False),
        "physics_cross_attention": (
            PhysicsGuidedCrossAttention(
                hidden_dim=hidden_dim,
                token_dim=hidden_dim,
                max_delay_bins=channel.max_delay_bins,
                max_abs_doppler_bins=channel.max_abs_doppler_bins,
            ),
            True,
        ),
    }

    results: dict[str, Any] = {
        "device": str(device),
        "config": config,
        "models": {},
    }
    for name, (model, is_cross) in models.items():
        print(f"\nTraining {name} on {device}")
        history = train_model(
            model,
            train_loader,
            device,
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            is_cross=is_cross,
        )
        metrics = evaluate(model, test_loader, device, is_cross, token_mode="oracle")
        evaluations = {"oracle": metrics}
        if is_cross:
            evaluations["shuffled"] = evaluate(
                model, test_loader, device, is_cross, token_mode="shuffled"
            )
            evaluations["null"] = evaluate(
                model, test_loader, device, is_cross, token_mode="null"
            )
        results["models"][name] = {
            "history": history,
            "test": metrics,
            "evaluations": evaluations,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        print(f"{name}: test NMSE={metrics['nmse_db']:.3f} dB")

    cross_nmse = results["models"]["physics_cross_attention"]["test"]["nmse_db"]
    tf_nmse = results["models"]["tf_only"]["test"]["nmse_db"]
    results["oracle_cross_attention_gain_db"] = float(tf_nmse - cross_nmse)
    cross_evals = results["models"]["physics_cross_attention"]["evaluations"]
    results["oracle_vs_shuffled_gain_db"] = float(
        cross_evals["shuffled"]["nmse_db"] - cross_evals["oracle"]["nmse_db"]
    )
    results["oracle_vs_null_gain_db"] = float(
        cross_evals["null"]["nmse_db"] - cross_evals["oracle"]["nmse_db"]
    )
    with (output_dir / "gate1_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nGate 1 result written to {output_dir / 'gate1_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gate 1 Oracle-token models")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "gate1.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gate1"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.train_size is not None:
        config["dataset"]["train_size"] = args.train_size
    if args.test_size is not None:
        config["dataset"]["test_size"] = args.test_size
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
