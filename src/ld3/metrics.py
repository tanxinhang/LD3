from __future__ import annotations

import numpy as np
import torch


def nmse_numpy(estimate: np.ndarray, truth: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(estimate - truth) ** 2))
    denominator = float(np.sum(np.abs(truth) ** 2))
    return numerator / max(denominator, np.finfo(float).eps)


def nmse_torch(estimate_ri: torch.Tensor, truth_ri: torch.Tensor) -> torch.Tensor:
    error = (estimate_ri - truth_ri).square().sum(dim=(1, 2, 3))
    power = truth_ri.square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
    return error / power


def nmse_loss(estimate_ri: torch.Tensor, truth_ri: torch.Tensor) -> torch.Tensor:
    return nmse_torch(estimate_ri, truth_ri).mean()
