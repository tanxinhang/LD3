from __future__ import annotations

import numpy as np

from .channel import PathSet, synthesize_tf_channel
from .config import OFDMConfig


def oracle_path_tokens(paths: PathSet, max_paths: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-width path tokens and a validity mask.

    Token fields: delay_bin, doppler_bin, normalized power, confidence,
    sigma_delay, sigma_doppler, communication relevance.
    """

    tokens = np.zeros((max_paths, 7), dtype=np.float32)
    valid = np.zeros(max_paths, dtype=bool)
    count = min(max_paths, len(paths.gains))
    order = np.argsort(paths.power)[::-1][:count]
    tokens[:count, 0] = paths.delay_bins[order]
    tokens[:count, 1] = paths.doppler_bins[order]
    tokens[:count, 2] = paths.normalized_power()[order]
    tokens[:count, 3] = 1.0
    tokens[:count, 4] = 0.0
    tokens[:count, 5] = 0.0
    tokens[:count, 6] = 1.0
    valid[:count] = True
    return tokens, valid


def oracle_parametric_reconstruction(ofdm: OFDMConfig, paths: PathSet) -> np.ndarray:
    """Perfect path-parameter reconstruction used as a diagnostic upper bound."""

    return synthesize_tf_channel(ofdm, paths)


def perturb_tokens(
    tokens: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    delay_std: float = 0.0,
    doppler_std: float = 0.0,
    false_paths: int = 0,
    miss_probability: float = 0.0,
    max_delay_bins: float = 12.0,
    max_abs_doppler_bins: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject controlled prior errors for Gate 2 experiments."""

    out = tokens.copy()
    out_valid = valid.copy()
    active = np.flatnonzero(out_valid)
    if active.size:
        out[active, 0] += rng.normal(0.0, delay_std, size=active.size)
        out[active, 1] += rng.normal(0.0, doppler_std, size=active.size)
        missed = rng.random(active.size) < miss_probability
        out_valid[active[missed]] = False
        out[active, 4] = delay_std
        out[active, 5] = doppler_std
        confidence = np.exp(-0.5 * (delay_std**2 + doppler_std**2))
        out[active, 3] = confidence

    free = np.flatnonzero(~out_valid)
    for idx in free[:false_paths]:
        out[idx, 0] = rng.uniform(0.0, max_delay_bins)
        out[idx, 1] = rng.uniform(-max_abs_doppler_bins, max_abs_doppler_bins)
        out[idx, 2] = rng.uniform(0.01, 0.1)
        out[idx, 3] = rng.uniform(0.05, 0.4)
        out[idx, 4:6] = 1.0
        out[idx, 6] = rng.uniform(0.0, 0.3)
        out_valid[idx] = True
    return out, out_valid
