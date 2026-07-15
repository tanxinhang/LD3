from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class DDGrid:
    delay_bins: np.ndarray
    doppler_bins: np.ndarray


@dataclass
class EstimatedPaths:
    delay_bins: np.ndarray
    doppler_bins: np.ndarray
    scores: np.ndarray
    gains: np.ndarray


def build_dd_grid(
    num_subcarriers: int,
    num_symbols: int,
    max_delay_bins: float,
    max_abs_doppler_bins: float,
    oversample_delay: int = 2,
    oversample_doppler: int = 4,
) -> DDGrid:
    delay_count = max(2, int(np.ceil(max_delay_bins * oversample_delay)) + 1)
    doppler_count = max(3, int(np.ceil(2 * max_abs_doppler_bins * oversample_doppler)) + 1)
    return DDGrid(
        delay_bins=np.linspace(0.0, max_delay_bins, delay_count),
        doppler_bins=np.linspace(
            -max_abs_doppler_bins,
            max_abs_doppler_bins,
            doppler_count,
        ),
    )


def masked_matched_filter_map(
    pilot_observations: np.ndarray,
    pilot_mask: np.ndarray,
    grid: DDGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a mask-aware DD matched-filter map.

    Unlike zero-filling followed by a 2-D FFT, this operation only evaluates
    the phase model at truly observed pilot locations. It therefore separates
    pilot-mask ambiguity from interpolation artifacts.
    """

    if pilot_observations.shape != pilot_mask.shape:
        raise ValueError("pilot_observations and pilot_mask shapes must match")
    n_idx, m_idx = np.nonzero(pilot_mask)
    if n_idx.size == 0:
        raise ValueError("pilot mask contains no observations")
    y = pilot_observations[pilot_mask]
    n_total, m_total = pilot_mask.shape

    delay_phase = np.exp(
        -2j
        * np.pi
        * n_idx[:, None]
        * grid.delay_bins[None, :]
        / n_total
    )
    doppler_phase = np.exp(
        2j
        * np.pi
        * m_idx[:, None]
        * grid.doppler_bins[None, :]
        / m_total
    )
    dictionary = (
        delay_phase[:, :, None] * doppler_phase[:, None, :]
    ).reshape(n_idx.size, -1)
    column_norm = np.linalg.norm(dictionary, axis=0, keepdims=True)
    dictionary = dictionary / np.maximum(column_norm, np.finfo(float).eps)

    correlations = dictionary.conj().T @ y
    score_map = np.abs(correlations).reshape(
        grid.delay_bins.size, grid.doppler_bins.size
    )
    gain_map = correlations.reshape(score_map.shape) / np.sqrt(n_idx.size)
    return score_map, gain_map


def detect_paths_nms(
    score_map: np.ndarray,
    gain_map: np.ndarray,
    grid: DDGrid,
    num_paths: int,
    delay_radius: int = 2,
    doppler_radius: int = 2,
    relative_threshold: float = 0.08,
) -> EstimatedPaths:
    """Select DD peaks with local non-maximum suppression."""

    if score_map.shape != gain_map.shape:
        raise ValueError("score_map and gain_map shapes must match")
    work = score_map.copy()
    maximum = float(np.max(work))
    delays: list[float] = []
    dopplers: list[float] = []
    scores: list[float] = []
    gains: list[complex] = []

    for _ in range(num_paths):
        flat_idx = int(np.argmax(work))
        i, j = np.unravel_index(flat_idx, work.shape)
        value = float(work[i, j])
        if value < relative_threshold * max(maximum, np.finfo(float).eps):
            break
        delays.append(float(grid.delay_bins[i]))
        dopplers.append(float(grid.doppler_bins[j]))
        scores.append(value / max(maximum, np.finfo(float).eps))
        gains.append(complex(gain_map[i, j]))

        i0, i1 = max(0, i - delay_radius), min(work.shape[0], i + delay_radius + 1)
        j0, j1 = max(0, j - doppler_radius), min(work.shape[1], j + doppler_radius + 1)
        work[i0:i1, j0:j1] = -np.inf

    return EstimatedPaths(
        delay_bins=np.asarray(delays, dtype=np.float64),
        doppler_bins=np.asarray(dopplers, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        gains=np.asarray(gains, dtype=np.complex128),
    )


def match_paths(
    true_delay: np.ndarray,
    true_doppler: np.ndarray,
    est: EstimatedPaths,
    delay_tolerance: float,
    doppler_tolerance: float,
) -> list[tuple[int, int]]:
    """Hungarian matching under normalized DD distance and hard tolerances."""

    if len(true_delay) == 0 or len(est.delay_bins) == 0:
        return []
    delay_error = np.abs(true_delay[:, None] - est.delay_bins[None, :])
    doppler_error = np.abs(true_doppler[:, None] - est.doppler_bins[None, :])
    cost = np.sqrt(
        (delay_error / max(delay_tolerance, 1e-12)) ** 2
        + (doppler_error / max(doppler_tolerance, 1e-12)) ** 2
    )
    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if (
            delay_error[r, c] <= delay_tolerance
            and doppler_error[r, c] <= doppler_tolerance
        ):
            matches.append((r, c))
    return matches


def identifiability_metrics(
    true_delay: np.ndarray,
    true_doppler: np.ndarray,
    true_power: np.ndarray,
    est: EstimatedPaths,
    delay_tolerance: float,
    doppler_tolerance: float,
) -> dict[str, float]:
    matches = match_paths(
        true_delay,
        true_doppler,
        est,
        delay_tolerance,
        doppler_tolerance,
    )
    matched_true = {r for r, _ in matches}
    recall = len(matches) / max(len(true_delay), 1)
    precision = len(matches) / max(len(est.delay_bins), 1)
    recovered_power = float(np.sum([true_power[i] for i in matched_true]))
    total_power = float(np.sum(true_power))

    if matches:
        delay_rmse = float(
            np.sqrt(
                np.mean(
                    [
                        (true_delay[r] - est.delay_bins[c]) ** 2
                        for r, c in matches
                    ]
                )
            )
        )
        doppler_rmse = float(
            np.sqrt(
                np.mean(
                    [
                        (true_doppler[r] - est.doppler_bins[c]) ** 2
                        for r, c in matches
                    ]
                )
            )
        )
    else:
        delay_rmse = float("nan")
        doppler_rmse = float("nan")

    return {
        "path_recall": recall,
        "path_precision": precision,
        "power_recovery": recovered_power / max(total_power, np.finfo(float).eps),
        "delay_rmse_bins": delay_rmse,
        "doppler_rmse_bins": doppler_rmse,
        "num_estimated": float(len(est.delay_bins)),
    }
