from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import Optional


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


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DD matched filter
# ---------------------------------------------------------------------------


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


def build_dd_dictionary(
    pilot_mask: np.ndarray,
    grid: DDGrid,
) -> np.ndarray:
    """Build the normalised DD dictionary matrix from a pilot mask.

    Returns A of shape (N_pilots, N_delay * N_doppler) with unit-norm columns.
    """
    n_idx, m_idx = np.nonzero(pilot_mask)
    if n_idx.size == 0:
        raise ValueError("pilot mask contains no observations")
    n_total, m_total = pilot_mask.shape

    delay_phase = np.exp(
        -2j * np.pi * n_idx[:, None] * grid.delay_bins[None, :] / n_total
    )
    doppler_phase = np.exp(
        2j * np.pi * m_idx[:, None] * grid.doppler_bins[None, :] / m_total
    )
    A = (delay_phase[:, :, None] * doppler_phase[:, None, :]).reshape(n_idx.size, -1)
    col_norm = np.linalg.norm(A, axis=0, keepdims=True)
    A = A / np.maximum(col_norm, np.finfo(float).eps)
    return A


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------


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


def refine_paths_quadratic(
    est: EstimatedPaths,
    score_map: np.ndarray,
    grid: DDGrid,
) -> EstimatedPaths:
    """Refine DD path estimates via local 2D quadratic interpolation.

    Fits a 2D quadratic to the local 3×3 patch of the log-power score map
    and solves for the sub-grid maximum.  Clamps offsets to ±0.5 fine-grid
    cells to stay within the interpolation neighbourhood.

    NOTE: this only adjusts positions; gains MUST be re-estimated (e.g. via
    estimated_support_ls_reconstruction) after refinement.  The caller should
    also verify that pilot residual decreases — peak interpolation ≠ optimal
    reconstruction.
    """
    refined_delays: list[float] = []
    refined_dopplers: list[float] = []
    refined_scores: list[float] = []
    refined_gains: list[complex] = []

    n_delay = grid.delay_bins.size
    n_doppler = grid.doppler_bins.size
    d_step = grid.delay_bins[1] - grid.delay_bins[0] if n_delay > 1 else 1.0
    f_step = grid.doppler_bins[1] - grid.doppler_bins[0] if n_doppler > 1 else 1.0

    # Work in log-power for stability (avoid strong sidelobe dominance)
    score_log = np.log10(np.maximum(np.abs(score_map), np.finfo(np.float64).eps))

    for l in range(len(est.delay_bins)):
        d0 = float(est.delay_bins[l])
        f0 = float(est.doppler_bins[l])

        # Nearest grid index (delay)
        i0 = int(np.argmin(np.abs(grid.delay_bins - d0)))

        # Nearest grid index (Doppler) — handle periodic wrap
        doppler_half = int(n_doppler // 2)
        doppler_diff = np.abs(grid.doppler_bins - f0)
        # For periodic axis, consider wrap-around
        doppler_diff_wrap = np.minimum(
            doppler_diff,
            np.abs(grid.doppler_bins - f0 + 2.0 * grid.doppler_bins[-1])
            if grid.doppler_bins[0] < 0 else doppler_diff,
        )
        j0 = int(np.argmin(doppler_diff_wrap))

        # Extract 3×3 patch (clamp to grid bounds; Doppler handles wrap below)
        i_start = max(0, i0 - 1)
        i_end = min(n_delay, i0 + 2)
        j_indices = np.arange(j0 - 1, j0 + 2)
        # Wrap Doppler indices periodically
        j_wrapped = np.mod(j_indices, n_doppler)

        patch = score_log[i_start:i_end, :][:, np.arange(len(j_wrapped))]
        for pi in range(i_end - i_start):
            for pj in range(len(j_wrapped)):
                patch[pi, pj] = score_log[i_start + pi, j_wrapped[pj]]

        if patch.size < 3:
            refined_delays.append(d0)
            refined_dopplers.append(f0)
            refined_scores.append(float(est.scores[l]))
            refined_gains.append(complex(est.gains[l]))
            continue

        # Fit quadratic: f(y, x) ≈ c0 + c1*y + c2*x + c3*y² + c4*x² + c5*y*x
        # where y = (delay_idx - i0) in fine-grid cells,
        #       x = (doppler_idx - j0) in fine-grid cells.
        ny, nx = patch.shape
        yy = np.arange(i_start - i0, i_start - i0 + ny, dtype=np.float64)
        xx = np.arange(-1, -1 + nx, dtype=np.float64)  # always [-1, 0, 1] for 3×3
        X, Y = np.meshgrid(xx, yy)
        z = patch.astype(np.float64).ravel()

        A = np.column_stack([
            np.ones(X.size),
            Y.ravel(), X.ravel(),
            (Y.ravel()) ** 2, (X.ravel()) ** 2,
            Y.ravel() * X.ravel(),
        ])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            refined_delays.append(d0)
            refined_dopplers.append(f0)
            refined_scores.append(float(est.scores[l]))
            refined_gains.append(complex(est.gains[l]))
            continue

        c0, c1, c2, c3, c4, c5 = coeffs

        # Must be concave-down in both directions (local maximum)
        det = 4.0 * c3 * c4 - c5 * c5
        if abs(det) > 1e-12 and c3 < -1e-12 and c4 < -1e-12:
            dx = (c5 * c2 - 2.0 * c4 * c1) / det
            dy = (c5 * c1 - 2.0 * c3 * c2) / det
            # Clamp to ±0.5 fine-grid cells (stay within interpolation patch)
            dx = max(-0.5, min(0.5, dx))
            dy = max(-0.5, min(0.5, dy))
            d_refined = d0 + dy * d_step
            f_refined = f0 + dx * f_step
            score_refined = float(
                10.0 ** (c0 + c1 * dy + c2 * dx + c3 * dy**2 + c4 * dx**2 + c5 * dy * dx)
            )
        else:
            d_refined = d0
            f_refined = f0
            score_refined = float(est.scores[l])

        # Clip delay to physical range; wrap Doppler
        d_refined = max(0.0, min(d_refined, grid.delay_bins[-1]))
        f_half = grid.doppler_bins[-1]
        f_refined = max(-f_half, min(f_refined, f_half))

        refined_delays.append(d_refined)
        refined_dopplers.append(f_refined)
        refined_scores.append(score_refined / max(float(np.max(np.abs(score_map))), np.finfo(float).eps))
        refined_gains.append(complex(est.gains[l]))

    return EstimatedPaths(
        delay_bins=np.asarray(refined_delays, dtype=np.float64),
        doppler_bins=np.asarray(refined_dopplers, dtype=np.float64),
        scores=np.asarray(refined_scores, dtype=np.float64),
        gains=np.asarray(refined_gains, dtype=np.complex128),
    )


def _solve_spd_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b for a real SPD matrix A using manual Cholesky."""
    import math
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = float(A[i, j])
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    s = np.finfo(np.float64).eps
                L[i, j] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = float(b[i])
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]
    return x


def refine_paths_variable_projection(
    est: EstimatedPaths,
    pilot_observations: np.ndarray,
    pilot_mask: np.ndarray,
    num_subcarriers: int,
    num_symbols: int,
    n_rounds: int = 3,
    n_probes: int = 8,
    step_delay: float = 0.1,
    step_doppler: float = 0.1,
    ridge_relative: float = 1e-6,
    search_method: str = "random",
) -> tuple[EstimatedPaths, dict]:
    """Refine DD path estimates via variable projection coordinate descent.

    search_method:
      "random" — bounded random probes, monotonic acceptance gate (legacy)
      "golden" — deterministic golden-section line search per axis

    For each path, tries perturbations, re-estimates gains via LS, and
    accepts if pilot residual decreases. Runs multiple rounds of coordinate
    descent over all paths.

    Returns (refined_paths, diagnostics).
    """
    n_paths = len(est.delay_bins)
    if n_paths == 0:
        return est, {"accepted": 0, "rounds": 0}

    n_idx, m_idx = np.nonzero(pilot_mask)
    if n_idx.size == 0:
        return est, {"accepted": 0, "rounds": 0}

    y = pilot_observations[pilot_mask].copy()
    N = num_subcarriers
    M = num_symbols

    # Current best positions and gains
    d_bins = est.delay_bins.copy()
    f_bins = est.doppler_bins.copy()

    # Pilot residual for current positions
    def _compute_residual(d, f):
        K = len(d)
        # Build dictionary via broadcasting (P × K) — safe, element-wise ops
        d_arr = np.asarray(d, dtype=np.float64)
        f_arr = np.asarray(f, dtype=np.float64)
        dph = np.exp(-2j * np.pi * n_idx[:, None] * d_arr[None, :] / N)
        fph = np.exp(2j * np.pi * m_idx[:, None] * d_arr[None, :] / M)
        A = dph * fph
        col_norm = np.sqrt(np.sum(np.abs(A) ** 2, axis=0))
        A = A / np.maximum(col_norm, np.finfo(float).eps)

        # Manual Gram + RHS: matches old -11.79 dB code path exactly.
        # Numpy matmul (A.H @ A) produces tiny BLAS-dependent numerical
        # differences that accumulate over 200+ CD iterations → worse VP.
        gram = np.zeros((K, K), dtype=np.complex128)
        rhs = np.zeros(K, dtype=np.complex128)
        for i in range(K):
            s_rhs = 0.0 + 0.0j
            for k in range(n_idx.size):
                s_rhs += A[k, i].conjugate() * y[k]
            rhs[i] = s_rhs
            for j in range(i, K):
                s = 0.0 + 0.0j
                for k in range(n_idx.size):
                    s += A[k, i].conjugate() * A[k, j]
                gram[i, j] = s
                if i != j:
                    gram[j, i] = s.conjugate()

        trace_gram = sum(float(gram[i, i].real) for i in range(K))
        ridge = ridge_relative * trace_gram / max(K, 1)
        for i in range(K):
            gram[i, i] += ridge

        # Solve real system (manual Cholesky — MKL-safe)
        dim = 2 * K
        G_real = np.zeros((dim, dim))
        rhs_real = np.zeros(dim)
        for i in range(K):
            rhs_real[i] = float(rhs[i].real)
            rhs_real[i + K] = float(rhs[i].imag)
            for j in range(K):
                g_re = float(gram[i, j].real)
                g_im = float(gram[i, j].imag)
                G_real[i, j] = g_re
                G_real[i, j + K] = -g_im
                G_real[i + K, j] = g_im
                G_real[i + K, j + K] = g_re

        g = _solve_spd_cholesky(G_real, rhs_real)
        g_complex = g[:K] + 1j * g[K:]

        # Manual residual
        resid = 0.0
        for k in range(n_idx.size):
            pred = 0.0 + 0.0j
            for l in range(K):
                pred += A[k, l] * g_complex[l]
            diff = y[k] - pred
            resid += float(diff.real ** 2 + diff.imag ** 2)
        return resid, A, g_complex

    # Initial residual
    resid_old, _, _ = _compute_residual(d_bins, f_bins)
    rng = np.random.default_rng(42)
    total_accepted = 0

    for round_idx in range(n_rounds):
        # Scale step down each round
        step_d = step_delay / (1.0 + round_idx)
        step_f = step_doppler / (1.0 + round_idx)
        accepted_this_round = 0

        for l in range(n_paths):
            best_d = d_bins[l]
            best_f = f_bins[l]
            best_resid = resid_old
            accepted_this = False

            if search_method == "golden":
                # --- Golden-section line search per axis ---
                phi = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618...

                def _golden_search_1d(
                    x0: float, direction: str, step: float,
                    lo_bound: float, hi_bound: float,
                    d_bins_base: np.ndarray, f_bins_base: np.ndarray,
                ) -> tuple[float, float]:
                    """Golden-section search along delay or doppler axis."""
                    a = max(lo_bound, x0 - step)
                    b = min(hi_bound, x0 + step)
                    if b - a < 0.001:  # already converged
                        return x0, best_resid

                    c = b - phi * (b - a)
                    d = a + phi * (b - a)
                    n_iter = max(8, int(math.log(0.001 / (b - a + 1e-8)) / math.log(phi)) + 3)

                    # Evaluate at c and d
                    for _ in range(n_iter // 2):
                        # Eval at c
                        d_trial = d_bins_base.copy()
                        f_trial = f_bins_base.copy()
                        if direction == "delay":
                            d_trial[l] = c
                        else:
                            f_trial[l] = c
                        rc, _, _ = _compute_residual(d_trial, f_trial)

                        # Eval at d
                        d_trial = d_bins_base.copy()
                        f_trial = f_bins_base.copy()
                        if direction == "delay":
                            d_trial[l] = d
                        else:
                            f_trial[l] = d
                        rd, _, _ = _compute_residual(d_trial, f_trial)

                        if rc < rd:
                            b = d
                            d = c
                            c = b - phi * (b - a)
                        else:
                            a = c
                            c = d
                            d = a + phi * (b - a)

                        if b - a < 0.001:
                            break

                    x_best = (a + b) / 2.0
                    # Final eval at best
                    d_trial = d_bins_base.copy()
                    f_trial = f_bins_base.copy()
                    if direction == "delay":
                        d_trial[l] = x_best
                    else:
                        f_trial[l] = x_best
                    r_best, _, _ = _compute_residual(d_trial, f_trial)
                    return x_best, r_best

                # Golden-section on delay axis
                new_d, resid_d = _golden_search_1d(
                    d_bins[l], "delay", step_d, 0.0, 12.0, d_bins, f_bins,
                )
                if resid_d < best_resid:
                    best_d, best_resid = new_d, resid_d
                    accepted_this = True

                # Golden-section on doppler axis
                new_f, resid_f = _golden_search_1d(
                    f_bins[l], "doppler", step_f, -3.0, 3.0, d_bins, f_bins,
                )
                if resid_f < best_resid:
                    best_f, best_resid = new_f, resid_f
                    accepted_this = True

            else:
                # --- Legacy random probes ---
                for _ in range(n_probes):
                    d_probe = d_bins[l] + step_d * (rng.uniform(-1, 1))
                    f_probe = f_bins[l] + step_f * (rng.uniform(-1, 1))
                    d_probe = max(0.0, d_probe)
                    f_probe = max(-3.0, min(3.0, f_probe))

                    d_trial = d_bins.copy()
                    f_trial = f_bins.copy()
                    d_trial[l] = d_probe
                    f_trial[l] = f_probe

                    resid_new, _, _ = _compute_residual(d_trial, f_trial)
                    if resid_new < best_resid:
                        best_resid = resid_new
                        best_d = d_probe
                        best_f = f_probe
                        accepted_this = True

            if accepted_this:
                d_bins[l] = best_d
                f_bins[l] = best_f
                resid_old = best_resid
                total_accepted += 1
                accepted_this_round += 1

            # Joint re-estimate gains after each accepted path update.
            # This provides the correct LS residual baseline for the NEXT
            # path's coordinate descent, preventing error accumulation.
            if accepted_this:
                resid_old, _, _ = _compute_residual(d_bins, f_bins)

        # Early stop if no path improved in this round
        if accepted_this_round == 0:
            break

    # Caller (estimated_support_ls_reconstruction) does fresh LS with refined
    # positions — no need for joint re-estimation here.

    # Don't rebuild full dict again; just return refined positions.
    # Gains will be re-estimated by the caller (estimated_support_ls_reconstruction).
    refined = EstimatedPaths(
        delay_bins=d_bins,
        doppler_bins=f_bins,
        scores=est.scores.copy(),
        gains=est.gains.copy(),  # stale — caller must re-estimate
    )

    diag = {
        "accepted": total_accepted,
        "rounds": round_idx + 1,  # actual rounds executed (may be < n_rounds)
        "requested_rounds": n_rounds,
        "resid_initial": float(resid_old),  # updated to final after all rounds
    }
    return refined, diag


def detect_paths_oracle_nms(
    score_map: np.ndarray,
    gain_map: np.ndarray,
    grid: DDGrid,
    true_delays: np.ndarray,
    true_dopplers: np.ndarray,
    delay_radius: int = 2,
    doppler_radius: int = 2,
) -> EstimatedPaths:
    """Oracle NMS: select the grid point closest to each true path.

    This is an ablation tool — it removes the peak-selection bottleneck so we
    can isolate off-grid leakage from NMS failure modes.
    """
    delays: list[float] = []
    dopplers: list[float] = []
    scores: list[float] = []
    gains_list: list[complex] = []

    for td, tdp in zip(true_delays, true_dopplers):
        i = int(np.argmin(np.abs(grid.delay_bins - td)))
        j = int(np.argmin(np.abs(grid.doppler_bins - tdp)))
        delays.append(float(grid.delay_bins[i]))
        dopplers.append(float(grid.doppler_bins[j]))
        scores.append(float(score_map[i, j]) / max(float(np.max(score_map)), np.finfo(float).eps))
        gains_list.append(complex(gain_map[i, j]))

    return EstimatedPaths(
        delay_bins=np.asarray(delays, dtype=np.float64),
        doppler_bins=np.asarray(dopplers, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        gains=np.asarray(gains_list, dtype=np.complex128),
    )


# ---------------------------------------------------------------------------
# Path matching
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core identifiability metrics
# ---------------------------------------------------------------------------


def identifiability_metrics(
    true_delay: np.ndarray,
    true_doppler: np.ndarray,
    true_power: np.ndarray,
    est: EstimatedPaths,
    delay_tolerance: float,
    doppler_tolerance: float,
    max_delay_bins: Optional[float] = None,
    max_abs_doppler_bins: Optional[float] = None,
) -> dict[str, float]:
    """Compute recall, precision, power recovery, RMSE, and advanced metrics.

    Extended with:
    - Penalized RMSE (tolerance as miss penalty — see manifest for values)
    - OSPA distance (p=2, c=1.0, normalised DD distance)
    - False alarm rate per estimated path
    """
    matches = match_paths(
        true_delay,
        true_doppler,
        est,
        delay_tolerance,
        doppler_tolerance,
    )
    matched_true = {r for r, _ in matches}
    matched_est = {c for _, c in matches}
    n_true = max(len(true_delay), 1)
    n_est = max(len(est.delay_bins), 1)

    recall = len(matches) / n_true
    precision = len(matches) / n_est
    recovered_power = float(np.sum([true_power[i] for i in matched_true]))
    total_power = float(np.sum(true_power))
    power_recovery = recovered_power / max(total_power, np.finfo(float).eps)

    # --- matched-path RMSE (only over successfully matched paths) ---
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

    # --- penalised RMSE: tolerance as miss penalty ---
    # Each missed path contributes the matching tolerance as its error.
    # This is more interpretable than the full search range: a missed path
    # is "at least one tolerance away" from any estimated path.
    n_miss = n_true - len(matches)
    if matches:
        delay_se = np.sum([(true_delay[r] - est.delay_bins[c]) ** 2 for r, c in matches])
        doppler_se = np.sum([(true_doppler[r] - est.doppler_bins[c]) ** 2 for r, c in matches])
    else:
        delay_se = 0.0
        doppler_se = 0.0
    delay_penalty = n_miss * (delay_tolerance ** 2)
    doppler_penalty = n_miss * (doppler_tolerance ** 2)
    penalized_delay_rmse = float(np.sqrt((delay_se + delay_penalty) / n_true))
    penalized_doppler_rmse = float(np.sqrt((doppler_se + doppler_penalty) / n_true))

    # --- OSPA distance ---
    ospa_val = _ospa_distance(
        true_delay, true_doppler, est, delay_tolerance, doppler_tolerance
    )

    # --- false alarm metrics ---
    # Under Known-K with Top-K output and NO early-exit: n_est == n_true,
    # so FP == FN == K - TP.  When NMS early-exits (peak below threshold),
    # n_est < n_true and FP < FN — the missing peak was never reported.
    # In Unknown-K (Gate 0-B), these two quantities decouple fully.
    n_false_alarms = n_est - len(matches)
    false_alarm_rate_per_est = n_false_alarms / max(n_est, 1)

    metrics = {
        "path_recall": recall,
        "path_precision": precision,
        "power_recovery": power_recovery,
        "delay_rmse_bins": delay_rmse,
        "doppler_rmse_bins": doppler_rmse,
        "penalized_delay_rmse_bins": penalized_delay_rmse,
        "penalized_doppler_rmse_bins": penalized_doppler_rmse,
        "ospa_distance": ospa_val,
        "num_estimated": float(len(est.delay_bins)),
        "num_missed": float(n_miss),
        "num_false_alarms": float(n_false_alarms),
        "false_alarm_rate": false_alarm_rate_per_est,
    }
    return metrics


# ---------------------------------------------------------------------------
# OSPA distance (joint localisation + cardinality error)
# ---------------------------------------------------------------------------


def _ospa_distance(
    true_delay: np.ndarray,
    true_doppler: np.ndarray,
    est: EstimatedPaths,
    delay_tolerance: float,
    doppler_tolerance: float,
    p: int = 2,
    c: float = 1.0,
) -> float:
    """Compute OSPA distance between true and estimated path sets.

    OSPA jointly penalises localisation error, missed detections, and false
    alarms in a single metric.

    Parameters
    ----------
    p : int
        Order of the metric (default 2).
    c : float
        Cut-off distance in normalised DD units.  Errors beyond `c` are capped.
        Default 1.0 means one tolerance-unit of error is the maximum per-path
        contribution from localisation.

    Implementation follows the standard OSPA definition:
        d_OSPA = [ 1/max(m,n) * ( min_sum + c^p * |m-n| ) ]^(1/p)
    where min_sum uses Hungarian assignment with distances capped at c,
    and m, n are the cardinalities of the true and estimated sets.
    """
    n_true = len(true_delay)
    n_est = len(est.delay_bins)

    if n_true == 0 and n_est == 0:
        return 0.0

    # Normalise DD coordinates to comparable units using tolerances
    norm_delay_true = true_delay / max(delay_tolerance, 1e-12)
    norm_doppler_true = true_doppler / max(doppler_tolerance, 1e-12)
    norm_delay_est = est.delay_bins / max(delay_tolerance, 1e-12)
    norm_doppler_est = est.doppler_bins / max(doppler_tolerance, 1e-12)

    true_points = np.stack([norm_delay_true, norm_doppler_true], axis=-1)
    est_points = np.stack([norm_delay_est, norm_doppler_est], axis=-1)

    if n_true == 0:
        # All estimates are false alarms
        return (c ** p * n_est) ** (1.0 / p)
    if n_est == 0:
        # All true paths are missed
        return (c ** p * n_true) ** (1.0 / p)

    # Pairwise distances, capped at c
    dist = cdist(true_points, est_points, metric="euclidean")
    dist = np.minimum(dist, c)

    # Hungarian assignment on (n_true x n_est) matrix
    row_ind, col_ind = linear_sum_assignment(dist)

    # Sum over assigned pairs + cardinality penalty
    loc_cost = float(np.sum(dist[row_ind, col_ind] ** p))
    card_cost = c ** p * abs(n_true - n_est)
    ospa = (1.0 / max(n_true, n_est) * (loc_cost + card_cost)) ** (1.0 / p)
    return ospa


# ---------------------------------------------------------------------------
# Dictionary coherence (global + far-field)
# ---------------------------------------------------------------------------


def dictionary_coherence(
    pilot_mask: np.ndarray,
    grid: DDGrid,
    nms_delay_radius: int = 2,
    nms_doppler_radius: int = 2,
) -> dict[str, float]:
    """Compute mutual coherence of the DD dictionary for a given pilot mask.

    Reports three quantities:
    - mu_max:  global maximum coherence (may be near 1 for oversampled grids)
    - mu_far:  maximum after excluding the local NMS neighbourhood around each
               column (more diagnostic for Comb grating lobes)
    - mu_p95, mu_p99:  upper percentiles of coherence distribution
    """
    A = build_dd_dictionary(pilot_mask, grid)
    G = np.abs(A.conj().T @ A)
    n_cols = G.shape[0]
    # Zero out diagonal
    G[np.arange(n_cols), np.arange(n_cols)] = 0.0

    mu_max = float(np.max(G))
    mu_mean = float(np.mean(G))

    # --- far-field coherence: exclude local DD neighbourhood ---
    # Map each column index back to its (delay_bin, doppler_bin) grid position
    n_delay = grid.delay_bins.size
    n_doppler = grid.doppler_bins.size
    G_far = G.copy()
    for i in range(n_cols):
        i_delay, i_doppler = divmod(i, n_doppler)
        for j in range(n_cols):
            j_delay, j_doppler = divmod(j, n_doppler)
            if (
                abs(i_delay - j_delay) <= nms_delay_radius
                and abs(i_doppler - j_doppler) <= nms_doppler_radius
            ):
                G_far[i, j] = 0.0
    mu_far = float(np.max(G_far))

    # Percentiles (over upper triangle for efficiency, but full is fine)
    upper = G[np.triu_indices(n_cols, k=1)]
    mu_p95 = float(np.percentile(upper, 95))
    mu_p99 = float(np.percentile(upper, 99))

    return {
        "mu_max": mu_max,
        "mu_far": mu_far,
        "mu_mean": mu_mean,
        "mu_p95": mu_p95,
        "mu_p99": mu_p99,
        "n_dictionary_columns": n_cols,
        "gram_psr": 1.0 / max(mu_max, np.finfo(float).eps),
    }


# ---------------------------------------------------------------------------
# Pilot ambiguity function
# ---------------------------------------------------------------------------


def pilot_ambiguity_function(
    pilot_mask: np.ndarray,
    grid: DDGrid,
) -> np.ndarray:
    """Compute the normalised pilot ambiguity function |A(Δτ, Δν)|.

    A(Δτ, Δν) = |Σ_{(n,m)∈P} exp(-j2π n Δτ / N) exp(j2π m Δν / M)|

    Normalised so that A(0, 0) = 1.
    """
    n_idx, m_idx = np.nonzero(pilot_mask)
    if n_idx.size == 0:
        raise ValueError("pilot mask contains no observations")
    n_total, m_total = pilot_mask.shape

    af = np.zeros((grid.delay_bins.size, grid.doppler_bins.size), dtype=np.float64)
    for i, dtau in enumerate(grid.delay_bins):
        for j, dnu in enumerate(grid.doppler_bins):
            phase = (
                -2j * np.pi * n_idx * dtau / n_total
                + 2j * np.pi * m_idx * dnu / m_total
            )
            af[i, j] = float(np.abs(np.sum(np.exp(phase))))
    # Normalise so zero-delay-zero-Doppler peak = 1
    af /= max(float(np.max(af)), np.finfo(float).eps)
    return af


def ambiguity_metrics(
    af: np.ndarray,
    delay_radius: int = 2,
    doppler_radius: int = 2,
) -> dict[str, float]:
    """Extract key metrics from a pilot ambiguity function.

    The mainlobe exclusion region uses the same radius as NMS so the AF
    sidelobe metrics correspond directly to the detector behaviour.

    Returns
    -------
    pslr_db : float
        Peak-to-maximum-sidelobe ratio (dB).  More negative = better.
    islr_db : float
        Integrated sidelobe ratio (dB).  More negative = better.
    max_far_sidelobe_delay : float
        Delay bin of the strongest far sidelobe.
    max_far_sidelobe_doppler : float
        Doppler bin of the strongest far sidelobe.
    mainlobe_width_delay_bins : float
        -3 dB mainlobe width in delay bins.
    mainlobe_width_doppler_bins : float
        -3 dB mainlobe width in Doppler bins.
    """
    peak_idx = np.unravel_index(np.argmax(af), af.shape)
    di, dj = peak_idx

    # Mainlobe: contiguous -3 dB region around the peak
    mainlobe_mask = af >= float(np.max(af)) / np.sqrt(2.0)

    # Sidelobe mask: exclude mainlobe AND local NMS neighbourhood
    sidelobe_mask = ~mainlobe_mask
    # Also zero out the NMS exclusion zone (same radius as the detector)
    i0, i1 = max(0, di - delay_radius), min(af.shape[0], di + delay_radius + 1)
    j0, j1 = max(0, dj - doppler_radius), min(af.shape[1], dj + doppler_radius + 1)
    far_mask = sidelobe_mask.copy()
    far_mask[i0:i1, j0:j1] = False

    # PSLR: peak-to-max-sidelobe ratio (using far-field sidelobes only)
    if np.any(far_mask):
        max_far_sidelobe = float(np.max(af[far_mask]))
        pslr_db = 20.0 * np.log10(max_far_sidelobe)  # normalised AF → peak = 1 = 0 dB
    else:
        max_far_sidelobe = 0.0
        pslr_db = float("-inf")

    # Far sidelobe location
    if max_far_sidelobe > 0:
        far_idx = np.unravel_index(np.argmax(af * far_mask.astype(np.float64)), af.shape)
        max_far_delay = float(far_idx[0])
        max_far_doppler = float(far_idx[1])
    else:
        max_far_delay = float("nan")
        max_far_doppler = float("nan")

    # ISLR
    mainlobe_energy = float(np.sum(af[mainlobe_mask] ** 2))
    sidelobe_energy = float(np.sum(af[sidelobe_mask] ** 2))
    islr_db = (
        10.0 * np.log10(sidelobe_energy / mainlobe_energy)
        if mainlobe_energy > 0
        else float("inf")
    )

    # Mainlobe widths
    ml_indices = np.argwhere(mainlobe_mask)
    if ml_indices.size > 0:
        ml_width_delay = float(ml_indices[:, 0].max() - ml_indices[:, 0].min())
        ml_width_doppler = float(ml_indices[:, 1].max() - ml_indices[:, 1].min())
    else:
        ml_width_delay = 0.0
        ml_width_doppler = 0.0

    return {
        "pslr_db": pslr_db,
        "islr_db": islr_db,
        "max_far_sidelobe_delay_bin": max_far_delay,
        "max_far_sidelobe_doppler_bin": max_far_doppler,
        "mainlobe_width_delay_bins": ml_width_delay,
        "mainlobe_width_doppler_bins": ml_width_doppler,
    }


# ---------------------------------------------------------------------------
# Confidence intervals and paired bootstrap
# ---------------------------------------------------------------------------


def confidence_interval(
    values: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute mean, standard error, and 95% confidence interval.

    SE = s / sqrt(N_eff), where N_eff counts only finite (non-NaN) values.
    This is critical for matched-path RMSE which can be NaN when no paths
    are matched at low SNR.
    """
    finite = values[np.isfinite(values)]
    n_eff = finite.size
    if n_eff < 2:
        return {
            "mean": float(np.mean(values)) if finite.size > 0 else float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_eff": int(n_eff),
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1))
    se = std / np.sqrt(n_eff)
    z = 1.96  # 95% CI
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_lower": mean - z * se,
        "ci_upper": mean + z * se,
        "n_eff": int(n_eff),
    }


def paired_bootstrap_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap test for the difference A - B.

    Returns the mean difference, its 95% CI, and the two-sided p-value.

    REQUIRES paired data: for each trial index t, values_a[t] and values_b[t]
    must share the same channel realisation and base noise.  This is ensured
    by the Gate 0 RNG design (separate channel/pilot/noise RNGs; channel and
    noise seeds do not include pattern_index).
    """
    finite_mask = np.isfinite(values_a) & np.isfinite(values_b)
    a = values_a[finite_mask]
    b = values_b[finite_mask]
    if a.size < 10:
        return {
            "mean_diff": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "n_pairs": int(a.size),
            "warning": "fewer than 10 finite pairs — CI unreliable",
        }

    diff = a - b
    mean_diff = float(np.mean(diff))
    rng = np.random.default_rng(42)
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(a.size, size=a.size, replace=True)
        bootstrap_diffs[i] = float(np.mean(diff[idx]))

    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))

    # Two-sided p-value with finite-sample correction (k+1)/(B+1).
    # Bootstrap CANNOT prove p=0; always report a floor of 1/(B+1).
    if mean_diff >= 0:
        k = int(np.sum(bootstrap_diffs <= 0))
    else:
        k = int(np.sum(bootstrap_diffs >= 0))
    p_raw = 2.0 * (k + 1) / (n_bootstrap + 1)
    p_value = min(p_raw, 1.0)

    # Human-readable p-value string for CSV (avoids misleading "0.000")
    if k == 0:
        p_str = f"<{2.0 / (n_bootstrap + 1):.1e}"
    elif p_value < 0.001:
        p_str = f"{p_value:.1e}"
    else:
        p_str = f"{p_value:.4f}"

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "p_value_str": p_str,
        "n_pairs": int(a.size),
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Net spectral efficiency
# ---------------------------------------------------------------------------


def net_spectral_efficiency(
    pilot_density: float,
    snr_db: float,
    channel_nmse: Optional[float] = None,
) -> dict[str, float]:
    """Compute net spectral efficiency accounting for pilot overhead.

    R_net = (1 - ρ_p) * log2(1 + γ_eff)

    where γ_eff = SNR / (1 + SNR * NMSE) when NMSE is available.

    Returns both the raw rate and the fraction of ideal (ρ_p=0, perfect CSI).
    """
    rho_p = pilot_density
    snr_linear = 10.0 ** (snr_db / 10.0)

    if channel_nmse is not None and channel_nmse > 0:
        gamma_eff = snr_linear / (1.0 + snr_linear * channel_nmse)
    else:
        gamma_eff = snr_linear

    data_fraction = 1.0 - rho_p
    r_net = data_fraction * np.log2(1.0 + gamma_eff)

    # Ideal: ρ_p=0, perfect CSI → log2(1 + SNR)
    r_ideal = np.log2(1.0 + snr_linear)

    return {
        "net_spectral_efficiency_bps_hz": float(r_net),
        "data_resource_fraction": float(data_fraction),
        "effective_snr_db": float(10.0 * np.log10(max(gamma_eff, 1e-12))),
        "efficiency_ratio": float(r_net / max(r_ideal, 1e-12)),
    }
