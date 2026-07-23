from __future__ import annotations

import math
import warnings
import numpy as np

from .channel import PathSet, synthesize_tf_channel
from .config import OFDMConfig
from .dd_estimation import EstimatedPaths


# ---------------------------------------------------------------------------
# Oracle path tokens
# ---------------------------------------------------------------------------


def oracle_path_tokens(paths: PathSet, max_paths: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-width path tokens and a validity mask.

    Token fields (legacy 7-dim, no complex gain):
      delay_bin, doppler_bin, normalized power, confidence,
      sigma_delay, sigma_doppler, communication relevance.

    For Gate 1-D1, use oracle_path_tokens_v2 which includes Re(α), Im(α).
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


def oracle_path_tokens_v2(paths: PathSet, max_paths: int) -> tuple[np.ndarray, np.ndarray]:
    """Create 9-dim path tokens WITH complex gain — Gate 1-D1.

    Token fields (9-dim):
      0: delay_bin
      1: doppler_bin
      2: normalized power |α|²
      3: confidence
      4: sigma_delay
      5: sigma_doppler
      6: communication relevance
      7: Re(α)    ← NEW
      8: Im(α)    ← NEW
    """

    tokens = np.zeros((max_paths, 9), dtype=np.float32)
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
    tokens[:count, 7] = paths.gains[order].real.astype(np.float32)
    tokens[:count, 8] = paths.gains[order].imag.astype(np.float32)
    valid[:count] = True
    return tokens, valid


def compute_path_quality(
    est: EstimatedPaths,
    ls_gains: np.ndarray,
    pilot_observations: np.ndarray,
    pilot_mask: np.ndarray,
    score_map: np.ndarray,
    num_subcarriers: int,
    num_symbols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-path quality metrics from DD detections + pilot data.

    Returns (confidence, sigma_tau, sigma_nu, relevance) — each shape (n_paths,).
    All values are in [0, 1] or bin units.

    Metrics:
      - Peak PSLR: local peak-to-sidelobe ratio (higher → more reliable)
      - Leave-one-out ΔJ: residual increase when this path is removed
      - Local dictionary coherence: max column correlation in neighbourhood
    """
    n_paths = len(est.delay_bins)
    if n_paths == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    n_idx, m_idx = np.nonzero(pilot_mask)
    y = pilot_observations[pilot_mask]
    N, M = num_subcarriers, num_symbols

    # --- 1. Peak PSLR per path ---
    peak_pslr = np.ones(n_paths)
    n_delay = score_map.shape[0]
    n_doppler = score_map.shape[1]
    for l in range(n_paths):
        i0 = int(np.argmin(np.abs(
            np.linspace(0, est.delay_bins[-1] if n_delay > 1 else 1.0, n_delay)
            - est.delay_bins[l]
        )))
        j0 = int(np.argmin(np.abs(
            np.linspace(-3.0, 3.0, n_doppler) - est.doppler_bins[l]
        )))
        peak_val = score_map[i0, j0]
        # Local sidelobe: max outside NMS radius in 3×3 window
        r = 2  # NMS radius
        i_start, i_end = max(0, i0 - r), min(n_delay, i0 + r + 1)
        j_start, j_end = max(0, j0 - r), min(n_doppler, j0 + r + 1)
        local = score_map[i_start:i_end, j_start:j_end].copy()
        local[i0 - i_start, j0 - j_start] = 0.0  # exclude peak
        max_sidelobe = float(np.max(local)) if local.size > 1 else 0.0
        peak_pslr[l] = peak_val / max(max_sidelobe, np.finfo(float).eps)

    # --- 2. Leave-one-out residual contribution ---
    # Full model residual (with normalised columns for consistency)
    A_full = np.zeros((n_idx.size, n_paths), dtype=np.complex128)
    for l in range(n_paths):
        dp = np.exp(-2j * np.pi * n_idx * est.delay_bins[l] / N)
        fp = np.exp(2j * np.pi * m_idx * est.doppler_bins[l] / M)
        A_full[:, l] = dp * fp
    # Residual uses the existing LS gains (already in unnormalised units)
    resid_full = float(np.sum(np.abs(y - A_full @ ls_gains) ** 2))

    loo_contrib = np.zeros(n_paths)
    for l in range(n_paths):
        # Remove path l, re-estimate gains
        mask_l = np.ones(n_paths, dtype=bool)
        mask_l[l] = False
        if mask_l.sum() == 0:
            loo_contrib[l] = 1.0
            continue
        A_loo = A_full[:, mask_l]
        # Normalise columns for consistent LS regularisation
        col_norms = np.sqrt(np.sum(np.abs(A_loo) ** 2, axis=0))
        A_loo_normed = A_loo / np.maximum(col_norms, np.finfo(float).eps)
        g_loo_normed = _ridge_ls(A_loo_normed, y)
        g_loo = g_loo_normed / np.maximum(col_norms, np.finfo(float).eps)
        resid_loo = float(np.sum(np.abs(y - A_loo @ g_loo) ** 2))
        loo_contrib[l] = (resid_loo - resid_full) / max(resid_loo, np.finfo(float).eps)
    loo_contrib = np.clip(loo_contrib, 0.0, 1.0)

    # --- 3. Per-path confidence from PSLR + LOO ---
    # Sigmoid-style mapping: high PSLR + high LOO → confidence near 1
    confidence = np.zeros(n_paths)
    for l in range(n_paths):
        # PSLR contribution: values > 3 → confident, < 1 → uncertain
        pslr_score = 1.0 / (1.0 + np.exp(-2.0 * (peak_pslr[l] - 2.0)))
        conf = 0.5 * pslr_score + 0.5 * loo_contrib[l]
        confidence[l] = float(np.clip(conf, 0.1, 0.95))

    # --- 4. Uncertainty estimates ---
    # σ_τ, σ_ν: inversely proportional to peak PSLR
    sigma_tau = 0.3 / np.clip(peak_pslr, 1.0, 10.0)
    sigma_nu = 0.2 / np.clip(peak_pslr, 1.0, 10.0)

    # --- 5. Relevance = LOO contribution ---
    relevance = loo_contrib.copy()

    return confidence, sigma_tau, sigma_nu, relevance


def estimated_path_tokens_v2(
    est: EstimatedPaths,
    ls_gains: np.ndarray,
    max_paths: int,
    confidence: np.ndarray | None = None,
    sigma_tau: np.ndarray | None = None,
    sigma_nu: np.ndarray | None = None,
    relevance: np.ndarray | None = None,
    score_map: np.ndarray | None = None,
    grid_delay: np.ndarray | None = None,
    grid_doppler: np.ndarray | None = None,
    attach_patch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create tokens from DD-estimated support + LS-estimated gains.

    When attach_patch=True, tokens are 18-dim:
      0-8: same 9 scalar fields
      9-17: flattened 3x3 score_map patch around NMS/OMP peak (normalised)

    The DDTokenRefiner uses the patch to predict Δτ,Δν from local
    score_map shape (peak curvature, asymmetry, sidelobes).
    """
    dim = 18 if attach_patch else 9
    tokens = np.zeros((max_paths, dim), dtype=np.float32)
    valid = np.zeros(max_paths, dtype=bool)
    n_paths = min(max_paths, len(est.delay_bins))
    order = np.argsort(np.abs(ls_gains))[::-1][:n_paths]
    tokens[:n_paths, 0] = est.delay_bins[order]
    tokens[:n_paths, 1] = est.doppler_bins[order]
    power = np.abs(ls_gains[order]) ** 2
    total_p = np.sum(power)
    tokens[:n_paths, 2] = power / max(total_p, np.finfo(float).eps)
    tokens[:n_paths, 3] = confidence[order] if confidence is not None else 0.7
    tokens[:n_paths, 4] = sigma_tau[order] if sigma_tau is not None else 0.2
    tokens[:n_paths, 5] = sigma_nu[order] if sigma_nu is not None else 0.1
    tokens[:n_paths, 6] = relevance[order] if relevance is not None else 1.0
    tokens[:n_paths, 7] = ls_gains[order].real.astype(np.float32)
    tokens[:n_paths, 8] = ls_gains[order].imag.astype(np.float32)
    valid[:n_paths] = True

    if attach_patch and score_map is not None and grid_delay is not None and grid_doppler is not None:
        n_d, n_f = score_map.shape
        for l in range(n_paths):
            tau = float(est.delay_bins[order[l]])
            nu = float(est.doppler_bins[order[l]])
            i0 = int(np.argmin(np.abs(grid_delay - tau)))
            j0 = int(np.argmin(np.abs(grid_doppler - nu)))
            patch = np.zeros(9, dtype=np.float32)
            pi = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ii = i0 + di
                    jj = (j0 + dj) % n_f  # doppler wrap
                    if 0 <= ii < n_d:
                        patch[pi] = score_map[ii, jj]
                    pi += 1
            pmax = np.max(patch)
            if pmax > 0:
                patch /= pmax  # normalise to [0, 1]
            tokens[l, 9:18] = patch

    return tokens, valid


# ---------------------------------------------------------------------------
# DD local patch extraction (token v3 enhancement)
# ---------------------------------------------------------------------------


def _nearest_grid_index(value: float, bins: np.ndarray) -> int:
    """Nearest-neighbour grid index for a continuous bin value."""
    return int(np.argmin(np.abs(bins - value)))


def _extract_dd_patch(
    score_map: np.ndarray,       # [n_delay, n_doppler]
    gain_map: np.ndarray,        # [n_delay, n_doppler] complex
    grid_delay: np.ndarray,      # [n_delay]
    grid_doppler: np.ndarray,    # [n_doppler]
    tau: float,
    nu: float,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (2R+1)×(2R+1) local patches around DD peak position.

    Returns (score_patch_flat, gain_patch_flat) each of length (2R+1)².
    Doppler is handled with periodic wrapping.
    """
    i0 = _nearest_grid_index(tau, grid_delay)
    j0 = _nearest_grid_index(nu, grid_doppler)

    n_delay = len(grid_delay)
    n_doppler = len(grid_doppler)

    i_start = max(0, i0 - radius)
    i_end = min(n_delay, i0 + radius + 1)

    # Doppler with wrapping
    j_indices = np.arange(j0 - radius, j0 + radius + 1)
    j_wrapped = np.mod(j_indices, n_doppler)

    patch_size = (2 * radius + 1) ** 2

    # Score patch (pad with zeros for out-of-bounds regions)
    score_patch = np.zeros(patch_size, dtype=np.float32)
    si = 0
    di = i0 - radius - i_start  # offset in delay dimension for centering
    dj = j0 - radius
    for ii in range(i_start, i_end):
        for jj_idx, jj in enumerate(j_wrapped):
            patch_idx = (ii - i_start) * (2 * radius + 1) + jj_idx
            score_patch[patch_idx] = score_map[ii, jj]
            si += 1

    # Gain patch (real + imag separated)
    gain_patch = np.zeros(patch_size * 2, dtype=np.float32)
    for ii in range(i_start, i_end):
        for jj_idx, jj in enumerate(j_wrapped):
            patch_idx = (ii - i_start) * (2 * radius + 1) + jj_idx
            g = gain_map[ii, jj]
            gain_patch[patch_idx] = float(g.real)
            gain_patch[patch_idx + patch_size] = float(g.imag)

    return score_patch, gain_patch


def estimated_path_tokens_v3(
    est: EstimatedPaths,
    ls_gains: np.ndarray,
    max_paths: int,
    score_map: np.ndarray,
    gain_map: np.ndarray,
    grid_delay: np.ndarray,
    grid_doppler: np.ndarray,
    confidence: np.ndarray | None = None,
    sigma_tau: np.ndarray | None = None,
    sigma_nu: np.ndarray | None = None,
    relevance: np.ndarray | None = None,
    patch_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create enhanced tokens from DD-estimated support + local DD spectrum patches.

    Token fields (9 base + (2R+1)² score + 2*(2R+1)² gain = 9+25+50=84 for R=2):
      0-8: same 9-dim as v2 (τ, ν, power, confidence, σ_τ, σ_ν, relevance, Re(α), Im(α))
      9-33: (2R+1)² score_map patch around peak (flattened)
      34-83: 2*(2R+1)² gain_map patch (real then imag, flattened)

    The local patches tell the model:
      - Peak sharpness (localisation confidence)
      - Nearby competitors / sidelobes (coherent false path risk)
      - DD energy distribution around the detected location
    """
    patch_size = (2 * patch_radius + 1) ** 2
    token_dim = 9 + patch_size + 2 * patch_size  # 9 + score + gain

    tokens = np.zeros((max_paths, token_dim), dtype=np.float32)
    valid = np.zeros(max_paths, dtype=bool)
    n_paths = min(max_paths, len(est.delay_bins))
    order = np.argsort(np.abs(ls_gains))[::-1][:n_paths]

    # Base 9-dim fields (same as v2)
    tokens[:n_paths, 0] = est.delay_bins[order]
    tokens[:n_paths, 1] = est.doppler_bins[order]
    power = np.abs(ls_gains[order]) ** 2
    total_p = np.sum(power)
    tokens[:n_paths, 2] = power / max(total_p, np.finfo(float).eps)
    tokens[:n_paths, 3] = confidence[order] if confidence is not None else 0.7
    tokens[:n_paths, 4] = sigma_tau[order] if sigma_tau is not None else 0.2
    tokens[:n_paths, 5] = sigma_nu[order] if sigma_nu is not None else 0.1
    tokens[:n_paths, 6] = relevance[order] if relevance is not None else 1.0
    tokens[:n_paths, 7] = ls_gains[order].real.astype(np.float32)
    tokens[:n_paths, 8] = ls_gains[order].imag.astype(np.float32)

    # DD local patches per path
    for l in range(n_paths):
        tau = est.delay_bins[order[l]]
        nu = est.doppler_bins[order[l]]
        score_patch, gain_patch = _extract_dd_patch(
            score_map, gain_map, grid_delay, grid_doppler,
            tau, nu, radius=patch_radius,
        )
        tokens[l, 9:9 + patch_size] = score_patch
        tokens[l, 9 + patch_size:9 + patch_size + 2 * patch_size] = gain_patch

    valid[:n_paths] = True
    return tokens, valid


# ---------------------------------------------------------------------------
# Oracle parametric reconstruction (upper bound / code-closure test)
# ---------------------------------------------------------------------------


def oracle_perfect_reconstruction(ofdm: OFDMConfig, paths: PathSet) -> np.ndarray:
    """Perfect reconstruction from true path parameters.

    This should approach numerical precision (~ -100 dB NMSE in float64) if
    the synthesis formulas are closed — it is a code-closure test, not an
    algorithm result.

    Returns the TF channel H[n, m] synthesised from true {τ, ν, α}.
    """
    return synthesize_tf_channel(ofdm, paths)


# ---------------------------------------------------------------------------
# Ridge LS helper — pure Python, ZERO MKL calls
# ---------------------------------------------------------------------------


def _col_norms(A: np.ndarray) -> np.ndarray:
    """Compute per-column L2 norms manually (no np.linalg.norm, no BLAS)."""
    n_rows, n_cols = A.shape
    norms = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        s = 0.0
        for i in range(n_rows):
            s += float((A[i, j].real ** 2) + (A[i, j].imag ** 2))
        norms[j] = math.sqrt(s)
    return norms


def _manual_gram_and_rhs(
    A: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gram = A^H @ A  and  rhs = A^H @ y  using pure-Python loops.

    Avoids np.dot / np.matmul / @ which call MKL BLAS ?GEMM and SIGABRT on
    the target Anaconda/Windows environment.
    """
    n_rows = A.shape[0]
    n_cols = A.shape[1]
    gram = np.zeros((n_cols, n_cols), dtype=np.complex128)
    rhs = np.zeros(n_cols, dtype=np.complex128)

    for i in range(n_cols):
        s_rhs = 0.0 + 0.0j
        for k in range(n_rows):
            s_rhs += A[k, i].conjugate() * y[k]
        rhs[i] = s_rhs

        for j in range(i, n_cols):
            s = 0.0 + 0.0j
            for k in range(n_rows):
                s += A[k, i].conjugate() * A[k, j]
            gram[i, j] = s
            if i != j:
                gram[j, i] = s.conjugate()

    return gram, rhs


def _ridge_ls(
    A: np.ndarray,
    y: np.ndarray,
    ridge_relative: float = 1e-6,
) -> np.ndarray:
    """Solve min ||A x - y||² + λ||x||² with trace-scaled regularisation.

    Columns of A should be approximately unit-norm so that ridge strength
    is comparable across different pilot densities (callers normalise A
    before passing it in).

    Uses PURE PYTHON linear algebra throughout — no BLAS, no LAPACK, no
    MKL of any kind.
    """
    n_cols = A.shape[1]
    if n_cols == 0:
        return np.zeros(0, dtype=np.complex128)

    gram, rhs = _manual_gram_and_rhs(A, y)

    # Trace (manual, no np.trace)
    trace_gram = 0.0
    for i in range(n_cols):
        trace_gram += float(gram[i, i].real)

    ridge = ridge_relative * trace_gram / max(n_cols, 1)
    for i in range(n_cols):
        gram[i, i] += ridge

    # Real equivalent: 2L × 2L real SPD system
    dim = 2 * n_cols
    G_real = np.zeros((dim, dim), dtype=np.float64)
    rhs_real = np.zeros(dim, dtype=np.float64)

    for i in range(n_cols):
        rhs_real[i] = float(rhs[i].real)
        rhs_real[i + n_cols] = float(rhs[i].imag)
        for j in range(n_cols):
            g_re = float(gram[i, j].real)
            g_im = float(gram[i, j].imag)
            G_real[i, j] = g_re
            G_real[i, j + n_cols] = -g_im
            G_real[i + n_cols, j] = g_im
            G_real[i + n_cols, j + n_cols] = g_re

    x_real = _solve_spd_cholesky(G_real, rhs_real)
    return x_real[:n_cols] + 1j * x_real[n_cols:]


def _solve_spd_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b for a real SPD matrix A using manual Cholesky.

    Uses ONLY Python float + math.sqrt — no numpy linear-algebra calls.

    Issues a RuntimeWarning if the matrix is numerically singular (diagonal
    element ≤ 0 during decomposition), which indicates the LS problem is
    ill-conditioned and the result should not be trusted.
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    singular = False
    for i in range(n):
        for j in range(i + 1):
            s = float(A[i, j])
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    singular = True
                    # Use a fallback small enough to not dominate but large
                    # enough to avoid amplifying errors catastrophically.
                    # The result is still unreliable; we warn below.
                    s = float(np.finfo(np.float64).eps)
                L[i, j] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    if singular:
        warnings.warn(
            "_solve_spd_cholesky: matrix is numerically singular — "
            "LS solution is unreliable.  This usually indicates "
            "near-identical DD dictionary columns (path ambiguity).",
            RuntimeWarning,
        )

    # Forward substitution: L @ y = b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = float(b[i])
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]

    # Back substitution: L^T @ x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]

    return x


def _dict_condition_estimate(A: np.ndarray) -> float:
    """Estimate the condition number of the LS dictionary A.

    Uses the ratio of largest to smallest column norm as a cheap proxy.
    Returns inf if any column has zero norm.
    """
    col_norms = _col_norms(A)
    c_min = float(np.min(col_norms))
    c_max = float(np.max(col_norms))
    if c_min < np.finfo(np.float64).eps:
        return float("inf")
    return c_max / c_min


# ---------------------------------------------------------------------------
# Oracle support + LS complex-gain
# ---------------------------------------------------------------------------


def oracle_support_ls_reconstruction(
    ofdm: OFDMConfig,
    paths: PathSet,
    pilot_observations: np.ndarray,
    pilot_mask: np.ndarray,
) -> np.ndarray:
    """Oracle DD support + LS complex-gain estimation.

    Given TRUE DD path locations {τ, ν}, estimate complex gains via
    trace-regularised LS on raw pilot observations, then reconstruct the
    full TF channel.

    Dictionary columns are normalised so that ridge regularisation strength
    is independent of pilot density and path count.

    NMSE(·, H_true) isolates the gain-estimation error Δ_gain:
        Δ_gain = NMSE(oracle_support_ls) - NMSE(oracle_perfect)

    Returns H_rec[n, m].
    """
    n_sc = ofdm.num_subcarriers
    n_sym = ofdm.num_symbols
    n_paths = len(paths.gains)

    if n_paths == 0:
        return np.zeros((n_sc, n_sym), dtype=np.complex128)

    n_idx, m_idx = np.nonzero(pilot_mask)
    if n_idx.size == 0:
        raise ValueError("pilot mask contains no observations")

    # Build dictionary at pilot locations using TRUE path parameters
    A_pilot = np.zeros((n_idx.size, n_paths), dtype=np.complex128)
    for l in range(n_paths):
        delay_phase = np.exp(-2j * np.pi * n_idx * paths.delay_bins[l] / n_sc)
        doppler_phase = np.exp(2j * np.pi * m_idx * paths.doppler_bins[l] / n_sym)
        A_pilot[:, l] = delay_phase * doppler_phase

    # Column-normalise so ridge strength is consistent across pilot densities
    _normalise_columns(A_pilot)

    y = pilot_observations[pilot_mask]
    g_hat = _ridge_ls(A_pilot, y)

    # Undo normalisation: each g_hat[l] was estimated with a normalised column,
    # so the true gain = g_hat[l] / original_norm[l].
    # Since we passed the normalised A to _ridge_ls, the solution is in the
    # normalised basis.  We compensate during reconstruction by using the
    # original (unnormalised) dictionary, which is what _synthesize_from_params
    # does anyway.  BUT:  g_hat is the gain for unit-norm columns.  The
    # true channel contribution of path l is  α_l * a_l (where ||a_l|| = √N_p).
    # After normalisation,  α_l * a_l = (α_l * ||a_l||) * (a_l / ||a_l||).
    # So g_hat[l] ≈ α_l * ||a_l||, and we must divide g_hat[l] by ||a_l||
    # when passing to _synthesize_from_params (which uses unnormalised phases).
    col_norms = _col_norms(
        _build_raw_dict(n_sc, n_sym, n_idx, m_idx,
                        paths.delay_bins, paths.doppler_bins)
    )
    g_hat = g_hat / np.maximum(col_norms, np.finfo(np.float64).eps)

    return _synthesize_from_params(
        n_sc, n_sym, paths.delay_bins, paths.doppler_bins, g_hat
    )


# ---------------------------------------------------------------------------
# DD-estimated support + LS complex-gain
# ---------------------------------------------------------------------------


def estimated_support_ls_reconstruction(
    ofdm: OFDMConfig,
    est: EstimatedPaths,
    pilot_observations: np.ndarray,
    pilot_mask: np.ndarray,
) -> np.ndarray:
    """DD-estimated support + LS complex-gain reconstruction.

    Uses DD-ESTIMATED (not true) path locations.  Dictionary columns are
    normalised for ridge-consistent LS.

    NMSE(·, H_true) isolates the support-estimation error Δ_support:
        Δ_support = NMSE(estimated_support_ls) - NMSE(oracle_support_ls)

    Returns H_rec[n, m].
    """
    n_sc = ofdm.num_subcarriers
    n_sym = ofdm.num_symbols
    n_paths = len(est.delay_bins)

    if n_paths == 0:
        return np.zeros((n_sc, n_sym), dtype=np.complex128)

    n_idx, m_idx = np.nonzero(pilot_mask)
    if n_idx.size == 0:
        raise ValueError("pilot mask contains no observations")

    # Build dictionary at pilot locations using ESTIMATED path parameters
    A_pilot = _build_raw_dict(
        n_sc, n_sym, n_idx, m_idx, est.delay_bins, est.doppler_bins,
    )

    # Estimate condition as a diagnostic (cheap)
    cond_est = _dict_condition_estimate(A_pilot)
    if cond_est > 1e6:
        warnings.warn(
            f"estimated_support_ls: dictionary condition estimate = {cond_est:.1e} "
            f"— near-identical DD columns may inflate NMSE.",
            RuntimeWarning,
        )

    _normalise_columns(A_pilot)

    y = pilot_observations[pilot_mask]
    g_hat = _ridge_ls(A_pilot, y)

    # Undo normalisation (see oracle_support_ls_reconstruction for explanation)
    col_norms = _col_norms(
        _build_raw_dict(n_sc, n_sym, n_idx, m_idx,
                        est.delay_bins, est.doppler_bins)
    )
    g_hat = g_hat / np.maximum(col_norms, np.finfo(np.float64).eps)

    return _synthesize_from_params(
        n_sc, n_sym, est.delay_bins, est.doppler_bins, g_hat
    )


def _build_raw_dict(
    n_sc: int,
    n_sym: int,
    n_idx: np.ndarray,
    m_idx: np.ndarray,
    delay_bins: np.ndarray,
    doppler_bins: np.ndarray,
) -> np.ndarray:
    """Build the (unnormalised) DD dictionary at pilot locations."""
    n_paths = len(delay_bins)
    A = np.zeros((n_idx.size, n_paths), dtype=np.complex128)
    for l in range(n_paths):
        delay_phase = np.exp(-2j * np.pi * n_idx * delay_bins[l] / n_sc)
        doppler_phase = np.exp(2j * np.pi * m_idx * doppler_bins[l] / n_sym)
        A[:, l] = delay_phase * doppler_phase
    return A


def _normalise_columns(A: np.ndarray) -> None:
    """In-place column normalisation to unit L2 norm."""
    norms = _col_norms(A)
    for j in range(A.shape[1]):
        if norms[j] > np.finfo(np.float64).eps:
            A[:, j] /= norms[j]


def _synthesize_from_params(
    n_sc: int,
    n_sym: int,
    delay_bins: np.ndarray,
    doppler_bins: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    """Synthesise TF channel from explicit {τ, ν, α} parameters."""
    H = np.zeros((n_sc, n_sym), dtype=np.complex128)
    n_arr = np.arange(n_sc, dtype=np.float64)[:, None]
    m_arr = np.arange(n_sym, dtype=np.float64)[None, :]
    for l in range(len(gains)):
        delay_phase = np.exp(-2j * np.pi * n_arr * delay_bins[l] / n_sc)
        doppler_phase = np.exp(2j * np.pi * m_arr * doppler_bins[l] / n_sym)
        H += gains[l] * delay_phase * doppler_phase
    return H


# ---------------------------------------------------------------------------
# Token perturbation / corruption (Gate 2)
# ---------------------------------------------------------------------------


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
    """Legacy wrapper — delegates to corrupt_tokens for backward compat."""
    return corrupt_tokens(
        tokens, valid, rng,
        delay_jitter_std=delay_std,
        doppler_jitter_std=doppler_std,
        random_false_paths=false_paths,
        drop_probability=miss_probability,
        max_delay_bins=max_delay_bins,
        max_abs_doppler_bins=max_abs_doppler_bins,
    )


def corrupt_tokens(
    tokens: np.ndarray,         # [L, 9] float32
    valid: np.ndarray,          # [L] bool
    rng: np.random.Generator,
    *,
    # --- Location perturbation ---
    delay_jitter_std: float = 0.0,
    doppler_jitter_std: float = 0.0,
    common_bias_delay: float = 0.0,
    common_bias_doppler: float = 0.0,
    # --- Token dropout ---
    drop_probability: float = 0.0,
    # --- False paths ---
    random_false_paths: int = 0,
    coherent_false_paths: int = 0,
    # For coherent false: true path positions (needed to place false near strong)
    true_delays: np.ndarray | None = None,
    true_dopplers: np.ndarray | None = None,
    # --- Gain perturbation ---
    gain_magnitude_factor: float | str = 1.0,
    gain_phase_rad: float = 0.0,
    # --- Permutation ---
    permute_paths: bool = False,
    # --- Null all tokens ---
    null_all: bool = False,
    # --- Grid bounds ---
    max_delay_bins: float = 12.0,
    max_abs_doppler_bins: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject controlled token errors for Gate 2 failure-boundary audit.

    Processing order (each step sees the output of the previous):
      1. Location jitter (independent + common bias)
      2. Gain magnitude error
      3. Gain phase error
      4. Token dropout (valid → False)
      5. Inject false paths (random + coherent) into free slots
      6. Path permutation
      7. Null-all override

    Returns (corrupted_tokens, corrupted_valid).  The input arrays are NOT
    modified in-place.

    Token layout (9-dim, v2):
      0: delay_bin      1: doppler_bin     2: normalised power
      3: confidence     4: sigma_delay     5: sigma_doppler
      6: relevance      7: Re(α)           8: Im(α)
    """
    out = tokens.copy()
    out_valid = valid.copy()
    max_paths = out.shape[0]
    active = np.flatnonzero(out_valid).copy()

    # ---- null-all shortcut ----
    if null_all:
        out_valid[:] = False
        return out, out_valid

    # ===================================================================
    # 1. Location perturbation
    # ===================================================================
    if active.size and (delay_jitter_std > 0 or doppler_jitter_std > 0
                        or common_bias_delay != 0 or common_bias_doppler != 0):
        # Independent jitter
        if delay_jitter_std > 0:
            out[active, 0] += rng.normal(0.0, delay_jitter_std, size=active.size)
        if doppler_jitter_std > 0:
            out[active, 1] += rng.normal(0.0, doppler_jitter_std, size=active.size)

        # Common bias
        out[active, 0] += common_bias_delay
        out[active, 1] += common_bias_doppler

        # Clip to physical range
        out[active, 0] = np.clip(out[active, 0], 0.0, max_delay_bins)
        out[active, 1] = np.clip(
            out[active, 1], -max_abs_doppler_bins, max_abs_doppler_bins,
        )

        # Update sigma fields + confidence based on total jitter
        total_std = math.sqrt(delay_jitter_std**2 + doppler_jitter_std**2
                              + common_bias_delay**2 + common_bias_doppler**2)
        out[active, 4] = np.clip(total_std, 0.0, 2.0)
        out[active, 5] = np.clip(total_std, 0.0, 2.0)
        confidence_decay = float(math.exp(-0.5 * total_std))
        out[active, 3] = np.clip(out[active, 3] * confidence_decay, 0.05, 1.0)

    # ===================================================================
    # 2. Gain magnitude error
    # ===================================================================
    if active.size and gain_magnitude_factor != 1.0:
        if isinstance(gain_magnitude_factor, str):
            # "uniform_0.5_2.0" → sample per-path from uniform range
            lo, hi = 0.5, 2.0
            factors = rng.uniform(lo, hi, size=active.size)
        else:
            factors = np.full(active.size, float(gain_magnitude_factor))
        out[active, 7] *= factors  # Re(α)
        out[active, 8] *= factors  # Im(α)
        # Update power field
        old_power = out[active, 2].copy()
        new_power = old_power * (factors ** 2)
        # Keep normalised power of ACTIVE tokens consistent; caller doesn't
        # rely on absolute normalisation but it's cleaner.
        out[active, 2] = new_power

    # ===================================================================
    # 3. Gain phase error
    # ===================================================================
    if active.size and gain_phase_rad != 0.0:
        if isinstance(gain_phase_rad, str):
            # "uniform_pi8_pi" → random phase ∈ [π/8, π]
            lo, hi = math.pi / 8, math.pi
            phases = rng.uniform(lo, hi, size=active.size)
        else:
            phases = np.full(active.size, float(gain_phase_rad))
        cos_p = np.cos(phases)
        sin_p = np.sin(phases)
        re = out[active, 7].copy()
        im = out[active, 8].copy()
        out[active, 7] = re * cos_p - im * sin_p
        out[active, 8] = re * sin_p + im * cos_p

    # ===================================================================
    # 4. Token dropout
    # ===================================================================
    if active.size and drop_probability > 0:
        drop_mask = rng.random(active.size) < drop_probability
        out_valid[active[drop_mask]] = False
        active = np.flatnonzero(out_valid)  # refresh

    # ===================================================================
    # 5. Inject false paths into free slots
    # ===================================================================
    total_false = random_false_paths + coherent_false_paths
    if total_false > 0:
        free = np.flatnonzero(~out_valid)
        n_free = len(free)
        n_inject = min(total_false, n_free)
        if n_inject < total_false:
            warnings.warn(
                f"corrupt_tokens: only {n_inject}/{total_false} false paths "
                f"injected ({n_free} free slots, {max_paths} max).",
                RuntimeWarning,
            )

        # Split free slots: first coherent, then random
        n_coherent_inject = min(coherent_false_paths, n_inject)
        n_random_inject = n_inject - n_coherent_inject

        # --- Coherent false paths ---
        if n_coherent_inject > 0 and true_delays is not None and true_dopplers is not None \
                and len(true_delays) > 0 and len(true_dopplers) > 0:
            # Pick the strongest true path as anchor
            strongest_idx = 0  # paths are sorted by power
            anchor_tau = float(true_delays[strongest_idx])
            anchor_nu = float(true_dopplers[strongest_idx])
            for idx in free[:n_coherent_inject]:
                # Small offset from the strongest true path
                out[idx, 0] = anchor_tau + rng.uniform(-0.5, 0.5)
                out[idx, 1] = anchor_nu + rng.uniform(-0.5, 0.5)
                out[idx, 0] = np.clip(out[idx, 0], 0.0, max_delay_bins)
                out[idx, 1] = np.clip(out[idx, 1], -max_abs_doppler_bins, max_abs_doppler_bins)
                # Plausible gain: 20-60% of the anchor gain
                scale = rng.uniform(0.2, 0.6)
                if true_delays is not None and len(true_delays) > strongest_idx:
                    # Use the anchor gain magnitude
                    out[idx, 7] = scale  # placeholder; real gain unknown
                    out[idx, 8] = 0.0
                out[idx, 2] = rng.uniform(0.05, 0.3)
                out[idx, 3] = rng.uniform(0.2, 0.5)  # medium confidence
                out[idx, 4] = 0.5
                out[idx, 5] = 0.5
                out[idx, 6] = rng.uniform(0.0, 0.5)
                out_valid[idx] = True

        # --- Random false paths ---
        if n_random_inject > 0:
            free_remaining = free[n_coherent_inject:n_inject]
            for idx in free_remaining:
                out[idx, 0] = rng.uniform(0.0, max_delay_bins)
                out[idx, 1] = rng.uniform(-max_abs_doppler_bins, max_abs_doppler_bins)
                out[idx, 2] = rng.uniform(0.01, 0.1)
                out[idx, 3] = rng.uniform(0.05, 0.4)
                out[idx, 4] = 1.0
                out[idx, 5] = 1.0
                out[idx, 6] = rng.uniform(0.0, 0.3)
                out[idx, 7] = 0.0
                out[idx, 8] = 0.0
                out_valid[idx] = True

    # ===================================================================
    # 6. Path permutation (shuffle order of valid tokens)
    # ===================================================================
    if permute_paths and active.size > 1:
        active = np.flatnonzero(out_valid)
        perm = rng.permutation(len(active))
        out[active] = out[active[perm]]
        # valid mask stays the same — only the ORDER of active slots changes

    return out, out_valid
