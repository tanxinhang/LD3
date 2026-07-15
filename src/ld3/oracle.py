from __future__ import annotations

import math
import numpy as np

from .channel import PathSet, synthesize_tf_channel
from .config import OFDMConfig
from .dd_estimation import EstimatedPaths


# ---------------------------------------------------------------------------
# Oracle path tokens
# ---------------------------------------------------------------------------


def oracle_path_tokens(paths: PathSet, max_paths: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-width path tokens and a validity mask.

    Token fields: delay_bin, doppler_bin, normalized power, confidence,
    sigma_delay, sigma_doppler, communication relevance.

    NOTE: this 7-dim token does NOT include complex gain (Re α, Im α).
    The model fix adding complex-gain tokens is planned for the next revision
    (Gate 1-D: learned fusion value).
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
        # rhs[i] = Σ_k conj(A[k,i]) * y[k]
        s_rhs = 0.0 + 0.0j
        for k in range(n_rows):
            s_rhs += A[k, i].conjugate() * y[k]
        rhs[i] = s_rhs

        for j in range(i, n_cols):
            # gram[i,j] = Σ_k conj(A[k,i]) * A[k,j]
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

    Uses PURE PYTHON linear algebra throughout — no BLAS, no LAPACK, no
    MKL of any kind.  This is the only way to survive the SIGABRT-happy
    Anaconda/Windows numpy build.
    """
    n_cols = A.shape[1]
    if n_cols == 0:
        return np.zeros(0, dtype=np.complex128)

    # Manual A^H A  and  A^H y  (no @, no np.dot)
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
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    # Cholesky: A = L @ L^T
    for i in range(n):
        for j in range(i + 1):
            s = float(A[i, j])
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    s = np.finfo(np.float64).eps
                L[i, j] = math.sqrt(s)       # math.sqrt, NOT np.sqrt
            else:
                L[i, j] = s / L[j, j]

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

    y = pilot_observations[pilot_mask]
    g_hat = _ridge_ls(A_pilot, y)

    # Reconstruct full TF channel with estimated gains
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

    Uses DD-ESTIMATED (not true) path locations.  This bridges Gate 0 and
    Gate 1: it measures how well DD-identified support translates to TF
    channel NMSE.

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

    A_pilot = np.zeros((n_idx.size, n_paths), dtype=np.complex128)
    for l in range(n_paths):
        delay_phase = np.exp(-2j * np.pi * n_idx * est.delay_bins[l] / n_sc)
        doppler_phase = np.exp(2j * np.pi * m_idx * est.doppler_bins[l] / n_sym)
        A_pilot[:, l] = delay_phase * doppler_phase

    y = pilot_observations[pilot_mask]
    g_hat = _ridge_ls(A_pilot, y)

    return _synthesize_from_params(
        n_sc, n_sym, est.delay_bins, est.doppler_bins, g_hat
    )


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
# Token perturbation (Gate 2)
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
