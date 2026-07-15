from __future__ import annotations

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
# Ridge LS helper
# ---------------------------------------------------------------------------


def _ridge_ls(
    A: np.ndarray,
    y: np.ndarray,
    ridge_relative: float = 1e-6,
) -> np.ndarray:
    """Solve min ||A x - y||² + λ||x||² with trace-scaled regularisation.

    λ = ridge_relative * tr(A^H A) / L   where L = number of columns.

    Uses real-valued decomposition of the augmented system to avoid MKL
    crashes on complex128 matrices (observed on Windows + Anaconda).
    """
    n_cols = A.shape[1]
    if n_cols == 0:
        return np.zeros(0, dtype=np.complex128)
    gram = A.conj().T @ A
    trace_gram = float(np.trace(gram).real)
    ridge = ridge_relative * trace_gram / max(n_cols, 1)
    sqrt_ridge = np.sqrt(max(ridge, np.finfo(float).eps))

    # Augmented system: [A; √λ·I] x ≈ [y; 0]
    A_aug = np.vstack([A, sqrt_ridge * np.eye(n_cols)])
    y_aug = np.concatenate([y, np.zeros(n_cols, dtype=A.dtype)])

    # Split into real-valued system to avoid MKL complex-path SIGABRT.
    # [Re(A)  -Im(A)] [Re(x)]   [Re(y)]
    # [Im(A)   Re(A)] [Im(x)] = [Im(y)]
    A_real = np.block([
        [A_aug.real, -A_aug.imag],
        [A_aug.imag,  A_aug.real],
    ])
    y_real = np.concatenate([y_aug.real, y_aug.imag])
    result_real, _residuals, _rank, _singulars = np.linalg.lstsq(
        A_real, y_real, rcond=None
    )
    n = n_cols
    return result_real[:n] + 1j * result_real[n:]


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
