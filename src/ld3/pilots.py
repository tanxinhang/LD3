from __future__ import annotations

import numpy as np


def make_pilot_mask(
    num_subcarriers: int,
    num_symbols: int,
    density: float,
    rng: np.random.Generator,
    pattern: str = "random",
) -> np.ndarray:
    """Construct a 2-D pilot mask.

    Random masks are preferred for Gate 0 because they reduce coherent DD
    ambiguities. Comb masks are provided to expose realistic aliasing failure
    modes rather than hiding them.
    """

    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    shape = (num_subcarriers, num_symbols)

    if pattern == "random":
        count = max(1, int(round(density * num_subcarriers * num_symbols)))
        flat = np.zeros(num_subcarriers * num_symbols, dtype=bool)
        flat[rng.choice(flat.size, size=count, replace=False)] = True
        return flat.reshape(shape)

    if pattern == "comb":
        stride = max(1, int(round(1.0 / np.sqrt(density))))
        mask = np.zeros(shape, dtype=bool)
        mask[::stride, ::stride] = True
        return mask

    raise ValueError(f"unknown pilot pattern: {pattern}")


def generate_noise_grid(
    shape: tuple[int, int],
    signal_power: float,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Generate a full-grid complex AWGN realisation.

    SNR is defined as signal_power / noise_variance, computed over the full
    grid, not just pilot positions.  This keeps the noise level identical
    regardless of pilot mask — critical for paired Random-vs-Comb comparisons.

    Returns (noise_grid, noise_variance).
    """
    noise_var = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
    return noise, noise_var


def observe_pilots(
    channel_tf: np.ndarray,
    mask: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    noise_grid: np.ndarray | None = None,
    noise_var: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return masked LS observations using unit-modulus pilots.

    Parameters
    ----------
    channel_tf : shape (N, M)
        Ground-truth TF channel.
    mask : shape (N, M) bool
        Pilot positions.
    snr_db : float
        Only used when noise_grid is None (legacy / non-paired mode).
    rng : np.random.Generator
        Only used when noise_grid is None.
    noise_grid : shape (N, M), optional
        Pre-generated noise grid.  When provided, snr_db and rng are ignored.
        Use this for paired Random-vs-Comb comparisons.
    noise_var : float, optional
        Pre-computed noise variance corresponding to noise_grid.

    Returns (observed, noise_variance).
    """
    if channel_tf.shape != mask.shape:
        raise ValueError("channel_tf and mask shapes must match")

    if noise_grid is not None:
        if noise_grid.shape != channel_tf.shape:
            raise ValueError("noise_grid and channel_tf shapes must match")
        noise = noise_grid
        nv = noise_var if noise_var is not None else float(np.mean(np.abs(noise_grid) ** 2))
    else:
        pilot_signal = channel_tf[mask]
        signal_power = float(np.mean(np.abs(pilot_signal) ** 2))
        nv = signal_power / (10.0 ** (snr_db / 10.0))
        noise = np.sqrt(nv / 2.0) * (
            rng.standard_normal(channel_tf.shape)
            + 1j * rng.standard_normal(channel_tf.shape)
        )

    observed = np.zeros_like(channel_tf, dtype=np.complex128)
    observed[mask] = channel_tf[mask] + noise[mask]
    return observed, nv
