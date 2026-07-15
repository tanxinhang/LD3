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


def observe_pilots(
    channel_tf: np.ndarray,
    mask: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Return masked LS observations using unit-modulus pilots."""

    if channel_tf.shape != mask.shape:
        raise ValueError("channel_tf and mask shapes must match")
    pilot_signal = channel_tf[mask]
    signal_power = float(np.mean(np.abs(pilot_signal) ** 2))
    noise_var = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(pilot_signal.shape)
        + 1j * rng.standard_normal(pilot_signal.shape)
    )
    observed = np.zeros_like(channel_tf, dtype=np.complex128)
    observed[mask] = pilot_signal + noise
    return observed, noise_var
