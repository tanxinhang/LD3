from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def nearest_smooth_interpolation(
    pilot_observations: np.ndarray,
    pilot_mask: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Simple deterministic TF baseline: nearest fill followed by smoothing."""

    if pilot_observations.shape != pilot_mask.shape:
        raise ValueError("pilot_observations and pilot_mask shapes must match")
    if not np.any(pilot_mask):
        raise ValueError("pilot mask contains no observations")

    missing = ~pilot_mask
    nearest_indices = distance_transform_edt(
        missing,
        return_distances=False,
        return_indices=True,
    )
    filled = pilot_observations[tuple(nearest_indices)]
    real = gaussian_filter(filled.real, sigma=sigma, mode="nearest")
    imag = gaussian_filter(filled.imag, sigma=sigma, mode="nearest")
    estimate = real + 1j * imag
    estimate[pilot_mask] = pilot_observations[pilot_mask]
    return estimate
