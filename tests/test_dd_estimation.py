import numpy as np

from ld3.channel import PathSet, synthesize_tf_channel
from ld3.config import OFDMConfig
from ld3.dd_estimation import (
    build_dd_grid,
    detect_paths_nms,
    identifiability_metrics,
    masked_matched_filter_map,
)


def test_full_observation_recovers_dominant_power() -> None:
    ofdm = OFDMConfig(num_subcarriers=64, num_symbols=16)
    paths = PathSet(
        delay_bins=np.array([2.0, 7.0]),
        doppler_bins=np.array([-1.0, 1.5]),
        gains=np.array([np.sqrt(0.75), np.sqrt(0.25) * 1j]),
    )
    truth = synthesize_tf_channel(ofdm, paths)
    mask = np.ones_like(truth, dtype=bool)
    grid = build_dd_grid(64, 16, 10.0, 3.0, 2, 4)
    score, gains = masked_matched_filter_map(truth, mask, grid)
    estimated = detect_paths_nms(score, gains, grid, num_paths=2, delay_radius=2, doppler_radius=2)
    metrics = identifiability_metrics(
        paths.delay_bins,
        paths.doppler_bins,
        paths.power,
        estimated,
        delay_tolerance=0.6,
        doppler_tolerance=0.35,
    )
    assert metrics["power_recovery"] > 0.95
    assert metrics["path_recall"] == 1.0
