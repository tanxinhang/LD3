import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from ld3.channel import PathSet, synthesize_tf_channel
from ld3.config import OFDMConfig
from ld3.dd_estimation import (
    ambiguity_metrics,
    build_dd_dictionary,
    build_dd_grid,
    confidence_interval,
    detect_paths_nms,
    detect_paths_oracle_nms,
    dictionary_coherence,
    identifiability_metrics,
    masked_matched_filter_map,
    paired_bootstrap_test,
    pilot_ambiguity_function,
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


def test_confidence_interval_with_nan() -> None:
    values = np.array([1.0, 2.0, float("nan"), 4.0, 5.0], dtype=np.float64)
    ci = confidence_interval(values)
    assert abs(ci["mean"] - 3.0) < 0.01
    assert ci["n_eff"] == 4  # NaN excluded
    assert ci["se"] > 0
    assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]


def test_paired_bootstrap() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 0.1, size=100)
    b = rng.normal(0.0, 0.1, size=100)
    bt = paired_bootstrap_test(a, b, n_bootstrap=2000)
    assert bt["mean_diff"] > 0
    assert bt["ci_lower"] < bt["mean_diff"] < bt["ci_upper"]
    assert bt["p_value"] < 0.05
    assert bt["n_pairs"] == 100


def test_ospa_distance() -> None:
    true_delay = np.array([2.0, 7.0])
    true_doppler = np.array([-1.0, 1.5])
    true_power = np.array([0.5, 0.5])
    from ld3.dd_estimation import EstimatedPaths
    # Perfect match
    est = EstimatedPaths(
        delay_bins=np.array([2.0, 7.0]),
        doppler_bins=np.array([-1.0, 1.5]),
        scores=np.array([1.0, 0.8]),
        gains=np.array([1.0 + 0j, 0.5 + 0j]),
    )
    metrics = identifiability_metrics(
        true_delay, true_doppler, true_power, est,
        delay_tolerance=0.75, doppler_tolerance=0.5,
    )
    assert metrics["ospa_distance"] < 0.1
    assert metrics["path_recall"] == 1.0
    assert metrics["path_precision"] == 1.0

    # Complete miss
    est_bad = EstimatedPaths(
        delay_bins=np.array([20.0, 25.0]),
        doppler_bins=np.array([10.0, 12.0]),
        scores=np.array([0.3, 0.2]),
        gains=np.array([0.1 + 0j, 0.1 + 0j]),
    )
    metrics_bad = identifiability_metrics(
        true_delay, true_doppler, true_power, est_bad,
        delay_tolerance=0.75, doppler_tolerance=0.5,
    )
    assert metrics_bad["path_recall"] == 0.0
    assert metrics_bad["ospa_distance"] > 0.0


def test_penalized_rmse_uses_tolerance() -> None:
    """Penalized RMSE should use tolerance as miss penalty, not full range."""
    true_delay = np.array([2.0, 7.0, 4.0])
    true_doppler = np.array([-1.0, 1.5, 0.0])
    true_power = np.array([0.5, 0.3, 0.2])
    from ld3.dd_estimation import EstimatedPaths
    # Only find 2 of 3 paths
    est = EstimatedPaths(
        delay_bins=np.array([2.0, 7.0]),
        doppler_bins=np.array([-1.0, 1.5]),
        scores=np.array([1.0, 0.8]),
        gains=np.array([1.0 + 0j, 0.5 + 0j]),
    )
    delay_tol = 0.75
    doppler_tol = 0.5
    metrics = identifiability_metrics(
        true_delay, true_doppler, true_power, est,
        delay_tolerance=delay_tol, doppler_tolerance=doppler_tol,
    )
    # Penalized RMSE should be larger than matched-only RMSE
    assert metrics["penalized_delay_rmse_bins"] > metrics["delay_rmse_bins"]
    assert metrics["num_missed"] == 1
    assert metrics["path_recall"] == 2 / 3
    # With tolerance-based penalty, the penalized value should be:
    # sqrt((0 + delay_tol^2) / 3) = delay_tol / sqrt(3)
    expected = delay_tol / np.sqrt(3)
    assert abs(metrics["penalized_delay_rmse_bins"] - expected) < 0.01


def test_dictionary_coherence_far_field() -> None:
    mask = np.ones((32, 8), dtype=bool)
    grid = build_dd_grid(32, 8, 6.0, 2.0, 2, 4)
    coh = dictionary_coherence(mask, grid, nms_delay_radius=2, nms_doppler_radius=2)
    assert 0.0 <= coh["mu_max"] <= 1.0
    assert "mu_far" in coh
    assert "mu_p95" in coh
    assert "mu_p99" in coh
    assert coh["n_dictionary_columns"] > 0
    # mu_far should be <= mu_max (far-field excludes local columns)
    assert coh["mu_far"] <= coh["mu_max"] + 1e-10


def test_pilot_ambiguity_function() -> None:
    rng = np.random.default_rng(42)
    from ld3.pilots import make_pilot_mask
    mask = make_pilot_mask(32, 8, 0.25, rng, "random")
    grid = build_dd_grid(32, 8, 6.0, 2.0, 2, 4)
    af = pilot_ambiguity_function(mask, grid)
    assert af.shape == (grid.delay_bins.size, grid.doppler_bins.size)
    assert abs(float(np.max(af)) - 1.0) < 0.01
    am = ambiguity_metrics(af, delay_radius=2, doppler_radius=2)
    assert "pslr_db" in am
    assert "max_far_sidelobe_delay_bin" in am


def test_oracle_nms() -> None:
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
    est = detect_paths_oracle_nms(score, gains, grid, paths.delay_bins, paths.doppler_bins)
    assert len(est.delay_bins) == 2
    for td in paths.delay_bins:
        assert np.min(np.abs(est.delay_bins - td)) < 1.0
