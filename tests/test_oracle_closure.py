"""Physical closure tests — must pass before any Gate 1 NMSE decomposition."""

import sys
from pathlib import Path

# Ensure src/ is on the path (consistent with experiment scripts)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from ld3.channel import generate_path_set, synthesize_tf_channel
from ld3.config import ChannelConfig, OFDMConfig
from ld3.oracle import (
    oracle_perfect_reconstruction,
    oracle_support_ls_reconstruction,
)
from ld3.metrics import nmse_numpy


def test_oracle_perfect_is_numerically_closed() -> None:
    """Oracle perfect MUST approach machine precision.

    If this fails, the synthesis / reconstruction formulas are inconsistent
    and every downstream NMSE decomposition is invalid.
    """
    ofdm = OFDMConfig(num_subcarriers=64, num_symbols=14)
    channel = ChannelConfig(
        num_paths=4,
        max_delay_bins=12.0,
        max_abs_doppler_bins=3.0,
        fractional_delay=True,
        fractional_doppler=True,
    )
    rng = np.random.default_rng(42)
    paths = generate_path_set(ofdm, channel, rng)
    h_true = synthesize_tf_channel(ofdm, paths)
    h_oracle = oracle_perfect_reconstruction(ofdm, paths)

    nmse = nmse_numpy(h_oracle, h_true)
    assert nmse < 1e-10, (
        f"Oracle perfect NMSE = {nmse:.3e} — should be < 1e-10.\n"
        f"Check: delay/Doppler sign convention, normalisation, path truncation."
    )


def test_oracle_support_ls_noiseless_is_closed() -> None:
    """Oracle support + LS with zero noise should approach perfect reconstruction.

    With no noise, LS gain estimation should recover the true gains exactly
    (up to regularisation tolerance).
    """
    ofdm = OFDMConfig(num_subcarriers=64, num_symbols=14)
    channel = ChannelConfig(
        num_paths=4,
        max_delay_bins=12.0,
        max_abs_doppler_bins=3.0,
        fractional_delay=True,
        fractional_doppler=True,
    )
    rng = np.random.default_rng(42)
    paths = generate_path_set(ofdm, channel, rng)
    h_true = synthesize_tf_channel(ofdm, paths)

    # No noise: observed = true channel at all positions
    mask = np.ones((ofdm.num_subcarriers, ofdm.num_symbols), dtype=bool)
    observed = h_true.copy()

    h_ls = oracle_support_ls_reconstruction(ofdm, paths, observed, mask)
    nmse = nmse_numpy(h_ls, h_true)

    # With full observation + no noise + trace-regularised LS, NMSE should be
    # very small (regularisation introduces a tiny bias but ≪ 1e-6)
    assert nmse < 1e-6, (
        f"Noiseless Oracle+LS NMSE = {nmse:.3e} — should be < 1e-6.\n"
        f"Check: ridge regularisation, LS formulation, complex-gain sign."
    )


def test_known_K_fp_equals_fn() -> None:
    """Under Known-K with Top-K output and no early-exit: FP == FN.

    This is a mathematical identity, not an empirical claim.
    When the detector outputs exactly K paths and TP < K:
      FN = K - TP,  FP = K - TP  →  FP == FN.
    """
    from ld3.dd_estimation import (
        EstimatedPaths,
        identifiability_metrics,
    )

    true_delay = np.array([2.0, 7.0, 4.0, 9.0])
    true_doppler = np.array([-1.0, 1.5, 0.0, -2.0])
    true_power = np.array([0.4, 0.3, 0.2, 0.1])

    # Simulate: K=4 estimates, 3 matched → 1 FP, 1 FN
    est = EstimatedPaths(
        delay_bins=np.array([2.0, 7.0, 4.0, 20.0]),  # 4th is wrong
        doppler_bins=np.array([-1.0, 1.5, 0.0, 10.0]),
        scores=np.array([1.0, 0.8, 0.6, 0.2]),
        gains=np.array([1.0 + 0j, 0.8 + 0j, 0.6 + 0j, 0.1 + 0j]),
    )

    metrics = identifiability_metrics(
        true_delay, true_doppler, true_power, est,
        delay_tolerance=0.75, doppler_tolerance=0.5,
    )
    assert metrics["path_recall"] == 0.75  # 3/4
    assert metrics["num_missed"] == 1
    assert metrics["num_false_alarms"] == 1, (
        f"num_false_alarms={metrics['num_false_alarms']}, expected 1.\n"
        f"Under Known-K with Top-K=4 and TP=3: FP MUST equal FN=1."
    )
