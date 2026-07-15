import numpy as np

from ld3.channel import generate_path_set, synthesize_tf_channel
from ld3.config import ChannelConfig, OFDMConfig


def test_channel_shape_and_finite() -> None:
    ofdm = OFDMConfig(num_subcarriers=32, num_symbols=8)
    channel = ChannelConfig(num_paths=3, max_delay_bins=8, max_abs_doppler_bins=2)
    paths = generate_path_set(ofdm, channel, np.random.default_rng(1))
    h = synthesize_tf_channel(ofdm, paths)
    assert h.shape == (32, 8)
    assert np.all(np.isfinite(h))
    assert np.isclose(np.sum(paths.power), 1.0)
