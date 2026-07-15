from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ChannelConfig, OFDMConfig


@dataclass
class PathSet:
    """Physical sparse paths represented in normalized DD-bin coordinates."""

    delay_bins: np.ndarray
    doppler_bins: np.ndarray
    gains: np.ndarray

    def __post_init__(self) -> None:
        self.delay_bins = np.asarray(self.delay_bins, dtype=np.float64)
        self.doppler_bins = np.asarray(self.doppler_bins, dtype=np.float64)
        self.gains = np.asarray(self.gains, dtype=np.complex128)
        if not (
            self.delay_bins.shape == self.doppler_bins.shape == self.gains.shape
        ):
            raise ValueError("delay_bins, doppler_bins, and gains must have equal shapes")

    @property
    def power(self) -> np.ndarray:
        return np.abs(self.gains) ** 2

    def normalized_power(self) -> np.ndarray:
        total = float(np.sum(self.power))
        return self.power / max(total, np.finfo(float).eps)


def generate_path_set(
    ofdm: OFDMConfig,
    channel: ChannelConfig,
    rng: np.random.Generator,
) -> PathSet:
    """Generate resolvable sparse paths with optional fractional offsets."""

    if channel.num_paths < 1:
        raise ValueError("num_paths must be positive")
    if channel.max_delay_bins >= ofdm.num_subcarriers:
        raise ValueError("max_delay_bins must be smaller than num_subcarriers")

    integer_delay = rng.choice(
        np.arange(int(np.ceil(channel.max_delay_bins))),
        size=channel.num_paths,
        replace=channel.num_paths > int(np.ceil(channel.max_delay_bins)),
    ).astype(np.float64)
    if channel.fractional_delay:
        integer_delay += rng.uniform(-0.45, 0.45, size=channel.num_paths)
    delay_bins = np.clip(integer_delay, 0.0, channel.max_delay_bins)

    doppler_bins = rng.uniform(
        -channel.max_abs_doppler_bins,
        channel.max_abs_doppler_bins,
        size=channel.num_paths,
    )
    if not channel.fractional_doppler:
        doppler_bins = np.round(doppler_bins)

    order = np.argsort(delay_bins)
    decay = np.exp(-channel.exponential_power_decay * np.arange(channel.num_paths))
    decay = decay / np.sum(decay)
    random_complex = (
        rng.standard_normal(channel.num_paths)
        + 1j * rng.standard_normal(channel.num_paths)
    ) / np.sqrt(2.0)
    gains = random_complex * np.sqrt(decay)

    if channel.rician_k_db is not None:
        k_lin = 10.0 ** (channel.rician_k_db / 10.0)
        gains *= np.sqrt(1.0 / (k_lin + 1.0))
        strongest = int(np.argmax(np.abs(gains)))
        gains[strongest] += np.sqrt(k_lin / (k_lin + 1.0))

    delay_bins = delay_bins[order]
    doppler_bins = doppler_bins[order]
    gains = gains[order]
    gains /= np.sqrt(np.sum(np.abs(gains) ** 2) + np.finfo(float).eps)
    return PathSet(delay_bins, doppler_bins, gains)


def synthesize_tf_channel(ofdm: OFDMConfig, paths: PathSet) -> np.ndarray:
    """Synthesize H[n,m] from DD-bin path parameters.

    delay_bins are normalized to 1/(N*Delta_f), while doppler_bins are
    normalized to 1/(M*T_sym). This keeps the implementation independent of
    a particular bandwidth while retaining the exact OFDM phase law.
    """

    n = np.arange(ofdm.num_subcarriers, dtype=np.float64)[:, None, None]
    m = np.arange(ofdm.num_symbols, dtype=np.float64)[None, :, None]
    delay = paths.delay_bins[None, None, :]
    doppler = paths.doppler_bins[None, None, :]

    phase_delay = -2j * np.pi * n * delay / ofdm.num_subcarriers
    phase_doppler = 2j * np.pi * m * doppler / ofdm.num_symbols
    response = np.exp(phase_delay + phase_doppler)
    return np.sum(response * paths.gains[None, None, :], axis=-1)


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Add complex AWGN and return noisy signal and noise variance."""

    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_var = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return signal + noise, noise_var
