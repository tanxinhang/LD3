from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OFDMConfig:
    """OFDM grid and carrier configuration."""

    num_subcarriers: int = 64
    num_symbols: int = 14
    subcarrier_spacing_hz: float = 120e3
    carrier_frequency_hz: float = 28e9
    cp_ratio: float = 0.07

    @property
    def useful_symbol_s(self) -> float:
        return 1.0 / self.subcarrier_spacing_hz

    @property
    def symbol_period_s(self) -> float:
        return self.useful_symbol_s * (1.0 + self.cp_ratio)

    @property
    def cp_duration_s(self) -> float:
        return self.useful_symbol_s * self.cp_ratio

    @property
    def bandwidth_hz(self) -> float:
        return self.num_subcarriers * self.subcarrier_spacing_hz

    @property
    def delay_resolution_s(self) -> float:
        """Delay-bin width in seconds: 1 / bandwidth."""
        return 1.0 / self.bandwidth_hz

    @property
    def doppler_resolution_hz(self) -> float:
        """Doppler-bin width in Hz: 1 / (M * T_sym)."""
        return 1.0 / (self.num_symbols * self.symbol_period_s)

    def delay_bins_to_seconds(self, bins: float) -> float:
        return bins * self.delay_resolution_s

    def doppler_bins_to_hz(self, bins: float) -> float:
        return bins * self.doppler_resolution_hz

    def doppler_hz_to_speed_mps(self, fd_hz: float) -> float:
        """Convert Doppler shift in Hz to line-of-sight speed in m/s."""
        SPEED_OF_LIGHT = 299_792_458.0
        return fd_hz * SPEED_OF_LIGHT / max(self.carrier_frequency_hz, 1.0)

    def validate_cp(self, max_delay_bins: float) -> dict[str, float]:
        """Check whether max delay fits within the cyclic prefix.

        Returns a dict with delay_s, cp_s, and a boolean `ok`.
        Issues a RuntimeWarning if max delay exceeds CP.
        """
        import warnings
        delay_s = self.delay_bins_to_seconds(max_delay_bins)
        cp_s = self.cp_duration_s
        ok = delay_s <= cp_s
        if not ok:
            warnings.warn(
                f"max_delay ({max_delay_bins} bins = {delay_s*1e6:.2f} μs) "
                f"exceeds CP ({cp_s*1e6:.2f} μs). "
                f"The current channel model is a TF surface abstraction "
                f"(no time-domain convolution, no ISI). "
                f"Results should be labelled accordingly.",
                RuntimeWarning,
            )
        return {"delay_s": delay_s, "cp_s": cp_s, "ok": ok}


@dataclass(frozen=True)
class ChannelConfig:
    """Sparse doubly-selective channel configuration."""

    num_paths: int = 4
    max_delay_bins: float = 12.0
    max_abs_doppler_bins: float = 3.0
    fractional_delay: bool = True
    fractional_doppler: bool = True
    rician_k_db: float | None = None
    exponential_power_decay: float = 0.25
