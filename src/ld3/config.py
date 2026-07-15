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
    def bandwidth_hz(self) -> float:
        return self.num_subcarriers * self.subcarrier_spacing_hz

    @property
    def delay_resolution_s(self) -> float:
        return 1.0 / self.bandwidth_hz

    @property
    def doppler_resolution_hz(self) -> float:
        return 1.0 / (self.num_symbols * self.symbol_period_s)


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
