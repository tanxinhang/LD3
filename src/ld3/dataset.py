from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .channel import generate_path_set, synthesize_tf_channel
from .config import ChannelConfig, OFDMConfig
from .interpolation import nearest_smooth_interpolation
from .dd_estimation import (
    build_dd_grid,
    detect_paths_nms,
    detect_paths_omp,
    masked_matched_filter_map,
    refine_paths_variable_projection,
)
from .oracle import (
    compute_path_quality,
    estimated_path_tokens_v2,
    estimated_path_tokens_v3,
    oracle_path_tokens,
    oracle_path_tokens_v2,
)
from .pilots import generate_noise_grid, make_pilot_mask, observe_pilots


@dataclass(frozen=True)
class DatasetConfig:
    size: int = 1024
    snr_min_db: float = -5.0
    snr_max_db: float = 20.0
    pilot_density: float = 0.125
    pilot_pattern: str = "random"
    max_paths: int = 8
    seed: int = 2036
    cache_in_memory: bool = True
    token_version: int = 1  # 1 = legacy 7-dim, 2 = 9-dim with Re(α), Im(α)
    token_source: str = "oracle"  # "oracle" | "estimated"
    token_refine: str = ""  # "" | "vp" — apply continuous refinement to estimated tokens
    token_vp_rounds: int = 3  # VP coordinate descent rounds
    token_vp_probes: int = 8  # VP random probes per path per round
    token_vp_fast: bool = False  # column-cached Gram update (~4x faster)
    dd_oversample_delay: int = 2   # DD grid oversampling factor (delay axis)
    dd_oversample_doppler: int = 4  # DD grid oversampling factor (doppler axis)
    detector_method: str = "nms"  # "nms" | "omp" — DD path detection algorithm


class SyntheticOFDMISACDataset(Dataset):
    """Deterministic-on-index synthetic dataset for Gate 1 experiments."""

    def __init__(
        self,
        ofdm: OFDMConfig,
        channel: ChannelConfig,
        cfg: DatasetConfig,
    ) -> None:
        self.ofdm = ofdm
        self.channel = channel
        self.cfg = cfg
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.cfg.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.cfg.cache_in_memory and index in self._cache:
            return self._cache[index]

        rng = np.random.default_rng([self.cfg.seed, index])
        paths = generate_path_set(self.ofdm, self.channel, rng)
        truth = synthesize_tf_channel(self.ofdm, paths)
        snr_db = rng.uniform(self.cfg.snr_min_db, self.cfg.snr_max_db)
        mask = make_pilot_mask(
            self.ofdm.num_subcarriers,
            self.ofdm.num_symbols,
            self.cfg.pilot_density,
            rng,
            self.cfg.pilot_pattern,
        )
        # Full-grid SNR definition (consistent with Gate 0 paired design)
        signal_power = float(np.mean(np.abs(truth) ** 2))
        noise_grid, noise_var = generate_noise_grid(
            truth.shape, signal_power, snr_db, rng,
        )
        observed, _ = observe_pilots(
            truth, mask, snr_db, rng,
            noise_grid=noise_grid, noise_var=noise_var,
        )
        initial = nearest_smooth_interpolation(observed, mask)
        tokens, valid = oracle_path_tokens(paths, self.cfg.max_paths)
        if self.cfg.token_source == "estimated":
            # Gate 1-E: DD-estimated support + LS gains as tokens
            grid = build_dd_grid(
                self.ofdm.num_subcarriers, self.ofdm.num_symbols,
                self.channel.max_delay_bins, self.channel.max_abs_doppler_bins,
                self.cfg.dd_oversample_delay, self.cfg.dd_oversample_doppler,
            )
            score_map, gain_map = masked_matched_filter_map(observed, mask, grid)
            if self.cfg.detector_method == "omp":
                est = detect_paths_omp(
                    score_map, gain_map, grid,
                    num_paths=self.channel.num_paths,
                    pilot_observations=observed, pilot_mask=mask,
                    num_subcarriers=self.ofdm.num_subcarriers,
                    num_symbols=self.ofdm.num_symbols,
                )
            else:
                est = detect_paths_nms(
                    score_map, gain_map, grid, num_paths=self.channel.num_paths,
                )
            if len(est.delay_bins) > 0:
                # Optional: VP continuous refinement
                if self.cfg.token_refine == "vp":
                    est, _vp_diag = refine_paths_variable_projection(
                        est,
                        pilot_observations=observed,
                        pilot_mask=mask,
                        num_subcarriers=self.ofdm.num_subcarriers,
                        num_symbols=self.ofdm.num_symbols,
                        n_rounds=self.cfg.token_vp_rounds,
                        n_probes=self.cfg.token_vp_probes,
                        fast_mode=self.cfg.token_vp_fast,
                    )
                # LS gain estimation at (possibly refined) DD positions
                from .oracle import _ridge_ls, _col_norms, _build_raw_dict
                n_idx, m_idx = np.nonzero(mask)
                A_raw = _build_raw_dict(
                    self.ofdm.num_subcarriers, self.ofdm.num_symbols,
                    n_idx, m_idx, est.delay_bins, est.doppler_bins,
                )
                norms = _col_norms(A_raw)
                for j in range(A_raw.shape[1]):
                    if norms[j] > 1e-15:
                        A_raw[:, j] /= norms[j]
                y = observed[mask]
                g_hat = _ridge_ls(A_raw, y)
                g_hat = g_hat / np.maximum(norms, np.finfo(float).eps)
                # Per-path quality metrics (LOO, PSLR, coherence)
                conf, sig_t, sig_n, rel = compute_path_quality(
                    est, g_hat, observed, mask, score_map,
                    self.ofdm.num_subcarriers, self.ofdm.num_symbols,
                )
                if self.cfg.token_version >= 3:
                    tokens, valid = estimated_path_tokens_v3(
                        est, g_hat, self.cfg.max_paths,
                        score_map, gain_map, grid.delay_bins, grid.doppler_bins,
                        confidence=conf, sigma_tau=sig_t,
                        sigma_nu=sig_n, relevance=rel,
                    )
                else:
                    tokens, valid = estimated_path_tokens_v2(
                        est, g_hat, self.cfg.max_paths,
                        confidence=conf, sigma_tau=sig_t,
                        sigma_nu=sig_n, relevance=rel,
                    )
            else:
                # No paths detected — empty tokens, no Oracle leak
                if self.cfg.token_version >= 3:
                    patch_size = (2 * 2 + 1) ** 2  # R=2 → 25
                    dim = 9 + patch_size + 2 * patch_size  # 84
                else:
                    dim = 9 if self.cfg.token_version >= 2 else 7
                tokens = np.zeros((self.cfg.max_paths, dim), dtype=np.float32)
                valid = np.zeros(self.cfg.max_paths, dtype=bool)
        elif self.cfg.token_version >= 2:
            tokens, valid = oracle_path_tokens_v2(paths, self.cfg.max_paths)

        tf_input = np.stack([initial.real, initial.imag, mask.astype(np.float64)], axis=0)
        target = np.stack([truth.real, truth.imag], axis=0)
        sample = {
            "tf_input": torch.tensor(tf_input, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "path_tokens": torch.tensor(tokens, dtype=torch.float32),
            "path_valid": torch.tensor(valid, dtype=torch.bool),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }
        if self.cfg.cache_in_memory:
            self._cache[index] = sample
        return sample
