from __future__ import annotations

import math
import torch
from torch import nn


class TFEncoder(nn.Module):
    """Lightweight TF encoder; deliberately avoids global quadratic attention."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
            ),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsGuidedCrossAttention(nn.Module):
    """Path-token cross attention with phase bias, uncertainty, and null token.

    Inputs
    ------
    tf_input: [B, 3, N, M] with real LS, imag LS, and pilot mask.
    path_tokens: [B, L, 7] fields documented in oracle.oracle_path_tokens.
    path_valid: [B, L] boolean.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        token_dim: int = 32,
        max_delay_bins: float = 12.0,
        max_abs_doppler_bins: float = 3.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_delay_bins = max_delay_bins
        self.max_abs_doppler_bins = max_abs_doppler_bins
        self.tf_encoder = TFEncoder(hidden_dim)
        self.query = nn.Linear(hidden_dim, token_dim, bias=False)
        self.token_encoder = nn.Sequential(
            nn.Linear(7, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.key = nn.Linear(token_dim, token_dim, bias=False)
        self.value = nn.Linear(token_dim, hidden_dim, bias=False)
        self.null_key = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.null_value = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.phase_weights = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.physics_scale = nn.Parameter(torch.tensor(1.0))
        self.uncertainty_scale = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(
        self,
        tf_input: torch.Tensor,
        path_tokens: torch.Tensor,
        path_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, _, n_subcarriers, n_symbols = tf_input.shape
        features = self.tf_encoder(tf_input)
        tf_tokens = features.permute(0, 2, 3, 1).reshape(batch, -1, self.hidden_dim)
        query = self.query(tf_tokens)

        encoded = self.token_encoder(path_tokens)
        keys = self.key(encoded)
        values = self.value(encoded)

        # Learned similarity.
        logits = torch.einsum("bqd,bld->bql", query, keys) / math.sqrt(keys.shape[-1])

        # Exact OFDM phase law in normalized DD-bin coordinates.
        n = torch.arange(n_subcarriers, device=tf_input.device, dtype=tf_input.dtype)
        m = torch.arange(n_symbols, device=tf_input.device, dtype=tf_input.dtype)
        delay = path_tokens[..., 0]
        doppler = path_tokens[..., 1]
        phase = (
            -2.0
            * math.pi
            * n[None, :, None, None]
            * delay[:, None, None, :]
            / n_subcarriers
            + 2.0
            * math.pi
            * m[None, None, :, None]
            * doppler[:, None, None, :]
            / n_symbols
        )
        phase_bias = (
            self.phase_weights[0] * torch.cos(phase)
            + self.phase_weights[1] * torch.sin(phase)
        ).reshape(batch, n_subcarriers * n_symbols, -1)

        confidence = path_tokens[..., 3].clamp(1e-4, 1.0)
        relevance = path_tokens[..., 6].clamp(1e-4, 1.0)
        uncertainty = path_tokens[..., 4] + path_tokens[..., 5]
        prior_bias = torch.log(confidence * relevance) - self.uncertainty_scale.abs() * uncertainty
        logits = logits + self.physics_scale * phase_bias + prior_bias[:, None, :]
        logits = logits.masked_fill(~path_valid[:, None, :], torch.finfo(logits.dtype).min)

        # Null token lets the model reject every DD candidate.
        null_key = self.null_key.expand(batch, -1, -1)
        null_value = self.null_value.expand(batch, -1, -1)
        null_logits = torch.einsum("bqd,bld->bql", query, null_key) / math.sqrt(keys.shape[-1])
        all_logits = torch.cat([logits, null_logits], dim=-1)
        all_values = torch.cat([values, null_value], dim=1)
        attention = torch.softmax(all_logits, dim=-1)
        cross = torch.einsum("bql,bld->bqd", attention, all_values)
        cross = cross.reshape(batch, n_subcarriers, n_symbols, self.hidden_dim).permute(0, 3, 1, 2)

        valid_float = path_valid.to(tf_input.dtype)
        denom = valid_float.sum(dim=1).clamp_min(1.0)
        mean_conf = (confidence * valid_float).sum(dim=1) / denom
        mean_unc = (uncertainty * valid_float).sum(dim=1) / denom
        quality = torch.stack([mean_conf, mean_unc], dim=1)[:, :, None, None]
        quality = quality.expand(-1, -1, n_subcarriers, n_symbols)
        gate = self.gate(torch.cat([features, quality], dim=1))
        fused = features + gate * cross

        correction = self.head(fused)
        initial = tf_input[:, :2]
        estimate = initial + correction
        diagnostics = {
            "attention": attention,
            "null_attention": attention[..., -1],
            "gate": gate,
        }
        return estimate, diagnostics


class TFOnlyEstimator(nn.Module):
    """Matched-capacity TF-only baseline."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = TFEncoder(hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(self, tf_input: torch.Tensor) -> torch.Tensor:
        return tf_input[:, :2] + self.head(self.encoder(tf_input))
