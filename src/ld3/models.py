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
    path_tokens: [B, L, D] where D is token_dim_in (7 for legacy, 9 for v2).
    path_valid: [B, L] boolean.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        token_dim: int = 32,
        token_dim_in: int = 7,
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
            nn.Linear(token_dim_in, token_dim),
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


# ---------------------------------------------------------------------------
# Gate 1-D1: Physical reconstruction + TF residual gated fusion
# ---------------------------------------------------------------------------


class PhysicalReconstructor(nn.Module):
    """Differentiable OFDM phase-law synthesis from DD path parameters.

    Given path tokens [τ, ν, Re(α), Im(α), ...], synthesises:
      H_phys[n,m] = Σ_l α_l · exp(−j2π·n·τ_l/N + j2π·m·ν_l/M)

    This is a hard-coded physics layer — no learned parameters.
    The model receives the exact complex superposition, avoiding the need
    to re-learn the OFDM phase law through softmax attention.
    """

    def __init__(
        self,
        num_subcarriers: int = 64,
        num_symbols: int = 14,
    ) -> None:
        super().__init__()
        n = torch.arange(num_subcarriers, dtype=torch.float32)
        m = torch.arange(num_symbols, dtype=torch.float32)
        self.register_buffer("n_grid", n[:, None])  # [N,1]
        self.register_buffer("m_grid", m[None, :])  # [1,M]
        self.N = num_subcarriers
        self.M = num_symbols

    def forward(
        self,
        tokens: torch.Tensor,       # [B, L, 9] — must have τ at [:,0], ν at [:,1], Re(α) at [:,7], Im(α) at [:,8]
        valid: torch.Tensor,        # [B, L]
    ) -> torch.Tensor:
        """Returns H_phys: [B, 2, N, M] (real, imag)."""
        batch, L, _ = tokens.shape
        device = tokens.device

        tau = tokens[:, :, 0]    # [B, L]
        nu = tokens[:, :, 1]     # [B, L]
        alpha_re = tokens[:, :, 7]  # [B, L]
        alpha_im = tokens[:, :, 8]  # [B, L]
        valid_f = valid.to(torch.float32)

        # Phase: -2π·n·τ/N + 2π·m·ν/M
        # Shapes: n_grid [N,1], m_grid [1,M], tau/nu [B, L]
        n_phase = (-2.0 * torch.pi / self.N) * torch.einsum(
            "nl,nm->nml", tau, self.n_grid.expand(-1, self.M)
        )  # [B, N, M, L] — wrong, let me just do it simply

        # Simple approach: loop-free broadcasting with explicit reshape
        _n = self.n_grid.squeeze(-1)           # [N]
        _m = self.m_grid.squeeze(0)            # [M]
        # Build the 2D phase grid per path
        # phase[b, n, m, l] = -2π·n·τ[b,l]/N + 2π·m·ν[b,l]/M
        delay_phase = -2.0 * torch.pi * torch.einsum(
            "n,bl->nbl", _n, tau
        ) / self.N  # [N, B, L]
        doppler_phase = 2.0 * torch.pi * torch.einsum(
            "m,bl->mbl", _m, nu
        ) / self.M  # [M, B, L]
        # Combine: [N, B, L] + [M, B, L] → broadcast to [N, M, B, L]
        phase = (delay_phase[:, None, :, :] + doppler_phase[None, :, :, :]).permute(2, 0, 1, 3)
        # phase: [B, N, M, L]

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Complex multiplication: α · exp(j·phase)
        # Re(H) = Σ [Re(α)·cos - Im(α)·sin] · valid
        # Im(H) = Σ [Re(α)·sin + Im(α)·cos] · valid
        h_real = torch.sum(
            (alpha_re[:, None, None, :] * cos_phase
             - alpha_im[:, None, None, :] * sin_phase) * valid_f[:, None, None, :],
            dim=-1,
        )  # [B, N, M]
        h_imag = torch.sum(
            (alpha_re[:, None, None, :] * sin_phase
             + alpha_im[:, None, None, :] * cos_phase) * valid_f[:, None, None, :],
            dim=-1,
        )  # [B, N, M]

        return torch.stack([h_real, h_imag], dim=1)  # [B, 2, N, M]


class PhysicalResidualEstimator(nn.Module):
    """TF–DD gated residual estimator — Gate 1-D1 target architecture.

    H_phys = PhysicalReconstructor(path_tokens)     ← explicit physics
    H_tf   = TFEncoder(tf_input)                     ← learned TF refinement
    Ĥ = g ⊙ H_phys + (1−g) ⊙ H_tf + ΔH              ← gated fusion
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        num_subcarriers: int = 64,
        num_symbols: int = 14,
    ) -> None:
        super().__init__()
        self.tf_encoder = TFEncoder(hidden_dim)
        self.physics = PhysicalReconstructor(num_subcarriers, num_symbols)

        # TF refinement head
        self.tf_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

        # Fusion gate: learns where to trust physics vs TF
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim + 4, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Residual correction
        self.residual = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(
        self,
        tf_input: torch.Tensor,      # [B, 3, N, M]  real-LS, imag-LS, mask
        path_tokens: torch.Tensor,   # [B, L, 9]  with Re(α), Im(α)
        path_valid: torch.Tensor,    # [B, L]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, _, N, M = tf_input.shape

        # 1. TF encoding
        tf_features = self.tf_encoder(tf_input)          # [B, H, N, M]
        H_tf = tf_input[:, :2] + self.tf_head(tf_features)  # [B, 2, N, M]

        # 2. Explicit physical reconstruction
        H_phys = self.physics(path_tokens, path_valid)    # [B, 2, N, M]

        # 3. Gated fusion
        gate_input = torch.cat([tf_features, H_phys, H_tf], dim=1)  # [B, H+4, N, M]
        g = self.gate(gate_input)                          # [B, 1, N, M]
        H_fused = g * H_phys + (1.0 - g) * H_tf            # [B, 2, N, M]

        # 4. Residual correction
        residual_input = torch.cat([tf_features, H_fused], dim=1)
        delta = self.residual(residual_input)              # [B, 2, N, M]
        H_out = H_fused + delta

        diagnostics = {
            "gate_mean": g.mean(),
            "gate": g,
        }
        return H_out, diagnostics
