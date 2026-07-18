from __future__ import annotations

import math
import torch
import torch.nn.functional as F
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
        tokens: torch.Tensor,       # [B, L, 9]
        valid: torch.Tensor,        # [B, L]
    ) -> torch.Tensor:
        """Returns H_phys: [B, 2, N, M] (real, imag)."""
        batch, L, _ = tokens.shape

        tau = tokens[:, :, 0]    # [B, L]
        nu = tokens[:, :, 1]     # [B, L]
        alpha_re = tokens[:, :, 7]  # [B, L]
        alpha_im = tokens[:, :, 8]  # [B, L]
        valid_f = valid.to(torch.float32)

        # Phase: -2π·n·τ/N + 2π·m·ν/M
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


def _build_quality_map(
    H_phys: torch.Tensor,      # [B, 2, N, M]
    H_tf: torch.Tensor,        # [B, 2, N, M]
    path_tokens: torch.Tensor, # [B, L, 9]
    path_valid: torch.Tensor,  # [B, L]
) -> torch.Tensor:
    """Build 4-channel spatial quality map for token-conditioned gating.

    Channels:
      0: |H_phys - H_tf|²  — discrepancy map (locally normalized)
      1: mean_token_confidence — expanded to spatial constant
      2: mean_token_uncertainty — sigma_delay + sigma_doppler, expanded
      3: valid_ratio — fraction of token slots that are valid [0, 1]

    The discrepancy channel is the most informative: it tells the gate
    WHERE physics and learned TF disagree, enabling spatial gating.
    The valid_ratio channel gives the gate an explicit all-tokens-invalid
    signal, enabling structural null fallback.

    All channels are normalised to [-1, 1] range for stable CNN input
    alongside TF features.
    """
    batch, _, N, M = H_phys.shape
    device = H_phys.device
    dtype = H_phys.dtype

    # --- Channel 0: |H_phys - H_tf|²  discrepancy ---
    diff = (H_phys - H_tf).square().sum(dim=1, keepdim=True)  # [B, 1, N, M]
    # Local normalisation: divide by per-sample mean for scale invariance
    diff_mean = diff.mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    discrepancy = diff / diff_mean  # values in [0, ~∞), typically [0, 10]
    discrepancy = torch.tanh(discrepancy * 0.5)  # soft clamp to [-1, 1]

    # --- Channel 1: mean token confidence ---
    valid_f = path_valid.to(dtype)
    denom = valid_f.sum(dim=1).clamp_min(1.0)  # [B]
    mean_conf = (path_tokens[:, :, 3].clamp(0.0, 1.0) * valid_f).sum(dim=1) / denom  # [B]
    confidence_map = mean_conf[:, None, None, None].expand(batch, 1, N, M)  # [B, 1, N, M]
    confidence_map = confidence_map * 2.0 - 1.0  # map [0,1] → [-1, 1]

    # --- Channel 2: mean token uncertainty ---
    sigma_tau = path_tokens[:, :, 4]   # [B, L]
    sigma_nu = path_tokens[:, :, 5]    # [B, L]
    uncertainty = (sigma_tau + sigma_nu).clamp(0.0, 2.0)  # per-token
    mean_unc = (uncertainty * valid_f).sum(dim=1) / denom  # [B]
    uncertainty_map = mean_unc[:, None, None, None].expand(batch, 1, N, M)  # [B, 1, N, M]
    uncertainty_map = uncertainty_map - 1.0  # center [0, 2] → [-1, 1]

    # --- Channel 3: valid_ratio ---
    valid_ratio = denom / float(path_tokens.shape[1])  # [B], fraction of valid slots
    valid_map = valid_ratio[:, None, None, None].expand(batch, 1, N, M)
    valid_map = valid_map * 2.0 - 1.0  # map [0, 1] → [-1, 1]

    return torch.cat([discrepancy, confidence_map, uncertainty_map, valid_map], dim=1)  # [B, 4, N, M]


class PhysicalResidualEstimator(nn.Module):
    """TF–DD gated residual estimator — Gate 1-D1 / Gate 2-C target architecture.

    H_phys = PhysicalReconstructor(path_tokens)     ← explicit physics
    H_tf   = TFEncoder(tf_input)                     ← learned TF refinement
    Ĥ = g ⊙ H_phys + (1−g) ⊙ H_tf + g ⊙ ΔH          ← coupled gated fusion

    Gate 2-C v2 improvements:
    - Coupled residual: gate controls both blend AND residual correction.
      g→0 ⇒ Ĥ = H_tf (clean structural fallback, ΔH suppressed).
    - Quality map v2: 4 channels (discrepancy, confidence, uncertainty, valid_ratio).
      valid_ratio gives the gate an explicit all-tokens-invalid signal.

    When use_quality_gate=True, the gate additionally receives
    a 3-channel token-quality map: discrepancy |H_phys-H_tf|², mean token
    confidence, and mean token uncertainty. This lets the gate learn to
    reject the physics branch when token quality is poor.
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        num_subcarriers: int = 64,
        num_symbols: int = 14,
        use_quality_gate: bool = False,
    ) -> None:
        super().__init__()
        self.use_quality_gate = use_quality_gate
        self.tf_encoder = TFEncoder(hidden_dim)
        self.physics = PhysicalReconstructor(num_subcarriers, num_symbols)

        # TF refinement head
        self.tf_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

        # Fusion gate: learns where to trust physics vs TF.
        # Gate 2-C v2 adds 4 quality channels → H+4 → H+8.
        gate_in_channels = hidden_dim + 4 + (4 if use_quality_gate else 0)
        self.gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Residual correction — zero-init so training starts at H_phys.
        self.residual = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        tf_input: torch.Tensor,      # [B, 3, N, M]  real-LS, imag-LS, mask
        path_tokens: torch.Tensor,   # [B, L, 9]  with Re(α), Im(α)
        path_valid: torch.Tensor,    # [B, L]
        return_components: bool = False,
        ablation_mode: str = "full",  # "full" | "fixed_blend_nores" | "spatial_nores" | "fixed_blend_res"
        fixed_lam: float = 0.80,      # λ for fixed_blend modes
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, _, N, M = tf_input.shape

        # 1. TF encoding
        tf_features = self.tf_encoder(tf_input)          # [B, H, N, M]
        H_tf = tf_input[:, :2] + self.tf_head(tf_features)  # [B, 2, N, M]

        # 2. Explicit physical reconstruction
        H_phys = self.physics(path_tokens, path_valid)    # [B, 2, N, M]

        # 3. Gated fusion
        if self.use_quality_gate:
            quality_map = _build_quality_map(H_phys, H_tf, path_tokens, path_valid)
            gate_input = torch.cat([tf_features, H_phys, H_tf, quality_map], dim=1)
        else:
            gate_input = torch.cat([tf_features, H_phys, H_tf], dim=1)
        g_raw = self.gate(gate_input)                     # [B, 1, N, M]

        # Hard null-fallback: when ALL tokens are invalid, force gate to zero.
        has_any_valid = path_valid.any(dim=1).to(g_raw.dtype)       # [B]
        has_any_valid = has_any_valid[:, None, None, None]           # [B, 1, 1, 1]
        g = has_any_valid * g_raw                                    # [B, 1, N, M]

        H_fused = g * H_phys + (1.0 - g) * H_tf            # [B, 2, N, M]

        # --- Gate × Residual 2×2 ablation ---
        if ablation_mode.startswith("fixed_blend"):
            lam = fixed_lam
            H_fused = lam * H_phys + (1.0 - lam) * H_tf
            g = torch.full_like(g, lam)  # constant gate for diagnostics

        # 4. Residual correction
        residual_input = torch.cat([tf_features, H_fused], dim=1)
        delta = self.residual(residual_input)              # [B, 2, N, M]

        if ablation_mode.endswith("_nores"):
            H_out = H_fused  # no residual
        else:
            # Coupled residual: gate controls both blend AND correction.
            # g→0: H_out = H_tf + 0 = H_tf (clean structural fallback)
            # g→1: H_out = H_phys + delta (full physics + correction)
            # This prevents residual from contributing when physics is shut off.
            H_out = H_fused + g * delta

        # --- Diagnostics ---
        with torch.no_grad():
            p_tf = H_tf.square().sum(dim=(1, 2, 3))
            p_fused = H_fused.square().sum(dim=(1, 2, 3))
            p_phys = H_phys.square().sum(dim=(1, 2, 3))
            p_delta = delta.square().sum(dim=(1, 2, 3))
            p_out = H_out.square().sum(dim=(1, 2, 3))
            phys_mix = (g * H_phys).square().sum(dim=(1, 2, 3))

        diagnostics = {
            "gate_mean": g.mean(),
            "gate": g,
            "p_tf_mean": p_tf.mean(),
            "p_fused_mean": p_fused.mean(),
            "p_phys_mean": p_phys.mean(),
            "p_delta_mean": p_delta.mean(),
            "p_out_mean": p_out.mean(),
            "phys_mix_mean": phys_mix.mean(),
            "frac_null": 1.0 - has_any_valid.float().mean(),
        }
        if return_components:
            diagnostics["H_phys"] = H_phys
            diagnostics["H_tf"] = H_tf
        return H_out, diagnostics


# ---------------------------------------------------------------------------
# Literature baselines (simplified versions for head-to-head comparison)
# ---------------------------------------------------------------------------


class AMMSEEstimator(nn.Module):
    """A-MMSE — Attention-Aided MMSE [Ha et al., 2024].

    Two-stage separable self-attention:
      1. Frequency encoder: MHA over N subcarriers (per-symbol)
      2. Time encoder: MHA over M symbols (per-subcarrier)
    No DD prior — pure TF-domain learning from interpolated LS + pilot mask.
    """

    def __init__(self, hidden_dim: int = 48, num_subcarriers: int = 64, num_symbols: int = 14) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N = num_subcarriers
        self.M = num_symbols

        self.input_proj = nn.Conv2d(3, hidden_dim, 3, padding=1)

        # Frequency attention: N subcarrier-tokens, each with H features
        self.freq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True,
        )
        self.freq_norm = nn.LayerNorm(hidden_dim)

        # Time attention: M symbol-tokens, each with H features
        self.time_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=2, batch_first=True,
        )
        self.time_norm = nn.LayerNorm(hidden_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(self, tf_input: torch.Tensor) -> torch.Tensor:
        batch, _, N, M = tf_input.shape
        x = self.input_proj(tf_input)  # [B, H, N, M]

        # --- Frequency attention: per-symbol, tokens = subcarriers ---
        # [B, H, N, M] → [B*M, N, H]
        x_f = x.permute(0, 3, 2, 1).contiguous().view(batch * M, N, self.hidden_dim)
        x_f, _ = self.freq_attn(x_f, x_f, x_f)
        x_f = self.freq_norm(x_f)  # norm after attention
        x_f = x_f.view(batch, M, N, self.hidden_dim).permute(0, 3, 2, 1)  # [B, H, N, M]

        # --- Time attention: per-subcarrier, tokens = symbols ---
        # [B, H, N, M] → [B*N, M, H]
        x_t = x.permute(0, 2, 3, 1).contiguous().view(batch * N, M, self.hidden_dim)
        x_t, _ = self.time_attn(x_t, x_t, x_t)
        x_t = self.time_norm(x_t)  # norm after attention
        x_t = x_t.view(batch, N, M, self.hidden_dim).permute(0, 3, 1, 2)  # [B, H, N, M]

        return tf_input[:, :2] + self.decoder(x_f + x_t + x)


class D2ANEstimator(nn.Module):
    """D2AN — Delay-Doppler Attention Network [Zhao et al., 2026].

    DD complex-exponential basis functions uniformly sampled in (τ, ν) space.
    FC network learns combination weights → DD attention map → TF estimate.
    """

    def __init__(self, hidden_dim: int = 48, num_subcarriers: int = 64, num_symbols: int = 14,
                 num_delay: int = 8, num_doppler: int = 6) -> None:
        super().__init__()
        self.N = num_subcarriers
        self.M = num_symbols
        self.num_delay = num_delay
        self.num_doppler = num_doppler
        self.num_bases = num_delay * num_doppler

        # DD basis: uniform grid
        tau = torch.linspace(0, 12.0, num_delay)
        nu = torch.linspace(-3.0, 3.0, num_doppler)
        n = torch.arange(num_subcarriers, dtype=torch.float32)
        m = torch.arange(num_symbols, dtype=torch.float32)

        # Build basis: [D*F, N*M] real-valued (cos + sin stacked)
        basis_cos = torch.zeros(self.num_bases, self.N * self.M)
        basis_sin = torch.zeros(self.num_bases, self.N * self.M)
        for d in range(num_delay):
            for f in range(num_doppler):
                idx = d * num_doppler + f
                phase = (-2.0 * torch.pi * n[:, None] * tau[d] / self.N
                         + 2.0 * torch.pi * m[None, :] * nu[f] / self.M)
                basis_cos[idx] = torch.cos(phase).reshape(-1)
                basis_sin[idx] = torch.sin(phase).reshape(-1)
        self.register_buffer("basis_cos", basis_cos)  # [D*F, N*M]
        self.register_buffer("basis_sin", basis_sin)

        self.input_proj = nn.Conv2d(3, hidden_dim, 3, padding=1)

        # FC: global features → basis combination weights
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_bases),
        )

        # Per-channel DD attention: basis → per-channel spatial weights
        self.basis_proj = nn.Conv2d(1, hidden_dim, 1)

        # Multi-scale DD attention
        self.attn_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(self, tf_input: torch.Tensor) -> torch.Tensor:
        batch, _, N, M = tf_input.shape
        features = self.input_proj(tf_input)  # [B, H, N, M]

        # Per-position DD attention: each (n,m) gets attention from DD basis
        # Pool features spatially → learned basis weights → DD attention map
        feat_pool = features.mean(dim=(2, 3))  # [B, H]
        # FC to produce basis weights
        w = self.weight_net(feat_pool)  # [B, D*F]

        # Project DD basis with learned weights to get attention map [B, 1, N, M]
        attn_real = w @ self.basis_cos  # [B, N*M]
        attn_imag = w @ self.basis_sin
        attn = (attn_real ** 2 + attn_imag ** 2).reshape(batch, 1, N, M)
        attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-8)

        # Per-channel attention: expand to H channels
        attn_h = self.basis_proj(attn)  # [B, H, N, M]

        # Fuse DD-attended features with original
        fused = self.attn_fusion(torch.cat([attn_h * features, features], dim=1))

        dec_input = torch.cat([fused, tf_input[:, :2]], dim=1)
        return tf_input[:, :2] + self.head(dec_input)
