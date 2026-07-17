# LD3 Research Report — Gate 0 & Gate 1 Final Results

Date: 2026-07-18 (final — literature baselines, 300-epoch convergence, Gitee mirror)

---

## 1. Simulation Configuration

### 1.1 OFDM Waveform

| Parameter | Value |
|---|---|
| Subcarriers (N) | 64 |
| OFDM Symbols (M) | 14 |
| Subcarrier spacing | 120 kHz |
| CP duration | 0.58 μs |
| **Delay bin width** | 0.130 μs |
| **Doppler bin width** | 8.01 kHz |
| TF grid size | 64 × 14 = 896 REs |

### 1.2 Channel Model

| Parameter | Value |
|---|---|
| Paths (K) | 4 (Known-K) |
| Max delay | 12 DD bins (~1.56 μs) |
| Max Doppler | ±3 DD bins (~±24.0 kHz) |
| Delay/Doppler | Fractional |
| Power profile | Exponential decay (factor 0.25) |
| Gains | Circular complex Gaussian, Rayleigh |

### 1.3 Important Caveat

The current channel model is a **TF-domain sparse channel-surface abstraction**.
No IFFT/CP/time-domain convolution. No ISI/ICI. This is a **physics-guided
DD-prior-assisted OFDM channel estimation prototype**.

### 1.4 SNR Definition

Unified: `SNR = mean(|H_true|² over full grid) / σ²_noise`.

---

## 2. Gate 0: DD Identifiability Audit

### 2.1 Random Pilots — Recommended Work Point

Random, density 0.125, 1000 trials/condition:

| SNR | Recall | Power Recovery | Delay RMSE | NMSE Est LS |
|---|---|---|---|---|
| −5 | 0.536 | 0.743 | 0.193 | 0.622 |
| 0 | 0.634 | 0.822 | 0.177 | 0.307 |
| 10 | 0.718 | 0.872 | 0.178 | 0.203 |
| 20 | 0.735 | 0.871 | 0.175 | 0.190 |

**High-SNR plateau** at SNR ≥ 10 dB. System transitions from noise-limited
to basis-mismatch-limited.

### 2.2 Random vs Comb

Δ = Random − Comb. Random wins on recall, NMSE, OSPA at ρ ≤ 0.125 (p < 1e-4).
Low-density Comb has μ_far = 1 (deterministic DD ambiguity).

### 2.3 Ablation at SNR = 20 dB

| Method | Recall | Delay RMSE | Doppler RMSE | NMSE Est LS |
|---|---|---|---|---|
| **I** (integer bins) | 0.767 | **0.070** | **0.070** | **0.116** |
| **F_oracle** (oracle discrete) | **1.000** | 0.145 | 0.070 | 0.086 |
| **F_vp** (variable projection) | 0.741 | **0.092** | **0.062** | 0.108 |
| **F_refine** (quadratic) | 0.730 | 0.164 | 0.104 | 0.176 |
| **F** (baseline 2×4) | 0.728 | 0.173 | 0.106 | 0.190 |
| OS 4×8 | 0.754 | 0.114 | 0.090 | 0.114 |
| OS 8×16 | 0.754 | 0.099 | 0.083 | 0.096 |

### 2.4 Pilot Density Scan (K=4, SNR=10 dB)

DD+LS NMSE as a function of pilot density. 500 trials per point.

| ρ | Pilots | DD+LS NMSE | Recall | Power Recovery |
|---|---|---|---|---|
| 0.03125 | 28 | 0.395 | 0.524 | 0.760 |
| 0.0625 | 56 | 0.256 | 0.629 | 0.811 |
| **0.125** | **112** | **0.201** | **0.727** | **0.876** |
| 0.25 | 224 | 0.175 | 0.759 | 0.872 |
| 0.5 | 448 | 0.168 | 0.791 | 0.888 |

**Key finding**: ρ=0.125 is the sweet spot. Doubling to ρ=0.25 improves NMSE
only 13% (0.201→0.175) at 2× pilot overhead. Further increase to ρ=0.5 yields
marginal gain (0.175→0.168). Below ρ=0.0625, recall drops below 0.63 and NMSE
degrades rapidly. Power recovery is more robust than recall — dominant paths are
found even with very few pilots.

### 2.5 Path Count Scan (ρ=0.125, SNR=10 dB)

DD+LS NMSE as K increases. 500 trials per point.

| K | DD+LS NMSE | Recall | Power Recovery |
|---|---|---|---|
| 4 | 0.201 | 0.727 | 0.876 |
| 6 | 0.265 | 0.595 | 0.792 |
| 8 | 0.305 | 0.523 | 0.750 |

**Key finding**: Recall degrades with K (0.73→0.60→0.52), but power recovery
stays above 0.75. The DD detector prioritises strong paths and misses weak ones.
This implies Physical Residual's learned residual has larger potential gain at
higher K, where DD support quality degrades and residual compensation matters more.

---

## 3. Gate 1: Learned Fusion Results

### 3.1 Model Architecture (Final)

```
H_phys = Σ_l α_l · exp(−j2π n τ_l / N + j2π m ν_l / M)    [explicit physics]
H_tf   = TFEncoder(tf_input) + refinement_head              [learned TF]
g      = SpatialGate(TF_features, H_phys, H_tf)             [spatial fusion]
Ĥ      = g ⊙ H_phys + (1−g) ⊙ H_tf + ΔH                     [zero-init residual]
```

Key design choices:
- **Zero-init residual**: ΔH = 0 at step 0 → training starts at H_phys
- **Complex-gain tokens**: 9-dim including Re(α), Im(α)
- **Spatial gate only**: per-path gate removed (CNN residual learns it implicitly)
- **No VP in final pipeline**: zero-init residual compensates position errors

### 3.2 Non-Learned Baselines (10 dB, ρ=0.125)

| Method | NMSE |
|---|---|
| Oracle perfect | −∞ dB (code closure) |
| **Oracle support + LS** | **−24.29 dB** |
| DD estimated + LS | −8.36 dB |
| Nearest interpolation | −1.41 dB |

### 3.3 Main Work Point (10 dB, ρ=0.125, Oracle Tokens)

| Model | NMSE | Notes |
|---|---|---|
| TF-only | −4.62 dB | CNN baseline |
| Cross-Attn (7-dim, legacy) | −6.97 dB | No complex gain |
| Cross-Attn (9-dim) | −9.32 dB | +2.3 dB free lunch from Re/Im α |
| Physical Residual (random init) | −9.6 dB | Before zero-init |
| **Physical Residual (zero-init)** | **−19.55 dB** | **+10.5 dB improvement** |
| Estimated Residual (zero-init) | −20.10 dB | Oracle tokens + est. pipeline overlap |

### 3.4 Estimated Tokens (DD + LS gains, no VP)

Three work points, 3 seeds each:

| Work Point | SNR | Density | TF-only | Cross-Attn | DD+LS | **Est. Residual** | Δ |
|---|---|---|---|---|---|---|---|
| Main | 10 dB | 0.125 | −4.62 | −7.97 | −8.36 | **−9.76** | **+1.40** |
| Boundary | 0 dB | 0.125 | −2.28 | −4.41 | −6.13 | **−6.45** | +0.32 |
| Stress | 5 dB | 0.0625 | −1.85 | −3.23 | −5.87 | **−6.21** | +0.34 |

Paired CI for Main: +1.40 dB [CI: +1.20, +1.61]. Statistically significant.

### 3.5 Multi-SNR Unified Model

Single model trained on SNR ∈ [−5, 20] dB. Per-SNR evaluation:

| SNR | Estimated Residual | DD+LS | Δ | Notes |
|---|---|---|---|---|
| −5 dB | −2.97 | −2.42 | −0.55 | Token quality too low |
| 0 dB | −6.08 | −6.19 | +0.11 | Near break-even |
| +5 dB | **−8.24** | −7.79 | **+0.45** | |
| +10 dB | **−9.50** | −8.37 | **+1.13** | |
| +15 dB | **−10.05** | −8.60 | **+1.45** | |
| +20 dB | **−10.17** | −8.58 | **+1.59** | |

Unified model: NMSE −7.57 dB (mixed-SNR test set).
SNR ≥ 0 dB: model consistently outperforms DD+LS.
−5 dB: token quality collapses (recall ~0.5) → model cannot recover.

### 3.6 Architecture Evolution (Ablation Summary)

| Change | Δ NMSE | Key Insight |
|---|---|---|
| 7-dim → 9-dim tokens (add Re/Im α) | **+2.3 dB** | Complex-gain tokens are necessary |
| Random init → Zero-init residual | **+10.5 dB** | Training must start from H_phys |
| No gate → Spatial gate | +0.5 dB | Mild benefit over pure residual |
| Add per-path gate | −1.0 dB | **Harmful** — CNN residual learns it |
| Add VP refinement | ~0 dB | Residual already compensates position errors |
| Add path quality metrics | ~0 dB | Validates hardcoded confidence≈0.7 was reasonable |

### 3.7 Token-Use Audit (Cross-Attn, 9-dim)

| Token Mode | NMSE | Δ from Oracle |
|---|---|---|
| Oracle (correct) | −9.32 dB | — |
| Shuffled | −5.91 dB | +3.41 dB |
| Null | −5.46 dB | +3.86 dB |

Model strongly depends on correct DD tokens.

---

## 4. Literature Comparison (Internal Diagnostic — Not SOTA Benchmark)

**Note:** Results in this section are *LD3-adapted baselines*, not necessarily
faithful reproductions of the cited methods. A-MMSE and D2AN were re-implemented
for the LD3 setting (ρ=0.125, 4-path fractional channel, 64×14 grid). The cited
papers use different channel models, pilot densities, and OFDM configurations.
These comparisons serve as architectural diagnostics within LD3's experimental
framework, not as formal SOTA comparisons.

### 4.1 Compared Methods

| Model | Paper | Paradigm | DD Prior | Token Source | Epochs |
|---|---|---|---|---|---|
| TF-only | — | CNN baseline | None | — | 300 |
| **A-MMSE (adapted)** | Ha et al., arXiv:2506.00452 | Two-stage Transformer attention | None | — | 300 |
| **D2AN (adapted)** | Zhao et al., IEEE WCL 2026 | DD complex-exponential basis → attention | Indirect (attention bias) | — | 300 |
| Cross-Attn (9-dim) | LD3 | DD path-token cross-attention | Direct (token features) | Oracle | 300 |
| **DD+LS** | LD3 (non-learned) | DD detection + LS gains | Direct (path parameters) | — | — |
| **Physical Residual** | LD3 | Explicit H_phys + zero-init residual | Direct (H_phys formula) | **Oracle** | 300 |

### 4.2 Head-to-Head Results

Same setting: ρ=0.125, 4 fractional paths, 10 dB SNR, 64×14 grid.

| Model | NMSE (dB) | Token Source | Paradigm |
|---|---|---|---|
| TF-only | −4.62 | — | TF interpolation |
| D2AN (adapted) | −5.68 | — | TF + DD soft attention |
| A-MMSE (adapted) | −5.83 | — | TF + Transformer attention |
| DD+LS (non-learned) | −8.36 | Estimated | DD explicit, no learning |
| Cross-Attn (9-dim) | −11.75 | Oracle | TF + DD path token attention |
| **Physical Residual** | **−20.10** | **Oracle** | **DD explicit + zero-init residual** |
| **Physical Residual (gate1_estimated)** | **−9.76** | **Estimated (DD+LS)** | **DD explicit + zero-init residual** |

### 4.3 Attribution of Gains

The 15+ dB gap between TF-domain methods and Physical Residual (Oracle) is NOT purely
a network architecture effect. It combines three distinct contributions:

| Transition | Δ NMSE | Source |
|---|---|---|
| TF-only → DD+LS | +3.74 dB | Explicit DD parameterization prior (Known-K) |
| DD+LS → Physical Residual (estimated) | **+1.40 dB** | **Learned residual (clean learning gain)** |
| Physical Residual (estimated) → (Oracle) | +10.34 dB | Token quality (true vs. estimated {τ,ν,α}) |

The +1.40 dB from DD+LS to estimated Physical Residual is the most reliable
measure of the learned fusion architecture's contribution — the rest comes from
the strong inductive bias of explicit path parameterization and Known-K prior.

### 4.4 Key Insight

Under the tested known-order four-path channel model, implicit TF-domain estimators
(A-MMSE adapted, D2AN adapted) plateaued near −6 dB. Explicit path-parametric
reconstruction (DD+LS) already achieves −8.4 dB without any learning. The learned
residual further improves this to −9.8 dB (estimated tokens) or −20.1 dB (Oracle
tokens). This demonstrates a **strong inductive-bias advantage** of enforcing the
sparse physical channel manifold, rather than a universal information-theoretic
limit for all TF-domain methods.

The result does **not** establish that "no TF-domain method can exceed −6 dB" —
the DD path parameters themselves are extracted from the same pilot observations,
proving the information *is* present in the TF data. The advantage comes from the
**representation**: compressing 896 TF coefficients into ~16 path parameters.

---

## 5. Gate Status

```
Gate 0-A    Known-K DD identifiability ................. PASS
Gate 0-A2   Comb structural aliasing (μ_far=1) ......... PASS
Gate 0-A3   Off-grid + peak selection bottlenecks ...... PASS
Gate 0-A5   VP continuous refinement ................... PASS (−39% NMSE vs baseline)
Gate 0-B    Unknown-K detection ........................ OPEN

Gate 1-A    Physical model closure ..................... PASS (nmse_perfect = 0)
Gate 1-B    Oracle continuous-support value ............ PASS (+22.9 dB)
Gate 1-C    DD estimated support value ................. PASS (−8.4 dB baseline)
Gate 1-D1   Oracle Physical Residual (zero-init) ...... PASS (−19.6 dB)
Gate 1-E2   Estimated-token Physical Residual .......... PASS (+1.4 dB vs DD+LS)
Gate 1-E3   Multi-SNR unified model .................... PASS (SNR ≥ 0 dB)
Gate 1-F    Per-path gate .............................. FAIL (−1.0 dB regression)

Gate 2      Safe degradation under corrupted priors .... OPEN
Gate 3      Full OFDM-ISAC waveform .................... OPEN
```

---

## 6. Reproducibility

### 6.1 Key Commands

```bash
# Physical closure
pytest tests/test_oracle_closure.py -v

# Gate 0 full sweep
python experiments/gate0_identifiability.py --config configs/gate0.yaml --trials 1000

# Gate 0 VP ablation
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_vp --ablation-vp

# Gate 1 — Oracle tokens (main work point)
python experiments/gate1_oracle.py --config configs/gate1_main.yaml --output-dir results/gate1_main --device cuda

# Gate 1 — Estimated tokens
python experiments/gate1_oracle.py --config configs/gate1_estimated.yaml --output-dir results/gate1_estimated --device cuda

# Gate 1 — Multi-SNR
python experiments/gate1_oracle.py --config configs/gate1_multisnr.yaml --output-dir results/gate1_multisnr --device cuda

# Gate 1 — Literature baselines (A-MMSE + D2AN + LD3)
python experiments/gate1_oracle.py --config configs/gate1_main.yaml --output-dir results/gate1_literature --device cuda
```

### 5.2 Config Index

| Config | Purpose |
|---|---|
| `configs/gate0.yaml` | Gate 0 full sweep (1000 trials) |
| `configs/gate0_ablation.yaml` | Gate 0 ablation (500 trials) |
| `configs/gate1_main.yaml` | Oracle tokens, 10 dB |
| `configs/gate1_estimated.yaml` | Estimated tokens (DD+LS), 10 dB |
| `configs/gate1_boundary_estimated.yaml` | Estimated tokens, 0 dB |
| `configs/gate1_stress_estimated.yaml` | Estimated tokens, 5 dB, ρ=0.0625 |
| `configs/gate1_multisnr.yaml` | Multi-SNR (−5 to +20 dB), estimated tokens |

### 5.3 RNG Seeding

```
Channel RNG:  [seed, density_idx, snr_idx, trial, 100]  — no pattern_index
Noise RNG:    [seed, snr_idx, density_idx, trial, 300]  — no pattern_index
Pilot RNG:    [seed, pattern_idx, density_idx, trial, 200]  — varies by pattern
```

Gate 1 multi-seed: base_seed + seed_idx × 1000 for training data.
Fixed test bank: base_seed + 10000. Fixed val bank: base_seed + 20000.
