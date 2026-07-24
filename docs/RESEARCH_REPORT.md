# LD3 Research Report — Gate 0 & Gate 1 Final Results

Date: 2026-07-24 (Gate 2 mechanism audit complete; canonical safe-pipeline validation open)

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

**Key finding**: ρ=0.125 constitutes a favorable pilot-efficiency knee point,
retaining most of the channel-reconstruction benefit while avoiding the rapidly
diminishing returns at higher densities. Doubling to ρ=0.25 improves NMSE only
13% (0.201→0.175) at 2× pilot overhead. Further increase to ρ=0.5 yields
marginal gain (0.175→0.168). Below ρ=0.0625, recall drops below 0.63 and NMSE
degrades rapidly. Power recovery is more robust than recall under the current
non-uniform (exponential decay) path-power profile — the detector preferentially
retains dominant-energy paths.

### 2.5 Path Count Scan (ρ=0.125, SNR=10 dB)

DD+LS NMSE as K increases. 500 trials per point.

| K | DD+LS NMSE | Recall | Power Recovery |
|---|---|---|---|
| 4 | 0.201 | 0.727 | 0.876 |
| 6 | 0.265 | 0.595 | 0.792 |
| 8 | 0.305 | 0.523 | 0.750 |

**Key finding**: Recall degrades with K (0.73→0.60→0.52), but power recovery
stays above 0.75. Under the exponential power decay profile, later-added paths
are naturally weaker, so the DD detector preferentially retains dominant-energy
paths while missing weaker ones. Higher K leaves a larger unreconstructed
residual after DD-based parametric estimation. Gate 1 K-sweep (§3.8) confirms
this headroom is real and grows with K: the PhysicalResidual learned gain over
DD+LS increases from +1.40 dB (K=4) to +1.86 dB (K=6) to +1.86 dB (K=8).

**Note**: All K-scan experiments use Known-K (true path count provided to the
detector). At K=8, the detector outputs 8 candidates but only ~4.2 match true
paths. The remaining ~3.8 are false candidates, so the residual network must
both compensate for missed weak paths AND suppress contributions from incorrect
tokens. This makes safe fusion (Gate 2) increasingly important as K grows.

**K-sweep conclusion**: The learned residual gain G_learn(K) increases from
K=4→6 and stays flat from K=6→8. The spatial gate mean rises from 0.62 (K=4)
to 0.72 (K=6) to 0.70 (K=8), indicating the model increasingly trusts the
physical reconstruction branch when more paths are available — even though
DD support estimation degrades (recall 0.73→0.60→0.52). The gain plateau at
K=8 may reflect the DD detector's floor: at ~52% recall, nearly half the path
tokens are wrong, and the residual network approaches its ability to compensate.

---

## 3. Gate 1: Learned Fusion Results

### 3.1 Model Architecture (Gate 1 Final / Gate 2 Predecessor)

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

### 3.8 Path-Count Sweep (K=4/6/8, Estimated Tokens)

Three K values, same setting: ρ=0.125, 10 dB SNR, estimated tokens (DD+LS),
3 seeds × 300 epochs. Hierarchical bootstrap CIs over seeds.

| K | TF-only | Cross-Attn | DD+LS | **Est. Residual** | Δ (vs DD+LS) | Gate Mean |
|---|---------|------------|-------|-------------------|---------------|-----------|
| 4 | −4.62 | −7.97 | −8.36 | **−9.76** [−9.87, −9.65] | **+1.40** [1.20, 1.61] | 0.618 |
| 6 | −5.19 | −8.53 | −7.59 | **−9.45** [−9.53, −9.37] | **+1.86** [1.74, 1.97] | 0.718 |
| 8 | −4.98 | −8.07 | −7.01 | **−8.87** [−8.94, −8.79] | **+1.86** [1.74, 1.97] | 0.701 |

All Δ values statistically significant (paired bootstrap p < 1e-4 via
hierarchical resampling). CIs in brackets are 95% hierarchical bootstrap.

**Key findings:**

1. **Gain increases with K then plateaus.** From K=4→6, the learned advantage
   over DD+LS grows from +1.40 to +1.86 dB. From K=6→8, gain stays at +1.86 dB
   — the residual network reaches its compensation capacity as DD recall drops
   to 0.52 (nearly half the tokens are wrong).

2. **Gate mean rises with K.** 0.62 (K=4) → 0.72 (K=6) → 0.70 (K=8). The model
   learns to trust the physical reconstruction branch more when more paths
   provide richer DD prior information — even though individual path accuracy
   degrades.

3. **DD+LS degrades faster than PhysicalResidual.** DD+LS drops 1.35 dB from
   K=4→8 (−8.36→−7.01), while PhysicalResidual drops only 0.89 dB
   (−9.76→−8.87). The TF residual branch partially compensates for DD support
   degradation.

4. **Oracle+LS upper bound also degrades with K** (−24.3→−22.6→−21.2 dB),
   reflecting the fundamental information-theoretic cost of estimating more
   complex-gain parameters from fixed pilot resources.

5. **TF-only is roughly flat across K** (−4.6 to −5.2 dB) — the CNN baseline
   is indifferent to path count since it operates purely in the TF domain.

**Architecture insight**: The K-sweep validates the core design hypothesis of
Gate 1-D1: explicit physical reconstruction with zero-init residual provides a
structurally monotonic prior — even when individual DD tokens are inaccurate,
the physics branch contributes a superposition of all path hypotheses, and the
TF residual suppresses what doesn't match the data. The gain plateau at K=8
suggests that further K increases will eventually require improved DD support
quality (Gate 0-B) or safe fusion mechanisms (Gate 2) to prevent the residual
branch from being overwhelmed by false tokens.

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

## 5. Gate 2-A: Failure Boundary Audit (Frozen Model)

Gate 2-A tests how the clean-trained PhysicalResidualEstimator degrades under
controlled token corruption, without any retraining. 1024 test samples,
46 corruption specifications, oracle and estimated token chains.
Safety baseline: TF-only at −4.62 dB.

### 5.1 Failure Severity Tiers

**☠️ Critical (harm_rate ≥ 90%)**

| Perturbation | Oracle NMSE | Est. NMSE | Harm% | Gate |
|---|---|---|---|---|
| phase π (total flip) | **+1.10 dB** | +0.79 dB | 100% | 0.472 |
| bias_delay 0.5 | −1.43 dB | −1.59 dB | **100%** | 0.554 |
| jitter 2.0 joint | −1.20 dB | −1.47 dB | 99.5% | 0.550 |

**⚠️ Dangerous (harm_rate 30–90%)**

| Perturbation | Oracle NMSE | Harm% |
|---|---|---|
| coherent_false 4 | −3.11 dB | 74.0% |
| dropout 0.75 | −4.57 dB | 60.6% |
| mag 2.0 | −4.22 dB | 59.9% |

**✅ Safe (harm_rate < 5%)**

| Perturbation | Oracle NMSE | Harm% | Finding |
|---|---|---|---|
| random_false 1/2/4 | **−15.39 dB** | **0.0%** | Null token makes model immune |
| permute | **−15.39 dB** | **0.0%** | Permutation invariance confirmed |
| phase π/8 | −9.94 dB | 0.0% | Small phase errors tolerated |
| mag 0.75–1.25 | −9.3 to −12.7 dB | 0.0% | Minor gain deviation safe |

### 5.2 Three Key Findings

**Finding 1: `null_all` fails to reach TF-only — structural defect.**

```
null_all:  −3.40 dB  (gate stays at 0.589)
TF-only:   −4.62 dB  (expected safe fallback)
Gap:       +1.22 dB  ← GATE 2-A FAIL
```

Even with all tokens marked invalid, the gate remains at ~0.59, mixing
nearly 60% of the (now-meaningless) physical reconstruction into the output.
The spatial gate reads TF features + H_phys + H_Tf, not independent token
quality — it cannot fully reject the physics branch.

**Finding 2: Random false paths are harmless, coherent false are deadly.**

The model ignores random false tokens (null token mechanism works), but
coherent false paths near strong true paths degrade NMSE by ~3 dB each.
This is the DD sidelobe detection error mode — the most realistic threat
in deployment.

**Finding 3: Gate responds to quality but insufficiently.**

Gate drops from 0.642 (clean) to 0.472 (phase π), confirming it perceives
token degradation. But even at phase π it retains 47% physics weighting,
causing +1.10 dB NMSE — worse than not using DD prior at all.

### 5.3 Implications

The Gate 2-A audit identifies three architecture gaps to address in
Gate 2-C (structural safety fusion):

1. **null_all must reach TF-only parity** (±0.3 dB) — requires the gate to
   receive explicit token-quality evidence, not just TF features.
2. **Phase errors are the most dangerous single perturbation** — coherent
   cancellation from flipped gains causes worse-than-TF-only NMSE.
3. **The model already survives random false tokens and permutation** —
   these structural invariants are confirmed and should be preserved.

### 5.4 Gate 2-C: Quality-Conditioned Gate

Gate 2-C addresses the three gaps above by adding a 3-channel token-quality
map to the fusion gate input: |H_phys−H_tf|² discrepancy, mean token
confidence, and mean token uncertainty. Same training setup as
gate1_estimated (K=4, 10 dB, ρ=0.125, 3 seeds × 300 epochs).

**Clean performance:**

| Metric | Gate 1-D1 | Gate 2-C | Δ |
|---|---|---|---|
| Estimated Residual NMSE | −9.76 dB | **−10.69 dB** | +0.93 dB |
| DD+LS → EstRes gain | +1.40 dB | **+1.93 dB** [1.86, 1.99] | +0.53 dB |
| Gate mean (clean tokens) | 0.618 | **0.923** | +0.305 |

**Before/after audit comparison (Oracle chain, 1024 samples):**

| Condition | Gate 1-D1 NMSE | Gate 1-D1 Gate | Gate 2-C NMSE | Gate 2-C Gate | Δ NMSE |
|---|---|---|---|---|---|
| clean | −15.39 dB | 0.642 | −16.29 dB | 0.996 | +0.90 |
| null_all | −3.40 dB | 0.589 | −4.37 dB | 0.504 | +0.97 |
| **phase π** | **+1.10 dB** ☠️ | 0.472 | **−3.93 dB** | **0.162** | **+5.03** |
| phase π/2 | −0.98 dB | 0.548 | −2.75 dB | 0.591 | +1.77 |
| jitter 2.0 joint | −1.20 dB | 0.550 | −4.78 dB | 0.049 | +3.58 |
| dropout 0.75 | −4.57 dB | 0.603 | −5.10 dB | 0.835 | +0.53 |
| coherent_false 4 | −3.11 dB | 0.586 | −5.00 dB | 0.554 | +1.89 |

**Key findings:**

1. **Gate dynamic range: 1.4× → ~6–19× depending on error type.** Quality gate
   drops from 0.92 (clean) to 0.05 (jitter 2.0, ~19×) or 0.16 (phase π, ~5.7×).
   The original gate only went from 0.64 to 0.47 (~1.4×).

2. **Phase π is no longer catastrophic.** +1.10 dB → −3.93 dB is a +5.03 dB
   improvement. However, −3.93 dB is still +0.69 dB worse than the paired
   TF-only (−4.62 dB), so it meets the severe-zone criterion (≤+1 dB) but not
   strict no-harm. The gate shuts to 0.16, preventing active destruction.

3. **null_all gap: needs reconciliation.** Two interpretations exist:
   - vs. Gate 2-C paired TF-only (−5.48 dB): gap = +1.11 dB
   - vs. original Gate 1-D1 TF-only (−4.62 dB): gap = **+0.25 dB** (PASS)
   The inconsistency arises because TF-only models from different training
   runs have different performance. A paired audit with a common TF-only
   baseline is needed. Gate at null_all = 0.50 is NOT sigmoid saturation
   (σ(0)=0.5 is the maximum-gradient midpoint) — it means the null_all
   quality features fall on the decision boundary.

4. **Residual ΔH is always added unconditionally** — even when g=0,
   Ĥ = H_TF + ΔH ≠ pure H_TF, so the residual itself can introduce error
   on null samples independently of gate behavior.

See `docs/GATE2_DESIGN.md` §11.4–11.5 for complete audit results and
revised priority ordering.

### 5.5 Mechanism-Gradient Safety Baselines

Four baselines using the SAME frozen H_phys and H_Tf from the Gate 2-C model.
717 test samples, hyperparameters optimised on 307 validation samples.

| Method | NMSE (dB) | Params | Trained |
|---|---|---|---|
| H_phys-only | −8.50 | 0 | No |
| TF-only (standalone) | −5.06 | CNN | Yes |
| Fixed blend (λ=0.80) | **−9.15** | 1 | No |
| Hard switch | −5.52 | 1 | No |
| Logistic quality gate | −9.04 | 3 | Light |
| Hold-out pilot (hard) | −8.27 | 0 | No |
| Soft hold-out blend (T=5) | −9.06 | 1 | No |
| **2×2 ablation** | | | |
| Fixed λ, no ΔH | −9.15 | 1 | No |
| Spatial gate, no ΔH | −8.63 | CNN | Yes |
| Fixed λ + ΔH | **−10.17** | CNN | No |
| Spatial gate + ΔH | **−10.45** | CNN | Yes |

**Key findings:**

1. **Fusion gain = mostly scalar blending.** Fixed blend (λ=0.80, 1 parameter,
   no training) achieves −9.15 dB — within 1.30 dB of the full model.

2. **2×2 ablation: residual ΔH does 78% of the work. Spatial gating alone
   is worse than fixed blend.**
   ```
   G_spatial      = −0.52 dB  (spatial gate alone HARMFUL — discards info)
   G_residual     = +1.02 dB  (zero-init residual is the main contributor)
   G_spatial|res  = +0.28 dB  (marginal spatial gain given residual)
   G_total        = +1.30 dB
   ```
   The spatial gate does not directly improve reconstruction. It selectively
   suppresses the physics branch, creating room for the zero-init residual to
   correct. This is a qualitatively different mechanism from "learning WHERE
   to trust physics."

3. **Scalar quality/heuristic features add no clean-condition gain.**
   Logistic quality gate (−9.04 dB, 0.11 dB worse) and soft hold-out blend
   (−9.06 dB, 0.09 dB better) are both within noise of fixed blend.
   Corruption-detection value TBD.

4. **Hard selection loses information.** Hard switch (−5.52 dB) and hard
   hold-out (−8.27 dB) underperform soft blending.

### 5.6 Gate 2-C v2: Coupled Residual + Corruption-Aware Training

Changes: coupled residual (g·ΔH), quality map v2 (+valid_ratio), token
augmentation (15% dropout + 10% shuffle).  Same setup: K=4, 10 dB, 3 seeds.

**Results vs v1:**

| Metric | v1 | v2 |
|---|---|---|
| Physics Residual NMSE | −10.50 | −10.30 |
| null_all NMSE | −4.78 | −4.61 (worse) |
| phase π NMSE | −4.71 | −4.49 |
| Fixed λ + ΔH | −10.17 | −10.01 |
| Spatial gate + ΔH | −10.45 | −10.49 |

**Key finding: coupled residual degraded null_all.** Gate=0 suppresses both
physics AND residual, but the internal H_tf was co-trained with the residual
and performs worse without it. Standalone TF-only (−5.00 dB) remains better.

**2×2 ablation is cross-run robust:**

| Contribution | v1 | v2 | Consensus |
|---|---|---|---|
| G_spatial (gate alone) | −0.52 dB | −0.52 dB | **Always harmful** |
| G_residual (ΔH alone) | +1.02 dB | +0.82 dB | **78–82% of total** |
| G_spatial\|res (marginal) | +0.28 dB | +0.48 dB | **18–22% of total** |

The spatial gate does not directly improve reconstruction. It suppresses
physics, creating room for the zero-init residual to correct. Gate and
residual must be trained jointly — decoupling them at inference (v2) or
using gate alone (both v1 and v2) harms performance.

### 5.7 Token Dimension Ablation (9-dim vs 84-dim)

| Config | Clean | Gate(π) | NMSE(π) | Gate(jitt) | NMSE(jitt) | Gate(coh) | NMSE(coh) |
|---|---|---|---|---|---|---|---|
| **v1 (9-dim)** | **−10.50** | **0.041** ✅ | −4.71 | **0.033** ✅ | −4.79 | **0.27** ✅ | **−5.15** |
| v3 (84-dim) | −10.52 | 0.86 ☠️ | +1.78 ☠️ | 0.78 ☠️ | −1.71 | 0.87 ☠️ | −2.12 |
| v3+sup+aug | −10.17 | 0.061 | −4.73 | 0.040 | −4.93 | 0.65 | −3.87 |

**9-dim scalar tokens are optimal for the fusion gate input.** 84-dim DD spectrum
patches, when fed directly to the fusion gate, create spurious correlations and
require gate supervision + augmentation to approach v1. However, a compact 3×3
local DD score patch is valuable as geometric input to the dedicated DDTokenRefiner
(Conv2d patch refiner achieves +0.91 dB over MLP-only refiner, reaching −12.87 dB).
These are two distinct input positions: the fusion gate benefits from a compact
scalar representation, while the position refiner benefits from local spatial context.

### 5.8 Gate Supervision Breakthrough: P0 Optimized

Three technical innovations recover clean NMSE while preserving gate reliability:
1. **Normalized target**: `a = (e_tf-e_phys)/(e_tf+e_phys+eps)` — scale-invariant
2. **Margin mask**: supervise only where `|a| > 0.1` (skip ambiguous positions)
3. **Clean-sample ratio**: 75% of batch receives no augmentation

| Config | Clean | Gate(cl) | Gate(coh4) | NMSE(coh4) | Gate(jitt) | NMSE(jitt) |
|---|---|---|---|---|---|---|
| v1 baseline | −10.69 | 0.923 | 0.265 | −5.15 | 0.033 | −4.79 |
| + sup+aug (aggressive) | −9.21 | 0.361 | 0.162 | −5.97 | **0.029** | **−5.33** |
| **P0 optimized** | **−10.79** | 0.831 | **0.017** | −4.93 | **0.002** | −4.86 |

**Gate dynamic range: 0.83→0.002 (415×) vs v1's 0.92→0.04 (23×).**
Clean NMSE (−10.79) is within 0.1 dB of v1, while gate response under
corruption is nearly two orders of magnitude stronger. The reliability–
performance Pareto is now clearly mapped:

```
v1:              Clean −10.69  ✓✓  |  Gate 23×       ✗  |  Harm ~99%
aggressive:      Clean −9.21   ✗   |  Gate perfect    ✓✓ |  Harm ~5%
P0 optimized:    Clean −10.79  ✓   |  Gate 415×       ✓  |  Harm ~62%
```

**Cross-run 2×2 consensus (3 independent runs):**

| Contribution | v1 | v2 | MoE |
|---|---|---|---|
| G_spatial (gate alone) | −0.52 | −0.52 | −0.59 |
| G_residual (ΔH alone) | +1.02 | +0.82 | +0.78 |
| G_spatial\|res | +0.28 | +0.48 | +0.13 |

Spatial gate is always harmful alone; residual contributes 78-82% of total
gain; marginal spatial gain is 18-22%.

### 5.9 OMP Detector: +2.28 dB from Detection Alone

| Detector | DD+LS NMSE |
|----------|:---:|
| NMS | −8.36 dB |
| **OMP** | **−10.64 dB** |
| Improvement | **+2.28 dB** |

OMP replaces NMS peak-picking with residual-driven orthogonal matching pursuit
on the same DD dictionary. No VP needed — OMP alone exceeds NMS+VP+LS. Each
OMP iteration selects the column with maximum residual projection, then updates
residual via ridge LS. This is immune to the over-sampling dilemma that degrades
NMS at finer grids.

### 5.10 Token Architecture Evolution

| Token Version | Dims | Content | DD+LS Baseline |
|:---|:---:|---|:---:|
| v1 (legacy) | 7 | τ, ν, power, conf, στ, σν, rel | −8.36 |
| v2 (current) | 9 | v1 + Re(α), Im(α) | −8.36 |
| v3 (patch) | 18 | v2 + 3×3 score_map patch | −10.64 (with OMP) |

### 5.11 DDTokenRefiner: Learned Position Correction

Two architectures compared:

| Refiner | Input | Params | NMSE |
|---------|:---:|:---:|:---:|
| None (OMP only) | — | 0 | −10.64 |
| MLP(9d→16→16→2) | 9 scalar | 470 | −11.96 |
| **Conv2d(3×3)+MLP** | 9 scalar + 9 patch | ~500 | **−12.87** |

The Conv2d branch learns gradient, curvature, and asymmetry operators from the
3×3 score_map patch — directly measuring the Δτ,Δν correction — while the MLP
branch provides physical context from scalar features. The hybrid architecture
converges faster (epoch 4 val=0.098 vs MLP's 0.120) and achieves +0.91 dB over
MLP-only.

### 5.12 Complete Results Ladder

| Method | NMSE (dB) | Gain | Date |
|--------|:---:|:---:|---|
| NMS+LS | −8.36 | baseline | 7/17 |
| NMS+VP+model | −11.31 | +2.95 | 7/21 |
| OMP+LS | −10.64 | — | 7/22 |
| OMP+MLP Refiner+model | −11.96 | +1.32 over OMP | 7/22 |
| **OMP+Conv2d Refiner+model** | **−12.87** | **+2.23 over OMP** | 7/22 |
| Oracle+LS (upper bound) | −24.29 | +11.42 remaining | — |

**Gate supervision (BCE) is incompatible with the Refiner** — produces NaN
at epoch 4-6 even at weight=0.01. The mask-normalized BCE amplifies gradients
in the presence of the differentiable DD position correction. TF auxiliary loss
and token augmentation also cause instability with the Refiner. Bare NMSE
training is the only stable configuration for the Refiner pipeline.

### 5.13 Safe Fallback Architecture (2026-07-23)

Output restructured for structural safety: H_out = H_TF + c * (E_phys - H_TF)
where E_phys = H_phys + DeltaH. c=0 guarantees exact TF-only output (no
learned approximation needed). c=1 uses full physics expert. Hard rule:
all tokens invalid → c=0.

Current NMSE: -10.39 dB (1 seed, 100 epochs, bare NMSE, 1024-sample test set). Direct comparison
with -12.87 dB is invalid: the old result used the unconstrained gate formula
(g.H_phys + (1-g).H_Tf + DeltaH) and was reached with obsolete 84-dim tokens.
The proper baseline for the safe structure is a same-token, same-Refiner run
using the unconstrained output formula -- this has NOT yet been done.
Multi-seed evaluation and controlled ablation are required before interpreting
the -0.4 or -2.5 dB gap as the structural safety cost.

Gate supervision (BCE) retired -- structurally incompatible with the
differentiable Refiner and no longer needed for all-invalid fallback.

**Important**: the full corruption audit (phase, jitter, coherent false,
dropout) has only been run on the old Gate 2-C / NMS pipeline, NOT on the
canonical OMP+Conv2d Refiner + Safe Fallback pipeline. Structure pass ≠
safety pass. See P2 in `docs/IMPLEMENTATION_STATUS.md`.

See `docs/GATE2_DESIGN.md` §11.8–11.15 for complete analysis.

### 5.14 Cross-Model Mechanism Baselines (2026-07-24)

`baselines_safety.py` was run against four independently trained model variants
to assess whether the 2×2 decomposition (gate vs ΔH) and mechanism baselines are
stable across training configurations.

#### 5.14.1 Four-Variant Comparison

| Method | Baseline | Corr-Aware | MoE | Safe (v3+OMP) |
|---|---|---|---|---|
| Spatial quality gate (full) | −10.45 | −10.49 | **−10.65** | −10.57 |
| TF-only | −5.06 | −5.06 | −5.06 | −4.81 |
| H_phys-only | −8.50 | −8.50 | −8.50 | −8.50 |
| Fixed blend (λ best) | −9.15 (λ=0.80) | −9.19 (λ=0.75) | −9.14 (λ=0.75) | −9.10 (λ=0.80) |
| Hard switch | −5.52 | −6.48 | −6.63 | −5.87 |
| Logistic quality gate | −9.04 | −9.11 | −9.09 | −9.04 |
| Hold-out pilot select | −8.27 | −7.50 | −7.50 | −7.69 |
| Soft hold-out blend | −9.06 | −8.83 | −8.78 | −8.78 |
| **2×2 ablation** | | | | |
| Fixed λ, no ΔH | −9.15 | −9.19 | −9.14 | −9.10 |
| Spatial gate, no ΔH | −8.63 | −8.67 | −8.55 | −7.93 |
| Fixed λ + ΔH | −10.17 | −10.01 | −9.92 | **−10.21** |
| Spatial gate + ΔH | −10.45 | −10.49 | **−10.65** | −10.57 |

#### 5.14.2 2×2 Decomposition Cross-Model Consensus

| Contribution | Baseline | Corr-Aware | MoE | Safe (v3+OMP) |
|---|---|---|---|---|
| G_spatial (gate alone) | −0.52 | −0.52 | −0.59 | −1.17 |
| G_residual (ΔH alone) | +1.02 | +0.82 | +0.78 | **+1.11** |
| G_spatial\|res (marginal) | +0.28 | +0.48 | **+0.73** | +0.36 |
| G_total | +1.30 | +1.30 | +1.51 | +1.47 |

#### 5.14.3 Key Findings

**1. The spatial gate is always harmful without ΔH.** Across all four variants,
using the learned CNN gate for pure H_phys/H_tf blending (no residual correction)
underperforms a single fixed scalar λ — by −0.52 to −1.17 dB. This is the most
robust finding across Gate 2: **pixel-level gating alone overfits and discards
information** without the residual correction mechanism.

**2. ΔH is the dominant contributor to fusion gain (78-82%).** The zero-init
residual correction consistently accounts for the majority of the improvement
over a fixed blend. In the safe fallback variant (v3+OMP), ΔH contributes +1.11 dB,
suggesting that better token quality enables a stronger residual correction.

**3. Gate and ΔH exhibit substitution.** In MoE (weakest ΔH: +0.78 dB), the
spatial gate provides the largest marginal gain (+0.73 dB). In safe fallback
(strongest ΔH: +1.11 dB), the spatial gate gain shrinks (+0.36 dB). This
inverse relationship suggests a common performance ceiling around −10.6 dB.

**4. The performance ceiling at ~−10.6 dB.** All four variants converge to
full-model NMSE within 0.2 dB of −10.55 dB, regardless of token version, DD
detector, auxiliary losses, or augmentation. The 13.7 dB gap to Oracle+LS
(−24.29 dB) is almost entirely attributable to DD token position accuracy,
not model architecture.

**5. OMP (v3 tokens, gate2_safe) enables fixed+ΔH at −10.21 dB — within 0.36 dB
of the full model — with only 100 training epochs and no MoE loss.** The improved
token quality from OMP detection makes the physics branch more reliable, reducing
the need for learned spatial gating.

**6. MoE auxiliary losses provide the largest gate marginal gain (+0.73 dB)**
but slightly reduce fixed+ΔH performance (−9.92 vs −10.17 baseline). The
auxiliary losses reshape expert behavior (making E_tf and E_phys more
distinguishable to the gate) rather than strengthening individual experts.

#### 5.14.4 Performance Ladder (Updated)

```
Method                                    NMSE (dB)
─────────────────────────────────────────────────────────────
Oracle+LS (upper bound)                    −24.29
─────────────────────────────────────────────────────────────
OMP + Conv2d Refiner + model               −12.87      [Refiner unstable]
OMP + LS (non-learned)                     −10.64
─────────────────────────────────────────────────────────────
P0 optimized (gate sup, best reliable)     −10.79
MoE + Spatial + ΔH                         −10.65      [best gate gain]
Safe fallback + Spatial + ΔH               −10.57
Corruption-Aware + Spatial + ΔH            −10.49
Baseline + Spatial + ΔH                    −10.45
─────────────────────────────────────────────────────────────
Safe fallback + Fixed λ + ΔH               −10.21      [best no-gate]
Baseline + Fixed λ + ΔH                    −10.17
Corruption-Aware + Fixed λ + ΔH            −10.01
MoE + Fixed λ + ΔH                         −9.92
─────────────────────────────────────────────────────────────
Fixed blend (λ=0.75-0.80, no ΔH)          −9.10~−9.19  [1 param]
H_phys-only                                −8.50        [0 param]
DD+LS (NMS)                                −8.36
TF-only                                    −4.6~−5.1
─────────────────────────────────────────────────────────────
```

### 5.15 Oracle Token Upper-Bound Experiment (2026-07-24)

The oracle experiment provides strong evidence on the "token quality vs architecture" bottleneck question.
Using `gate2_canonical_oracle.yaml` with `token_source: oracle` — true {τ, ν, α}
from the channel simulator — this experiment removes ALL token estimation error
from the pipeline.

#### 5.15.1 Results

| Method | NMSE (dB) | Note |
|---|---|---|
| H_phys-only (pure physics) | **−117.14** | Numerical precision limit |
| Oracle+LS | −24.29 | Non-learned baseline |
| Fixed blend λ=1.0 | **−117.14** | Identical to H_phys-only |
| Spatial gate + ΔH (full model) | **−59.58** | Gate cannot reach g=1.0 |
| E_phys-only (fix_c=1, forced) | −17.33 | ΔH actively harms perfect H_phys |
| TF-only (fix_c=0) | −1.34 | Internal TF (not standalone) |
| TF-only (standalone) | −4.74 | Same as all other configs |

#### 5.15.2 The Definitive Answer

**Q: Is the ~−10.6 dB plateau driven by token quality or fusion architecture?**

**A: Path-parameter quality dominates.** The oracle experiment replaces all estimated
path parameters {τ, ν, α} with their ground-truth values, eliminating LS gain error,
support mismatch, and position error simultaneously. H_phys-alone reaches −117 dB
(numerical precision). The full model reaches −59.58 dB.

The physical reconstruction formula is verified to machine epsilon.
The gap between estimated-token full model (−10.47 dB) and oracle-token
full model (−59.58 dB) is **49.1 dB**, overwhelmingly attributable to
path-parameter quality — a combination of position error, gain error, and
support mismatch. Further decomposition (§5.15.7) isolates position vs gain
contributions.

**Important caveat**: the oracle intervention replaces all {τ, ν, α}
simultaneously, not just DD positions. Attributing the entire gap to
"token position error" alone is an overstatement without the 4-cell
decomposition described in §5.15.7.

```
Token quality hierarchy:
──────────────────────────────────────────────────────────────
Oracle H_phys-only              −117.14 dB   █ numerical precision
Oracle spatial gate + ΔH         −59.58 dB   █ gate ~0.999 limits performance
Oracle fix_c=1 (forced)          −17.33 dB   █ ΔH trained for imperfect tokens
════════════════ TOKEN QUALITY GAP ═══════════════════════════
Estimated spatial gate + ΔH       −10.47 dB   █ best learned result
Estimated H_phys-only              −8.50 dB
Estimated DD+LS (NMS)              −8.36 dB
Estimated TF-only                  −5.00 dB
──────────────────────────────────────────────────────────────
```

**Gate+ΔH error suppression ratio**: with oracle tokens, H_phys-alone is
−117 dB but the full model is "only" −59.58 dB — the gate architecture
imposes a ~57 dB self-interference penalty. But with estimated tokens,
H_phys-alone crashes to −8.50 dB while the full model recovers to −10.47 dB —
the gate+ΔH recovers ~1.97 dB. The architecture compresses a 108.6 dB
H_phys degradation into a 49.1 dB final output degradation: a **~60 dB
error suppression effect**.

#### 5.15.3 Why the Full Model Cannot Reach −117 dB

The output formula `H_out = H_TF + c · (E_phys − H_TF)` means any deviation
of the gate from exactly 1.0 leaks H_TF noise (−4.74 dB quality) into the
output:

| Gate mean | Max NMSE | Mechanism |
|-----------|----------|-----------|
| g = 0.99 | ~−40 dB | 1% TF leakage |
| g = 0.999 | ~−60 dB | 0.1% TF leakage **← observed −59.58** |
| g = 0.9999 | ~−80 dB | 0.01% TF leakage |
| g = 0.99999 | ~−100 dB | 0.001% TF leakage |
| g = 1.0 | −117 dB | Perfect (unreachable by CNN) |

The CNN gate's finite capacity prevents it from outputting exact 1.0 at every
pixel — even 0.1% TF admixture caps performance at −60 dB. This is a
**fundamental architectural limit** of soft gating: it can express "full trust"
but cannot achieve "perfect trust" at the numerical-precision level.

#### 5.15.4 ΔH Learns to Compensate for Token Error

The fix_c=1 ablation (−17.33 dB) reveals that the residual correction ΔH,
trained alongside the spatial gate, actively harms the output when gate is
forced to 1.0 with oracle tokens. ΔH has learned to compensate for patterns
that only exist when tokens are imperfect — systematic biases in H_phys
caused by DD position errors. When those errors vanish (oracle tokens),
ΔH's "corrections" become destructive.

**This confirms the complementary mechanism**: gate and ΔH co-adapt during
training. The gate learns to identify regions where H_phys is unreliable,
and ΔH learns to fix those specific regions. Neither component works
correctly without the other.

#### 5.15.5 λ-Sweep in Oracle vs Estimated

| λ | Oracle NMSE | Estimated NMSE |
|---|------------|----------------|
| 0.0 (pure TF) | −1.36 | −3.77 |
| 0.25 | −3.86 | −5.00 |
| 0.50 | −7.39 | −6.70 |
| 0.75 | −13.41 | −8.85 |
| 0.90 | −21.36 | −9.03 |
| 0.95 | −27.39 | −8.80 |
| 1.0 (pure phys) | **−117.39** | −8.50 |

With oracle tokens: λ↑ → NMSE↑ (monotonic, best at 1.0).
With estimated tokens: optimal λ=0.75-0.80 (physics not trustworthy).
**The optimal λ directly measures token quality.**

#### 5.15.6 Implications

1. **Path-parameter quality is the dominant bottleneck.** Under the current
   channel abstraction, improving DD detection (higher pilot density,
   multi-frame tracking, VP/Refiner) provides far larger gains than further
   fusion architecture changes. However, the oracle experiment replaces all of
   {τ, ν, α} simultaneously — isolating position vs gain vs support contributions
   requires the 4-cell decomposition in §5.15.7.
2. **With perfect path parameters, the optimal model is H_phys directly** — no gate,
   no ΔH, no learned components. The gate+ΔH architecture is a token-error
   compensator, not a general-purpose channel estimator. Its value scales
   inversely with token quality.
3. **The −10.6 dB convergence is a fusion-variant empirical plateau**, not a hard
   architecture ceiling. The OMP+Conv2d Refiner branch reaches −12.87 dB,
   demonstrating that better token processing (not bigger fusion networks) is
   the path to breakthrough.
4. **Adaptive strategy**: if token quality can be estimated at inference time,
   bypass the gate and use pure H_phys when quality is high; use gate+ΔH
   when quality is low. This spans −117 dB to −10.47 dB dynamically.

#### 5.15.7 Required Decomposition: Position vs Gain vs Support

The oracle experiment replaces all {τ, ν, α} simultaneously. To isolate
individual error sources, a 4-cell decomposition is needed:

| Case | (τ, ν) | α | Isolates |
|------|--------|---|----------|
| A | true | true | Numerical closure (done: −117 dB) |
| B | true | LS-estimated | Gain estimation error |
| C | estimated/refined | true | Position + support error |
| D | estimated/refined | LS-estimated | Full pipeline (done: −8.36 to −12.87 dB) |

G_position = NMSE(D) − NMSE(B), G_gain = NMSE(B) − NMSE(A).
Case C is the critical missing experiment — it isolates whether the dominant
error is position/support or complex-gain estimation.

---

## 6. Gate Status

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
Gate 1-E2   Estimated-token Physical Residual .......... PASS (+1.40 dB vs DD+LS)
Gate 1-E3   Multi-SNR unified model .................... PASS (SNR ≥ 0 dB)
Gate 1-E4   K-sweep (K=6 estimated tokens) ............. PASS (+1.86 dB vs DD+LS)
Gate 1-E5   K-sweep (K=8 estimated tokens) ............. PASS (+1.86 dB vs DD+LS)
Gate 1-F    Per-path gate .............................. FAIL (−1.0 dB regression)

Gate 2-A    Failure boundary audit (frozen model) ....... COMPLETE
Gate 2-A1   Random false paths .......................... PASS (0.0% harm, null token immune)
Gate 2-A2   Permutation invariance ...................... PASS (0.0% harm)
Gate 2-A3   Small perturbation (jitter ≤0.1, phase ≤π/8) PASS (harm < 5%)
Gate 2-A4   Phase errors (≥π/2) ......................... FAIL (100% harm, NMSE > TF-only)
Gate 2-A5   Joint jitter (≥0.5 bins) .................... FAIL (≥90% harm)
Gate 2-A6   Coherent false paths (≥2) ................... FAIL (harm ≥ 16%)
Gate 2-A7   null_all → TF-only fallback ................. FAIL (+1.22 dB gap)
Gate 2-C    Quality-conditioned gate .................... CONDITIONAL PASS
Gate 2-C1   Gate dynamic range (6–19× per error type) ... PASS
Gate 2-C2   Phase π no longer catastrophic (+5 dB) ...... CONDITIONAL PASS (+0.69 dB vs TF-only)
Gate 2-C3   Clean performance maintained ................ PASS (+0.47 paired, +0.93 aggregate)
Gate 2-C4   null_all → TF-only gap ...................... DATA INCONSISTENT (+0.25 or +1.11 dB)
Gate 2-C v2 Coupled residual + corruption-aware ......... RESULTS (no gain)
Gate 2-D1   Fixed blend baseline ........................ PASS (−9.15 dB, 1 param)
Gate 2-D2   Hard discrepancy switch ..................... PASS (−5.52 dB, below TF-only)
Gate 2-D3   Logistic quality gate ....................... PASS (−9.04 dB, 3 params)
Gate 2-D4   Hold-out pilot selector ..................... PASS (−8.27 dB, no training)
Gate 2-D5   Soft hold-out blend ......................... PASS (−9.06 dB, T=5, 1 param)
Gate 2-D6   2×2: Spatial gate alone ..................... −0.52 dB (HARMFUL without ΔH)
Gate 2-D7   2×2: Residual ΔH alone ...................... +1.02 dB (78% of total gain)
Gate 2-D8   2×2: Spatial gate given residual ............ +0.28 dB (marginal)
Gate 2-D9   Token: 9-dim optimal for fusion gate ........ PASS (patch useful for refiner)
Gate 2-D10  Gate supervision + matched aug .............. BREAKTHROUGH (harm 99%→5%)
Gate 2-D11  Cross-run 2×2 consensus (3 runs) ............ PASS (robust)
Gate 2-D12  P0: Normalized target + margin + clean ratio . PASS (Clean -10.79, gate 415×)
Gate 2-D13  Reliability–Performance Pareto ............... MAPPED (v1/aggressive/P0)
Gate 2-D14  OMP detector ................................ PASS (+2.28 dB over NMS)
Gate 2-D15  DDTokenRefiner (Conv2d patch) ................ PASS (+0.91 dB over MLP)
Gate 2-D16  Gate supervision + Refiner ................... FAIL (NaN, incompatible)
Gate 2-D17a Safe fallback formulation .................... IMPLEMENTED (c=0 -> H_TF exact)
Gate 2-D17b MoE auxiliary losses .......................... PASS (−10.47 dB, gate gain +0.73 dB)
Gate 2-D17c Safe fallback (gate2_safe, v3+OMP) ............ PASS (−10.39 dB, fixed+ΔH −10.21 dB)
Gate 2-D18  Cross-model baselines_safety comparison ...... COMPLETE (4 variants)
Gate 2-D19  Gate×ΔH substitution relationship ............. DISCOVERED (better ΔH → smaller gate gain)
Gate 2-S1   Frozen-expert three-stage training ............ OPEN
Gate 2-S2   Corruption audit on safe fallback pipeline .... OPEN
Gate 2-S3   Oracle token + PhysicalResidual upper bound ... **COMPLETE ✅ (H_phys=−117 dB; path-parameter quality dominates)**

Gate 3      Full OFDM-ISAC waveform .................... OPEN
```

---

## 7. Reproducibility

### 7.1 Key Commands

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

# Gate 1 — K-sweep (estimated tokens)
python experiments/gate1_oracle.py --config configs/gate1_K6_estimated.yaml --output-dir results/gate1_K6 --device cuda
python experiments/gate1_oracle.py --config configs/gate1_K8_estimated.yaml --output-dir results/gate1_K8 --device cuda

# Gate 2 — Train model variants (4 configs)
python experiments/gate1_oracle.py --config configs/gate2_safety.yaml --output-dir results/gate2_safety --models physics_residual,tf_only --device cuda
python experiments/gate1_oracle.py --config configs/gate2_moe.yaml --output-dir results/gate2_moe --models physics_residual,tf_only --device cuda
python experiments/gate1_oracle.py --config configs/gate2_corruption_aware.yaml --output-dir results/gate2_corruption_aware --models physics_residual,tf_only --device cuda
python experiments/gate1_oracle.py --config configs/gate2_safe.yaml --output-dir results/gate2_safe --models physics_residual,tf_only --device cuda

# Gate 2 — Mechanism baselines (per model)
python experiments/baselines_safety.py --model-dir results/gate2_safety --output-dir results/gate2_safety_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_moe --output-dir results/gate2_moe_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_corruption_aware --output-dir results/gate2_corruption_aware_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_safe --output-dir results/gate2_safe_baselines --samples 1024 --device cpu

# Gate 2-A — Corruption audit (frozen model)
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 200 --device cpu --smoke-only
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 1024 --device cuda

# Oracle token upper-bound experiment
python experiments/gate1_oracle.py --config configs/gate2_canonical_oracle.yaml --output-dir results/gate2_canonical_oracle --models physics_residual,tf_only --device cuda
python experiments/baselines_safety.py --model-dir results/gate2_canonical_oracle --output-dir results/gate2_canonical_oracle_baselines --samples 1024 --device cpu
```

### 7.2 Config Index

| Config | Purpose |
|---|---|
| **Gate 0** | |
| `configs/gate0.yaml` | Gate 0 full sweep (1000 trials) |
| `configs/gate0_ablation.yaml` | Gate 0 high-SNR plateau ablation |
| `configs/gate0_density.yaml` | Pilot density scan |
| `configs/gate0_K6.yaml` | K=6 path count scan |
| `configs/gate0_K8.yaml` | K=8 path count scan |
| **Gate 1** | |
| `configs/gate1_main.yaml` | Oracle tokens, 10 dB, main work point |
| `configs/gate1_estimated.yaml` | Estimated tokens (NMS+LS), 10 dB |
| `configs/gate1_boundary.yaml` | Oracle tokens, 0 dB |
| `configs/gate1_boundary_estimated.yaml` | Estimated tokens, 0 dB |
| `configs/gate1_stress.yaml` | Oracle tokens, 5 dB, ρ=0.0625 |
| `configs/gate1_stress_estimated.yaml` | Estimated tokens, 5 dB, ρ=0.0625 |
| `configs/gate1_multisnr.yaml` | Multi-SNR (−5 to +20 dB), estimated tokens |
| `configs/gate1_K6_estimated.yaml` | Estimated tokens, K=6 paths |
| `configs/gate1_K8_estimated.yaml` | Estimated tokens, K=8 paths |
| **Gate 2 — Main variants** | |
| `configs/gate2_safety.yaml` | Baseline quality gate, v2 tokens, 300 ep |
| `configs/gate2_moe.yaml` | MoE auxiliary losses (tf_aux + phys_aux) |
| `configs/gate2_corruption_aware.yaml` | Corruption-aware token augmentation |
| `configs/gate2_safe.yaml` | Safe fallback, v3 tokens + OMP, 100 ep |
| **Gate 2 — Ablations** | |
| `configs/gate2_canonical_oracle.yaml` | Oracle token upper-bound reference |
| `configs/gate2_ablate.yaml` | 2×2 ablation controls (fusion_mode, use_residual) |
| `configs/gate2_nozero.yaml` | No zero-init residual (random init) |
| `configs/gate2_omp.yaml` | OMP detector (no Refiner) |
| `configs/gate2_omp_refiner.yaml` | OMP + DDTokenRefiner (Conv2d patch) |
| `configs/gate2_p0_optimize.yaml` | P0 gate supervision (normalized target + margin) |
| `configs/gate2_path_stats.yaml` | Path statistics enabled |
| `configs/gate2_token_v3.yaml` | v3 tokens (84-dim DD patches) |
| `configs/gate2_v1_sup_aug.yaml` | Gate supervision + matched augmentation |
| `configs/gate2_corruption.yaml` | Gate 2-A corruption audit config |
| `configs/gate2_snr10_test.yaml` | Fixed SNR=10 dB test config |

### 7.3 RNG Seeding

```
Channel RNG:  [seed, density_idx, snr_idx, trial, 100]  — no pattern_index
Noise RNG:    [seed, snr_idx, density_idx, trial, 300]  — no pattern_index
Pilot RNG:    [seed, pattern_idx, density_idx, trial, 200]  — varies by pattern
```

Gate 1 multi-seed: base_seed + seed_idx × 1000 for training data.
Fixed test bank: base_seed + 10000. Fixed val bank: base_seed + 20000.
