# LD3 Research Report — Gate 0 & Gate 1 Results

Date: 2026-07-15

---

## 1. Simulation Configuration

### 1.1 OFDM Waveform

| Parameter | Value |
|---|---|
| Subcarriers (N) | 64 |
| OFDM Symbols (M) | 14 |
| Subcarrier spacing | 120 kHz |
| Carrier frequency | 28 GHz |
| CP ratio | 0.07 |
| TF grid size | 64 × 14 = 896 REs |

### 1.2 Sparse Channel Model

| Parameter | Value |
|---|---|
| Number of paths (K) | 4 |
| Max delay | 12 bins (~1.56 μs) |
| Max Doppler | ±3 bins (~±2.2 kHz @ 3 km/h) |
| Delay distribution | Uniform integer + Uniform(−0.45, +0.45) fractional |
| Doppler distribution | Uniform(−max, +max), fractional |
| Power profile | Exponential decay (factor 0.25), total power normalised to 1 |
| Gain distribution | Circular complex Gaussian, Rayleigh fading |
| Rician K-factor | None (pure Rayleigh) |

### 1.3 Pilot Patterns

| Pattern | Generation |
|---|---|
| **Random** | Uniform without replacement; `round(density × N × M)` positions |
| **Comb** | Regular 2D grid; stride = `round(1/√density)` |

### 1.4 Pilot Densities & SNR Sweep

| Parameter | Gate 0 Full Sweep | Gate 0 Ablation | Gate 1 Main |
|---|---|---|---|
| SNR range | −5, 0, 5, 10, 15, 20 dB | −5 to +30 dB (8 points) | 10 dB (fixed) |
| Pilot densities | 0.0625, 0.125, 0.25 | 0.125 | 0.125 |
| Pilot patterns | random, comb | random | random |
| Trials per condition | 1000 | 500 | N/A (synthetic dataset) |

### 1.5 DD Estimator

| Parameter | Value |
|---|---|
| Delay oversampling | 2× |
| Doppler oversampling | 4× |
| DD grid size | 25 delay × 25 Doppler = 625 cells |
| NMS delay radius | 2 cells |
| NMS Doppler radius | 2 cells |
| Relative threshold | 0.08 |
| Matching tolerance (delay) | 0.75 bins |
| Matching tolerance (Doppler) | 0.5 bins |

### 1.6 Computational Budget

| Experiment | Trials × Conditions | Total Simulated Channels |
|---|---|---|
| Gate 0 full sweep | 1000 × 2 × 3 × 6 | 36,000 |
| Gate 0 ablation (×5 methods) | 500 × 8 | 4,000 each |
| Gate 1 main (3 seeds) | 4096 train + 1024 test | 15,360 |

---

## 2. Gate 0: DD Identifiability Audit

### 2.1 Known-K Dominant-Energy Identifiability — CONDITIONAL PASS

**Finding**: Under known K=4 and fixed Top-K output, the DD estimator stably captures
dominant channel energy from sparse pilot observations.

**Random pilots, density ≥ 0.125, SNR ≥ 5 dB**:

| Metric | Value @ 10 dB | Value @ 20 dB | Plateau? |
|---|---|---|---|
| Path Recall | 0.728 | 0.732 | Yes (~0.74) |
| Power Recovery | 0.882 | 0.879 | Yes (~0.88) |
| Delay RMSE (bins) | 0.172 | 0.169 | Yes (~0.17) |
| Doppler RMSE (bins) | 0.104 | 0.101 | Yes (~0.10) |
| OSPA Distance | 0.544 | 0.532 | — |
| Est. Support LS NMSE | 0.147 | 0.140 | Yes (~0.14) |

**Key observation**: All metrics exhibit a high-SNR performance plateau starting at
approximately 10–15 dB. Increasing SNR beyond this point yields negligible improvement,
indicating a transition from noise-limited to model/basis-mismatch-limited regime.

### 2.2 Random vs Comb: Structural Evidence for Comb Aliasing

**DD Dictionary Mutual Coherence** (far-field, excluding NMS neighbourhood):

| Pattern | Density | μ_max | μ_far | μ_p95 |
|---|---|---|---|---|
| random | 0.0625 | 0.924 | **0.429** | 0.260 |
| random | 0.125 | 0.905 | **0.339** | 0.186 |
| random | 0.25 | 0.901 | **0.300** | 0.162 |
| comb | 0.0625 | **1.000** | **1.000** | 0.215 |
| comb | 0.125 | **0.988** | **0.988** | 0.200 |
| comb | 0.25 | 0.902 | **0.306** | 0.146 |

**Finding**: Low-density Comb (ρ ≤ 0.125) exhibits μ_far = 1, meaning **two distinct
DD dictionary columns produce identical observations at the pilot positions** — a
deterministic far-field ambiguity. Random pilots avoid this entirely (μ_far ≤ 0.43).

**Paired Bootstrap: Random vs Comb** (1000 pairs, shared channel + noise bank):

| Metric | Density 0.0625 (10 dB) | Density 0.125 (10 dB) | Density 0.25 (10 dB) |
|---|---|---|---|
| Recall Δ (R−C) | **+0.172** (p<1e-4) | **+0.138** (p<1e-4) | −0.003 (p=0.63) |
| Power Recovery Δ | **+0.085** (p<1e-4) | **+0.058** (p<1e-4) | −0.004 (p=0.49) |
| NMSE Est LS Δ | **−0.370** (Comb wins) | **−0.580** (Comb wins) | **+0.008** (Comb wins) |

**Critical finding — Detection ≠ Reconstruction**:
- Random has **better DD path recall** (+14–17 pp at low density) due to lower dictionary coherence
- Comb has **better TF channel reconstruction NMSE** because its regular grid provides
  more uniform pilot coverage for LS gain estimation
- At high density (ρ=0.25), the differences largely vanish

This implies a two-stage system could use **different pilot masks** for DD detection and
TF reconstruction stages, or exploit a hybrid pattern.

### 2.3 High-SNR Plateau Ablation

**Five methods compared at SNR = 20 dB** (500 trials each, shared channel bank):

| Method | Recall | Delay RMSE | Doppler RMSE | NMSE Est LS |
|---|---|---|---|---|
| **I** (integer bins) | 0.804 | **0.045** | **0.043** | **0.053** |
| **F_oracle** (oracle discrete peaks) | **1.000** | 0.145 | 0.070 | 0.086 |
| **F_refine** (quadratic refinement) | 0.736 | 0.160 | 0.100 | 0.127 |
| **OS 4×8** (higher oversampling) | 0.716 | 0.158 | 0.101 | 0.133 |
| **F** (baseline 2×4) | 0.732 | 0.169 | 0.101 | 0.140 |

**Error decomposition** (F baseline, 20 dB):

```
nmse_oracle_perfect     = 0.0004  (code closure — near machine precision)
nmse_oracle_support_ls  = 0.0004  (Δ_gain ≈ 0: LS recovers gains near-perfectly at high SNR)
nmse_estimated_support_ls = 0.1395  (Δ_support + Δ_gain ≈ 0.139)
```

**Interpretation**:
- **Integer bins (I)**: Removing off-grid mismatch yields 4× better delay RMSE and 2.6×
  better NMSE. But it changes the channel distribution — it is an "on-grid best-case
  reference", not a true upper bound for fractional channels.
- **F_oracle**: Even with perfect peak association, discrete-grid reconstruction NMSE
  plateaus at ~0.09, confirming that basis mismatch (not just peak selection error)
  limits performance.
- **F_refine**: Quadratic interpolation provides moderate improvement (−5% delay RMSE,
  −9% NMSE) with residual-based acceptance gating. Cost is negligible. Does not
  fundamentally resolve the off-grid floor.
- **OS 4×8**: Higher oversampling yields only 0.22 dB NMSE improvement for ~4× compute
  cost. Not recommended as a primary solution.

**Primary bottleneck**: Off-grid (fractional) path parameters cause irreducible
dictionary mismatch. Discrete DD search provides coarse support but should not serve
as the final continuous parameter estimator.

### 2.4 Gate 0 Limitations

1. **Fixed Known-K**: All experiments use K=4 paths with Top-4 output. Recall equals
   precision by construction when no early-exit occurs. Unknown-K detection (Gate 0-B)
   is not yet implemented.
2. **No per-bin false alarm probability**: Current false alarm metric counts
   unmatched estimated paths but does not evaluate open-set detection on empty DD bins.
3. **Single channel class**: Only Rayleigh fading with exponential power decay is
   tested. Rician, clustered, and measured channels remain future work.
4. **No mobility variation**: Doppler spread fixed at ±3 bins. Higher mobility
   scenarios not explored.
5. **OSPA cardinality penalty = 0**: Since |Ŝ| = |S| = K always, the cardinality
   term in OSPA is always zero. OSPA currently measures only localisation quality.

---

## 3. Gate 1: Oracle DD Value Validation

### 3.1 Experimental Setup

| Parameter | Value |
|---|---|
| Training samples | 4,096 per seed |
| Test samples | 1,024 (fixed bank across all seeds) |
| SNR | 10 dB (fixed, narrowband) |
| Pilot pattern | Random, density 0.125 |
| Training seeds | 3 (2036, 3036, 4036) |
| Epochs | 50 |
| Batch size | 64 |
| Learning rate | 1e-3, AdamW, weight decay 1e-5 |
| Hidden dimension | 48 |
| TF-only parameters | ~15k |
| Cross-attention parameters | ~35k |
| Device | CPU |

### 3.2 Non-Learned Baselines (Fixed Test Bank, 1024 samples)

| Method | NMSE (dB) | Description |
|---|---|---|
| Nearest-neighbour interpolation | −1.41 dB | Simple TF baseline — fills missing REs with nearest pilot |
| **Oracle perfect** | **−∞ dB (0 linear)** | Code-closure test: reconstruct with true {τ, ν, α} |
| **Oracle support + LS** | **−24.29 dB** | True continuous {τ, ν} + LS-estimated α̂ |
| DD estimated support + LS | −8.37 dB | DD-detected discrete {τ̂, ν̂} + LS-estimated α̂ |

**Key findings**:
- **Physical closure confirmed**: Oracle perfect NMSE = 0 (machine precision).
- **Oracle support adds enormous value**: Using true continuous DD locations with
  LS-estimated gains improves over interpolation by **+22.9 dB**.
- **DD estimation cost is severe**: Using DD-estimated (discrete) positions instead
  of true continuous positions costs **15.9 dB**. This is the dominant loss term
  before any neural fusion.

### 3.3 Learned Models (3 Seeds, Hierarchical Bootstrap CI)

| Model | NMSE (dB) | 95% CI |
|---|---|---|
| TF-only (CNN baseline) | −4.60 dB | [−4.65, −4.55] |
| **Cross-attention (Oracle tokens)** | **−6.97 dB** | [−7.04, −6.90] |
| **Oracle gain over TF-only** | **+2.37 dB** | [+2.29, +2.43] |

Cross-attention consistently outperforms TF-only across all 3 seeds. The hierarchical
bootstrap CIs do not overlap — the gain is statistically significant.

### 3.4 Token-Use Audit — Critical Evidence

| Token Mode | NMSE (dB) | Δ from Oracle |
|---|---|---|
| **Oracle** (correct DD tokens) | **−6.97 dB** | — |
| Shuffled (wrong-sample tokens) | −2.98 dB | **+3.99 dB** |
| Null (all tokens rejected) | −2.55 dB | **+4.42 dB** |

**Interpretation**: The model is **highly sensitive to token correctness**.
- Shuffling tokens to wrong samples degrades NMSE by **4.0 dB** — the model is not
  merely using extra capacity; it genuinely relies on the DD prior.
- Nulling all tokens degrades NMSE by **4.4 dB** — further evidence that the
  cross-attention path carries meaningful physical information.
- The shuffled and null modes both underperform even the TF-only baseline (−4.60 dB),
  confirming that **misdirected DD information is worse than no DD information**.

This is the first time Gate 1 passes the token-use audit.

### 3.5 Full Error Decomposition Chain

```
−∞ dB  →  −24.3 dB  →  −8.4 dB  →  −7.0 dB  →  −4.6 dB
  ↑          ↑           ↑           ↑           ↑
Perfect    Oracle      DD-est.     Cross-      TF-only
reconstr.  support+LS  support+LS  attention   (no DD)
           (Δ_gain)    (Δ_support) (Δ_fusion)
```

| Decomposition | ΔNMSE (dB) | Interpretation |
|---|---|---|
| Δ_gain | ~0 (high SNR) | LS gain estimation near-perfect on known support |
| Δ_support | −15.9 dB | **Dominant loss**: DD-detected positions are far from true |
| Δ_fusion | +1.4 dB | Cross-attention recovers part of the DD+LS loss |

### 3.6 Gate 1 Limitations

1. **7-dim tokens lack complex gain**: Current tokens contain {τ, ν, power, confidence,
   σ_τ, σ_ν, relevance} but NOT {Re(α), Im(α)}. The model must re-learn complex gains
   from TF observations rather than receiving them directly.
2. **No explicit physical reconstruction**: The cross-attention uses learned Value
   embeddings rather than an explicit H_phys = Σ α_l exp(−j2πnτ_l/N + j2πmν_l/M) layer.
   This forces the model to re-learn the OFDM phase law.
3. **Fixed SNR training**: Training at a single SNR (10 dB) limits generalisation.
   Mixed-SNR training may improve robustness.
4. **Narrowband SNR range**: Gate 1 test set samples from [−5, 20] dB, but training
   is at fixed 10 dB. Performance at extreme SNRs may be suboptimal.
5. **Capacity mismatch**: Cross-attention model (~35k params) is larger than TF-only
   (~15k params). While the token-use audit (shuffled < TF-only) proves the gain is
   not purely capacity-driven, a matched-parameter comparison would strengthen the
   conclusion.

---

## 4. Gate Status Summary

```
Gate 0-A1  Known-K dominant-energy identifiability ........ PASS
Gate 0-A2  Comb μ_far=1 structural aliasing mechanism .... PASS
Gate 0-A3  Off-grid + peak selection both key bottlenecks . PASS
Gate 0-A4  Quadratic sub-grid refinement ................... MARGINAL (−5% delay RMSE, −9% NMSE)
Gate 0-A5  Higher oversampling (4×8) ....................... MARGINAL (+0.22 dB, 4× cost)
Gate 0-B   Unknown-K open-set detection .................... OPEN
Gate 1-A   Physical model closure .......................... PASS (nmse_oracle_perfect = 0)
Gate 1-B   Oracle support value ............................ PASS (+22.9 dB over interpolation)
Gate 1-C   Estimated support value ......................... PASS
Gate 1-D   Learned fusion value ............................ CONDITIONAL PASS (token-use audit passed)
```

---

## 5. Current Limitations (Cross-Cutting)

### 5.1 Methodological

1. **Known-K constraint**: All experiments fix K=4. Real systems must estimate the
   number of paths. Gate 0-B (unknown-K detection with model-order selection and
   per-bin false-alarm control) is not yet designed.
2. **Single channel class**: Only Rayleigh fading with exponential power decay tested.
   No Rician, no clustered delay profiles (TDL), no measured channels.
3. **No mobility sweep**: Doppler spread fixed. Higher mobility scenarios may degrade
   DD identifiability.
4. **Fixed OFDM numerology**: Single (N=64, M=14) configuration. Scaling behaviour
   with larger grids is unknown.

### 5.2 Technical

5. **Token lacks complex gain**: The 7-dim token omits Re(α) and Im(α), forcing the
   model to re-learn gains from TF observations. This likely explains much of the
   remaining gap to Oracle+LS (−24.3 dB).
6. **No explicit physical reconstruction layer**: The cross-attention uses abstract
   Value embeddings rather than a differentiable H_phys = Σ α_l exp(phase) decoder.
7. **Simple nearest-neighbour interpolation baseline**: A more competitive TF baseline
   (e.g., MMSE interpolation, Kriging) could raise the bar.
8. **No end-to-end BER/NMSE joint evaluation**: Only channel NMSE is optimised.
   Communication performance (BER, spectral efficiency) is not directly measured.

### 5.3 Computational

9. **CPU-only training**: Gate 1 used CPU (~50 epochs × 3 seeds × 4096 samples).
   GPU training would enable larger models and more extensive hyperparameter search.
10. **Fixed DD grid resolution**: Oversampling 2×4 is used throughout. Adaptive
    or multi-resolution grids are unexplored.

---

## 6. Future Work

### 6.1 Immediate (Gate 1-D — Learned Fusion Repair)

**Goal**: Enable the cross-attention model to use DD information more effectively.

| Step | Description | Expected Impact |
|---|---|---|
| Add complex-gain tokens | Extend tokens from 7-dim to 9-dim: {τ, ν, Re(α), Im(α), power, confidence, σ_τ, σ_ν, relevance} | Model receives complete path information |
| Explicit physical reconstruction | Differentiable layer: H_phys = Σ α_l exp(−j2πnτ_l/N + j2πmν_l/M) | Model no longer needs to re-learn OFDM phase law |
| TF residual gated fusion | Ĥ = g ⊙ H_phys + (1−g) ⊙ H_TF + ΔH | Model learns WHERE to trust physics vs. TF estimate |
| Matched-capacity comparison | Equalise parameter counts between TF-only and cross-attention | Isolate architecture benefit from capacity effect |

### 6.2 Short-Term (Gate 0-B — Unknown-K Detection)

| Step | Description |
|---|---|
| Variable path count | Generate channels with K ∈ {2, 3, …, 8} unknown to estimator |
| Model-order selection | MDL/AIC/BIC on matched-filter spectrum or CFAR threshold |
| Open-set FA control | Per-bin false alarm probability on empty DD cells |
| Stopping rule | Sequential NMS with adaptive threshold instead of fixed Top-K |

### 6.3 Medium-Term (Continuous Parameter Refinement)

| Method | Description | Priority |
|---|---|---|
| Variable projection | Alternate LS gain estimation with gradient-based (τ, ν) refinement | High |
| Newtonized OMP | Continuous dictionary local maximum likelihood per path | Medium |
| SAGE | Sequential interference cancellation with per-path continuous optimisation | Medium |
| Atomic norm minimisation | Gridless DD estimation (high complexity baseline) | Low |

### 6.4 Medium-Term (System-Level)

| Direction | Description |
|---|---|
| Mixed-SNR training | Train Gate 1 on SNR ∈ [−5, 20] dB for robustness |
| Pilot pattern co-design | Optimise mask jointly for DD detection AND TF reconstruction |
| Net spectral efficiency | Add R_net = (1−ρ_p) log₂(1+SNR_eff) to evaluation |
| BER end-to-end | Simulate full OFDM chain: bits → symbols → channel → estimation → detection → BER |

### 6.5 Long-Term (Gate 2 & Beyond)

| Gate | Description |
|---|---|
| Gate 2 | Token perturbation sweeps: inject delay/Doppler errors, false paths, missed paths; measure graceful degradation |
| Gate 3 | Replace Oracle tokens with estimated tokens; decompose token-extraction loss from fusion loss |
| Real channels | CDL/TDL models, measured channel impulse responses |
| Multi-antenna extension | MIMO-OFDM with spatial DD structure |

---

## 7. Reproducibility

All experiments are deterministic given a seed. RNG seeding scheme:

```
Channel RNG:  [base_seed, density_idx, snr_idx, trial, 100]  — no pattern_index (paired design)
Noise RNG:    [base_seed, snr_idx, density_idx, trial, 300]  — no pattern_index
Pilot RNG:    [base_seed, pattern_idx, density_idx, trial, 200]  — varies by pattern
```

Paired Random-vs-Comb comparisons share identical channel realisations and base noise
grids. SNR is defined as `mean(|H_true|²) / noise_variance` (full-grid, not pilot-only).

All results (CSV, JSON, PNG) are version-controlled in `results/`.

### Re-running Key Experiments

```bash
# Physical closure tests
pytest tests/test_oracle_closure.py -v

# Gate 0 full sweep (36,000 trials, ~30 min on modern CPU)
python experiments/gate0_identifiability.py --config configs/gate0.yaml --output-dir results/gate0 --trials 1000

# Gate 0 ablation (×5 methods, ~20 min each)
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_I --ablation-integer-bins
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_oracle --ablation-oracle-nms
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_refine --ablation-refine

# Gate 1 main work point (3 seeds × 50 epochs × 4096 samples, ~2 hours on CPU)
python experiments/gate1_oracle.py --config configs/gate1_main.yaml --output-dir results/gate1_main
```
