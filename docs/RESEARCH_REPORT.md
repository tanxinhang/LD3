# LD3 Research Report — Gate 0 & Gate 1 Results

Date: 2026-07-16 (revised — P0 physical consistency fixes)

---

## 1. Simulation Configuration

### 1.1 OFDM Waveform

| Parameter | Value |
|---|---|
| Subcarriers (N) | 64 |
| OFDM Symbols (M) | 14 |
| Subcarrier spacing (Δf) | 120 kHz |
| Useful symbol duration (T_u) | 8.33 μs |
| CP ratio | 0.07 |
| CP duration (T_cp) | 0.58 μs |
| Symbol period (T_sym) | 8.92 μs |
| Bandwidth | 7.68 MHz |
| **Delay bin width** | 0.130 μs |
| **Doppler bin width** | 8.01 kHz |
| Carrier frequency | 28 GHz |
| TF grid size | 64 × 14 = 896 REs |

### 1.2 Important Physical Caveat

**The current channel model is a TF-domain sparse channel-surface abstraction.**
It synthesises H[n,m] directly in the time-frequency domain via the 2D phase law;
there is **no IFFT → CP insertion → time-domain multipath convolution → CP removal → FFT** chain.
Therefore:

- **No ISI/ICI is modelled**, even when the maximum path delay exceeds the CP duration.
- With the default `max_delay_bins=12` (~1.56 μs) and CP ~0.58 μs, physical OFDM
  would experience significant ISI. The current model does **not** capture this.
- Results should be understood as **sparse 2D parameterised channel-surface recovery**,
  not as full OFDM receiver performance.

A `RuntimeWarning` is now issued when `max_delay_bins` exceeds the equivalent CP duration.
The abstraction is valid for studying DD identifiability and physics-guided estimation,
but its limits must be explicitly acknowledged.

### 1.3 Sparse Channel Model

| Parameter | Value |
|---|---|
| Number of paths (K) | 4 (fixed, Known-K) |
| Max delay | 12 DD bins (~1.56 μs) |
| Max Doppler | ±3 DD bins (~±24.0 kHz) |
| Delay distribution | Uniform integer + Uniform(−0.45, +0.45) fractional |
| Doppler distribution | Uniform(−max, +max), fractional |
| Power profile | Exponential decay (factor 0.25), total power normalised to 1 |
| Gain distribution | Circular complex Gaussian, Rayleigh fading |
| Rician K-factor | None (pure Rayleigh) |

### 1.4 Pilot Patterns

| Pattern | Generation |
|---|---|
| **Random** | Uniform without replacement; exactly `round(ρ × N × M)` positions |
| **Comb** | Regular 2D grid; trimmed to **same exact pilot count** as Random for fair comparison |

### 1.5 Pilot Densities & SNR Sweep

| Parameter | Gate 0 Full Sweep | Gate 0 Ablation | Gate 1 Main |
|---|---|---|---|
| SNR range | −5, 0, 5, 10, 15, 20 dB | −5 to +30 dB (8 points) | 10 dB (fixed) |
| Pilot densities | 0.0625, 0.125, 0.25 | 0.125 | 0.125 |
| Pilot patterns | random, comb | random | random |
| Trials per condition | 1000 | 500 | N/A (synthetic dataset) |

**SNR definition** (unified across Gate 0 and Gate 1):
SNR = `mean(|H_true|² over full TF grid) / noise_variance`.  The noise grid is
pre-generated over the full TF grid before pilot masking, ensuring identical noise
level regardless of pilot pattern or density.

### 1.6 DD Estimator

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

### 1.7 RNG Design (Paired Comparisons)

```
Channel RNG:  [base_seed, density_idx, snr_idx, trial, 100]  — no pattern_index
Noise RNG:    [base_seed, snr_idx, density_idx, trial, 300]  — no pattern_index
Pilot RNG:    [base_seed, pattern_idx, density_idx, trial, 200]  — varies by pattern
```

Random and Comb trials at the same (density, snr, trial) share identical channel
realisations and base noise grids.  This enables valid **paired bootstrap** comparison.

---

## 2. Gate 0: DD Identifiability Audit

### 2.1 Known-K Dominant-Energy Identifiability — CONDITIONAL PASS

**Finding**: Under known K=4 and fixed Top-K output, the DD estimator stably captures
dominant channel energy from sparse pilot observations.

**Random pilots, density ≥ 0.125, SNR ≥ 5 dB** (1000 trials per condition):

| Metric | Value @ 10 dB | Value @ 20 dB | Plateau? |
|---|---|---|---|
| Path Recall | 0.725 | 0.753 | Yes (~0.74) |
| Power Recovery | 0.882 | 0.881 | Yes (~0.88) |
| Delay RMSE (bins) | 0.170 | 0.169 | Yes (~0.17) |
| Doppler RMSE (bins) | 0.104 | 0.101 | Yes (~0.10) |
| OSPA Distance | 0.544 | 0.518 | — |
| DD-Est. Support LS NMSE | 0.146 | 0.136 | Yes (~0.14) |

**Key observation**: All metrics exhibit a high-SNR performance plateau starting at
approximately 10–15 dB. Increasing SNR beyond this point yields negligible improvement,
indicating a transition from noise-limited to model/basis-mismatch-limited regime.

### 2.2 Random vs Comb: Structural Evidence for Comb Aliasing

**DD Dictionary Far-Field Mutual Coherence** (excluding NMS neighbourhood):

| Pattern | Density | μ_max | **μ_far** | μ_p95 |
|---|---|---|---|---|
| random | 0.0625 | 0.924 | **0.429** | 0.260 |
| random | 0.125 | 0.905 | **0.339** | 0.186 |
| random | 0.25 | 0.901 | **0.300** | 0.162 |
| comb | 0.0625 | **1.000** | **1.000** | 0.215 |
| comb | 0.125 | **0.988** | **0.988** | 0.200 |
| comb | 0.25 | 0.902 | **0.306** | 0.146 |

**Finding**: Low-density Comb (ρ ≤ 0.125) exhibits μ_far = 1, meaning **two distinct
DD dictionary columns produce identical observations at the pilot positions** — a
deterministic far-field ambiguity. Random pilots avoid this entirely.

**Paired Bootstrap: Random vs Comb** (1000 pairs, shared channel + noise bank).
Δ = Random − Comb.  For recall and power recovery, positive Δ favours Random.
For NMSE, negative Δ favours Random (lower NMSE is better).

| Metric | ρ=0.0625, 10 dB | ρ=0.125, 10 dB | ρ=0.25, 10 dB |
|---|---|---|---|
| Recall Δ (R−C) | **+0.172** (p<1e-4) | **+0.138** (p<1e-4) | −0.003 (p=0.63) |
| Power Recovery Δ | **+0.085** (p<1e-4) | **+0.058** (p<1e-4) | −0.004 (p=0.49) |
| NMSE Est LS Δ (R−C) | **−0.370** (Random wins) | **−0.580** (Random wins) | +0.008 (Comb wins, tiny) |

**Corrected interpretation**: Random dominates at low-to-medium density on **all three
metrics** — better DD path recall AND better TF reconstruction NMSE.  Only at high
density (ρ=0.25) does Comb achieve a statistically significant but practically tiny
advantage (~0.008 linear NMSE, ~0.1 dB).  The previous conclusion that "Comb has better
reconstruction" was based on a sign error in reading the NMSE differences.  Random
wins on both detection and reconstruction at the densities relevant for sparse pilot
operation.

### 2.3 High-SNR Plateau Ablation

**Five methods compared at SNR = 20 dB** (500 trials each, shared channel bank):

| Method | Recall | Delay RMSE | Doppler RMSE | NMSE Est LS |
|---|---|---|---|---|
| **I** (integer bins) | 0.804 | **0.045** | **0.043** | **0.053** |
| **F_oracle** (oracle discrete peaks) | **1.000** | 0.145 | 0.070 | 0.086 |
| **F_refine** (quadratic refinement) | 0.736 | 0.160 | 0.100 | 0.127 |
| **OS 4×8** (higher oversampling) | 0.716 | 0.158 | 0.101 | 0.133 |
| **F** (baseline 2×4) | 0.732 | 0.169 | 0.101 | 0.140 |

**Linear NMSE decomposition** (F baseline, 20 dB):

| Component | NMSE (linear) | Interpretation |
|---|---|---|
| `nmse_oracle_perfect` | < 1e-12 | Code closure — numerically exact |
| `nmse_oracle_support_ls` | 3.7e-4 | True continuous {τ,ν} + LS gain |
| `nmse_estimated_support_ls` | 0.140 | DD-estimated discrete {τ̂,ν̂} + LS gain |

At high SNR, LS gain estimation on known support is near-perfect (NMSE ~ 10⁻⁴).
The gap from ~0 to 0.14 is dominated by DD support estimation error (discrete grid
positions vs. true continuous parameters).

### 2.4 Gate 0 Limitations

1. **Fixed Known-K**: All experiments use K=4 paths with Top-4 output. Recall equals
   precision by construction. Unknown-K detection (Gate 0-B) is not yet implemented.
2. **No per-bin false alarm probability**: Open-set detection on empty DD cells is
   untested.
3. **Single channel class**: Only Rayleigh fading with exponential power decay.
4. **Single OFDM numerology**: One (N=64, M=14) configuration.
5. **OSPA cardinality penalty = 0**: Since |Ŝ| = K always, OSPA reflects only
   localisation quality, not over/under-detection.
6. **Not a full OFDM waveform simulation**: No IFFT/FFT, CP, time-domain convolution,
   ISI/ICI, QAM data, or BER.
7. **Not a full ISAC simulation**: No target echo, two-way propagation, RCS, clutter,
   or sensing detection task. The system is a **physics-guided DD-prior-assisted
   OFDM channel estimation prototype**.

---

## 3. Gate 1: Oracle DD Value Validation

### 3.1 Experimental Setup

| Parameter | Value |
|---|---|
| Training samples | 4,096 per seed |
| Test samples | 1,024 (fixed bank across all seeds) |
| SNR | 10 dB (fixed, both train and test) |
| Pilot pattern | Random, density 0.125 |
| Training seeds | 3 (2036, 3036, 4036) |
| Epochs | 50 |
| Batch size | 64 |
| Optimiser | AdamW, lr=1e-3, weight_decay=1e-5 |
| Hidden dimension | 48 |
| TF-only parameters | ~15k |
| Cross-attention parameters | ~35k |
| Device | CPU |
| SNR definition | Full-grid (unified with Gate 0) |

**Limitation**: Cross-SNR generalisation has not been evaluated — both training and
test are at 10 dB.  There is no validation set; models are tested after a fixed
50 epochs without early stopping on held-out data.

### 3.2 Non-Learned Baselines (Fixed Test Bank, 1024 samples)

| Method | NMSE (linear) | NMSE (dB) | Description |
|---|---|---|---|
| Nearest-neighbour interpolation | 0.723 | −1.41 | Fill missing REs with nearest pilot |
| **Oracle perfect** | **< 1e-12** | **< −120 dB** | Code-closure: reconstruct with true {τ, ν, α} |
| **Oracle support + LS** | **0.0037** | **−24.3 dB** | True continuous {τ, ν} + LS-estimated α̂ |
| DD estimated support + LS | 0.146 | **−8.4 dB** | DD-detected discrete {τ̂, ν̂} + LS-estimated α̂ |

**Key findings**:
- **Physical closure confirmed**: Oracle perfect NMSE is numerically zero.
- **Oracle continuous support adds enormous value**: Using true {τ,ν} with LS gains
  improves over interpolation by **+22.9 dB**.
- **DD estimation cost is severe**: Using DD-estimated discrete positions instead
  of true continuous positions costs **+15.9 dB loss** (going from −24.3 to −8.4 dB).

### 3.3 Learned Models (3 Seeds, Paired Hierarchical Bootstrap CI)

| Model | NMSE (dB) | 95% CI |
|---|---|---|
| TF-only (CNN baseline) | −4.60 | [−4.65, −4.55] |
| **Cross-attention (Oracle tokens)** | **−6.97** | [−7.04, −6.90] |
| **Oracle gain over TF-only** | **+2.37 dB** | paired CI (see §3.4) |

Cross-attention consistently outperforms TF-only across all 3 seeds.

### 3.4 Token-Use Audit

| Token Mode | NMSE (dB) | Δ from Oracle |
|---|---|---|
| **Oracle** (correct DD tokens) | **−6.97** | — |
| Shuffled (wrong-sample tokens) | −2.98 | **+3.99 dB** |
| Null (all tokens rejected) | −2.55 | **+4.42 dB** |
| TF-only (no DD branch) | −4.60 | — |

**Interpretation**:
- The model is **strongly dependent on correct Oracle tokens**. Shuffling or nulling
  degrades NMSE by ~4 dB, and the degraded modes underperform even the TF-only baseline
  (−4.60 dB). This proves the model genuinely relies on the DD prior, not merely on
  extra capacity.
- **However**, the model does not yet exhibit safe degradation: when given wrong or
  missing tokens, it performs worse than having no DD branch at all. Robust rejection
  of corrupted priors has not been demonstrated.

### 3.5 Error Decomposition

**Two independent chains** — the following are NOT connected serially (cross-attention
uses Oracle tokens, while DD+LS uses estimated discrete support):

**Physical estimation chain:**

```
Oracle perfect (< −120 dB)
  └─ L_gain ≈ 0 dB (at 10 dB SNR):   Oracle support+LS (−24.3 dB)
       └─ L_support ≈ +15.9 dB:        DD estimated support+LS (−8.4 dB)
```

**Learned fusion chain:**

```
Nearest interpolation (−1.41 dB)
  └─ TF-only CNN (−4.60 dB):          +3.2 dB from learned TF refinement
       └─ Oracle-token Cross-Attn (−6.97 dB):  +2.4 dB from DD prior
```

**Critical observation**: DD estimated support + LS (−8.4 dB) **outperforms**
Oracle-token Cross-Attention (−6.97 dB).  The non-learned physical parametric
reconstruction is **1.4 dB better** than the current learned fusion model, despite
the latter receiving perfect (Oracle) tokens.  This confirms that the bottleneck
is not "whether neural networks can help" but rather that **simple, correct physical
reconstruction still beats abstract Softmax cross-attention** when the latter lacks
complex-gain tokens and an explicit H_phys decoder.

### 3.6 Gate 1 Limitations

1. **7-dim tokens lack complex gain**: Current tokens are {τ, ν, power, confidence,
   σ_τ, σ_ν, relevance}. Without {Re(α), Im(α)}, the model must re-learn complex
   gains from TF observations.
2. **No explicit physical reconstruction**: The cross-attention uses abstract Value
   embeddings rather than a differentiable H_phys = Σ α_l exp(phase) layer.
3. **Fixed SNR**: Both training and test at 10 dB. Cross-SNR generalisation unknown.
4. **No validation set**: Models are tested after fixed 50 epochs. Overfitting not
   monitored.
5. **Capacity mismatch**: Cross-attention (~35k) is larger than TF-only (~15k).
   Token-use audit partially addresses this concern, but matched-param comparison
   would further strengthen conclusions.
6. **No training-time token augmentation**: Oracle tokens are always perfect during
   training. Shuffled/null tokens are test-time out-of-distribution inputs.

---

## 4. Gate Status Summary

```
Gate 0-A1  Known-K dominant-energy identifiability ... CONDITIONAL PASS
Gate 0-A2  Low-density Comb structural aliasing ....... PASS (μ_far=1, pending equal-pilot-count rerun)
Gate 0-A3  Off-grid + peak selection both bottlenecks .. PASS
Gate 0-A4  Residual-gated quadratic refinement ......... MARGINAL (−5% delay RMSE, −9% NMSE)
Gate 0-A5  Higher oversampling (4×8) ................... MARGINAL (+0.22 dB, 4× cost)
Gate 0-B   Unknown-K open-set detection ................ OPEN

Gate 1-A   Physical reconstruction closure ............. PASS (nmse_oracle_perfect < 1e-12)
Gate 1-B   Oracle continuous-support value ............. PASS (+22.9 dB over interpolation)
Gate 1-C   Estimated-support LS value .................. PASS (−8.4 dB; strongest practical baseline)
Gate 1-D0  Legacy Oracle-token Cross-Attention ......... PRELIM PASS vs TF-only (+2.4 dB, token audit passed)
                                                  FAIL vs estimated-support LS (−1.4 dB gap)
Gate 1-D1  Complex-gain physical residual fusion ....... OPEN
Gate 1-E   Learned fusion with estimated tokens ........ OPEN

Gate 2     Imperfect-prior safe degradation ............ OPEN
Gate 3     Full OFDM-ISAC waveform validation ........... OPEN
```

---

## 5. Current Limitations (Cross-Cutting)

### 5.1 Methodological

1. **Fixed Known-K**: All experiments fix K=4. Real systems must estimate path count.
2. **Single channel class**: Only Rayleigh with exponential power decay tested.
3. **No mobility sweep**: Doppler spread fixed at ±3 bins (~±24 kHz). Higher mobility
   unexplored.
4. **Single OFDM numerology**: One (N=64, M=14) configuration.
5. **Not a full OFDM waveform**: No IFFT/CP/ISI/ICI/QAM/BER. Channel model is a TF
   surface abstraction.
6. **Not a full ISAC chain**: No sensing target, RCS, or detection task. System is
   a channel-estimation prototype.

### 5.2 Technical

7. **Token lacks complex gain**: 7-dim → must add {Re(α), Im(α)} for Gate 1-D1.
8. **No explicit physical reconstruction**: Softmax cross-attention with abstract
   Values cannot express complex multi-path superposition.
9. **No validation set in Gate 1**: Training runs fixed epochs without best-model
   selection.
10. **No training-time token augmentation**: Oracle tokens always perfect; model
    not trained to reject corrupted priors.

### 5.3 Computational

11. **CPU-only training**: Gate 1 trained on CPU. GPU would enable larger models.
12. **Fixed DD grid**: Oversampling 2×4 throughout. Adaptive/multi-resolution grids
    unexplored.

---

## 6. Future Work

### 6.1 Immediate (Gate 1-D1 — Physics-Repaired Fusion)

**Goal**: Eclipse the DD estimated support + LS baseline (−8.4 dB) with a learned model.

| Step | Description |
|---|---|
| Complex-gain tokens | Extend from 7-dim to 9-dim: add Re(α), Im(α) |
| Physical reconstruction | Differentiable H_phys = Σ α_l exp(−j2πnτ_l/N + j2πmν_l/M) |
| TF residual gated fusion | Ĥ = g ⊙ H_phys + (1−g) ⊙ H_TF + ΔH |
| Training-time token augmentation | Add token dropout, shuffle, perturbation for robustness |
| Validation set | Split train/val/test; select best model on val NMSE |
| Matched-parameter comparison | Equalise TF-only and cross-attention parameter counts |

**Gate 1-D1 pass criteria** (at main work point):
- NMSE < estimated support + LS (−8.4 dB)
- Oracle gain over TF-only ≥ +0.5 dB (paired CI)
- Null-token NMSE within 0.3 dB of TF-only (safe degradation)

### 6.2 Short-Term (Continuous Parameter Refinement)

| Method | Description |
|---|---|
| Variable projection | Alternate LS gain + gradient-based (τ,ν) refinement |
| Newtonized OMP | Continuous-dictionary local ML per path |
| SAGE | Sequential interference cancellation with per-path optimisation |

### 6.3 Medium-Term

| Direction | Description |
|---|---|
| Gate 0-B | Unknown-K: model-order selection, CFAR threshold, open-set FA control |
| Mixed-SNR training | Train on SNR ∈ [−5, 20] dB for robust generalisation |
| Net spectral efficiency | R_net = (1−ρ_p) log₂(1+SNR_eff) in evaluation |
| Full waveform chain | IFFT → CP → time-domain convolution → FFT → estimation → BER |

### 6.4 Long-Term

| Gate | Description |
|---|---|
| Gate 2 | Token perturbation sweeps: inject errors, measure graceful degradation |
| Gate 3 | Replace Oracle tokens with estimated tokens; decompose extraction vs fusion loss |
| Real channels | CDL/TDL models, measured impulse responses |
| MIMO extension | Multi-antenna OFDM with spatial DD structure |
| Full ISAC | Add sensing path, target detection, joint communication–sensing metrics |

---

## 7. Reproducibility

### 7.1 Key Commands

```bash
# Physical closure tests
pytest tests/test_oracle_closure.py -v

# Gate 0 full sweep (36,000 trials)
python experiments/gate0_identifiability.py --config configs/gate0.yaml --output-dir results/gate0 --trials 1000

# Gate 0 ablation (×5 methods)
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_I --ablation-integer-bins
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_oracle --ablation-oracle-nms
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_refine --ablation-refine
python experiments/gate0_identifiability.py --config configs/gate0_oversampling.yaml --output-dir results/gate0_os_4x8

# Gate 1 main work point (3 seeds)
python experiments/gate1_oracle.py --config configs/gate1_main.yaml --output-dir results/gate1_main
```

### 7.2 RNG Seeding

All experiments are deterministic given a base seed.  See §1.7 for the three-RNG
paired-design scheme.  Gate 1 multi-seed uses base_seed + seed_idx × 1000 for
training data, with a fixed test bank at base_seed + 10000.
