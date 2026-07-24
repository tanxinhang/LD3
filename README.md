# LD3 — Physics-Guided TF–DD OFDM-ISAC Validation

This repository implements validation gates for a robust TF–DD cross-domain
OFDM-ISAC channel estimator: from DD identifiability (Gate 0) through Oracle
fusion (Gate 1) to safe degradation under corrupted priors (Gate 2).

---

## Implemented scope

### Gate 0 — DD prior identifiability

The Gate 0 pipeline answers whether sparse DD paths are recoverable from noisy,
sparsely sampled OFDM pilot observations.

Implemented components:

- fractional delay and fractional Doppler sparse channel generator;
- random and comb two-dimensional pilot patterns;
- **mask-aware DD matched filter**, avoiding interpolation before DD analysis;
- Top-K peak selection with non-maximum suppression;
- **Oracle NMS** (ablation: ideal peak selection to isolate NMS bottleneck);
- Hungarian path matching;
- path recall, precision, delay/Doppler RMSE;
- **penalised RMSE** (miss penalty for undetected paths);
- **OSPA distance** (joint localisation + cardinality error);
- true-path power recovery ratio;
- **false alarm count and rate**;
- **pilot ambiguity function** analysis (grating lobes, PSLR, ISLR);
- **DD dictionary mutual coherence** analysis;
- **95% confidence intervals** on all summary metrics;
- **paired bootstrap** Random-vs-Comb comparison;
- **high-SNR plateau ablation** (integer bins, oracle NMS);
- SNR, pilot-density, and pilot-pattern sweeps;
- CSV, JSON, and figure outputs.

**Gate 0 conclusion (conditional):** Under known K=4, Random pilots with density
≥ 1/8 and SNR ≥ ~5 dB recover ~84-88% of true path power. Gate 0-B (Unknown-K
detection) remains open.

---

### Gate 1 — Oracle DD value test

Compares TF-only baselines, physics-guided TF–DD cross-attention, and a
Physical Residual Estimator with zero-init residual correction. Three work points:

| Work point | Config | Pilot | Density | SNR | Purpose |
|---|---|---|---|---|---|
| Main | `configs/gate1_main.yaml` | Random | 0.125 | 10 dB | Primary value judgment |
| Boundary | `configs/gate1_boundary.yaml` | Random | 0.125 | 0 dB | Low-SNR robustness |
| Stress | `configs/gate1_stress.yaml` | Random | 0.0625 | 5 dB | Sparse-pilot robustness |

Key results at 10 dB, ρ=0.125, estimated tokens:

| Method | NMSE |
|---|---|
| Oracle+LS (upper bound) | −24.29 dB |
| Physical Residual (Oracle tokens) | −19.55 dB |
| DD+LS (NMS, non-learned) | −8.36 dB |
| Physical Residual (estimated) | −9.76 dB (+1.40 dB vs DD+LS) |
| TF-only | −4.62 dB |

---

### Gate 2 — Safe degradation under corrupted priors

Gate 2 addresses the core safety question: _if DD detection produces wrong
tokens, does the fusion model degrade gracefully or catastrophically?_

#### Architecture

```
H_out = H_TF + c · (E_phys − H_TF)     ← structural safe fallback

where:
  H_phys  = explicit physics reconstruction from DD path tokens
  E_phys  = H_phys + ΔH                ← zero-init residual correction
  c(x,y)  = spatial confidence gate ∈ [0,1]
            c=0 ⇒ Ĥ = H_TF  (exact, guaranteed)
            c=1 ⇒ Ĥ = E_phys (full physics expert)
  all tokens invalid ⇒ c=0 (hard null-fallback)
```

#### Four model variants

| Variant | Config | Key difference |
|---|---|---|
| **Baseline** | `gate2_safety.yaml` | v2 tokens, 300 ep, gate_k=1 |
| **MoE** | `gate2_moe.yaml` | Auxiliary expert losses (tf_aux + phys_aux) |
| **Corruption-Aware** | `gate2_corruption_aware.yaml` | Token augmentation (dropout + shuffle) |
| **Safe Fallback** | `gate2_safe.yaml` | v3+OMP tokens, path_stats, gate_k=3, 100 ep |

#### Cross-model 2×2 ablation consensus

| Component | Baseline | Corr-Aware | MoE | Consensus |
|---|---|---|---|---|
| **Spatial gate alone** (no ΔH) | **−0.52** | **−0.52** | **−0.59** | **Always harmful** |
| Fixed λ + ΔH gain | +1.02 | +0.82 | +0.78 | **78-82% of total** |
| Spatial gate + ΔH (marginal) | +0.28 | +0.48 | +0.73 | **18-22% of total** |

Gate and ΔH are complementary — gate suppresses physics to create correction
room for ΔH; they must be trained jointly. Gate without ΔH is consistently
worse than a single scalar blend weight.

#### Mechanism baselines (all models, frozen H_phys + H_tf)

| Method | NMSE range | Params | Trained |
|---|---|---|---|
| H_phys-only | −8.50 | 0 | No |
| TF-only (standalone) | −4.7~−5.1 | CNN | Yes |
| Fixed blend (λ=0.75~0.80) | −9.1~−9.2 | 1 | No |
| Hard switch | −5.9~−6.6 | 1 | No |
| Logistic quality gate | −9.0~−9.1 | 3 | Light |
| Hold-out pilot select | −7.5~−8.3 | 0 | No |
| Soft hold-out blend | −8.8~−9.1 | 1 | No |
| Fixed λ + ΔH | −9.9~−10.2 | CNN | No |
| **Spatial gate + ΔH** | **−10.5~−10.7** | CNN | Yes |

---

## Repository structure

```text
configs/
  gate0.yaml                   # Gate 0 main sweep
  gate0_K6.yaml, gate0_K8.yaml # Path-count scans
  gate0_ablation.yaml          # High-SNR plateau ablation
  gate0_density.yaml           # Pilot density scan
  gate0_oversampling.yaml      # DD oversampling scan
  gate1.yaml                   # Default Gate 1
  gate1_main.yaml              # Main work point (Oracle)
  gate1_estimated.yaml         # Estimated tokens, 10 dB
  gate1_boundary.yaml          # Boundary (0 dB)
  gate1_stress.yaml            # Stress (5 dB, ρ=0.0625)
  gate1_multisnr.yaml          # Multi-SNR (−5~+20 dB)
  gate1_K6_estimated.yaml      # K=6 estimated
  gate1_K8_estimated.yaml      # K=8 estimated
  gate2_safety.yaml            # Gate 2 baseline model
  gate2_moe.yaml               # MoE with auxiliary losses
  gate2_corruption_aware.yaml  # Corruption-aware training
  gate2_safe.yaml              # Safe fallback (v3+OMP)
  gate2_canonical_oracle.yaml  # Oracle token reference
  gate2_ablate.yaml            # 2×2 ablation controls
  gate2_omp.yaml               # OMP detector config
  gate2_omp_refiner.yaml       # OMP + DDTokenRefiner
  gate2_p0_optimize.yaml       # P0 gate-supervision optimized
  ...                          # Additional ablation configs
experiments/
  gate0_identifiability.py     # Gate 0 DD identifiability sweep
  gate1_oracle.py              # Gate 1/2 model training
  gate2_corruption.py          # Gate 2-A corruption audit
  baselines_safety.py          # Mechanism-gradient safety baselines
src/ld3/
  channel.py                   # Sparse channel generation & TF synthesis
  config.py                    # OFDMConfig, ChannelConfig
  dataset.py                   # SyntheticOFDMISACDataset
  dd_estimation.py             # DD matched filter, NMS, OMP, metrics, OSPA, CI
  interpolation.py             # Nearest-smooth TF interpolation
  metrics.py                   # NMSE (numpy + torch)
  models.py                    # TFEncoder, TFOnlyEstimator, PhysicalResidualEstimator, DDTokenRefiner
  oracle.py                    # Oracle tokens, Oracle+LS, perturbation
  pilots.py                    # Pilot mask generation & observation
  baselines.py                 # Safety baselines (fixed blend, hard switch, logistic gate, hold-out)
tests/
  test_channel.py
  test_dd_estimation.py
  test_models.py
  test_oracle_closure.py       # P0: physical model closure
docs/
  IMPLEMENTATION_STATUS.md     # Full implementation status matrix
  EXPERIMENTS.md               # Experiment guide and config reference
  GATE2_DESIGN.md              # Gate 2 design document & empirical results
  RESEARCH_REPORT.md           # Final research report (Gate 0 + 1 + 2)
```

---

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1

pip install -e .[dev]
```

## Run tests

```bash
pytest -q
```

---

## Gate 0

### Smoke run

```bash
python experiments/gate0_identifiability.py --trials 3
```

### Full sweep

```bash
python experiments/gate0_identifiability.py \
  --config configs/gate0.yaml \
  --output-dir results/gate0
```

Outputs: `gate0_trials.csv`, `gate0_summary.csv`, `gate0_paired_bootstrap.csv`,
`gate0_coherence.csv`, `gate0_ambiguity.csv`, `ambiguity_*.png`,
`gate0_power_recovery_*.png`, `gate0_recall_*.png`, `gate0_ospa_*.png`,
`manifest.json`.

### Ablation: high-SNR plateau

```bash
# Case I: integer bins (no off-grid leakage)
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml \
  --output-dir results/gate0_ablation_I \
  --ablation-integer-bins

# Case F: fractional bins
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml \
  --output-dir results/gate0_ablation_F

# Case F+Oracle: oracle NMS
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml \
  --output-dir results/gate0_ablation_F_oracle \
  --ablation-oracle-nms
```

---

## Gate 1

### Smoke run

```bash
python experiments/gate1_oracle.py \
  --epochs 1 --train-size 64 --test-size 32
```

### Main work point (Oracle tokens, 3 seeds)

```bash
python experiments/gate1_oracle.py \
  --config configs/gate1_main.yaml \
  --output-dir results/gate1_main --device cuda
```

### Estimated tokens

```bash
python experiments/gate1_oracle.py \
  --config configs/gate1_estimated.yaml \
  --output-dir results/gate1_estimated --device cuda
```

### Key diagnostics

```
oracle_cross_attention_gain_db = TF-only NMSE(dB) - physics-cross NMSE(dB)
oracle_vs_shuffled_gain_db     = shuffled NMSE(dB) - oracle NMSE(dB)
oracle_vs_null_gain_db         = null NMSE(dB) - oracle NMSE(dB)
```

---

## Gate 2

### Step 1: Train model variants

```bash
# Baseline
python experiments/gate1_oracle.py --config configs/gate2_safety.yaml --output-dir results/gate2_safety --models physics_residual,tf_only --device cuda

# MoE
python experiments/gate1_oracle.py --config configs/gate2_moe.yaml --output-dir results/gate2_moe --models physics_residual,tf_only --device cuda

# Corruption-aware
python experiments/gate1_oracle.py --config configs/gate2_corruption_aware.yaml --output-dir results/gate2_corruption_aware --models physics_residual,tf_only --device cuda

# Safe fallback
python experiments/gate1_oracle.py --config configs/gate2_safe.yaml --output-dir results/gate2_safe --models physics_residual,tf_only --device cuda
```

### Step 2: Mechanism baselines

```bash
python experiments/baselines_safety.py --model-dir results/gate2_safety --output-dir results/gate2_safety_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_moe --output-dir results/gate2_moe_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_corruption_aware --output-dir results/gate2_corruption_aware_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_safe --output-dir results/gate2_safe_baselines --samples 1024 --device cpu
```

### Step 3: Corruption audit (optional)

```bash
# Smoke test
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 200 --device cpu --smoke-only

# Full sweep
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 1024 --device cuda
```

---

## Research gates

The recommended progression:

1. **Gate 0-A:** establish DD-identifiable operating region (known K) ✅
2. **Gate 0-B:** unknown-K detection and open-set false alarm control ❌
3. **Gate 1:** prove Oracle DD information improves over TF-only processing ✅
4. **Gate 2:** mechanism audit ✅ (canonical safe-pipeline validation open)
5. **Gate 3:** full OFDM-ISAC waveform ❌

---

## Gate status summary

```
Gate 0-A: Known-K DD identifiability ............... PASS
Gate 0-B: Unknown-K detection ...................... OPEN

Gate 1-A: Physical model closure ................... PASS (nmse_perfect = 0)
Gate 1-B: Oracle support value ..................... PASS (+22.9 dB)
Gate 1-C: DD estimated support value ............... PASS (−8.4 dB NMS, −10.6 dB OMP)
Gate 1-D: Oracle Physical Residual (zero-init) ..... PASS (−19.6 dB)
Gate 1-E: Estimated Physical Residual .............. PASS (+1.40 dB vs DD+LS)
Gate 1-F: Per-path gate ............................ FAIL (−1.0 dB regression)

Gate 2-A: Failure boundary audit ................... COMPLETE
Gate 2-C: Quality-conditioned gate ................. CONDITIONAL PASS
Gate 2-D1: Fixed blend baseline .................... PASS (−9.15 dB, 1 param)
Gate 2-D6: 2×2 ablation (3-run consensus) .......... PASS (ΔH 78-82%, gate 18-22%)
Gate 2-D9: Token dimension: 9-dim optimal .......... PASS
Gate 2-D12: OMP detector ........................... PASS (+2.28 dB vs NMS)
Gate 2-D13: DDTokenRefiner (Conv2d) ................ PASS (+0.91 dB over MLP)
Gate 2-D14: Gate supervision + Refiner ............. FAIL (NaN, incompatible)
Gate 2-D15a: Safe fallback formulation ............. IMPLEMENTED (c=0 → H_TF exact)
Gate 2-D15b: MoE auxiliary losses .................. PASS (−10.47 dB)
Gate 2-D15c: Safe fallback (gate2_safe) ............ PASS (−10.39 dB)
Gate 2-D16: Cross-model baselines comparison ....... COMPLETE (4 variants)
Gate 2-D17: **Oracle token upper-bound** ........... **PASS (H_phys=−117 dB → path-parameter quality dominates)**

Gate 3: Full OFDM-ISAC waveform .................... OPEN
```
