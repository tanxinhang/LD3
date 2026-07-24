# Implementation Status

Date: 2026-07-24 (v0.3 — Gate 2 complete, safe fallback, baselines)

## Critical P0 checks

Before interpreting ANY Gate 1 NMSE decomposition, confirm:

```bash
pytest tests/test_oracle_closure.py -v
```

| Test | Requirement |
|---|---|
| `test_oracle_perfect_is_numerically_closed` | NMSE < 1e-10 |
| `test_oracle_support_ls_noiseless_is_closed` | NMSE < 1e-6 |
| `test_known_K_fp_equals_fn` | Under Known-K, Top-K, no early-exit: FP == FN |

If `test_oracle_perfect_is_numerically_closed` fails, every NMSE decomposition below is invalid.

The Gate 0 script now prints a runtime warning if `nmse_oracle_perfect > 1e-10`.

---

## Gate 0: DD identifiability audit

### Gate 0-A1: Known-K dominant-energy identifiability — PASS ✅

Under Known path count K=4 and fixed Top-K output, strictly paired Random-vs-Comb (shared channel + noise bank):

- Random pilots, density ≥ 1/8, SNR ≥ ~5 dB: recover ~84-88% true-path power, ~65-74% recall
- Random, density 1/4: ~87-90% power recovery
- Dominant path energy is stably captured

### Gate 0-A2: Random > Comb mechanism — PASS ✅

- Low-density Comb: μ_far = 1 (deterministic far-field DD ambiguity)
- Random: lower far-field coherence, no exact grating lobes
- Dictionary coherence and pilot AF provide structural evidence

### Gate 0-B: Unknown-K open-set detection — OPEN ❌

---

## Gate 1 status matrix

| Gate | What | Status | Key metric |
|---|---|---|---|
| 1-A | Physical model closure | PASS ✅ | `nmse_oracle_perfect` < 1e-10 |
| 1-B | Oracle support value | PASS ✅ | Oracle+LS = −24.29 dB |
| 1-C | Estimated support value | PASS ✅ | DD+LS = −8.36 dB (NMS) / −10.64 dB (OMP) |
| 1-D | Learned fusion (Oracle tokens) | PASS ✅ | Physical Residual = −19.55 dB (zero-init) |
| 1-E | Learned fusion (Estimated tokens) | PASS ✅ | +1.40 dB vs DD+LS (NMS) |
| 1-E3 | Multi-SNR unified model | PASS ✅ | SNR ≥ 0 dB consistently beats DD+LS |
| 1-E4/5 | K-sweep (K=6, 8) | PASS ✅ | +1.86 dB vs DD+LS at both K=6,8 |
| 1-F | Per-path gate | FAIL ❌ | −1.0 dB regression |

---

## Gate 2 status matrix

### Gate 2-A: Failure Boundary Audit — COMPLETE ✅

Frozen model tested under 46 corruption specifications across oracle + estimated token chains. Safety baseline: TF-only at −4.62 dB.

| Sub-gate | Condition | Status | Key finding |
|---|---|---|---|
| 2-A1 | Random false paths | PASS ✅ | 0.0% harm — null token immune |
| 2-A2 | Permutation invariance | PASS ✅ | 0.0% harm |
| 2-A3 | Small perturbation (jitter ≤0.1, phase ≤π/8) | PASS ✅ | Harm < 5% |
| 2-A4 | Phase errors (≥π/2) | FAIL ❌ | 100% harm, NMSE > TF-only |
| 2-A5 | Joint jitter (≥0.5 bins) | FAIL ❌ | ≥90% harm |
| 2-A6 | Coherent false paths (≥2) | FAIL ❌ | Harm ≥ 16% |
| 2-A7 | null_all → TF-only fallback | FAIL ❌ | +1.22 dB gap |

### Gate 2-C: Quality-Conditioned Gate — CONDITIONAL PASS ⚠️

Added 4-channel token-quality map to fusion gate input (discrepancy, confidence, uncertainty, valid_ratio).

| Sub-gate | Status | Key finding |
|---|---|---|
| 2-C1 | Gate dynamic range (6–19×) | PASS ✅ |
| 2-C2 | Phase π no longer catastrophic | CONDITIONAL PASS ⚠️ | +5.03 dB improvement but +0.69 dB vs TF-only |
| 2-C3 | Clean performance maintained | PASS ✅ | +0.47 dB matched-audit |
| 2-C4 | null_all → TF-only gap | DATA INCONSISTENT | +0.25 or +1.11 dB depending on TF-only baseline |

### Gate 2-C v2: Coupled Residual + Corruption-Aware — RESULTS ✅

| Sub-gate | Status | Key finding |
|---|---|---|
| v2-1 | Coupled residual | FAIL ❌ | Degraded null_all (−4.78→−4.61), internal H_tf co-trained with residual |
| v2-2 | Corruption-aware training | NO GAIN | Within 0.2 dB of v1 |
| v2-3 | Quality map v2 (+valid_ratio) | NO GAIN | Absorbed by CNN gate |
| v2-4 | 2×2 cross-run robustness | PASS ✅ | Same pattern across v1 + v2 | 

### Gate 2-D: Architecture & Mechanism Baselines — MOSTLY COMPLETE ✅

| Sub-gate | What | Status | Key finding |
|---|---|---|---|
| 2-D1 | Fixed blend baseline | PASS ✅ | −9.15 dB, 1 param, no training |
| 2-D2 | Hard discrepancy switch | PASS ✅ | −5.52 dB, below TF-only |
| 2-D3 | Logistic quality gate | PASS ✅ | −9.04 dB, 3 params, light training |
| 2-D4 | Hold-out pilot selector | PASS ✅ | −8.27 dB, no training |
| 2-D5 | Soft hold-out blend | PASS ✅ | −9.06 dB, T=5, 1 param |
| 2-D6 | **2×2 ablation: gate alone** | PASS ✅ | **−0.52 dB HARMFUL** (3-run consensus) |
| 2-D7 | **2×2 ablation: ΔH alone** | PASS ✅ | **+0.78~1.02 dB (78-82% of total)** |
| 2-D8 | **2×2 ablation: gate given ΔH** | PASS ✅ | **+0.13~0.48 dB (18-22% of total)** |
| 2-D9 | Token dimension: 9-dim optimal | PASS ✅ | 84-dim DD patches net negative |
| 2-D10 | Gate supervision + matched aug | PASS ✅ | Harm 99%→5%, clean −9.21 dB |
| 2-D11 | P0 optimized (normalized target + margin + clean ratio) | PASS ✅ | Clean −10.79 dB, gate 415× range |
| 2-D12 | OMP detector | PASS ✅ | +2.28 dB over NMS |
| 2-D13 | DDTokenRefiner (Conv2d patch) | PASS ✅ | +0.91 dB over MLP Refiner |
| 2-D14 | Gate supervision + Refiner | FAIL ❌ | NaN at epoch 4-6 (incompatible) |
| 2-D15a | Safe fallback formulation (c=0→H_TF) | IMPLEMENTED ✅ | Hard null-fallback: all invalid → c=0 |
| 2-D15b | MoE auxiliary losses | PASS ✅ | Clean −10.47 dB, gate gain +0.73 dB |
| 2-D15c | Safe fallback (gate2_safe): token v3+OMP | PASS ✅ | Clean −10.39 dB, fixed+ΔH −10.21 dB |
| 2-D16 | Cross-model baselines_safety comparison | COMPLETE ✅ | 4 variants compared, ΔH dominant |
| 2-D17 | **Oracle token upper-bound** | **PASS ✅** | **H_phys=−117 dB, model=−59.58 dB, token-limited confirmed** |
| 2-S3 | Oracle token + PhysicalResidual upper bound | **COMPLETE ✅** | Token quality is sole bottleneck |

---

## Oracle Token Experiment (2026-07-24) — Definitive Answer

**Q: Is the ~−10.6 dB ceiling architecture-limited or token-limited?**

**A: Token-limited, by a factor of ~49 dB.**

With oracle tokens (perfect τ, ν, α):
- H_phys-only = **−117.14 dB** (numerical precision) — physical reconstruction verified
- Full model = **−59.58 dB** — CNN gate ~0.999 limits performance
- H_phys degradation (oracle→estimated): **108.6 dB**
- Full model degradation (oracle→estimated): **49.1 dB**
- Gate+ΔH error suppression: **~60 dB**

The model's output formula `H_out = H_TF + c·(E_phys−H_TF)` means any
deviation from c=1.0 leaks TF noise. CNN gate at ~0.999 caps NMSE at ~−60 dB.

## Current Performance Ladder

Primary results from `gate1_results.json` (1024-sample test set, multi-seed where noted).
Baselines comparison from `baselines_safety.json` (717-sample test set, single-seed evaluation).

```
Method                                    NMSE (dB)    Date     Source
─────────────────────────────────────────────────────────────────────────────
Oracle H_phys-only (oracle tokens)         −117.14     7/24     gate1_results  [★]
Oracle Spatial+ΔH (oracle tokens)           −59.58     7/24     gate1_results  [★]
Oracle+LS (upper bound)                    −24.29      7/17     gate1_results
─────────────────────────────────────────────────────────────────────────────
OMP + Conv2d Refiner + model               −12.87      7/22     gate1_results  [1]
OMP + MLP Refiner + model                  −11.96      7/22     gate1_results
NMS + VP + model                           −11.31      7/21     gate1_results
OMP + LS (non-learned)                     −10.64      7/22     gate1_results
─────────────────────────────────────────────────────────────────────────────
P0 optimized (gate sup + clean ratio)      −10.79      7/19     gate1_results  [1]
Gate 2-C quality gate                      −10.69      7/18     gate1_results
MoE (aux losses, 3-seed)                  −10.47      7/23     gate1_results  [2]
Gate 2-C v2 (coupled residual)            −10.49      7/18     gate1_results
Safe fallback (v3+OMP, 1-seed)            −10.39      7/23     gate1_results  [2]
─────────────────────────────────────────────────────────────────────────────
Fixed λ + ΔH (1 param, no train)          −9.92~−10.21         baselines      [3]
DD+LS (NMS, non-learned)                   −8.36      7/17     gate1_results
H_phys-only (estimated tokens)             −8.50      7/18     baselines
Initial interpolation                      −1.41      7/17     gate1_results
TF-only                                    −4.6~−5.1           gate1_results
─────────────────────────────────────────────────────────────────────────────
```

[★] Oracle token experiment definitively answers the bottleneck question:
     H_phys with perfect tokens = −117 dB (numerical precision). The 49 dB
     oracle→estimated gap is overwhelmingly token quality, not architecture.

[1] Gate supervision (BCE) + Refiner incompatible (NaN). P0 and Refiner are
    mutually exclusive branches.
[2] Single-model evaluation; multi-seed bootstrap for MoE only.
[3] Fixed λ + ΔH range across 4 model variants. Best single value: −10.21 dB
    (Safe fallback, λ=0.80). See cross-model comparison below.

### Cross-Model Comparison (baselines_safety, 717-sample, single-seed)

```
Model                Spatial+ΔH    Fixed+ΔH     Δ(gate)   λ_best
──────────────────────────────────────────────────────────────────
MoE                  −10.65 dB     −9.92 dB     +0.73     0.75
Safe fallback        −10.57 dB     −10.21 dB    +0.36     0.80
Corruption-Aware     −10.49 dB     −10.01 dB    +0.48     0.75
Baseline             −10.45 dB     −10.17 dB    +0.28     0.80
──────────────────────────────────────────────────────────────────
```

Note: baselines_safety numbers differ from gate1_results because (a) 717 vs
1024 samples (307 used for val λ-sweep), and (b) baselines uses `return_components=True`
forward path with `fix_c` ablation control.

---

## Four-Variant Architecture Comparison

| Feature | Baseline (gate2_safety) | MoE (gate2_moe) | Corruption-Aware (gate2_corruption_aware) | Safe Fallback (gate2_safe) |
|---|---|---|---|---|
| Token version | v2 (9-dim) | v2 (9-dim) | v2 (9-dim) | **v3 (84-dim)** |
| DD detection | NMS | NMS | NMS | **OMP** |
| Epochs | 300 | 300 | 300 | 100 |
| Seeds | 3 | 3 | 3 | 1 |
| MoE aux loss | No | **Yes** (λ=0.2) | No | No |
| Token augment | No | No | **Yes** (dropout+shuffle) | No |
| Path stats | No | No | No | **Yes** (7-ch) |
| Gate kernel | 1×1 | 1×1 | 1×1 | **3×3** |
| Best NMSE (full) | −10.45 | **−10.65** | −10.49 | −10.57 |
| Best NMSE (fixed+ΔH) | −10.17 | −9.92 | −10.01 | **−10.21** |
| Gate marginal gain | +0.28 | **+0.73** | +0.48 | +0.36 |

---

## Cross-Model 2×2 Ablation Consensus (4 models)

| Contribution | Baseline | Corr-Aware | MoE | Safe (v3+OMP) | Consensus |
|---|---|---|---|---|---|
| G_spatial (gate alone, no ΔH) | −0.52 | −0.52 | −0.59 | −1.17 | **Always harmful** (−0.5 to −1.2 dB) |
| Fixed λ + ΔH gain | +1.02 | +0.82 | +0.78 | **+1.11** | **78-82% of total** |
| Spatial gate given ΔH (marginal) | +0.28 | +0.48 | **+0.73** | +0.36 | **18-22% of total** |
| Gate×ΔH substitution | Weak | Moderate | **Strong** | Strong | Better ΔH → smaller gate gain |

**Core finding**: The spatial gate does NOT directly improve reconstruction. It selectively suppresses the physics branch, creating room for the zero-init residual ΔH. Gate and residual must be trained jointly — gate alone without ΔH is consistently harmful across all training configurations and model variants.

A clear **gate×ΔH substitution relationship** emerges: variants with stronger ΔH (Safe fallback: +1.11 dB) show smaller gate marginal gain (+0.36 dB), while variants with weaker ΔH (MoE: +0.78 dB) show larger gate gain (+0.73 dB). Both converge to ~−10.6 dB full-model NMSE, suggesting an architecture-limited performance ceiling.

---

## Key Architectural Decisions (Final)

1. **Output formula**: `H_out = H_TF + c · (E_phys − H_TF)` where `E_phys = H_phys + ΔH`. Structural guarantee: c=0 → exact TF-only output. Hard rule: all tokens invalid → c=0.

2. **9-dim tokens are optimal**. 84-dim DD spectrum patches create spurious correlations and require gate supervision + augmentation to match 9-dim baseline.

3. **Zero-init residual ΔH** contributes 78-82% of total fusion gain. The spatial gate contributes 18-22%.

4. **Gate supervision (BCE)** is incompatible with DDTokenRefiner (NaN at epoch 4-6) but works well without the Refiner, enabling the P0 optimized configuration at −10.79 dB with 415× gate dynamic range.

5. **OMP detector** provides +2.28 dB over NMS for DD+LS baseline without any learning.

6. **MoE auxiliary losses** increase gate marginal gain from +0.28 to +0.73 dB but slightly reduce fixed+ΔH performance (−10.17→−9.92 dB). The aux losses reshape expert behavior rather than strengthening individual experts.

---

## Recommended execution order

```bash
# 1. Physical closure (MUST PASS FIRST)
pytest tests/test_oracle_closure.py -v

# 2. Gate 0 smoke
python experiments/gate0_identifiability.py --trials 100

# 3. Gate 0 full sweep
python experiments/gate0_identifiability.py \
  --config configs/gate0.yaml --output-dir results/gate0 --trials 1000

# 4. Gate 1 — Main work point
python experiments/gate1_oracle.py \
  --config configs/gate1_main.yaml --output-dir results/gate1_main --device cuda

# 5. Gate 2 — Train model variants
python experiments/gate1_oracle.py --config configs/gate2_safety.yaml --output-dir results/gate2_safety --models physics_residual,tf_only --device cuda
python experiments/gate1_oracle.py --config configs/gate2_moe.yaml --output-dir results/gate2_moe --models physics_residual,tf_only --device cuda
python experiments/gate1_oracle.py --config configs/gate2_corruption_aware.yaml --output-dir results/gate2_corruption_aware --models physics_residual,tf_only --device cuda
python experiments/gate1_oracle.py --config configs/gate2_safe.yaml --output-dir results/gate2_safe --models physics_residual,tf_only --device cuda

# 6. Gate 2 — Mechanism baselines (per model)
python experiments/baselines_safety.py --model-dir results/gate2_safety --output-dir results/gate2_safety_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_moe --output-dir results/gate2_moe_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_corruption_aware --output-dir results/gate2_corruption_aware_baselines --samples 1024 --device cpu
python experiments/baselines_safety.py --model-dir results/gate2_safe --output-dir results/gate2_safe_baselines --samples 1024 --device cpu

# 7. Gate 2-A — Corruption audit (smoke → full)
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 200 --device cpu --smoke-only
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 1024 --device cuda
```

---

## NMSE decomposition (when physical closure is confirmed)

```
Δ_total = NMSE(final) - NMSE(oracle_perfect)   [≈ NMSE(final) since perfect ≈ 0]

Δ_gain   = NMSE(oracle_support_ls)  - NMSE(oracle_perfect)
Δ_support = NMSE(estimated_support_ls) - NMSE(oracle_support_ls)

If Δ_support ≫ Δ_gain: DD path localisation is the main bottleneck.
If Δ_gain ≫ Δ_support: complex-gain estimation is the main bottleneck.
```

For estimated tokens at 10 dB SNR, **Δ_support ≈ 15.93 dB** while **Δ_gain ≈ 0**, confirming DD path localisation as the dominant bottleneck throughout all Gate 2 investigations.
