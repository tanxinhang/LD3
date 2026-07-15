# Implementation Status

Date: 2026-07-15 (v0.2.1 — paired design fix + diagnostic baselines)

## Gate 0: DD identifiability audit

### Gate 0-A: Known-K identifiability — CONDITIONAL PASS ✅

In 1000 Monte Carlo trials with fixed K=4 paths and Top-(K=4) output,
using **strictly paired** Random-vs-Comb comparison (shared channel + noise bank):

| Condition | Power Recovery | Path Recall | Notes |
|---|---|---|---|
| Random, ρ=1/8, SNR≥5 dB | ~84-88% | ~65-74% | Main working region |
| Random, ρ=1/4, SNR≥5 dB | ~87-90% | ~70-80% | Higher density |
| Random vs Comb Δ | paired bootstrap | paired bootstrap | Shared channel + noise |

**Key design properties (v0.2.1):**
- Channel RNG: `[seed, density_idx, snr_idx, trial, 100]` — no pattern_index
- Noise RNG: `[seed, snr_idx, density_idx, trial, 300]` — no pattern_index
- Pilot RNG: `[seed, pattern_idx, density_idx, trial, 200]` — varies by pattern
- SNR defined over full-grid channel power (not pilot-only)
- Paired bootstrap is valid: each trial index pairs identical channel/noise under different masks

### Gate 0-B: Unknown-K detection — NOT YET RUN ❌

Required before claiming DD path detection is "fully available":
- Variable path count estimation
- Open-set false alarm probability per DD bin
- Stopping rule / model-order selection

## Gate 1: Oracle DD value validation

### Gate 1-A: Physical model closure — READY

`nmse_oracle_perfect`: reconstruct using true {τ, ν, α}. Should approach numerical
precision (~ -100 dB). If > -80 dB, check sign conventions and normalisation.

### Gate 1-B: Oracle support value — READY

`nmse_oracle_support_ls`: true {τ, ν} + LS-estimated α̂.
Δ_gain = NMSE(oracle_support_ls) - NMSE(oracle_perfect).

### Gate 1-C: Estimated support value — READY

`nmse_estimated_support_ls`: DD-estimated {τ̂, ν̂} + LS-estimated α̂.
Δ_support = NMSE(estimated_support_ls) - NMSE(oracle_support_ls).

### Gate 1-D: Learned fusion value — NOT YET REPAIRED ❌

Current model uses 7-dim tokens (delay, Doppler, power, confidence, σ_τ, σ_ν, relevance)
**without complex gain**. Without (Re α, Im α), the cross-attention cannot express
multi-path complex superposition.

**Required for next revision:**
1. Add complex-gain tokens: `[τ, ν, Re(α), Im(α), confidence, σ_τ, σ_ν, relevance]`
2. Implement explicit physical reconstruction layer:
   H_phys[n,m] = Σ_l α_l e^{-j2π n τ_l / N} e^{j2π m ν_l / M}
3. TF residual gated fusion:
   Ĥ = g ⊙ H_phys + (1-g) ⊙ H_TF + ΔH

### Gate 1 status matrix

| Gate | What | Status | Key metric |
|---|---|---|---|
| 1-A | Physical model closure | READY | `nmse_oracle_perfect` < -80 dB |
| 1-B | Oracle support value | READY | `nmse_oracle_support_ls` vs initial |
| 1-C | Estimated support value | READY | `nmse_estimated_support_ls` vs oracle |
| 1-D | Learned fusion value | NOT YET REPAIRED | Requires complex-gain tokens + physical reconstruction |

## New metrics (v0.2.1)

| Metric | Description |
|---|---|
| `penalized_delay_rmse_bins` | RMSE with **tolerance** as miss penalty (not full range) |
| `penalized_doppler_rmse_bins` | Same for Doppler |
| `ospa_distance` | OSPA (p=2, c=1.0, normalised DD, Hungarian assignment) |
| `num_missed` / `num_false_alarms` | Decomposed detection errors |
| `false_alarm_rate` | False alarms per estimated path |
| `mu_max` / `mu_far` / `mu_p95` / `mu_p99` | DD dictionary coherence (far-field excludes NMS neighbourhood) |
| `pslr_db` / `islr_db` | Pilot AF metrics (far-field PSLR) |
| `max_far_sidelobe_delay_bin` / `_doppler_bin` | Strongest grating lobe location |
| `nmse_oracle_perfect` | Code-closure test (~ numerical precision) |
| `nmse_oracle_support_ls` | True DD + estimated gains |
| `nmse_estimated_support_ls` | Estimated DD + estimated gains |
| CI columns | `_se`, `_ci95_lower`, `_ci95_upper`, `_n_eff` |
| Paired bootstrap | Valid paired design (shared channel + noise bank) |
| Hierarchical bootstrap | Seed-level + sample-level CI (Gate 1 multi-seed) |

## Gate 0 ablation: High-SNR plateau diagnosis

| Case | Delay/Doppler | NMS | CLI flag |
|---|---|---|---|
| I | Integer bins | Standard | `--ablation-integer-bins` |
| F | Fractional bins | Standard | (default) |
| F+Oracle | Fractional bins | Oracle | `--ablation-oracle-nms` |

Config: `configs/gate0_ablation.yaml`

## Recommended Gate 1 work points

| Work point | Config | Pilot | Density | SNR |
|---|---|---|---|---|
| Main | `configs/gate1_main.yaml` | Random | 0.125 | 10 dB |
| Boundary | `configs/gate1_boundary.yaml` | Random | 0.125 | 0 dB |
| Stress | `configs/gate1_stress.yaml` | Random | 0.0625 | 5 dB |

## Next steps (priority order)

1. Run Gate 1 at main work point → inspect Δ_gain and Δ_support
2. Correlate `power_recovery` with `nmse_estimated_support_ls` → does Gate 0 predict Gate 1?
3. If Δ_support is large but Δ_gain is small: focus on DD estimation quality
4. If Δ_gain is large: investigate ridge regularisation, path count mismatch
5. Only after 1-4: add complex-gain tokens + physical reconstruction (Gate 1-D)
