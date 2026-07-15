# Implementation Status

Date: 2026-07-15 (v0.2.2 — physical closure tests + P0 fixes)

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

## Gate 0: DD identifiability audit

### Gate 0-A1: Known-K dominant-energy identifiability — PASS ✅

Under Known path count K=4 and fixed Top-K output, with strictly paired Random-vs-Comb comparison (shared channel + noise bank):

- Random pilots, density ≥ 1/8, SNR ≥ ~5 dB: recover ~84-88% true-path power, ~65-74% recall
- Random, density 1/4: ~87-90% power recovery
- Dominant path energy is stably captured

### Gate 0-A2: Random > Comb mechanism — PASS ✅

- Low-density Comb: μ_far = 1 (deterministic far-field DD ambiguity)
- Random: lower far-field coherence, no exact grating lobes
- Dictionary coherence and pilot AF provide structural evidence

### Gate 0-B: Unknown-K open-set detection — OPEN ❌

Not yet run. Required for:
- Path count estimation
- Open-set false alarm per DD bin
- Stopping rule / model-order selection

## Known-K FP/FN identity

Under Known-K with fixed Top-K output and no NMS early-exit:
- `num_missed = K - TP`
- `num_false_alarms = K - TP`
- Therefore `FP == FN` by construction

When NMS early-exits (peak below `relative_threshold`), `num_false_alarms < num_missed` — the weak path was never reported, so it's a miss without a corresponding false alarm.

## OSPA scope (current)

With |Ŝ| = |S| = 4 (Known-K, no early-exit):
- Cardinality penalty c^p|m-n| = 0
- OSPA reflects matched-pair localisation + capping at distance c
- Does NOT yet capture over/under-detection cardinality errors
- Will become more informative in Gate 0-B (Unknown-K)

OSPA parameters: p=2, c=1.0, normalised DD distance (÷ tolerance).

## Gate 1 status matrix

| Gate | What | Status | Key metric |
|---|---|---|---|
| 1-A | Physical model closure | REQUIRES `pytest test_oracle_closure.py` PASS | `nmse_oracle_perfect` < 1e-10 |
| 1-B | Oracle support value | READY (after 1-A passes) | `nmse_oracle_support_ls` vs initial |
| 1-C | Estimated support value | READY (after 1-A passes) | Δ_support = Est+LS − Oracle+LS |
| 1-D | Learned fusion value | NOT YET IMPLEMENTED | Requires complex-gain tokens + physical reconstruction |

## Recommended execution order

```bash
# 1. Physical closure (MUST PASS FIRST)
pytest tests/test_oracle_closure.py -v

# 2. Gate 0 smoke (check runtime oracle-perfect warning)
python experiments/gate0_identifiability.py --trials 100

# 3. Gate 0 full sweep
python experiments/gate0_identifiability.py \
  --config configs/gate0.yaml --output-dir results/gate0 --trials 1000

# 4. Gate 0 ablation (high-SNR plateau)
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_I \
  --ablation-integer-bins
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_oracle \
  --ablation-oracle-nms

# 5. Gate 1 (only after step 1 passes)
python experiments/gate1_oracle.py \
  --config configs/gate1_main.yaml --output-dir results/gate1_main
```

## NMSE decomposition (when physical closure is confirmed)

```
Δ_total = NMSE(final) - NMSE(oracle_perfect)   [≈ NMSE(final) since perfect ≈ 0]

Δ_gain   = NMSE(oracle_support_ls)  - NMSE(oracle_perfect)
Δ_support = NMSE(estimated_support_ls) - NMSE(oracle_support_ls)

If Δ_support ≫ Δ_gain: DD path localisation is the main bottleneck.
If Δ_gain ≫ Δ_support: complex-gain estimation is the main bottleneck.
```

## Gate 1-D (next revision — NOT in this code)

Required model changes:
1. `dataset.py`: update token dim 7 → 9 (add Re(α), Im(α))
2. `models.py`: explicit physical reconstruction layer H_phys[n,m] = Σ α_l exp(...)
3. `models.py`: TF residual gated fusion Ĥ = g⊙H_phys + (1-g)⊙H_TF + ΔH
