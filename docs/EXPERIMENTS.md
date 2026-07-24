# LD3 Experiment Guide

Date: 2026-07-24

## Config Reference

| Config | Token | SNR | Note |
|---|---|---|---|
| `gate1_main.yaml` | **Oracle** | 10 dB | True {τ,ν,α} |
| `gate1_estimated.yaml` | **Estimated** | 10 dB | DD + LS gains |
| `gate1_boundary_estimated.yaml` | Estimated | 0 dB | |
| `gate1_stress_estimated.yaml` | Estimated | 5 dB, ρ=0.0625 | |
| `gate1_multisnr.yaml` | Estimated | −5~+20 dB | |
| `gate1_K6_estimated.yaml` | Estimated, K=6 | 10 dB | K-sweep |
| `gate1_K8_estimated.yaml` | Estimated, K=8 | 10 dB | K-sweep |

---

## P0: Verify Results (0 compute)

```powershell
python -c "
import json
for name, path in [
    ('Oracle', 'results/gate1_main/gate1_results.json'),
    ('Estimated', 'results/gate1_estimated/gate1_results.json'),
    ('Boundary', 'results/gate1_boundary_estimated/gate1_results.json'),
    ('Stress', 'results/gate1_stress_estimated/gate1_results.json'),
]:
    try:
        d = json.load(open(path))
        ddls = d['non_learned_baselines']['test']['nmse_estimated_support_ls']['nmse_db']
        parts = [f'DD+LS={ddls:+.1f}']
        for mk in ['physics_residual']:
            vs = [sd[mk]['test']['nmse_db'] for sn,sd in d['seeds'].items() if mk in sd]
            if vs: parts.append(f'{mk}={sum(vs)/len(vs):+.1f}')
        print(f'{name:12s}: {\" \".join(parts)}')
    except: print(f'{name}: NOT FOUND')
"
```

Oracle (−20 dB) ≠ Estimated (−10 dB). If they look the same, you're reading the wrong config.

---

## P1: Baseline Table (1 GPU run)

```powershell
# Literature baselines + full model comparison (6 models × 3 seeds)
python experiments/gate1_oracle.py --config configs/gate1_main.yaml --output-dir results/gate1_literature --device cuda
```

⏱ ~2h on GPU. Expected output:

| Model | NMSE |
|---|---|
| tf_only | −4.62 |
| ammse | −5.8 |
| d2an | −5.7 |
| physics_cross_attention | −11.8 |
| physics_residual | −20.0 |

> `gate1_main.yaml` uses Oracle tokens. For estimated-token results, also run P0's `gate1_estimated` config.

---

## P2: Pilot Density Scan (1 CPU run)

### Config

Save as `configs/gate0_density.yaml`:

```yaml
seed: 2036
trials: 500
ofdm: {num_subcarriers: 64, num_symbols: 14, subcarrier_spacing_hz: 120000.0, carrier_frequency_hz: 28000000000.0, cp_ratio: 0.07}
channel: {num_paths: 4, max_delay_bins: 12.0, max_abs_doppler_bins: 3.0, fractional_delay: true, fractional_doppler: true, exponential_power_decay: 0.25}
sweep: {snr_db: [10], pilot_density: [0.03125, 0.0625, 0.125, 0.25, 0.5], pilot_pattern: [random]}
estimator: {oversample_delay: 2, oversample_doppler: 4, delay_tolerance_bins: 0.75, doppler_tolerance_bins: 0.5, nms_delay_radius: 2, nms_doppler_radius: 2, relative_threshold: 0.08}
```

### Run

```powershell
python experiments/gate0_identifiability.py --config configs/gate0_density.yaml --output-dir results/gate0_density --trials 500
```

⏱ ~15 min CPU (5 densities × 500 trials = 2500).

### View

```powershell
python -c "
import csv
with open('results/gate0_density/gate0_summary.csv') as f:
    for r in csv.DictReader(f):
        print(f'ρ={float(r[\"pilot_density\"]):.4f} ({int(float(r[\"num_pilots\"]))} pilots): DD+LS NMSE={float(r[\"nmse_estimated_support_ls_mean\"]):.4f}, recall={float(r[\"path_recall_mean\"]):.3f}')
"
```

---

## P3: Path Count Scan (2 CPU runs)

### Configs

Save as `configs/gate0_K6.yaml`:

```yaml
seed: 2036
trials: 500
ofdm: {num_subcarriers: 64, num_symbols: 14, subcarrier_spacing_hz: 120000.0, carrier_frequency_hz: 28000000000.0, cp_ratio: 0.07}
channel: {num_paths: 6, max_delay_bins: 12.0, max_abs_doppler_bins: 3.0, fractional_delay: true, fractional_doppler: true, exponential_power_decay: 0.25}
sweep: {snr_db: [-5, 0, 5, 10, 15, 20], pilot_density: [0.125], pilot_pattern: [random]}
estimator: {oversample_delay: 2, oversample_doppler: 4, delay_tolerance_bins: 0.75, doppler_tolerance_bins: 0.5, nms_delay_radius: 2, nms_doppler_radius: 2, relative_threshold: 0.08}
```

`configs/gate0_K8.yaml` — same but `num_paths: 8`.

### Run

```powershell
python experiments/gate0_identifiability.py --config configs/gate0_K6.yaml --output-dir results/gate0_K6 --trials 500
python experiments/gate0_identifiability.py --config configs/gate0_K8.yaml --output-dir results/gate0_K8 --trials 500
```

⏱ ~20 min each CPU. Can parallel run.

### View

```powershell
python -c "
import csv
for K,path in [(4,'results/gate0'), (6,'results/gate0_K6'), (8,'results/gate0_K8')]:
    try:
        with open(f'{path}/gate0_summary.csv') as f:
            for r in csv.DictReader(f):
                if float(r['snr_db'])==10 and r['pilot_pattern']=='random':
                    print(f'K={K} ρ=0.125: DD+LS NMSE={float(r[\"nmse_estimated_support_ls_mean\"]):.4f}, recall={float(r[\"path_recall_mean\"]):.3f}')
    except: print(f'K={K}: NOT FOUND')
"
```

---

## P4: Gate 1 K-Sweep (2 GPU runs)

### Run

```powershell
# K=6 with estimated tokens
python experiments/gate1_oracle.py --config configs/gate1_K6_estimated.yaml --output-dir results/gate1_K6 --device cuda

# K=8 with estimated tokens
python experiments/gate1_oracle.py --config configs/gate1_K8_estimated.yaml --output-dir results/gate1_K8 --device cuda
```

⏱ ~3h each GPU. Can parallel run on 2 GPUs.

### View

```powershell
python -c "
import json, math
for K,path in [(4,'results/gate1_estimated'), (6,'results/gate1_K6'), (8,'results/gate1_K8')]:
    d = json.load(open(f'{path}/gate1_results.json'))
    hb = d['hierarchical_bootstrap']
    ddls = d['non_learned_baselines']['test']['nmse_estimated_support_ls']['nmse_db']
    er = hb['physics_residual_nmse_linear']
    er_db = 10*math.log10(er['mean'])
    gain = hb['physics_residual_vs_ddls_paired_gain_linear']
    print(f'K={K}: DD+LS={ddls:+.1f}, PhysRes={er_db:+.1f} [{er[\"ci_lower\"]:.4f}, {er[\"ci_upper\"]:.4f}], gain={gain[\"mean_diff\"]:.4f} [{gain[\"ci_lower\"]:.4f}, {gain[\"ci_upper\"]:.4f}], gate={hb[\"physics_residual_gate_mean\"]:.3f}')
"
```

Expected output:

| K | DD+LS | Est. Residual | Gain vs DD+LS | Gate Mean |
|---|-------|---------------|----------------|-----------|
| 4 | −8.36 | −9.76 | +0.040 [0.038, 0.043] | 0.618 |
| 6 | −7.59 | −9.45 | +0.061 [0.059, 0.063] | 0.718 |
| 8 | −7.01 | −8.87 | +0.069 [0.067, 0.072] | 0.701 |

---

## P5: Gate 2 Model Training (4 GPU runs)

Four independently trained model variants for comparison.

### Run

```powershell
# Baseline (v2 tokens, 300 ep, quality gate)
python experiments/gate1_oracle.py --config configs/gate2_safety.yaml --output-dir results/gate2_safety --models physics_residual,tf_only --device cuda

# MoE (auxiliary losses tf_aux=0.2, phys_aux=0.2)
python experiments/gate1_oracle.py --config configs/gate2_moe.yaml --output-dir results/gate2_moe --models physics_residual,tf_only --device cuda

# Corruption-aware (token augmentation: dropout + shuffle)
python experiments/gate1_oracle.py --config configs/gate2_corruption_aware.yaml --output-dir results/gate2_corruption_aware --models physics_residual,tf_only --device cuda

# Safe fallback (v3 tokens, OMP, path_stats, gate_k=3, 100 ep)
python experiments/gate1_oracle.py --config configs/gate2_safe.yaml --output-dir results/gate2_safe --models physics_residual,tf_only --device cuda
```

⏱ 30-90 min each GPU. All 4 can parallel run.

### View

```powershell
python -c "
import json, math
for name in ['gate2_safety', 'gate2_moe', 'gate2_corruption_aware', 'gate2_safe']:
    try:
        d = json.load(open(f'results/{name}/gate1_results.json'))
        seeds = d['seeds']
        for sn, sd in seeds.items():
            if 'physics_residual' in sd:
                pr = sd['physics_residual']['test']
                gate = sd['physics_residual'].get('gate_mean_val', 'N/A')
                print(f'{name:30s} {sn}: PhysicsRes={pr[\"nmse_db\"]:+.2f}, gate={gate}')
    except Exception as e: print(f'{name}: {e}')
"
```

Expected:

| Model | NMSE | Gate Mean | Notes |
|---|---|---|---|
| gate2_safety | −10.45 | 0.92 | 3-seed baseline |
| gate2_moe | −10.47 | 0.99 | Highest clean gate |
| gate2_corruption_aware | −10.49 | — | With augmentation |
| gate2_safe | −10.39 | — | v3+OMP, 1 seed, 100 ep |

---

## P6: Mechanism Baselines (4 CPU runs, parallel)

`baselines_safety.py` evaluates the SAME frozen models with 5 mechanism baselines
+ 2×2 factorial ablation. No training needed — reads model weights and runs
forward passes.

### Run

```powershell
python experiments/baselines_safety.py --model-dir results/gate2_safety --output-dir results/gate2_safety_baselines --samples 1024 --device cpu

python experiments/baselines_safety.py --model-dir results/gate2_moe --output-dir results/gate2_moe_baselines --samples 1024 --device cpu

python experiments/baselines_safety.py --model-dir results/gate2_corruption_aware --output-dir results/gate2_corruption_aware_baselines --samples 1024 --device cpu

python experiments/baselines_safety.py --model-dir results/gate2_safe --output-dir results/gate2_safe_baselines --samples 1024 --device cpu
```

⏱ ~1-2 min each CPU. Can parallel all 4.

### Output per model

`baselines_safety.json`:
- 5 mechanism baselines: fixed blend, hard switch, logistic quality gate, hold-out pilot selector, soft hold-out blend
- 2×2 factorial ablation: `{fixed/spatial} × {nores/res}`
- Summary comparison table

### View cross-model comparison

```powershell
python -c "
import json, sys
models = {
    'Baseline': 'results/gate2_safety_baselines',
    'MoE': 'results/gate2_moe_baselines',
    'Corr-Aware': 'results/gate2_corruption_aware_baselines',
    'Safe (v3+OMP)': 'results/gate2_safe_baselines',
}
print(f'{\"Method\":<28s} {\"Baseline\":>10s} {\"MoE\":>10s} {\"Corr-Aware\":>10s} {\"Safe\":>10s}')
print('-' * 70)
methods = [
    'spatial_quality_gate_nmse_db', 'tf_only_nmse_db', 'phys_only_nmse_db',
    'fixed_blend_test_nmse_db', 'hard_switch_test_nmse_db',
    'logistic_quality_gate_nmse_db', 'holdout_pilot_selector_nmse_db',
    'soft_holdout_blend_test_nmse_db',
]
labels = ['Spatial gate (full)', 'TF-only', 'H_phys-only',
          'Fixed blend', 'Hard switch', 'Logistic gate',
          'Hold-out pilot', 'Soft holdout']
for method, label in zip(methods, labels):
    vals = []
    for name, path in models.items():
        try:
            d = json.load(open(f'{path}/baselines_safety.json'))
            if method == 'fixed_blend_test_nmse_db':
                v = d['fixed_blend']['test']['nmse_db']
            elif method == 'hard_switch_test_nmse_db':
                v = d['hard_switch']['test']['nmse_db']
            elif method == 'soft_holdout_blend_test_nmse_db':
                v = d['soft_holdout_blend']['test']['nmse_db']
            elif method in d:
                v = d[method]
            else:
                v = None
            vals.append(f'{v:+.2f}' if v else 'N/A')
        except: vals.append('N/A')
    print(f'{label:<28s} {\" \".join(f\"{v:>10s}\" for v in vals)}')
"
```

Expected 2×2 decomposition:

| Contribution | Baseline | MoE | Corr-Aware | Safe | Consensus |
|---|---|---|---|---|---|
| Spatial gate alone | −0.52 | −0.59 | −0.52 | −1.17 | **Always harmful** |
| Fixed λ + ΔH | +1.02 | +0.78 | +0.82 | +1.11 | **78-82% of total** |
| Spatial + ΔH (marginal) | +0.28 | +0.73 | +0.48 | +0.36 | **18-22% of total** |

---

## P7: Gate 2-A Corruption Audit (1 GPU run)

Tests how a frozen model degrades under controlled token corruption.

### Run

```powershell
# Smoke test (fast, 200 samples)
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 200 --device cpu --smoke-only

# Full sweep (1024 samples, 46 corruption specs × 2 chains)
python experiments/gate2_corruption.py --model-dir results/gate2_safety --output-dir results/gate2_corruption --samples 1024 --device cuda
```

⏱ Smoke: ~2 min CPU. Full: ~15 min GPU.

---

## Gate 2 Config Reference

| Config | Token | Detector | Epochs | Key Feature |
|---|---|---|---|---|
| `gate2_safety.yaml` | v2 (9-dim) | NMS | 300 | Baseline quality gate |
| `gate2_moe.yaml` | v2 (9-dim) | NMS | 300 | Aux losses (tf+phys) |
| `gate2_corruption_aware.yaml` | v2 (9-dim) | NMS | 300 | Token augmentation |
| `gate2_safe.yaml` | v3 (84-dim) | OMP | 100 | Safe fallback, path_stats |
| `gate2_canonical_oracle.yaml` | Oracle | — | 100 | Upper-bound reference |
| `gate2_omp.yaml` | v3 | OMP | 100 | OMP only (no Refiner) |
| `gate2_omp_refiner.yaml` | v3 | OMP | 100 | OMP + Conv2d Refiner |
| `gate2_p0_optimize.yaml` | v2 | NMS | 300 | Gate supervision P0 |
| `gate2_ablate.yaml` | v2 | NMS | 300 | 2×2 ablation controls |

---

## Checklist

```
Gate 0 & 1:
☐ P0: Verify provenance (0 compute)
☐ P1: Literature baselines (~2h GPU)
☐ P2: Density scan      (~15min CPU)
☐ P3: K=6 scan          (~20min CPU)
☐ P3: K=8 scan          (~20min CPU, parallel)
☐ P4: Gate 1 K=6        (~3h GPU)
☐ P4: Gate 1 K=8        (~3h GPU, parallel)

Gate 2 (4-variant model training):
☐ P5: gate2_safety (baseline)           (~45 min GPU)
☐ P5: gate2_moe (MoE aux losses)        (~45 min GPU)
☐ P5: gate2_corruption_aware (token aug) (~45 min GPU)
☐ P5: gate2_safe (v3+OMP)               (~20 min GPU)
☐ P6: baselines_safety × 4 variants     (~10 min CPU, parallel)

Gate 2 (audit, optional):
☐ P7: gate2_corruption (smoke)          (~2 min CPU)
☐ P7: gate2_corruption (full)           (~15 min GPU)

Then:
☐ git add results/ && git commit -m "Gate 2 final experiments" && git push
☐ Update docs/ (IMPLEMENTATION_STATUS.md, RESEARCH_REPORT.md, EXPERIMENTS.md, GATE2_DESIGN.md, README.md)
```
