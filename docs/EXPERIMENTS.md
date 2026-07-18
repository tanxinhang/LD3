# LD3 Experiment Guide

Date: 2026-07-18

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
        for mk in ['estimated_residual', 'physics_residual']:
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
| estimated_residual | −20.0 |

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
    er = hb['estimated_residual_nmse_linear']
    er_db = 10*math.log10(er['mean'])
    gain = hb['estimated_residual_vs_ddls_paired_gain_linear']
    print(f'K={K}: DD+LS={ddls:+.1f}, EstRes={er_db:+.1f} [{er[\"ci_lower\"]:.4f}, {er[\"ci_upper\"]:.4f}], gain={gain[\"mean_diff\"]:.4f} [{gain[\"ci_lower\"]:.4f}, {gain[\"ci_upper\"]:.4f}], gate={hb[\"estimated_residual_gate_mean\"]:.3f}')
"
```

Expected output:

| K | DD+LS | Est. Residual | Gain vs DD+LS | Gate Mean |
|---|-------|---------------|----------------|-----------|
| 4 | −8.36 | −9.76 | +0.040 [0.038, 0.043] | 0.618 |
| 6 | −7.59 | −9.45 | +0.061 [0.059, 0.063] | 0.718 |
| 8 | −7.01 | −8.87 | +0.069 [0.067, 0.072] | 0.701 |

---

## Checklist

```
☐ P0: Verify provenance (0 compute)
☐ P1: Literature baselines (~2h GPU)
☐ P2: Density scan      (~15min CPU)
☐ P3: K=6 scan          (~20min CPU)
☐ P3: K=8 scan          (~20min CPU, parallel)
☐ P4: Gate 1 K=6        (~3h GPU)
☐ P4: Gate 1 K=8        (~3h GPU, parallel)

Then:
☐ git add results/ && git commit -m "Final experiments" && git push
☐ Update RESEARCH_REPORT.md
```
