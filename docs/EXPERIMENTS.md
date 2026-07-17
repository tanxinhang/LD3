# LD3 Experiment Guide

Date: 2026-07-18

## Quick Reference: Config → Token Source

| Config | Token Source | SNR |
|---|---|---|
| `configs/gate1_main.yaml` | **Oracle** {τ,ν,α} | 10 dB |
| `configs/gate1_estimated.yaml` | **Estimated** (DD + LS) | 10 dB |
| `configs/gate1_boundary_estimated.yaml` | Estimated | 0 dB |
| `configs/gate1_stress_estimated.yaml` | Estimated | 5 dB, ρ=0.0625 |
| `configs/gate1_multisnr.yaml` | Estimated | −5 to +20 dB |

---

## P0: Clear Result Provenance (Verify Existing Data)

This costs zero compute — just verify which NMSE numbers come from which config.

```powershell
python -c "
import json
for name, path in [
    ('Oracle tokens', 'results/gate1_main/gate1_results.json'),
    ('Estimated tokens', 'results/gate1_estimated/gate1_results.json'),
    ('Boundary estimated', 'results/gate1_boundary_estimated/gate1_results.json'),
    ('Stress estimated', 'results/gate1_stress_estimated/gate1_results.json'),
    ('Multi-SNR', 'results/gate1_multisnr/gate1_results.json'),
]:
    try:
        with open(path) as f: d = json.load(f)
        seeds = d['seeds']
        for model_key in ['estimated_residual', 'physics_residual']:
            vals = []
            for sn, sd in seeds.items():
                if model_key in sd:
                    vals.append(sd[model_key]['test']['nmse_db'])
            if vals:
                mean_nmse = sum(vals)/len(vals)
                print(f'{name:25s} {model_key:25s}: {mean_nmse:+.2f} dB')
        # DD+LS baseline
        ddls_db = d['non_learned_baselines']['test']['nmse_estimated_support_ls']['nmse_db']
        print(f'{name:25s} DD+LS baseline            : {ddls_db:+.2f} dB')
    except FileNotFoundError:
        print(f'{name:25s}: NOT FOUND — need to run first')
    print()
"
```

Expected output should clearly distinguish Oracle (−19~−21 dB) from Estimated (−9~−10 dB).

---

## P1: Complete the Baseline Table

### P1a: DD+VP+LS (Get VP-improved DD baseline)

First check if VP results exist:

```powershell
python -c "
import csv
try:
    with open('results/gate0_ablation_F_vp/gate0_summary.csv') as f:
        for r in csv.DictReader(f):
            snr = float(r['snr_db'])
            nmse = float(r['nmse_estimated_support_ls_mean'])
            rec = float(r['path_recall_mean'])
            delay = float(r['delay_rmse_bins_mean'])
            print(f'VP @ SNR={snr:3.0f}: DD+VP+LS NMSE={nmse:.4f}, recall={rec:.3f}, delay_rmse={delay:.4f}')
except FileNotFoundError:
    print('NOT FOUND — run:')
    print(r'python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_vp --ablation-vp')
"
```

If not found, run:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_ablation_F_vp --ablation-vp
```

### P1b: Literature Baselines (Gate 1 with A-MMSE + D2AN)

```powershell
python experiments/gate1_oracle.py --config configs/gate1_main.yaml --output-dir results/gate1_literature --device cuda
```

Expected output:
```
ammse: test NMSE ≈ −5.8 dB
d2an: test NMSE ≈ −5.7 dB
physics_cross_attention: test NMSE ≈ −11.8 dB
physics_residual: test NMSE ≈ −20.0 dB
estimated_residual: test NMSE ≈ −20.0 dB
```

**Note:** Gate1_main uses Oracle tokens. For true estimated-token comparison, also run:
```powershell
python experiments/gate1_oracle.py --config configs/gate1_estimated.yaml --output-dir results/gate1_estimated --device cuda
```

---

## P2: Pilot Density Scan

### Step 1: Create config

Create `configs/gate0_density.yaml`:

```yaml
seed: 2036
trials: 500
ofdm:
  num_subcarriers: 64
  num_symbols: 14
  subcarrier_spacing_hz: 120000.0
  carrier_frequency_hz: 28000000000.0
  cp_ratio: 0.07
channel:
  num_paths: 4
  max_delay_bins: 12.0
  max_abs_doppler_bins: 3.0
  fractional_delay: true
  fractional_doppler: true
  exponential_power_decay: 0.25
sweep:
  snr_db: [10]
  pilot_density: [0.03125, 0.0625, 0.125, 0.25, 0.5]
  pilot_pattern: [random]
estimator:
  oversample_delay: 2
  oversample_doppler: 4
  delay_tolerance_bins: 0.75
  doppler_tolerance_bins: 0.5
  nms_delay_radius: 2
  nms_doppler_radius: 2
  relative_threshold: 0.08
```

### Step 2: Run

```powershell
python experiments/gate0_identifiability.py --config configs/gate0_density.yaml --output-dir results/gate0_density --trials 500
```

### Step 3: View results

```powershell
python -c "
import csv
with open('results/gate0_density/gate0_summary.csv') as f:
    for r in csv.DictReader(f):
        density = float(r['pilot_density'])
        nmse = float(r['nmse_estimated_support_ls_mean'])
        rec = float(r['path_recall_mean'])
        power = float(r['power_recovery_mean'])
        npilots = float(r['num_pilots'])
        print(f'ρ={density:.4f} ({npilots:.0f} pilots): DD+LS NMSE={nmse:.4f}, recall={rec:.3f}, power_rec={power:.3f}')
"
```

Expected: NMSE should degrade gracefully as pilot density decreases.

---

## P3: Path Count Scan (K = 2, 4, 6, 8)

### Step 1: Create configs

Create `configs/gate0_K2.yaml` through `configs/gate0_K8.yaml` by copying `configs/gate0_ablation.yaml` and changing `channel.num_paths`.

Or use a single config and override via CLI (if supported):

```powershell
# K=2
python experiments/gate0_identifiability.py --config configs/gate0_ablation.yaml --output-dir results/gate0_K2 --trials 500

# K=6 — need to manually edit config or create new config
# cp configs/gate0_ablation.yaml configs/gate0_K6.yaml
# edit: num_paths: 6, max_paths: 12 (in dataset for Gate 1)

# K=8
# edit: num_paths: 8, max_paths: 16
```

For now, create configs manually:

```yaml
# configs/gate0_K6.yaml — key changes from gate0_ablation.yaml
channel:
  num_paths: 6
sweep:
  pilot_density: [0.125]
  pilot_pattern: [random]
```

```yaml
# configs/gate0_K8.yaml
channel:
  num_paths: 8
sweep:
  pilot_density: [0.125]
  pilot_pattern: [random]
```

### Step 2: Run

```powershell
python experiments/gate0_identifiability.py --config configs/gate0_K6.yaml --output-dir results/gate0_K6 --trials 500
python experiments/gate0_identifiability.py --config configs/gate0_K8.yaml --output-dir results/gate0_K8 --trials 500
```

### Step 3: Compare

```powershell
python -c "
import csv
for K, path in [(2,'results/gate0_K2'), (4,'results/gate0_ablation_F'), (6,'results/gate0_K6'), (8,'results/gate0_K8')]:
    try:
        with open(f'{path}/gate0_summary.csv') as f:
            for r in csv.DictReader(f):
                if float(r['snr_db']) == 10:
                    print(f'K={K}: DD+LS NMSE={float(r[\"nmse_estimated_support_ls_mean\"]):.4f}, recall={float(r[\"path_recall_mean\"]):.3f}')
    except FileNotFoundError:
        print(f'K={K}: NOT FOUND')
"
```

---

## P4: TF Parametric Reconstruction (Decisive Baseline)

This experiment tests whether the advantage comes from DD-domain specifically, or from explicit path parameterization in general. Estimate path parameters from TF grid interpolation instead of DD detection, then feed to the same Physical Residual network.

(This requires new code — to be implemented.)

---

## Complete Run Checklist

```
☐ P0: Provenance verification (0 compute)
☐ P1a: DD+VP+LS baseline check
☐ P1b: Literature baselines (gate1_literature, ~2h GPU)
☐ P2: Pilot density scan (gate0_density, ~1h CPU)
☐ P3: Path count scan K=6,8 (gate0_K6, gate0_K8, ~30min each CPU)
☐ P4: TF parametric baseline (needs implementation)

After all runs:
☐ Push results: git add results/ && git commit -m "Final experiment results" && git push
☐ Update RESEARCH_REPORT.md with final numbers
```
