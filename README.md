# LD3 — Physics-Guided TF–DD OFDM-ISAC Validation

This repository implements the first two validation gates for a robust
TF–DD cross-domain OFDM-ISAC channel estimator. It intentionally starts with
identifiability and controlled Oracle experiments instead of immediately
building a large end-to-end network.

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

The 0.8 power-recovery line in the plot is a configurable research decision
reference, not a universal physical threshold.

### Gate 0 conclusion (conditional)

**Gate 0-A (Known-K identifiability): CONDITIONAL PASS.** Under known path count
K=4 and fixed Top-K output, Random pilots with density ≥ 1/8 and SNR ≥ ~5 dB
recover ~84-88% of true path power and identify ~65-74% of true paths. The DD
estimator stably captures dominant channel energy, providing a basis for Gate 1.

**Gate 0-B (Unknown-K detection): NOT YET RUN.** Open-set false alarm control,
path-count estimation, and complex-gain reconstruction remain as future work.

### Gate 1 — Oracle DD value test

The Gate 1 implementation compares:

- a **lightweight TF-only estimator** (matched-capacity baseline);
- a **physics-guided TF–DD cross-attention estimator** using Oracle path tokens;
- **Oracle support + LS complex-gain** (non-learned parametric baseline);
- **DD-estimated support + LS complex-gain** (bridges Gate 0 → Gate 1);
- **Oracle perfect reconstruction** (diagnostic upper bound).

The cross-attention model includes:

- path tokens containing delay, Doppler, power, confidence, uncertainty, and
  communication relevance;
- exact normalized OFDM phase-law bias;
- confidence/relevance bias and uncertainty penalty;
- a **null token**, allowing all DD candidates to be rejected;
- a learned confidence gate;
- a lightweight convolutional TF encoder rather than global quadratic TF
  attention.

Three pre-defined work points:

| Work point | Config | Pilot | Density | SNR | Purpose |
|---|---|---|---|---|---|
| Main | `configs/gate1_main.yaml` | Random | 0.125 | 10 dB | Primary value judgment |
| Boundary | `configs/gate1_boundary.yaml` | Random | 0.125 | 0 dB | Low-SNR robustness |
| Stress | `configs/gate1_stress.yaml` | Random | 0.0625 | 5 dB | Sparse-pilot robustness |

Only normalized channel NMSE is used for the first training closure. Residual
path refinement, BER loss, and multi-task loss weighting are deliberately left
out until Gates 0 and 1 pass.

## Repository structure

```text
configs/
  gate0.yaml              # Main Gate 0 sweep
  gate0_ablation.yaml     # High-SNR plateau ablation
  gate1.yaml              # Default Gate 1 (small)
  gate1_main.yaml         # Main work point
  gate1_boundary.yaml     # Boundary work point
  gate1_stress.yaml       # Stress work point
experiments/
  gate0_identifiability.py
  gate1_oracle.py
src/ld3/
  channel.py              # Sparse channel generation & TF synthesis
  config.py               # OFDMConfig, ChannelConfig
  dataset.py              # SyntheticOFDMISACDataset
  dd_estimation.py        # DD matched filter, NMS, metrics, OSPA, CI, bootstrap
  interpolation.py        # Nearest-smooth TF interpolation baseline
  metrics.py              # NMSE (numpy + torch)
  models.py               # TFEncoder, PhysicsGuidedCrossAttention, TFOnlyEstimator
  oracle.py               # Oracle tokens, Oracle+LS reconstruction, perturbation
  pilots.py               # Pilot mask generation & observation
tests/
  test_channel.py
  test_dd_estimation.py
  test_models.py
```

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -e .[dev]
```

## Run tests

```bash
pytest -q
```

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

Outputs:

- `gate0_trials.csv` — per-trial metrics
- `gate0_summary.csv` — mean, std, SE, 95% CI per condition
- `gate0_paired_bootstrap.csv` — Random-vs-Comb paired comparison
- `gate0_coherence.csv` — DD dictionary mutual coherence
- `gate0_ambiguity.csv` — pilot ambiguity function metrics
- `ambiguity_*.png` — pilot AF plots
- `gate0_power_recovery_*.png`, `gate0_recall_*.png`, `gate0_ospa_*.png`
- `manifest.json`

### Ablation: high-SNR plateau

```bash
# Case I: integer bins (no off-grid leakage)
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml \
  --output-dir results/gate0_ablation_I \
  --ablation-integer-bins

# Case F: fractional bins (off-grid leakage present)
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml \
  --output-dir results/gate0_ablation_F

# Case F+Oracle: oracle NMS (isolate NMS bottleneck)
python experiments/gate0_identifiability.py \
  --config configs/gate0_ablation.yaml \
  --output-dir results/gate0_ablation_F_oracle \
  --ablation-oracle-nms
```

## Gate 1

### Smoke run

```bash
python experiments/gate1_oracle.py \
  --epochs 1 \
  --train-size 64 \
  --test-size 32
```

### Main work point (3 seeds, 50 epochs)

```bash
python experiments/gate1_oracle.py \
  --config configs/gate1_main.yaml \
  --output-dir results/gate1_main
```

The output JSON includes non-learned baselines (Oracle perfect, Oracle+LS,
DD+LS) plus learned model results with per-seed and aggregated metrics.

### Key diagnostics

```text
oracle_cross_attention_gain_db = TF-only NMSE(dB) - physics-cross NMSE(dB)
oracle_vs_shuffled_gain_db     = shuffled NMSE(dB) - oracle NMSE(dB)
oracle_vs_null_gain_db         = null NMSE(dB) - oracle NMSE(dB)
```

A positive `oracle_cross_attention_gain_db` indicates Oracle DD tokens added
value. `oracle_vs_shuffled_gain_db` and `oracle_vs_null_gain_db` should also
be positive — otherwise apparent gain may come only from extra model capacity.

The Oracle+LS baseline (non-learned) provides a diagnostic floor: if even
perfect DD support + LS gains cannot beat TF-only, the issue is support
quality, not model architecture.

## Research gates

The recommended progression is:

1. **Gate 0-A:** establish DD-identifiable operating region (known K) ✅
2. **Gate 0-B:** unknown-K detection and open-set false alarm control
3. **Gate 1:** prove Oracle DD information improves over TF-only processing
4. **Gate 2:** inject DD position errors, misses, false peaks, and path mismatch
5. **Gate 3:** replace Oracle tokens with estimated tokens and decompose token
   extraction loss from fusion loss

Do not add residual-path refinement before Gate 1 is consistently positive.

### Gate 1 token-use audit

The trained cross-domain model is evaluated three ways:

- `oracle`: correct path tokens;
- `shuffled`: tokens are moved to another sample in the batch;
- `null`: all physical path tokens are rejected, leaving only the learned null token.

A credible Gate 1 result should not only beat TF-only processing. It should also
perform better with correct Oracle tokens than with shuffled or null tokens.
Otherwise the apparent gain may come only from extra model capacity.
