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
- Hungarian path matching;
- path recall, precision, delay/Doppler RMSE;
- true-path power recovery ratio;
- SNR, pilot-density, and pilot-pattern sweeps;
- CSV, JSON, and figure outputs.

The 0.8 power-recovery line in the plot is a configurable research decision
reference, not a universal physical threshold.

### Gate 1 — Oracle DD value test

The Gate 1 implementation compares:

- a lightweight TF-only estimator;
- a physics-guided TF–DD cross-attention estimator using Oracle path tokens.

The cross-attention model includes:

- path tokens containing delay, Doppler, power, confidence, uncertainty, and
  communication relevance;
- exact normalized OFDM phase-law bias;
- confidence/relevance bias and uncertainty penalty;
- a **null token**, allowing all DD candidates to be rejected;
- a learned confidence gate;
- a lightweight convolutional TF encoder rather than global quadratic TF
  attention.

Only normalized channel NMSE is used for the first training closure. Residual
path refinement, BER loss, and multi-task loss weighting are deliberately left
out until Gates 0 and 1 pass.

## Repository structure

```text
configs/
  gate0.yaml
  gate1.yaml
experiments/
  gate0_identifiability.py
  gate1_oracle.py
src/ld3/
  channel.py
  config.py
  dataset.py
  dd_estimation.py
  interpolation.py
  metrics.py
  models.py
  oracle.py
  pilots.py
tests/
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

## Gate 0 smoke run

```bash
python experiments/gate0_identifiability.py --trials 3
```

Full configured sweep:

```bash
python experiments/gate0_identifiability.py \
  --config configs/gate0.yaml \
  --output-dir results/gate0
```

Primary outputs:

- `results/gate0/gate0_trials.csv`
- `results/gate0/gate0_summary.csv`
- `results/gate0/manifest.json`
- `results/gate0/gate0_power_recovery_random.png`
- `results/gate0/gate0_power_recovery_comb.png`

## Gate 1 smoke run

```bash
python experiments/gate1_oracle.py \
  --epochs 1 \
  --train-size 64 \
  --test-size 32
```

Full configured run:

```bash
python experiments/gate1_oracle.py \
  --config configs/gate1.yaml \
  --output-dir results/gate1
```

The key diagnostic is:

```text
oracle_cross_attention_gain_db = TF-only NMSE(dB) - physics-cross NMSE(dB)
```

A positive value indicates that Oracle DD tokens added measurable value beyond
the matched TF-only network. One run is not sufficient for a paper claim;
repeat across seeds and report paired confidence intervals.

## Research gates

The recommended progression is:

1. **Gate 0:** establish a DD-identifiable operating region;
2. **Gate 1:** prove Oracle DD information improves over TF-only processing;
3. **Gate 2:** inject DD position errors, misses, false peaks, and path mismatch;
4. **Gate 3:** replace Oracle tokens with estimated tokens and decompose token
   extraction loss from fusion loss.

Do not add residual-path refinement before Gate 1 is consistently positive.

### Gate 1 token-use audit

The trained cross-domain model is evaluated three ways:

- `oracle`: correct path tokens;
- `shuffled`: tokens are moved to another sample in the batch;
- `null`: all physical path tokens are rejected, leaving only the learned null token.

A credible Gate 1 result should not only beat TF-only processing. It should also
perform better with correct Oracle tokens than with shuffled or null tokens.
Otherwise the apparent gain may come only from extra model capacity.
