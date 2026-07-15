# Implementation Status

Date: 2026-07-15

## Implemented research slice

This initial repository revision implements the minimum validation path for the
proposed physics-guided TF–DD OFDM-ISAC estimator.

### Gate 0: DD identifiability audit

Implemented and tested:

- sparse fractional-delay/fractional-Doppler TF channel synthesis;
- random and comb pilot masks;
- mask-aware DD matched filtering directly on observed pilots;
- configurable DD-grid oversampling;
- Top-K peak selection with non-maximum suppression;
- Hungarian true/estimated path matching;
- path precision, recall, delay RMSE, Doppler RMSE, and recovered true-path
  power ratio;
- parameter sweep and CSV/JSON/PNG outputs.

A test exposed that an NMS radius of one oversampled grid cell can select a
main-lobe neighbor as a second path. The default was therefore changed to two
cells. The radius remains configurable because it depends on grid oversampling
and observation aperture.

### Gate 1: Oracle DD value skeleton

Implemented and tested:

- deterministic synthetic training dataset;
- lightweight matched-capacity TF-only baseline;
- Oracle path tokens containing delay, Doppler, power, confidence,
  uncertainty, and relevance;
- normalized OFDM phase-law attention bias;
- confidence/relevance prior and uncertainty penalty;
- learned null token that can reject every physical candidate;
- confidence gate;
- model checkpoints and JSON metrics;
- Oracle, shuffled-token, and all-null-token evaluations.

Only channel NMSE is optimized. Multi-task losses and residual path refinement
are intentionally excluded from this revision.

## Verification performed

```text
pytest: 4 passed
Gate 0 smoke: completed and generated all expected outputs
Gate 1 smoke: completed on CPU and generated both checkpoints and JSON metrics
```

## Current Gate 1 diagnosis

A five-epoch diagnostic run with 64 training and 32 test samples produced:

| Evaluation | NMSE |
|---|---:|
| TF-only | -0.684 dB |
| Physics cross-attention with Oracle tokens | -0.517 dB |
| Oracle cross-attention gain over TF-only | -0.167 dB |
| Oracle advantage over shuffled tokens | +0.011 dB |
| Oracle advantage over all-null tokens | +0.033 dB |

These values are not paper results. The dataset and training duration are far
too small. They show that the code path is operational, but Gate 1 has **not**
yet passed: the cross-domain model does not currently outperform the TF-only
baseline, and its sensitivity to correct token identity is weak.

This negative result should be retained as the starting diagnostic. The next
work should investigate token amplitude/complex-gain information, stronger
structure-level fusion, capacity matching, and longer paired-seed training
before adding more modules.

## Recommended next revision

1. Add a controlled complex-gain token or a parametric path reconstruction
   branch so Oracle DD information can produce a physically complete channel
   contribution.
2. Compare three models under matched parameter/FLOP budgets:
   TF-only, direct parametric-plus-residual, and cross-attention.
3. Train across at least three seeds and report paired confidence intervals.
4. Add Gate 2 token perturbation sweeps only after Oracle tokens produce a
   reproducible positive contribution.
