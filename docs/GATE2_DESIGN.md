# Gate 2 Design — Safe Degradation Under Corrupted Priors

Date: 2026-07-18

## Core Principle

> **先审计，不改模型；先找最危险失效模式，再设计可靠度机制。**

Gate 2 首先是失效诊断，不是直接优化。

---

## 1. Two Experiment Chains

### Chain A: Oracle Token + Artificial Perturbation

\[
T_{\mathrm{oracle}} \rightarrow \mathcal C_\eta(T_{\mathrm{oracle}}) \rightarrow \widehat H
\]

Purpose: isolate causal impact of single token error types.

### Chain B: Estimated Token + Additional Perturbation

\[
T_{\mathrm{estimated}} \rightarrow \mathcal C_\eta(T_{\mathrm{estimated}}) \rightarrow \widehat H
\]

Purpose: simulate real deployment — estimated tokens already contain position
errors, false peaks, and LS gain errors. This chain determines whether the
final system is safe.

> ⚠️ Oracle-only perturbation gives overly optimistic degradation curves.

---

## 2. Safety Metrics (Beyond Mean NMSE)

Safety baseline is **per-sample TF-only**, not a fixed threshold:

\[
R_i = \mathrm{NMSE}_{\mathrm{fusion},i} - \mathrm{NMSE}_{\mathrm{TF\text{-}only},i}
\]

\[
\overline R = \frac{1}{N}\sum_i R_i
\]

Required metrics:

| Metric | Definition |
|--------|-----------|
| Mean NMSE | Standard |
| Paired 95% CI | Hierarchical bootstrap over seeds |
| **Harm rate** | \(P(R_i > 0)\) — fraction of samples where fusion is WORSE than TF-only |
| Worst-10% NMSE | Mean NMSE over top decile of \(R_i\) |
| Max degradation | \(\max_i R_i\) |
| Degradation AUC | Area under \(R_i\) vs perturbation strength curve |

> A few severe collapses can be hidden by average NMSE.

---

## 3. Gate Diagnostics

Current spatial gate reads TF features, \(H_{\rm phys}\), and \(H_{\rm TF}\) —
it does NOT explicitly read token reliability. It can only judge indirectly
through inconsistency between the two reconstructions.

Record per perturbation level:

\[
g_{\mathrm{mean}},\quad g_{\mathrm{p10}},\quad g_{\mathrm{p50}},\quad g_{\mathrm{p90}}
\]

And:

\[
\operatorname{corr}\left(g_{\mathrm{mean}}, -\mathrm{NMSE}_{H_{\rm phys}}\right)
\]

If tokens get worse but gate does not decrease → gate is a spatial mixer,
not a reliability gate.

---

## 4. Perturbation Catalog

### 4.1 Location Jitter

| Type | Distribution | Purpose |
|------|-------------|---------|
| Independent delay | \(\Delta\tau_l \sim \mathcal{N}(0, \sigma_\tau^2)\) | Random estimation error |
| Independent Doppler | \(\Delta\nu_l \sim \mathcal{N}(0, \sigma_\nu^2)\) | |
| Joint | Both simultaneously | Combined effect |
| Common bias | \(\Delta\tau_l = b_\tau, \Delta\nu_l = b_\nu\) | Sync error, systematic mismatch |

Sweep: {0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0} bins

### 4.2 Token Dropout

Drop rate: {0, 0.25, 0.5, 0.75, 1.0}

Dropped tokens replaced with null/zero padding. At rate=1.0 (all dropped),
model must fall back to TF-only.

### 4.3 False Paths

| Type | Description |
|------|------------|
| **Random false** | Random τ, ν far from true paths |
| **Coherent false** | Near strong paths: \(\tilde\tau = \tau_{\rm strong} + \delta_\tau\), \(\tilde\nu = \nu_{\rm strong} + \delta_\nu\) |

Coherent false paths mimic DD sidelobe detection errors — much harder.

Count: {0, 1, 2, 4}

### 4.4 Gain Magnitude Error

\[
\hat\alpha_l \leftarrow a_l \hat\alpha_l, \quad a_l \in \{0.5, 0.75, 1.25, 1.5, 2.0\}
\]

### 4.5 Gain Phase Error

\[
\hat\alpha_l \leftarrow \hat\alpha_l e^{j\phi_l}, \quad \phi_l \in \left\{\frac{\pi}{8}, \frac{\pi}{4}, \frac{\pi}{2}, \pi\right\}
\]

Phase errors are typically more dangerous than magnitude errors — they cause
coherent cancellation.

Full sign flip (\(\phi=\pi\)) is the most extreme case.

### 4.6 Path Permutation

Shuffle token order. If reordering alone degrades performance, the model
has an unreasonable token-index dependency.

### 4.7 Null Token (All Invalid)

All physical tokens marked invalid. Model must rely entirely on null token.

### 4.8 K Mismatch

Detected path count ≠ true path count.

---

## 5. Three-Phase Gate 2

### Gate 2-A: Clean-Trained Failure Diagnosis

Model trained on clean tokens only. Test with injected corruption.

> How much natural robustness does the current model have?

**Do NOT retrain.** We need to find the real failure threshold.

### Gate 2-B: Corruption-Aware Training

Train with random injection of:
- Token dropout
- False paths
- Parameter perturbation
- Gain phase/magnitude error

> Can data augmentation teach the model to reject wrong priors?

Training may include TF-only distillation or safety constraints.

### Gate 2-C: Structural Safety Fusion

If data augmentation is insufficient, modify architecture:

\[
\widehat H = qH_{\rm phys} + (1-q)H_{\rm TF} + q\Delta H_{\rm phys} + (1-q)\Delta H_{\rm TF}
\]

where \(q = f_{\rm quality}(T, Y_p)\) explicitly inputs:
- Token confidence
- Independent pilot residual
- Consistency of \(H_{\rm phys}\) with pilot observations
- Discrepancy between \(H_{\rm phys}\) and \(H_{\rm TF}\)

This gives the gate genuine reliability semantics.

---

## 6. Gate 2 Pass Criteria (Tiered)

### Normal Deployment Zone

Conditions: dropout ≤ 50%, jitter ≤ 0.5 bins, false ≤ 2 paths, phase error ≤ π/4

Requirement: \(\mathrm{NMSE}_{\rm fusion} \le \mathrm{NMSE}_{\rm TF\text{-}only}\),
paired CI upper bound ≤ +0.3 dB.

### Severe Corruption Zone

Requirement: \(\mathrm{NMSE}_{\rm fusion} - \mathrm{NMSE}_{\rm TF\text{-}only} \le 1\text{ dB}\).

### Null Token

Requirement: \(|\mathrm{NMSE}_{\rm null} - \mathrm{NMSE}_{\rm TF\text{-}only}| < 0.3\text{ dB}\).

---

## 7. Recommended Smoke Test (Minimal First Step)

Fixed main work point, 100–200 paired samples, test only:

1. Token dropout
2. Coherent false tokens
3. Phase error
4. Joint location jitter

Goal: find the most dangerous failure mode, then expand to full sweep.

---

## 8. Priority Ordering

```
1. Gate 2-A: Failure boundary audit          ← HIGHEST scientific value
2. Self-verifying tokens + quality gate       ← HIGHEST innovation value
3. Boundary/Stress K=6/8                      ← Robustness validation
4. Corruption-aware training                  ← Safe degradation
5. Unknown-K                                  ← After token error handling works
6. Joint VP / Gauss-Newton                    ← Clean-token precision
7. Per-path gate redesign                     ← Based on Gate 2 findings
8. Larger networks / more epochs              ← LOWEST priority
```

---

## 9. Beyond Gate 2: Self-Verifying Tokens

If Gate 2 reveals collapse at certain corruption levels, the fix is NOT more
convolution layers. Give the model evidence to judge token quality.

Minimal viable version — four quantities per path:

\[
q_l = [\mathrm{PSLR}_l, \Delta J_l^{\mathrm{LOO}}, \mu_l, J_{\mathrm{check},l}]
\]

The independent check residual is the most critical:

\[
J_{\mathrm{check}} = \left\| y_{\mathcal P_{\rm check}} - A_{\mathcal P_{\rm check}}(\hat\tau,\hat\nu)\hat\alpha \right\|^2
\]

### Implementation approach

1. Aggregate path quality into a quality map
2. Feed quality map into spatial gate
3. Let gate learn when to trust physics reconstruction
4. Verify that null/wrong tokens → NMSE approaches TF-only

Do NOT directly multiply confidence onto path gains (\(c_l \hat\alpha_l\)) —
low confidence ≠ the path amplitude should be smaller. That produces biased
channel estimates.

---

## 10. Per-Path Gate Note

The −1.0 dB regression is likely a conceptual issue, not just gradient tuning:

\[
c_l \neq \text{path amplitude scaling factor}
\]

Proper uses of per-path reliability:
- Conditioning the spatial gate
- Determining residual strength
- Deciding whether to trigger VP
- Deciding token dropout or candidate re-estimation
- NOT directly scaling complex gains

Gate 1-F is deprioritized unless Gate 2 proves single-path errors are the
dominant collapse mode.
