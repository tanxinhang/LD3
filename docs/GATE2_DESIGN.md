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

---

## 11. Gate 2-A Empirical Results (2026-07-18)

Full sweep: 1024 samples × 46 corruption specs × 2 chains (oracle + estimated).
Safety baseline: TF-only = −4.62 dB.

### 11.1 Failure Severity Tiers

**☠️ Critical (harm_rate ≥ 90%)**

| Perturbation | Oracle NMSE | Est. NMSE | Harm% | Gate | Note |
|---|---|---|---|---|---|
| phase π | **+1.10 dB** | +0.79 dB | 100% | 0.472 | Active destruction |
| phase π/2 | −0.98 dB | −1.25 dB | 100% | 0.548 | Coherent cancellation |
| jitter 2.0 joint | −1.20 dB | −1.47 dB | 99.5% | 0.550 | Total location chaos |
| bias_delay 0.5 | −1.43 dB | −1.59 dB | **100%** | 0.554 | Systematic bias — gate blind |
| jitter 1.5 joint | −1.27 dB | −1.54 dB | 99.0% | 0.552 | |

**⚠️ Dangerous (harm_rate 30–90%)**

| Perturbation | Oracle NMSE | Harm% |
|---|---|---|
| coherent_false 4 | −3.11 dB | 74.0% |
| mag 2.0 | −4.22 dB | 59.9% |
| dropout 0.75 | −4.57 dB | 60.6% |
| jitter 0.5 joint | −2.05 dB | 89.9% |

**✅ Safe (harm_rate < 5%)**

| Perturbation | Oracle NMSE | Harm% | Key Finding |
|---|---|---|---|
| **random_false 1/2/4** | **−15.39 dB** | **0.0%** | Model completely immune — null token works |
| **permute** | **−15.39 dB** | **0.0%** | Permutation invariance confirmed |
| phase π/8 | −9.94 dB | 0.0% | Small phase errors tolerated |
| jitter 0.1 joint | −9.00 dB | 3.0% | Small jitter tolerated |
| mag 0.75–1.25 | −9.3 to −12.7 | 0.0% | Minor gain deviation safe |

### 11.2 Key Findings

**1. Random false paths: completely harmless.** Model uses null token to
ignore them. Coherent false paths (near strong true paths) are the real
threat — each added coherent false degrades ~3 dB.

**2. `null_all` fails to reach TF-only — structural defect.**

```
null_all:  −3.40 dB  (gate = 0.589)
TF-only:   −4.62 dB  (expected safe fallback)
Gap:       +1.22 dB  ← GATE 2-A FAIL
```

Even with all tokens marked invalid, gate stays at 0.59 — nearly 60% of
the physical reconstruction is still mixed in. The spatial gate reads
TF features + H_phys + H_TF, not independent token quality signals.

**3. Gate does respond to token quality — but not enough.**

| Condition | Gate mean | NMSE |
|---|---|---|
| clean | 0.642 | −15.39 dB |
| jitter 0.5 | 0.565 | −2.05 dB |
| phase π | **0.472** | +1.10 dB |
| null_all | 0.589 | −3.40 dB |

Gate drops from 0.64 to 0.47 under phase reversal, confirming it
perceives token degradation. But the drop is insufficient — at
phase π, nearly half the physical branch output is still fused in.

**4. Location jitter: joint > delay ≈ Doppler > bias.**

Joint jitter is consistently worse than single-axis, and systematic
bias (common offset) is harder for the gate to detect than random
jitter of equal magnitude — random errors create inconsistency
between H_phys and H_TF that the gate can exploit.

**5. Oracle vs Estimated chain asymmetry.**

Oracle clean (−15.39 dB) >> Estimated clean (−10.22 dB) — the 5 dB
gap comes entirely from token quality. Under strong corruption both
chains converge to the same floor (~−3.4 dB null_all), confirming
that the model's safety ceiling is architecture-limited, not
token-quality-limited.

### 11.3 Updated Priority

Based on the null_all result, the priority ordering is revised:

```
1. Gate 2-C: Structural safety fusion (fix null_all → TF-only gap)
   └── Quality-conditioned gate with explicit token reliability inputs
2. Self-verifying tokens (check residual, PSLR, LOO → per-path quality)
3. Corruption-aware training (with phase/coherent/jitter augmentation)
4. Boundary/Stress K=6/8
5. Unknown-K
```

The null_all exposure is the single most actionable finding: the
current gate is a spatial mixer, not a reliability gate. Fixing this
requires architectural change — the gate must receive explicit token
quality evidence, not just TF features and reconstructions.

### 11.4 Gate 2-C Results: Quality-Conditioned Gate (2026-07-18)

Implemented `_build_quality_map()` — 3-channel spatial map feeding into the
fusion gate: |H_phys−H_tf|² discrepancy, mean token confidence, and mean
token uncertainty.  Training: same setup as gate1_estimated (K=4, 10 dB,
ρ=0.125, estimated tokens, 3 seeds × 300 epochs).

**Training results (clean):**

| Metric | Gate 1-D1 (baseline) | Gate 2-C (quality gate) |
|---|---|---|
| Estimated Residual NMSE | −9.76 dB | **−10.69 dB** |
| DD+LS → EstRes gain | +1.40 dB [1.20, 1.61] | **+1.93 dB [1.86, 1.99]** |
| Gate mean (clean) | 0.618 | **0.923** |

Clean performance improved +0.93 dB with no extra parameters except the
3 quality-map channels fed into the gate CNN.

**Audit results (full perturbation sweep):**

| Condition | Gate 1-D1 NMSE | Gate 1-D1 Gate | Gate 2-C NMSE | Gate 2-C Gate | Δ NMSE |
|---|---|---|---|---|---|
| clean (est) | −10.22 dB | 0.635 | **−10.69 dB** | 0.923 | +0.47 |
| null_all | −3.40 dB | 0.589 | **−4.37 dB** | 0.504 | +0.97 |
| phase π | **+1.10 dB** ☠️ | 0.472 | **−3.93 dB** | 0.162 | **+5.03** |
| phase π/2 | −0.98 dB | 0.548 | **−2.75 dB** | 0.591 | +1.77 |
| jitter 2.0 joint | −1.20 dB | 0.550 | **−4.78 dB** | 0.049 | +3.58 |
| dropout 0.75 | −4.57 dB | 0.603 | **−5.10 dB** | 0.835 | +0.53 |
| coherent_false 4 | −3.11 dB | 0.586 | **−5.00 dB** | 0.554 | +1.89 |

**Three key findings:**

1. **Gate becomes a genuine reliability indicator.** Quality gate drops from
   0.92 (clean) to 0.05 (jitter 2.0, ~19× dynamic range) and 0.16 (phase π,
   ~5.7×). The original gate had only ~1.4× range (0.64→0.47). The gate can
   now nearly fully shut off the physics branch for the most severe errors.

2. **Phase π no longer catastrophic.** Went from +1.10 dB (actively harmful,
   worse than no DD prior) to −3.93 dB — a +5.03 dB improvement. However,
   −3.93 dB is still +0.69 dB worse than the paired TF-only baseline (−4.62 dB
   in the original audit), so the result meets the "severe corruption zone"
   criterion (≤+1 dB) but not the strict "no-harm" criterion.

3. **null_all gap: DATA NEEDS RECONCILIATION.** Two interpretations exist:
   - Using the Gate 2-C paired TF-only (−5.48 dB): null_all (−4.37) → gap =
     **+1.11 dB**. This is a conservative estimate but uses a TF-only trained
     in the same Gate 2-C run (which may have different seed luck).
   - Using the original Gate 1-D1 TF-only (−4.62 dB, same test config): gap =
     **+0.25 dB**, which would actually **PASS** the ±0.3 dB criterion.
   The unresolved question: which TF-only model is the "correct" safety
   baseline? The Gate 2-C TF-only (−5.48 dB) is objectively better trained,
   but this makes the null_all gap look worse through no fault of the quality
   gate. A paired audit where both old and new models are evaluated against
   the SAME TF-only is needed to resolve this.

   Gate at null_all = 0.50. This is NOT sigmoid saturation (σ(0) = 0.5 is
   the sigmoid's maximum-gradient midpoint). It means the quality map's
   features place the null_all case near the decision boundary (z ≈ 0).
   The quality map's discrepancy channel |H_phys−H_TF|² is small when both
   reconstructions are near zero (all tokens invalid), reducing its
   discriminative power. The fix is an explicit all_tokens_invalid signal,
   not a temperature parameter.

4. **Residual ΔH is always added unconditionally** — even when g=0, the
   output is Ĥ = H_TF + ΔH, not pure H_TF. This means the residual can
   introduce its own error on null_all samples, independent of the gate.
   The next audit must decompose null_all error into: internal H_TF quality,
   gating residual, residual ΔH contribution, and standalone TF-only.

**Revised conclusion:**

```
Gate 2-C: CONDITIONAL PASS
  ✅ Gate responds to token quality (jitter ~19×, phase ~6× dynamic range)
  ✅ Phase π no longer catastrophic (+1.10 → −3.93 dB, +5.03 dB)
  ✅ Clean performance maintained (+0.47 dB matched-audit; +0.93 dB vs prior aggregate)
  ⚠️ null_all gap: +0.25 to +1.11 dB depending on TF-only baseline — needs reconciliation
  ⚠️ Gate at null_all = 0.50 — decision boundary, not saturation
  ⚠️ Residual ΔH unconditionally added even when g=0
```

### 11.5 Updated Priority (post Gate 2-C)

```
P0: Reconcile null_all baseline (same TF-only for old + new audit)
P1: Structural hard fallback — v = I[any valid], q = v·σ(z)
    └── Guarantees Ĥ → H_TF when all tokens invalid
P2: Null error decomposition — measure H_TF, fused, residual, final
P3: Quality map v2 — add valid_ratio, all-null flag, check residual
P4: Corruption-aware training — phase/coherent/bias augmentation
P5: Boundary/Stress K=6/8 — validate robustness under difficulty
```

### 11.6 Mechanism-Gradient Baselines (2026-07-18)

Four baselines from simple to complex, all using the SAME frozen H_phys and
H_Tf from the Gate 2-C model.  717 test samples, hyperparameters optimised on
307 validation samples.

| Method | NMSE (dB) | Params | Trained | Key insight |
|---|---|---|---|---|
| H_phys-only | −8.50 | 0 | No | Deployable DD-estimated physics baseline |
| TF-only (standalone) | −5.06 | CNN | Yes | Lower bound (no DD prior) |
| Fixed blend (λ=0.80) | **−9.15** | 1 | No | 80% physics + 20% TF |
| Hard switch | −5.52 | 1 | No | Threshold on |H_phys−H_TF|² |
| Logistic quality gate | −9.04 | 3 | Light | 0.11 dB worse than fixed blend (within noise) |
| Hold-out pilot select | −8.27 | 0 | No | Split pilots, pick min check residual |
| **Spatial quality gate** | **−10.45** | CNN | Yes | Gate 2-C: spatial gate + residual ΔH |

**Key findings:**

1. **The bulk of fusion gain comes from simple scalar blending.**
   Fixed blend (λ=0.80, 1 parameter, no training) achieves −9.15 dB —
   within 1.30 dB of the full spatial quality gate (−10.45 dB) and
   +4.09 dB over TF-only. The physics branch dominates (80% weight).

2. **Global quality features provide negligible clean-condition NMSE gain
   when compressed into a single scalar.** Logistic quality gate (−9.04 dB)
   is 0.11 dB *worse* than the validation-tuned fixed blend (−9.15 dB).
   The difference is within noise — at the scalar level, the three quality
   features add no discriminative power beyond a single blend ratio.
   q_mean ≈ 0.80 confirms the model is essentially learning λ = 0.80.
   **Their corruption-detection value must be assessed separately via a
   full corruption sweep on all mechanism baselines.**

3. **Spatial gating + residual ΔH improves +1.30 dB over fixed blend.**
   The gap from Fixed blend (−9.15, no spatial gate, no residual) to the
   full model (−10.45, spatial gate + zero-init ΔH) is +1.30 dB.
   **A gate–residual 2×2 factorial ablation is required to isolate the
   pure spatial-gating contribution from the residual correction
   contribution.** The fixed blend is a post-hoc baseline on frozen model
   branches; an end-to-end fixed-blend model trained from scratch would
   provide the fairest comparison.

4. **"Clever" non-learned baselines underperform simple blending.** Hard
   switch (−5.52 dB, barely above TF-only) and hold-out pilot selector
   (−8.27 dB, below H_phys-only) fail because binary selection loses
   information. Also, the hold-out selector splits pilots, reducing the
   effective estimation budget. **A soft hold-out blend (check-pilot
   residual as softmax temperature) should be tested before concluding
   that hold-out verification has no value.**

**Five-layer decomposition:**

| Layer | Transition | Δ NMSE | Mechanism |
|---|---|---|---|
| 1 | TF-only → H_phys-only | +3.44 dB | DD physical prior |
| 2 | H_phys-only → Fixed blend | +0.65 dB | Soft fusion synergy |
| 3 | Fixed blend → Logistic gate | −0.11 dB | Scalar quality = no clean gain |
| 4 | Fixed blend → Spatial gate + ΔH | +1.30 dB | Spatial gating + residual (to be split) |
| 5 | Spatial gate → + Hard fallback | — | Structural null_all safety (±0.18 dB) |
