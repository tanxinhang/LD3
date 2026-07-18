"""Mechanism-gradient safety baselines for Gate 2.

From simplest to most complex:
  1. Fixed blend:     Ĥ = λ·H_phys + (1-λ)·H_Tf,  λ chosen on val set
  2. Hard switch:     Ĥ = H_phys if D < θ else H_Tf,  θ chosen on val set
  3. Logistic gate:   q = σ(w·x + b),  Ĥ = q·H_phys + (1-q)·H_Tf
  4. Hold-out pilot:  split pilots → pick min check-residual branch

Baselines 1, 2, 4 require NO training.  Baseline 3 requires light logistic
regression training on the validation set (few seconds, no GPU needed).
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray


# ===========================================================================
# Shared helpers
# ===========================================================================


def _nmse_db(H_est: NDArray, H_true: NDArray) -> float:
    num = float(np.sum(np.abs(H_est - H_true) ** 2))
    den = float(np.sum(np.abs(H_true) ** 2))
    return float(10.0 * math.log10(max(num / max(den, 1e-30), 1e-30)))


def _nmse_linear(H_est: NDArray, H_true: NDArray) -> float:
    num = float(np.sum(np.abs(H_est - H_true) ** 2))
    den = float(np.sum(np.abs(H_true) ** 2))
    return num / max(den, np.finfo(float).eps)


# ===========================================================================
# 1. Fixed blend
# ===========================================================================


def evaluate_fixed_blend(
    H_phys: NDArray,       # [N_samp, N_sc, N_sym] complex
    H_tf: NDArray,         # [N_samp, N_sc, N_sym] complex
    H_true: NDArray,       # [N_samp, N_sc, N_sym] complex
    lam: float,
) -> dict:
    """Evaluate fixed-blend estimator at a single λ."""
    H_est = lam * H_phys + (1.0 - lam) * H_tf
    nmses = np.array([_nmse_linear(H_est[i], H_true[i]) for i in range(len(H_true))])
    return {
        "lam": lam,
        "nmse_linear": float(np.mean(nmses)),
        "nmse_db": float(10.0 * math.log10(max(float(np.mean(nmses)), 1e-30))),
        "nmse_std": float(np.std(nmses, ddof=1)),
    }


def optimise_fixed_blend(
    H_phys_val: NDArray,
    H_tf_val: NDArray,
    H_true_val: NDArray,
    lambdas: list[float] | None = None,
) -> tuple[float, dict]:
    """Sweep λ on validation set, return (best_lam, sweep_results)."""
    if lambdas is None:
        lambdas = [round(i * 0.05, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00
    best_lam = 0.0
    best_nmse = float("inf")
    sweep = []
    for lam in lambdas:
        r = evaluate_fixed_blend(H_phys_val, H_tf_val, H_true_val, lam)
        sweep.append(r)
        if r["nmse_linear"] < best_nmse:
            best_nmse = r["nmse_linear"]
            best_lam = lam
    return best_lam, {"sweep": sweep, "best_lam": best_lam}


# ===========================================================================
# 2. Hard discrepancy switch
# ===========================================================================


def _discrepancy_map(H_phys: NDArray, H_tf: NDArray) -> NDArray:
    """Per-sample normalised discrepancy: mean(|H_phys - H_tf|² / |H_tf|²)."""
    diff = np.abs(H_phys - H_tf) ** 2
    denom = np.abs(H_tf) ** 2 + np.finfo(float).eps
    # Per-sample mean discrepancy
    return np.mean(diff / denom, axis=(1, 2))  # [N_samp]


def evaluate_hard_switch(
    H_phys: NDArray,
    H_tf: NDArray,
    H_true: NDArray,
    theta: float,
) -> dict:
    """Evaluate hard discrepancy switch at a single threshold θ."""
    D = _discrepancy_map(H_phys, H_tf)
    n_samp = len(H_true)
    nmses = np.zeros(n_samp)
    for i in range(n_samp):
        H_est = H_phys[i] if D[i] < theta else H_tf[i]
        nmses[i] = _nmse_linear(H_est, H_true[i])
    use_phys_frac = float(np.mean(D < theta))
    return {
        "theta": theta,
        "nmse_linear": float(np.mean(nmses)),
        "nmse_db": float(10.0 * math.log10(max(float(np.mean(nmses)), 1e-30))),
        "nmse_std": float(np.std(nmses, ddof=1)),
        "frac_phys_selected": use_phys_frac,
        "frac_tf_selected": 1.0 - use_phys_frac,
    }


def optimise_hard_switch(
    H_phys_val: NDArray,
    H_tf_val: NDArray,
    H_true_val: NDArray,
    thetas: list[float] | None = None,
) -> tuple[float, dict]:
    """Sweep θ on validation set, return (best_theta, sweep_results)."""
    if thetas is None:
        thetas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_theta = thetas[0]
    best_nmse = float("inf")
    sweep = []
    for theta in thetas:
        r = evaluate_hard_switch(H_phys_val, H_tf_val, H_true_val, theta)
        sweep.append(r)
        if r["nmse_linear"] < best_nmse:
            best_nmse = r["nmse_linear"]
            best_theta = theta
    return best_theta, {"sweep": sweep, "best_theta": best_theta}


# ===========================================================================
# 3. Logistic quality gate
# ===========================================================================


def _quality_features(H_phys: NDArray, H_tf: NDArray,
                      tokens: NDArray, valid: NDArray) -> NDArray:
    """Extract per-sample quality features (same as Gate 2-C quality map).

    Returns [N_samp, 3]: [D_mean, confidence_mean, uncertainty_mean]
    """
    n_samp = len(H_phys)
    feats = np.zeros((n_samp, 3), dtype=np.float64)
    for i in range(n_samp):
        # Discrepancy mean
        diff = np.abs(H_phys[i] - H_tf[i]) ** 2
        denom = np.abs(H_tf[i]) ** 2 + np.finfo(float).eps
        D_mean = float(np.mean(diff / denom))

        # Mean confidence over valid tokens
        v = valid[i]
        if v.sum() > 0:
            conf_mean = float(np.mean(tokens[i, v, 3]))
            unc_mean = float(np.mean(tokens[i, v, 4] + tokens[i, v, 5]))
        else:
            conf_mean = 0.0
            unc_mean = 2.0  # max uncertainty when no valid tokens

        feats[i] = [D_mean, conf_mean, unc_mean]
    return feats


class LogisticQualityGate:
    """Scalar logistic regression on quality features → per-sample blend weight.

    q = σ(w0*D + w1*conf + w2*unc + b)
    Ĥ = q·H_phys + (1-q)·H_tf
    """

    def __init__(self):
        self.w = np.zeros(3, dtype=np.float64)
        self.b = 0.0

    def _sigmoid(self, z: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X: NDArray, H_phys: NDArray, H_tf: NDArray,
            H_true: NDArray, lr: float = 0.01, n_iter: int = 2000):
        """Train via gradient descent to minimise per-sample NMSE."""
        n_samp = X.shape[0]
        w, b = self.w.copy(), self.b
        best_nmse = float("inf")
        best_w, best_b = w.copy(), b

        for it in range(n_iter):
            z = X @ w + b
            q = self._sigmoid(z)
            # Loss: mean NMSE over samples
            losses = np.zeros(n_samp)
            dq_dz = q * (1.0 - q)  # [n_samp]
            grad_w = np.zeros(3)
            grad_b = 0.0

            for i in range(n_samp):
                H_est = q[i] * H_phys[i] + (1.0 - q[i]) * H_tf[i]
                err = H_est - H_true[i]
                loss_num = float(np.sum(np.abs(err) ** 2))
                loss_den = float(np.sum(np.abs(H_true[i]) ** 2)) + 1e-30
                losses[i] = loss_num / loss_den

                # dLoss/dq[i]: derivative of NMSE w.r.t. q[i]
                diff = H_phys[i] - H_tf[i]  # dĤ/dq
                # dNMSE/dq = 2 * Re(Σ err* · diff) / ||H_true||²
                dloss_dq = 2.0 * float(np.real(np.sum(np.conj(err) * diff))) / loss_den

                # Chain rule: dq/dz = q(1-q)
                grad_w += dloss_dq * dq_dz[i] * X[i]
                grad_b += dloss_dq * dq_dz[i]

            # Gradient step
            w -= lr * grad_w / n_samp
            b -= lr * grad_b / n_samp

            nmse_lin = float(np.mean(losses))
            if nmse_lin < best_nmse:
                best_nmse = nmse_lin
                best_w, best_b = w.copy(), b

        self.w, self.b = best_w, best_b
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Return per-sample blend weight q ∈ (0, 1)."""
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def evaluate(self, X: NDArray, H_phys: NDArray, H_tf: NDArray,
                 H_true: NDArray) -> dict:
        """Evaluate on test set, return metrics."""
        q = self.predict(X)
        n_samp = len(H_true)
        nmses = np.zeros(n_samp)
        for i in range(n_samp):
            H_est = q[i] * H_phys[i] + (1.0 - q[i]) * H_tf[i]
            nmses[i] = _nmse_linear(H_est, H_true[i])
        return {
            "nmse_linear": float(np.mean(nmses)),
            "nmse_db": float(10.0 * math.log10(max(float(np.mean(nmses)), 1e-30))),
            "nmse_std": float(np.std(nmses, ddof=1)),
            "q_mean": float(np.mean(q)),
            "q_std": float(np.std(q)),
            "weights": self.w.tolist(),
            "bias": float(self.b),
        }


# ===========================================================================
# 4. Hold-out pilot selector
# ===========================================================================


def evaluate_holdout_pilot_selector(
    pilot_observations: NDArray,    # [N_samp, N_sc, N_sym] complex
    pilot_mask: NDArray,            # [N_samp, N_sc, N_sym] bool
    H_phys: NDArray,                # [N_samp, N_sc, N_sym] complex
    H_tf: NDArray,                  # [N_samp, N_sc, N_sym] complex
    H_true: NDArray,                # [N_samp, N_sc, N_sym] complex
    holdout_fraction: float = 0.3,
    rng_seed: int = 42,
) -> dict:
    """Select H_phys or H_Tf per-sample based on hold-out pilot residual.

    For each sample:
      1. Split pilot locations into P_fit (70%) and P_check (30%).
      2. Compute |y - H_branch|² on P_check for both branches.
      3. Select the branch with lower check residual.

    This is the closest non-learned analogue to "self-verifying tokens".
    """
    rng = np.random.default_rng(rng_seed)
    n_samp = len(H_true)
    nmses = np.zeros(n_samp)
    selections = np.zeros(n_samp, dtype=int)  # 0 = phys, 1 = tf
    check_resid_phys = np.zeros(n_samp)
    check_resid_tf = np.zeros(n_samp)

    for i in range(n_samp):
        # Get pilot indices for this sample
        n_idx, m_idx = np.nonzero(pilot_mask[i])
        n_pilots = len(n_idx)
        if n_pilots < 2:
            # Not enough pilots to split → fall back to TF-only
            H_est = H_tf[i]
            selections[i] = 1
        else:
            n_check = max(1, int(n_pilots * holdout_fraction))
            perm = rng.permutation(n_pilots)
            fit_idx = perm[n_check:]
            check_idx = perm[:n_check]

            # Check residuals
            y = pilot_observations[i, n_idx, m_idx]

            # Phys branch residual on check pilots
            phys_check_err = np.abs(
                y[check_idx] - H_phys[i, n_idx[check_idx], m_idx[check_idx]]
            ) ** 2
            check_resid_phys[i] = float(np.sum(phys_check_err))

            # TF branch residual on check pilots
            tf_check_err = np.abs(
                y[check_idx] - H_tf[i, n_idx[check_idx], m_idx[check_idx]]
            ) ** 2
            check_resid_tf[i] = float(np.sum(tf_check_err))

            if check_resid_phys[i] < check_resid_tf[i]:
                H_est = H_phys[i]
                selections[i] = 0
            else:
                H_est = H_tf[i]
                selections[i] = 1

        nmses[i] = _nmse_linear(H_est, H_true[i])

    frac_phys = float(np.mean(selections == 0))
    return {
        "nmse_linear": float(np.mean(nmses)),
        "nmse_db": float(10.0 * math.log10(max(float(np.mean(nmses)), 1e-30))),
        "nmse_std": float(np.std(nmses, ddof=1)),
        "frac_phys_selected": frac_phys,
        "frac_tf_selected": 1.0 - frac_phys,
        "holdout_fraction": holdout_fraction,
        "mean_check_resid_phys": float(np.mean(check_resid_phys)),
        "mean_check_resid_tf": float(np.mean(check_resid_tf)),
    }
