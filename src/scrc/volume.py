"""Conformal intervals on the organ volume (patient level).

Two score functions:
  * relative residual  s = |V - V_hat| / V_hat            ->  V_hat * (1 ± q)
  * CQR on the MC / prompt-perturbation volume quantiles   ->  [lo - q, hi + q]
Both have a standard (exchangeable) and a weighted (covariate-shift) version.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def weighted_conformal_quantile(scores, w_cal, w_test, alpha) -> np.ndarray:
    """Quantile of  sum_i p_i delta_{s_i} + p_{n+1} delta_{+inf}  (one per test weight)."""
    scores = np.asarray(scores, float)
    order = np.argsort(scores)
    s, w = scores[order], np.asarray(w_cal, float)[order]
    w_test = np.atleast_1d(np.asarray(w_test, float))
    cw = np.cumsum(w)
    out = np.empty(len(w_test))
    for j, wt in enumerate(w_test):
        tot = cw[-1] + wt
        idx = np.searchsorted(cw / tot, 1 - alpha, side="left")
        out[j] = s[idx] if idx < len(s) else np.inf
    return out


def relative_scores(v_true, v_hat):
    return np.abs(v_true - v_hat) / np.maximum(v_hat, 1e-6)


def relative_interval(v_hat, q):
    return v_hat * (1 - q), v_hat * (1 + q)


def cqr_scores(v_true, lo, hi):
    return np.maximum(lo - v_true, v_true - hi)


def cqr_interval(lo, hi, q):
    return lo - q, hi + q


def coverage_width(v_true, lo, hi):
    lo, hi = np.broadcast_arrays(lo, hi)
    cov = float(np.mean((v_true >= lo) & (v_true <= hi)))
    width = float(np.mean(np.where(np.isfinite(hi - lo), hi - lo, np.nan)))
    return cov, width
