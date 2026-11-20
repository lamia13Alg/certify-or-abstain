r"""Patient-level abstention ("refer to the radiologist").

A patient is ACCEPTED when its uncertainty score u(x) <= tau. Two guarantees:

1. Joint risk (works with covariate-shift weights, exact finite-sample):
       E[ L(x) * 1{u(x) <= tau} ] <= alpha
   The loss  L_i(tau) = l_i * 1{u_i <= tau}  is NON-DECREASING in tau, so we run
   CRC on  -tau  (largest tau whose adjusted risk stays below alpha).

2. Selective (conditional) risk, high probability, Learn-then-Test
   (Angelopoulos, Bates, Candès, Jordan, Lei 2021):
       P( E[ L | accepted ] <= alpha ) >= 1 - delta
   using Hoeffding–Bentkus p-values and fixed-sequence testing over tau.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def joint_risk_tau(
    loss: np.ndarray,
    u: np.ndarray,
    alpha: float,
    taus: np.ndarray,
    w_cal: np.ndarray | None = None,
    w_test: float | np.ndarray | None = None,
    B: float = 1.0,
) -> np.ndarray:
    """Largest tau with (weighted) CRC-adjusted joint risk <= alpha.

    loss : [n] per-patient loss at a fixed mask threshold (in [0, B])
    u    : [n] uncertainty scores
    taus : ascending grid
    Returns one tau per test weight (or a single tau when unweighted).
    """
    taus = np.asarray(taus)
    L = loss[:, None] * (u[:, None] <= taus[None, :])     # [n, m], non-decreasing in tau
    n = len(loss)
    if w_cal is None:
        adj = n / (n + 1) * L.mean(0) + B / (n + 1)
        ok = adj <= alpha
        ok = np.logical_and.accumulate(ok)                # feasible prefix
        return np.array([taus[ok.nonzero()[0].max()] if ok.any() else -np.inf])
    w_cal = np.asarray(w_cal, float)
    w_test = np.atleast_1d(np.asarray(w_test, float))
    S = w_cal.sum()
    num = w_cal @ L
    denom = S + w_test
    adj = num[None, :] / denom[:, None] + (B * w_test / denom)[:, None]
    ok = np.logical_and.accumulate(adj <= alpha, axis=1)
    out = np.full(len(w_test), -np.inf)
    has = ok.any(1)
    last = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    out[has] = taus[last[has]]
    return out


def hb_pvalue(rhat: float, n: int, alpha: float) -> float:
    """Hoeffding–Bentkus p-value for H0: E[L] > alpha, losses in [0, 1]."""
    if n == 0:
        return 1.0
    a = min(rhat, alpha)

    def h(a, b):
        a = np.clip(a, 1e-12, 1 - 1e-12)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    p_h = np.exp(-n * h(a, alpha)) if rhat < alpha else 1.0
    p_b = np.e * binom.cdf(np.ceil(n * rhat), n, alpha)
    return float(min(p_h, p_b, 1.0))


def ltt_selective_tau(
    loss: np.ndarray,
    u: np.ndarray,
    alpha: float,
    delta: float,
    taus: np.ndarray,
    min_accept: int = 5,
) -> float:
    """Learn-then-Test, fixed sequence from the most conservative tau (accept few)
    to the most permissive one. Returns the largest certified tau (-inf if none)."""
    best = -np.inf
    for tau in np.sort(taus):
        acc = u <= tau
        n_acc = int(acc.sum())
        if n_acc < min_accept:
            continue
        p = hb_pvalue(float(loss[acc].mean()), n_acc, alpha)
        if p <= delta:
            best = tau
        else:
            break
    return best
