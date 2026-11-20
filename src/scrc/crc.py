r"""Conformal risk control (CRC) at the patient level, with and without shift.

Notation
--------
Calibration patients i = 1..n (source centres), test patient n+1 (target centre).
L_i(lambda) in [0, B], non-increasing in lambda (e.g. patient-level FNR).

Standard CRC  (Angelopoulos, Bates, Fisch, Lei, Schuster, 2022/2024)
    lambda_hat = inf{ lambda : n/(n+1) * Rhat_n(lambda) + B/(n+1) <= alpha }
    guarantees  E[ L_{n+1}(lambda_hat) ] <= alpha  under exchangeability.

Weighted CRC under covariate shift
    Assume P_target(y | x) = P_source(y | x) and w(x) = dP_target/dP_source (x),
    where x is a patient-level covariate (here: the embedding of the volume).
    Normalised weights
        p_i(x) = w(X_i) / (sum_j w(X_j) + w(x)),   p_{n+1}(x) = w(x) / (...)
    and
        lambda_hat(x) = inf{ lambda : sum_i p_i(x) L_i(lambda) + p_{n+1}(x) B <= alpha }.
    With the true w this gives E_target[ L_{n+1}(lambda_hat(X_{n+1})) ] <= alpha
    (weighted exchangeability argument of Tibshirani et al. 2019 applied to the
    CRC fixed point; the lambda depends on the test patient through w(x)).

    With ESTIMATED weights w_hat the guarantee degrades gracefully:
        E_target[L] <= alpha + B * TV(P_target, Q_{w_hat}),
    where Q_{w_hat} is the source distribution reweighted by w_hat
    (cf. Lei & Candès 2021, Thm 3, for the coverage case). Truncating the
    weights trades a small bias for a large variance reduction.
"""
from __future__ import annotations

import numpy as np


def _first_feasible(adj_risk: np.ndarray, alpha: float) -> np.ndarray:
    """Index of the smallest lambda with adj_risk <= alpha (last index if none).
    adj_risk: [..., m] (non-increasing in the last axis up to noise)."""
    ok = adj_risk <= alpha
    # conservative monotone version: lambda is feasible only if every larger lambda
    # is feasible too (identical to the plain rule when the risk is monotone)
    ok = np.flip(np.logical_and.accumulate(np.flip(ok, -1), axis=-1), -1)
    m = adj_risk.shape[-1]
    idx = np.where(ok.any(-1), ok.argmax(-1), m - 1)
    return idx


def crc_index(losses: np.ndarray, alpha: float, B: float = 1.0) -> int:
    """Standard CRC. losses: [n, m] on an ascending lambda grid. Returns grid index."""
    n = losses.shape[0]
    adj = n / (n + 1) * losses.mean(0) + B / (n + 1)
    return int(_first_feasible(adj, alpha))


def weighted_crc_index(
    losses: np.ndarray,
    w_cal: np.ndarray,
    w_test: np.ndarray,
    alpha: float,
    B: float = 1.0,
) -> np.ndarray:
    """Weighted CRC, one lambda per test patient.

    losses : [n, m]   calibration loss curves (source)
    w_cal  : [n]      weights of calibration patients
    w_test : [k]      weights of test patients
    returns : [k] grid indices
    """
    w_cal = np.asarray(w_cal, np.float64)
    w_test = np.atleast_1d(np.asarray(w_test, np.float64))
    S = w_cal.sum()
    num = w_cal @ losses                      # [m]
    denom = S + w_test                        # [k]
    adj = num[None, :] / denom[:, None] + (B * w_test / denom)[:, None]
    return _first_feasible(adj, alpha)


def truncate_weights(w: np.ndarray, max_w: float | None = None, q: float | None = None) -> np.ndarray:
    """Clip weights at an absolute value or at a quantile of the calibration weights."""
    w = np.asarray(w, np.float64).copy()
    if q is not None:
        max_w = np.quantile(w, q)
    if max_w is not None:
        w = np.minimum(w, max_w)
    return w


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, np.float64)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))


def realized_risk(test_losses: np.ndarray, idx) -> float:
    """Mean test loss given one index (int) or one index per test patient."""
    idx = np.broadcast_to(np.asarray(idx), (test_losses.shape[0],))
    return float(test_losses[np.arange(test_losses.shape[0]), idx].mean())
