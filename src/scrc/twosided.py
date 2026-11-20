r"""Two-sided certify-or-abstain risk control (FNR *and* FDR), weighted or not.

Per patient i and lambda on the ascending grid (larger lambda = larger mask):
    FNR_i(lambda)   non-increasing in lambda
    FDR*_i(lambda)  = max_{lambda' <= lambda} FDR_i(lambda')   (monotone envelope >= FDR)

Weighted CRC gives, for a test patient with weight w(x):
    lam_lo(x) = inf{ lambda : weighted-adjusted FNR  <= alpha_fnr }       (mask large enough)
    lam_hi(x) = sup{ lambda : weighted-adjusted FDR* <= alpha_fdr }       (mask small enough)
Decision:
    certify with lambda*(x) = lam_lo(x)   if lam_lo(x) <= lam_hi(x)
    abstain (refer to the radiologist)    otherwise.

Guarantee (abstention costs 0 in the joint risk):
    E[ FNR(lambda*)  1{certified} ] <= E[ FNR(lam_lo) ]   <= alpha_fnr
    E[ FDR(lambda*)  1{certified} ] <= E[ FDR*(lam_hi) ]  <= alpha_fdr
because on the certified event lambda* = lam_lo <= lam_hi and FDR* is monotone.
Both statements hold under covariate shift with the true weights (weighted CRC),
and degrade by B * TV(true, estimated reweighting) with estimated weights.
"""
from __future__ import annotations

import numpy as np

from .crc import weighted_crc_index


def fdr_envelope(fdr: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(fdr, axis=-1)


def weighted_crc_upper_index(losses_nondecreasing, w_cal, w_test, alpha, B=1.0):
    """Largest grid index whose weighted-adjusted risk of a NON-DECREASING loss <= alpha
    (-1 if none). Implemented by flipping the grid."""
    flipped = losses_nondecreasing[:, ::-1]
    idx = weighted_crc_index(flipped, w_cal, w_test, alpha, B)   # smallest feasible in flipped grid
    m = losses_nondecreasing.shape[1]
    up = m - 1 - np.asarray(idx)
    # weighted_crc_index returns the last index when nothing is feasible: detect it
    w_cal = np.asarray(w_cal, float)
    w_test = np.atleast_1d(np.asarray(w_test, float))
    num = w_cal @ flipped
    denom = w_cal.sum() + w_test
    adj_at = num[np.asarray(idx)] / denom + B * w_test / denom
    return np.where(adj_at <= alpha, up, -1)


def certify_or_abstain(fnr_cal, fdr_cal, w_cal, w_test, alpha_fnr, alpha_fdr, B=1.0):
    """Returns (lambda index per test patient, certified boolean per test patient)."""
    env = fdr_envelope(fdr_cal)
    lo = np.asarray(weighted_crc_index(fnr_cal, w_cal, w_test, alpha_fnr, B))
    hi = weighted_crc_upper_index(env, w_cal, w_test, alpha_fdr, B)
    cert = lo <= hi
    return lo, cert


def mixture_weights(r: np.ndarray, n_source: int, k_target: int) -> np.ndarray:
    """Weights for a calibration pool made of n source patients (density p) and k labeled
    target patients (density q), with r = q/p:  w = q/m,  m = (n p + k q)/(n + k).
    Up to a constant: w(x) = r(x) / (n + k r(x)). Applied to every calibration and test point.
    (Exact when the pool is an i.i.d. sample of the mixture m; with fixed counts n, k this is
    the standard approximation — to be stated as such in the paper.)"""
    r = np.asarray(r, float)
    return r / (n_source + k_target * r)
