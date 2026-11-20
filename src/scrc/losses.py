"""Patient-level loss curves L_i(lambda) from the histogram summaries.

Convention
----------
lambda in [0, 1] on the grid lambda_k = k / n_bins (k = 0..n_bins), ascending.
Threshold t_k = 1 - lambda_k, predicted mask S_lambda = {v : s_v >= t_k}.
Larger lambda  ->  larger mask  ->  FNR non-increasing (required by CRC).

lambda = 0 : empty mask (FNR = 1);  lambda = 1 : every voxel (FNR = 0).
"""
from __future__ import annotations

import numpy as np


def lambda_grid(n_bins: int) -> np.ndarray:
    return np.arange(n_bins + 1) / n_bins


def _rev_cum(h: np.ndarray) -> np.ndarray:
    """counts[..., k] = sum of the last k bins  (shape [..., n_bins + 1])."""
    c = np.cumsum(h[..., ::-1], axis=-1)
    zero = np.zeros(h.shape[:-1] + (1,), dtype=c.dtype)
    return np.concatenate([zero, c], axis=-1)


def curves(hist_all: np.ndarray, hist_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (TP(lambda), |S_lambda|) for histograms of shape [..., n_bins]."""
    return _rev_cum(hist_pos), _rev_cum(hist_all)


def fnr_curve(hist_all, hist_pos, n_pos) -> np.ndarray:
    """False-negative rate (1 - recall) of the patient, shape [..., n_bins + 1].
    Non-increasing in lambda, bounded in [0, 1]. Patients with no foreground get 0."""
    tp, _ = curves(hist_all, hist_pos)
    n_pos = np.asarray(n_pos, dtype=np.float64)
    denom = np.maximum(n_pos, 1.0)[..., None]
    out = 1.0 - tp / denom
    out = np.where((n_pos <= 0)[..., None], 0.0, out)
    return np.clip(out, 0.0, 1.0)


def fdr_curve(hist_all, hist_pos) -> np.ndarray:
    tp, sz = curves(hist_all, hist_pos)
    return np.where(sz > 0, 1.0 - tp / np.maximum(sz, 1), 0.0)


def dice_curve(hist_all, hist_pos, n_pos) -> np.ndarray:
    tp, sz = curves(hist_all, hist_pos)
    n_pos = np.asarray(n_pos, dtype=np.float64)[..., None]
    return np.where(sz + n_pos > 0, 2 * tp / np.maximum(sz + n_pos, 1), 1.0)


def volume_curve(hist_all, voxel_volume_ml) -> np.ndarray:
    """Predicted volume (ml) as a function of lambda."""
    sz = _rev_cum(hist_all)
    return sz * np.asarray(voxel_volume_ml, dtype=np.float64)[..., None]


def loss_matrix(data: dict, beta_index: int = 0, loss: str = "fnr") -> np.ndarray:
    """Loss matrix [n_cases, n_lambda] for one score family (beta)."""
    ha = data["hist_all"][:, beta_index]
    hp = data["hist_pos"][:, beta_index]
    if loss == "fnr":
        return fnr_curve(ha, hp, data["n_pos"])
    if loss == "fdr":
        return fdr_curve(ha, hp)
    if loss == "1-dice":
        return 1.0 - dice_curve(ha, hp, data["n_pos"])
    raise ValueError(loss)
