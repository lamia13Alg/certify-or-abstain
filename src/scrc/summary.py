"""Patient-level summaries of a 3D segmentation output.

Storing full probability maps for every patient, every fold and every stochastic
sample is too heavy for Kaggle. Every quantity needed by the conformal layer
(FNR, FDR, Dice, predicted volume, for ANY threshold) can be recovered exactly
from two histograms of the voxel scores:

    hist_all[b] = number of voxels whose score falls in bin b
    hist_pos[b] = number of *foreground* voxels whose score falls in bin b

With thresholds on the bin edges, the predicted mask at threshold t is
{v : score_v >= t}, so TP(t) and |S_t| are reverse cumulative sums.

Scores
------
Given K stochastic probability maps (MC-dropout, test-time augmentation,
prompt perturbations of a promptable foundation model, ...), we build a family
of penalised scores  s_beta = clip(mean - beta * std, 0, 1).  beta = 0 is the
plain mean probability. Each beta gives its own pair of histograms.
"""
from __future__ import annotations

import numpy as np

N_BINS_DEFAULT = 1000


def bin_edges(n_bins: int = N_BINS_DEFAULT) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _hist(values: np.ndarray, n_bins: int) -> np.ndarray:
    # bin index = floor(v * n_bins), value 1.0 goes into the last bin
    idx = np.minimum((values * n_bins).astype(np.int64), n_bins - 1)
    return np.bincount(idx.ravel(), minlength=n_bins).astype(np.int64)


def entropy_binary(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def summarize_case(
    prob_samples: np.ndarray,
    label: np.ndarray | None,
    voxel_volume_ml: float,
    betas=(0.0, 0.5, 1.0, 2.0),
    n_bins: int = N_BINS_DEFAULT,
    roi: np.ndarray | None = None,
) -> dict:
    """Summarise one patient.

    Parameters
    ----------
    prob_samples : array (K, D, H, W) of foreground probabilities in [0, 1].
        K = 1 is allowed (deterministic model, std = 0).
    label : binary array (D, H, W) or None for unlabeled cases.
    voxel_volume_ml : volume of one voxel in millilitres.
    betas : penalisation factors for s = mean - beta * std.
    roi : optional boolean mask restricting the voxels that are counted
        (e.g. the FOV after cropping). Voxels outside are ignored.

    Returns
    -------
    dict with
        hist_all  (n_betas, n_bins) int64
        hist_pos  (n_betas, n_bins) int64   (zeros if label is None)
        n_pos     int    number of foreground voxels (-1 if unlabeled)
        vol_true_ml float (nan if unlabeled)
        vol_samples_ml (K,) predicted volume of each sample at threshold 0.5
        unc : dict of patient-level uncertainty features
    """
    ps = np.asarray(prob_samples, dtype=np.float32)
    if ps.ndim == 3:
        ps = ps[None]
    mean = ps.mean(0)
    std = ps.std(0) if ps.shape[0] > 1 else np.zeros_like(mean)

    if roi is not None:
        roi = roi.astype(bool)
        mean_r, std_r = mean[roi], std[roi]
        lab_r = label[roi].astype(bool) if label is not None else None
        ps_r = ps[:, roi]
    else:
        mean_r, std_r = mean.ravel(), std.ravel()
        lab_r = label.astype(bool).ravel() if label is not None else None
        ps_r = ps.reshape(ps.shape[0], -1)

    betas = tuple(float(b) for b in betas)
    hist_all = np.zeros((len(betas), n_bins), np.int64)
    hist_pos = np.zeros((len(betas), n_bins), np.int64)
    for j, b in enumerate(betas):
        s = np.clip(mean_r - b * std_r, 0.0, 1.0)
        hist_all[j] = _hist(s, n_bins)
        if lab_r is not None:
            hist_pos[j] = _hist(s[lab_r], n_bins)

    vol_samples = (ps_r >= 0.5).sum(1) * voxel_volume_ml
    pred_half = mean_r >= 0.5
    ent = entropy_binary(mean_r)
    # uncertainty features (patient level) used by the selective layer
    soft_region = mean_r >= 0.1
    unc = {
        "vol_mean_ml": float(vol_samples.mean()),
        "vol_cv": float(vol_samples.std() / (vol_samples.mean() + 1e-8)),
        "entropy_pred": float(ent[pred_half].mean()) if pred_half.any() else 1.0,
        "entropy_soft": float(ent[soft_region].mean()) if soft_region.any() else 1.0,
        "std_pred": float(std_r[pred_half].mean()) if pred_half.any() else 1.0,
        "boundary_ratio": float(((mean_r > 0.2) & (mean_r < 0.8)).sum() / (pred_half.sum() + 1.0)),
        "max_prob": float(mean_r.max()) if mean_r.size else 0.0,
    }
    return {
        "betas": np.array(betas),
        "hist_all": hist_all,
        "hist_pos": hist_pos,
        "n_pos": int(lab_r.sum()) if lab_r is not None else -1,
        "vol_true_ml": float(lab_r.sum() * voxel_volume_ml) if lab_r is not None else float("nan"),
        "voxel_volume_ml": float(voxel_volume_ml),
        "vol_samples_ml": vol_samples.astype(np.float64),
        "unc": unc,
    }


def stack_summaries(summaries: list[dict], meta: list[dict] | None = None) -> dict:
    """Stack per-case summaries into arrays (n_cases first)."""
    out = {
        "betas": summaries[0]["betas"],
        "hist_all": np.stack([s["hist_all"] for s in summaries]),
        "hist_pos": np.stack([s["hist_pos"] for s in summaries]),
        "n_pos": np.array([s["n_pos"] for s in summaries]),
        "vol_true_ml": np.array([s["vol_true_ml"] for s in summaries]),
        "voxel_volume_ml": np.array([s["voxel_volume_ml"] for s in summaries]),
        "vol_samples_ml": np.stack([s["vol_samples_ml"] for s in summaries]),
    }
    keys = list(summaries[0]["unc"].keys())
    out["unc_names"] = np.array(keys)
    out["unc"] = np.array([[s["unc"][k] for k in keys] for s in summaries])
    if meta is not None:
        for k in meta[0]:
            out[k] = np.array([m[k] for m in meta])
    return out
