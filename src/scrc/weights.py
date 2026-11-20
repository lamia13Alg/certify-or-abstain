"""Density-ratio estimation w(x) = dP_target / dP_source in embedding space.

Only UNLABELED target images are needed. A probabilistic classifier separates
source-calibration embeddings (label 0) from target embeddings (label 1);
by Bayes,  w(x) = (n_s / n_t) * c(x) / (1 - c(x)),  c(x) = P(target | x).

Cross-fitting: every patient (source or target) receives a weight predicted by a
classifier that did NOT see it during training, which avoids the optimistic
(over-separated) weights obtained in-sample.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _make_clf(n_components: int | None, C: float, seed: int):
    steps = [StandardScaler()]
    if n_components:
        steps.append(PCA(n_components=n_components, random_state=seed))
    steps.append(LogisticRegression(C=C, max_iter=5000))
    return make_pipeline(*steps)


def estimate_weights(
    emb_source: np.ndarray,
    emb_target: np.ndarray,
    n_folds: int = 5,
    n_components: int | None = 8,
    C: float = 0.1,
    clip_prob: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cross-fitted weights for source (calibration) and target patients.

    Returns (w_source, w_target, info) where info contains the cross-fitted AUC
    of the domain classifier (0.5 = no detectable shift).
    """
    from sklearn.metrics import roc_auc_score

    Xs, Xt = np.asarray(emb_source, float), np.asarray(emb_target, float)
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    n_comp = None
    if n_components:
        n_comp = int(min(n_components, X.shape[1], int(len(X) * (n_folds - 1) / n_folds) - 1))
    prob = np.zeros(len(X))
    k = int(min(n_folds, len(Xs), len(Xt)))
    skf = StratifiedKFold(n_splits=max(k, 2), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = _make_clf(n_comp, C, seed)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    prob = np.clip(prob, clip_prob, 1 - clip_prob)
    ratio = len(Xs) / len(Xt)
    w = ratio * prob / (1 - prob)
    info = {"domain_auc": float(roc_auc_score(y, prob)), "n_components": n_comp}
    return w[: len(Xs)], w[len(Xs):], info


def gaussian_true_weights(x: np.ndarray, mu_s: np.ndarray, mu_t: np.ndarray) -> np.ndarray:
    """Exact density ratio N(mu_t, I) / N(mu_s, I) (used for synthetic checks)."""
    x = np.atleast_2d(x)
    return np.exp(x @ (mu_t - mu_s) - 0.5 * (mu_t @ mu_t - mu_s @ mu_s))


def loo_weights(score_source, score_target, C: float = 1.0, clip_prob: float = 1e-3, n_iter: int = 50):
    """Deterministic leave-one-out cross-fitted weights for a one-dimensional score.

    For every patient (source or target), a logistic regression of domain (target = 1) on the
    standardised score is fitted on all OTHER patients (L2 penalty 1/(2C) on the slope, as in
    scikit-learn), and evaluated at that patient. All n fits are solved together by Newton's method.
    Unlike K-fold cross-fitting, the result does not depend on a random fold assignment.
    Returns (w_source, w_target).
    """
    xs = np.asarray(score_source, float).ravel()
    xt = np.asarray(score_target, float).ravel()
    x = np.r_[xs, xt]
    y = np.r_[np.zeros(len(xs)), np.ones(len(xt))]
    n = len(x)
    z = (x - x.mean()) / (x.std() + 1e-12)
    M = 1.0 - np.eye(n)                      # row i: training mask of the fit that leaves i out
    a = np.zeros(n)
    b = np.zeros(n)
    lam = 1.0 / C
    for _ in range(n_iter):
        eta = a[:, None] + b[:, None] * z[None, :]
        p = 1.0 / (1.0 + np.exp(-eta))
        r = (p - y[None, :]) * M
        wgt = p * (1 - p) * M
        ga = r.sum(1)
        gb = (r * z[None, :]).sum(1) + lam * b
        haa = wgt.sum(1) + 1e-9
        hab = (wgt * z[None, :]).sum(1)
        hbb = (wgt * z[None, :] ** 2).sum(1) + lam
        det = haa * hbb - hab ** 2
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da
        b -= db
        if max(np.abs(da).max(), np.abs(db).max()) < 1e-10:
            break
    prob = 1.0 / (1.0 + np.exp(-(a + b * z)))
    prob = np.clip(prob, clip_prob, 1 - clip_prob)
    # each left-out fit sees n_s or n_s - 1 sources: use the class ratio of its own training set
    ns = len(xs) - (y == 0)
    nt = len(xt) - (y == 1)
    w = (ns / nt) * prob / (1 - prob)
    return w[: len(xs)], w[len(xs):]
