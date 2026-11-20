"""Synthetic study of weight mis-estimation and of violations of Assumption 1 (Online Resource 2, S6).

Score U: source N(0, 1), target N(s, 1); true weight w(u) = exp(s u - s^2 / 2).
Patient hardness h = gamma U + d_h + noise (d_h = 0 in the source); loss L(lambda) = (1 - lambda)^exp(-h),
non-increasing in lambda (an FNR-like curve). With d_h = 0, Assumption 1 holds exactly.
Rules (weighted CRC at level alpha, B = 1):
  * standard (w = 1)          * true weights          * estimated weights (cross-fitted logistic, n_t unlabeled)
  * tempered weights w^tau: tau < 1 under-corrects, tau > 1 over-corrects; the reweighted source score is
    N(tau s, 1), so the theorem's slack is TV(N(s,1), N(tau s,1)) = 2 Phi(|1 - tau| s / 2) - 1
  * violation of Assumption 1 (d_h > 0), true weights; slack delta = sup_{u, lambda} E_Q[L|u] - E_P[L|u]

    python scripts/exp_misweight.py --reps 400 --out results/misweight
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import crc, weights  # noqa: E402

LAM = np.linspace(0, 1, 401)
GAMMA, NOISE = 0.8, 0.3


def draw(n, mu, d_h, rng):
    u = rng.normal(mu, 1.0, n)
    h = GAMMA * u + d_h + NOISE * rng.normal(size=n)
    return u, (1.0 - LAM[None, :]) ** np.exp(-h)[:, None]


def sup_delta(d_h, s, rng):
    """sup over u in the target bulk and lambda of the conditional excess loss."""
    if d_h == 0:
        return 0.0
    eps = NOISE * rng.normal(size=4000)
    best = 0.0
    for u in np.linspace(s - 2.5, s + 2.5, 41):
        lp = ((1 - LAM[None, :]) ** np.exp(-(GAMMA * u + eps))[:, None]).mean(0)
        lq = ((1 - LAM[None, :]) ** np.exp(-(GAMMA * u + d_h + eps))[:, None]).mean(0)
        best = max(best, float((lq - lp).max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--n_cal", type=int, default=60)
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--out", default="results/misweight")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for s in (0.5, 1.0, 1.5):
        w_true = lambda u: np.exp(s * u - s * s / 2)  # noqa: E731
        configs = [("standard (w = 1)", 0.0, None, 0.0), ("true weights", 1.0, None, 0.0)]
        configs += [(f"tempered, tau = {t}", t, None, 0.0) for t in (0.5, 1.5, 2.0)]
        configs += [(f"estimated, n_t = {n}", None, n, 0.0) for n in (15, 40, 150)]
        configs += [(f"violation, d_h = {d}", 1.0, None, d) for d in (0.1, 0.3)]
        for name, tau, n_t, d_h in configs:
            risks, trivial = [], []
            for r in range(a.reps):
                rng = np.random.default_rng(10_000 * r + int(100 * s))
                uc, Lc = draw(a.n_cal, 0.0, 0.0, rng)
                ue, Le = draw(a.n_eval, s, d_h, rng)
                if n_t is None:
                    wc, we = w_true(uc) ** tau, w_true(ue) ** tau
                else:
                    ut, _ = draw(n_t, s, 0.0, rng)
                    m = LogisticRegressionWeights(uc, ut, rng)
                    wc, we = m(uc), m(ue)
                idx = crc.weighted_crc_index(Lc, wc, we, a.alpha)
                idx = np.broadcast_to(np.asarray(idx), (a.n_eval,))
                risks.append(float(Le[np.arange(a.n_eval), idx].mean()))
                trivial.append(float(np.mean(idx == len(LAM) - 1)))
            if tau is not None and n_t is None and d_h == 0:
                slack = 2 * norm.cdf(abs(1 - tau) * s / 2) - 1
            elif d_h > 0:
                slack = sup_delta(d_h, s, np.random.default_rng(0))
            else:
                slack = np.nan
            rr = np.array(risks)
            rows.append({"shift": s, "rule": name, "risk": rr.mean(),
                         "risk_se": rr.std(ddof=1) / np.sqrt(len(rr)),
                         "bound": a.alpha + slack if np.isfinite(slack) else np.nan,
                         "frac_full_mask": float(np.mean(trivial))})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "misweight.csv"), index=False)


def LogisticRegressionWeights(uc, ut, rng):
    """Weights fitted on the source calibration scores and n_t separate unlabeled target scores."""
    from sklearn.linear_model import LogisticRegression
    X = np.r_[uc, ut][:, None]
    y = np.r_[np.zeros(len(uc)), np.ones(len(ut))]
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
    r = len(uc) / len(ut)

    def w(u):
        p = np.clip(clf.predict_proba(np.asarray(u)[:, None])[:, 1], 1e-3, 1 - 1e-3)
        return r * p / (1 - p)
    return w


if __name__ == "__main__":
    main()
