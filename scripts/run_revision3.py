"""Third batch of review analyses (CPU only).

  tv        : estimate of TV(target law of U, source law reweighted by w_hat), histogram on 8 pooled-quantile bins,
              point estimate and bootstrap 95% upper bound, for cross-fitted weights
  volume    : Spearman correlation between U and the predicted volume (source calibration / target)
  C         : sensitivity of our rule to the classifier regularisation C in {0.1, 1, 10}
  inductive : weights fitted on the calibration patients and the first half of the target patients
              (in random arrival order), applied to the second half, which is certified patient by patient
  power     : target-sample size needed for a one-sided 95% upper bound below the target, given the
              observed joint risk and per-patient standard deviation of our rule

    python scripts/run_revision3.py --mnms runs_mnms --out results/revision3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from run_revision import certify, metrics  # noqa: E402
from run_revision2 import SETS  # noqa: E402
from scrc import losses, weights  # noqa: E402


def load(root, site, phase):
    z = np.load(os.path.join(root, site, "summaries.npz"))
    D = {k: z[k] for k in z.files}
    role = D["role"].copy()
    if phase != "all":
        role[D["phase"] != phase] = "excluded"
    ha, hp = D["hist_all"][:, 0], D["hist_pos"][:, 0]
    names = list(D["unc_names"])
    return dict(cal=np.where(role == "cal")[0], tst=np.where(role == "test")[0],
                F=losses.fnr_curve(ha, hp, D["n_pos"]), FD=losses.fdr_curve(ha, hp),
                x=np.log(D["unc"][:, names.index("entropy_pred")] + 1e-6)[:, None],
                vol=D["unc"][:, names.index("vol_mean_ml")])


def tv_hat(xc, wc, xt, nb=8):
    edges = np.quantile(np.r_[xc, xt], np.linspace(0, 1, nb + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pc = np.array([wc[(xc > edges[k]) & (xc <= edges[k + 1])].sum() for k in range(nb)]) / wc.sum()
    pt = np.array([((xt > edges[k]) & (xt <= edges[k + 1])).mean() for k in range(nb)])
    return 0.5 * np.abs(pc - pt).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prostate", default="data/prostate_summaries/strong")
    ap.add_argument("--mnms", required=True)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--out", default="results/revision3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    roots = {"PROSTATE": a.prostate, "MNMS": a.mnms}
    rng = np.random.default_rng(0)
    rows = []
    for tag, root, sites, ph, af, ad in SETS:
        for s in sites:
            S = load(roots[root], s, ph)
            F, FD, x, cal, tst = S["F"], S["FD"], S["x"], S["cal"], S["tst"]
            row = {"tag": tag, "site": s, "n_cal": len(cal), "n_test": len(tst), "af": af, "ad": ad}
            # TV
            wc, wt = weights.loo_weights(x[cal, 0], x[tst, 0], C=1.0)
            row["tv"] = tv_hat(x[cal, 0], wc, x[tst, 0])
            row["tv_unweighted"] = tv_hat(x[cal, 0], np.ones(len(cal)), x[tst, 0])
            bs = []
            for _ in range(a.B):
                cb, tb = rng.choice(cal, len(cal)), rng.choice(tst, len(tst))
                w1, _ = weights.loo_weights(x[cb, 0], x[tb, 0], C=1.0)
                bs.append(tv_hat(x[cb, 0], w1, x[tb, 0]))
            row["tv_ub95"] = float(np.quantile(bs, 0.95))
            # U vs volume
            row["rho_cal"] = spearmanr(x[cal, 0], S["vol"][cal]).correlation
            row["rho_test"] = spearmanr(x[tst, 0], S["vol"][tst]).correlation
            # C sensitivity
            for C in (0.1, 1.0, 10.0):
                w1, w2 = weights.loo_weights(x[cal, 0], x[tst, 0], C=C)
                m = metrics(F, FD, tst, *certify(F, FD, cal, tst, w1, w2, af, ad))
                row[f"C{C:g}_cert"], row[f"C{C:g}_fnr"], row[f"C{C:g}_fdr"] = m["cert"], m["fnr"], m["fdr"]
            # inductive
            ind = []
            for r in range(a.R):
                perm = rng.permutation(tst)
                early, late = perm[: len(perm) // 2], perm[len(perm) // 2:]
                X = np.r_[x[cal, 0], x[early, 0]][:, None]
                y = np.r_[np.zeros(len(cal)), np.ones(len(early))]
                clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
                ratio = len(cal) / len(early)
                w = lambda ids: ratio * (lambda p: p / (1 - p))(np.clip(clf.predict_proba(x[ids])[:, 1], 1e-3, 1 - 1e-3))  # noqa
                wcal, _ = weights.loo_weights(x[cal, 0], x[early, 0], C=1.0)
                ind.append(metrics(F, FD, late, *certify(F, FD, cal, late, wcal, w(late), af, ad)))
            ind = pd.DataFrame(ind)
            for k in ("cert", "fnr", "fdr"):
                row[f"ind_{k}"] = ind[k].mean()
            # power: n needed for a one-sided 95% UB of the joint FNR below alpha
            idx, cert = certify(F, FD, cal, tst, wc, wt, af, ad)
            lfn = F[tst, idx] * cert
            row["risk"], row["sd"] = lfn.mean(), lfn.std(ddof=1)
            gap = af - lfn.mean()
            row["n_needed"] = float(np.ceil((1.645 * lfn.std(ddof=1) / gap) ** 2)) if gap > 0 and lfn.std() > 0 else np.nan
            rows.append(row)
            print("done", tag, s, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev3.csv"), index=False)


if __name__ == "__main__":
    main()
