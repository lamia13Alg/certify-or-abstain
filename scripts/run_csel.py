"""Label-free choice of the regularisation C of the domain classifier.

For each setting, C in {0.01, 0.1, 1, 10, 100} is chosen to maximise the
leave-one-out log-likelihood of the domain classifier (source vs target score),
which uses no segmentation label; the rule is then rerun with that C.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from run_revision3 import load, SETS, certify, metrics  # noqa: E402
from scrc import weights  # noqa: E402

GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def loo_loglik(xs, xt, C):
    X = np.r_[xs, xt]; X = ((X - X.mean()) / X.std())[:, None]
    y = np.r_[np.zeros(len(xs)), np.ones(len(xt))]
    ll = 0.0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        p = LogisticRegression(C=C, max_iter=5000).fit(X[m], y[m]).predict_proba(X[i:i + 1])[0, 1]
        p = np.clip(p, 1e-3, 1 - 1e-3)
        ll += np.log(p if y[i] else 1 - p)
    return ll


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prostate", default="data/prostate_summaries/strong")
    ap.add_argument("--mnms", required=True)
    ap.add_argument("--out", default="results/v5")
    a = ap.parse_args()
    roots = {"PROSTATE": a.prostate, "MNMS": a.mnms}
    rows = []
    for tag, root, sites, ph, af, ad in SETS:
        for s in sites:
            S = load(roots[root], s, ph)
            F, FD, x, cal, tst = S["F"], S["FD"], S["x"], S["cal"], S["tst"]
            lls = [loo_loglik(x[cal, 0], x[tst, 0], C) for C in GRID]
            C = GRID[int(np.argmax(lls))]
            w1, w2 = weights.loo_weights(x[cal, 0], x[tst, 0], C=C)
            m = metrics(F, FD, tst, *certify(F, FD, cal, tst, w1, w2, af, ad))
            rows.append({"tag": tag, "site": s, "C": C, "cert": m["cert"], "fnr": m["fnr"], "fdr": m["fdr"]})
            print(rows[-1], flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "csel.csv"), index=False)


if __name__ == "__main__":
    main()
