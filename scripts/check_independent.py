"""Independent re-implementation of standard CRC and of the leave-one-out score-weighted rule.

Uses only scikit-learn and numpy (not scrc.weights / scrc.twosided): one scikit-learn logistic fit per
left-out patient, and the two thresholds computed directly from their definitions (Eqs. 3-4). Compares
fraction certified, joint FNR and joint FDR with results/v5/rev_*.csv, and checks that the validation,
calibration and test patients are disjoint in every setting.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from run_revision2 import SETS, load  # noqa: E402


def rule(F, FD, cal, tst, wc, wt, af, ad):
    env = np.maximum.accumulate(FD[cal], 1)
    ce, fn, fd = [], [], []
    for j, t in enumerate(tst):
        den = wc.sum() + wt[j]
        rl = wc @ F[cal] / den + wt[j] / den
        rh = wc @ env / den + wt[j] / den
        f_lo, f_hi = np.where(rl <= af)[0], np.where(rh <= ad)[0]
        lo = f_lo.min() if len(f_lo) else F.shape[1] - 1
        hi = f_hi.max() if len(f_hi) else -1
        c = lo <= hi
        ce.append(c); fn.append(F[t, lo] * c); fd.append(FD[t, lo] * c)
    return np.mean(ce), np.mean(fn), np.mean(fd)


def loo(xc, xt):
    X = np.r_[xc, xt]; X = (X - X.mean()) / X.std()
    y = np.r_[np.zeros(len(xc)), np.ones(len(xt))]
    w = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        p = LogisticRegression(C=1.0, max_iter=5000).fit(X[m][:, None], y[m]).predict_proba(X[i:i + 1, None])[0, 1]
        p = np.clip(p, 1e-3, 1 - 1e-3)
        w[i] = (y[m] == 0).sum() / (y[m] == 1).sum() * p / (1 - p)
    return w[:len(xc)], w[len(xc):]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prostate", default="data/prostate_summaries/strong")
    ap.add_argument("--mnms", required=True)
    ap.add_argument("--results", default="results/v5")
    a = ap.parse_args()
    roots = {"PROSTATE": a.prostate, "MNMS": a.mnms}
    worst = 0.0
    for tag, root, sites, ph, af, ad in SETS:
        ref = pd.read_csv(os.path.join(a.results, f"rev_{tag}.csv"))
        for s in sites:
            S = load(roots[root], s, ph)
            F, FD, x, val, cal, tst = S["F"], S["FD"], S["x"][:, 0], S["val"], S["cal"], S["tst"]
            assert not (set(val) & set(cal) or set(val) & set(tst) or set(cal) & set(tst)), s
            for meth, (wc, wt) in (("standard", (np.ones(len(cal)), np.ones(len(tst)))), ("ours", loo(x[cal], x[tst]))):
                got = rule(F, FD, cal, tst, wc, wt, af, ad)
                r = ref[(ref.site == s) & (ref.method == meth)].iloc[0]
                d = max(abs(got[0] - r.cert), abs(got[1] - r.fnr), abs(got[2] - r.fdr))
                worst = max(worst, d)
                print(f"{tag:2s} {s:5s} {meth:8s} cert {got[0]:.3f}/{r.cert:.3f}  fnr {got[1]:.4f}/{r.fnr:.4f}  fdr {got[2]:.4f}/{r.fdr:.4f}")
    print(f"largest absolute difference: {worst:.1e}")


if __name__ == "__main__":
    main()
