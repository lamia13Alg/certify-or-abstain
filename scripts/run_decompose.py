"""Why is a patient referred? Decomposition of the referral decision of the score-weighted rule.

For each setting (leave-one-out weights, C = 1): fraction certified by the full rule, fraction certified
when the target term p_{n+1} is removed (B = 0), median p_{n+1}, effective sample size of the
calibration weights, and the median grid thresholds lambda_lo and lambda_hi of the full rule.
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from run_revision3 import load  # noqa: E402
from run_revision2 import SETS  # noqa: E402
from scrc import weights, twosided  # noqa: E402


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
            m = F.shape[1]
            wc, wt = weights.loo_weights(x[cal, 0], x[tst, 0], C=1.0)
            wc = np.atleast_2d(wc)
            if wc.shape[0] == 1:
                wc = np.repeat(wc, len(tst), 0)
            r = {"tag": tag, "site": s}
            out = {}
            for B in (1.0, 0.0):
                lo, hi, ce = [], [], []
                for j in range(len(tst)):
                    l, c = twosided.certify_or_abstain(F[cal], FD[cal], wc[j], wt[j], af, ad, B=B)
                    h = twosided.weighted_crc_upper_index(np.maximum.accumulate(FD[cal], 1), wc[j], wt[j], ad, B)
                    lo.append(int(np.atleast_1d(l)[0])); hi.append(int(np.atleast_1d(h)[0])); ce.append(bool(np.atleast_1d(c)[0]))
                out[B] = (np.array(lo), np.array(hi), np.array(ce))
            lo, hi, ce = out[1.0]
            r["cert"] = ce.mean(); r["cert_noP"] = out[0.0][2].mean()
            p = wt / (wc.sum(1) + wt)
            r["p_test_med"] = float(np.median(p))
            r["ess"] = float(np.median(wc.sum(1) ** 2 / (wc ** 2).sum(1)))
            r["lam_lo_med"] = float(np.median(lo)) / (m - 1)
            r["lam_hi_med"] = float(np.nanmedian(np.where(hi < 0, np.nan, hi))) / (m - 1) if (hi >= 0).any() else np.nan
            r["hi_infeasible"] = float((hi < 0).mean())
            rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "decompose.csv"), index=False)


if __name__ == "__main__":
    main()
