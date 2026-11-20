"""Is a per-centre excess over alpha real, or sampling noise?

The guarantee is on the expectation over calibration AND test patients. For each target
centre we resample both (patients with replacement), re-estimate the score weights,
re-run the two-sided procedure and record the joint FNR / FDR on the resampled test set.
Output: point estimate on the observed data, bootstrap mean, 2.5-97.5 % interval, and the
fraction of resamples above alpha.

    python scripts/run_bootstrap.py --runs "runs/*" --phase ED --alpha_fnr 0.15 --alpha_fdr 0.25
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import losses, twosided, weights  # noqa: E402


def evaluate(F, FD, x, cal, tst, af, ad, C, seed):
    out = {}
    m = F.shape[1]
    r_c, r_t, _ = weights.estimate_weights(x[cal], x[tst], n_components=None, C=C, seed=seed)
    for name, (wc, wt) in {"standard": (np.ones(len(cal)), np.ones(len(tst))),
                           "score-weighted": (r_c, r_t)}.items():
        idx, cert = twosided.certify_or_abstain(F[cal], FD[cal], wc, wt, af, ad)
        idx = np.clip(np.broadcast_to(idx, (len(tst),)), 0, m - 1)
        cert = np.broadcast_to(cert, (len(tst),))
        out[name] = (float(np.mean(F[tst, idx] * cert)), float(np.mean(FD[tst, idx] * cert)), float(cert.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--alpha_fnr", type=float, default=0.15)
    ap.add_argument("--alpha_fdr", type=float, default=0.25)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--C_score", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for p in sorted({q for pat in a.runs for q in glob.glob(pat)}):
        f = os.path.join(p, "summaries.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        D = {k: z[k] for k in z.files}
        role = D["role"].copy()
        if a.phase != "all" and "phase" in D:
            role[D["phase"] != a.phase] = "excluded"
        cal, tst = np.where(role == "cal")[0], np.where(role == "test")[0]
        ha, hp = D["hist_all"][:, 0], D["hist_pos"][:, 0]
        F = losses.fnr_curve(ha, hp, D["n_pos"])
        FD = losses.fdr_curve(ha, hp)
        names = list(D["unc_names"])
        x = np.log(D["unc"][:, names.index("entropy_pred")] + 1e-6)[:, None]
        point = evaluate(F, FD, x, cal, tst, a.alpha_fnr, a.alpha_fdr, a.C_score, 0)
        boot = {k: [] for k in point}
        for b in range(a.B):
            cb, tb = rng.choice(cal, len(cal)), rng.choice(tst, len(tst))
            res = evaluate(F, FD, x, cb, tb, a.alpha_fnr, a.alpha_fdr, a.C_score, b)
            for k, v in res.items():
                boot[k].append(v)
        for k in point:
            arr = np.array(boot[k])
            rows.append({"site": str(D["target_site"]), "method": k, "n_cal": len(cal), "n_test": len(tst),
                         "fnr": point[k][0], "fnr_boot_mean": arr[:, 0].mean(),
                         "fnr_lo": np.quantile(arr[:, 0], 0.025), "fnr_hi": np.quantile(arr[:, 0], 0.975),
                         "p(fnr>alpha)": float(np.mean(arr[:, 0] > a.alpha_fnr)),
                         "fdr": point[k][1], "fdr_boot_mean": arr[:, 1].mean(),
                         "fdr_hi": np.quantile(arr[:, 1], 0.975), "cert": point[k][2]})
        print("done", D["target_site"], flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
