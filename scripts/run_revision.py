"""Analyses requested in review (CPU only, from summaries.npz).

Per centre setting, with a bootstrap over calibration and target patients:
  * joint risk (refusal = 0 loss) AND conditional risk among certified patients, fraction certified
  * one-sided 95% upper bounds and Bonferroni (16 settings) bounds
  * 'split' variant of our rule whose weights satisfy the theorem exactly: the score classifier is fitted
    on the source VALIDATION patients and one half of the target patients, then applied to the
    calibration patients and the OTHER half (and vice versa)
  * retrospective check of Assumption 1 with target labels: conditional shift
        delta = mean_target[ L(0.5) - f_source(U) ],  f_source = OLS fit of L(0.5) on log U (source calibration)
  * risk-coverage sweeps: ours over target levels, uncertainty gate over quantiles

    python scripts/run_revision.py --runs "runs/*" --phase ED --alpha_fnr 0.15 --alpha_fdr 0.25 --tag ED --out results/revision
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import losses, twosided, weights  # noqa: E402


def certify(F, FD, cal, tst, wc, wt, af, ad):
    m = F.shape[1]
    idx, cert = twosided.certify_or_abstain(F[cal], FD[cal], wc, wt, af, ad)
    idx = np.clip(np.broadcast_to(idx, (len(tst),)), 0, m - 1)
    return idx, np.broadcast_to(cert, (len(tst),)).copy()


def split_weights(x, val, cal, tst, C, rng):
    """Weights fitted on (source validation, half A of target), applied to (calibration, half B)."""
    perm = rng.permutation(len(tst))
    halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
    wt = np.zeros(len(tst))
    wc_list = []
    for a, b in ((0, 1), (1, 0)):
        A = tst[halves[a]]
        X = np.r_[x[val, 0], x[A, 0]][:, None]
        y = np.r_[np.zeros(len(val)), np.ones(len(A))]
        clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
        r = len(val) / len(A)

        def w(ids):
            p = np.clip(clf.predict_proba(x[ids])[:, 1], 1e-3, 1 - 1e-3)
            return r * p / (1 - p)
        wt[halves[b]] = w(tst[halves[b]])
        wc_list.append((halves[b], w(cal)))
    return wt, wc_list


def ours_split(F, FD, x, val, cal, tst, af, ad, C, rng):
    wt, wc_list = split_weights(x, val, cal, tst, C, rng)
    idx = np.zeros(len(tst), int)
    cert = np.zeros(len(tst), bool)
    for hb, wc in wc_list:
        i, c = certify(F, FD, cal, tst[hb], wc, wt[hb], af, ad)
        idx[hb], cert[hb] = i, c
    return idx, cert


def gate(F, FD, x, cal, tst, af, ad, q):
    thr = np.quantile(x[cal, 0], q)
    cin = cal[x[cal, 0] <= thr]
    idx, cert = certify(F, FD, cin, tst, np.ones(len(cin)), np.ones(len(tst)), af, ad)
    return idx, cert & (x[tst, 0] <= thr)


def metrics(F, FD, tst, idx, cert):
    fnr, fdr = F[tst, idx], FD[tst, idx]
    n = max(cert.sum(), 1)
    return {"fnr": float((fnr * cert).mean()), "fdr": float((fdr * cert).mean()), "cert": float(cert.mean()),
            "fnr_c": float((fnr * cert).sum() / n) if cert.any() else np.nan,
            "fdr_c": float((fdr * cert).sum() / n) if cert.any() else np.nan}


def delta(L05, x, cal, tst):
    A = np.c_[np.ones(len(cal)), x[cal, 0]]
    beta = np.linalg.lstsq(A, L05[cal], rcond=None)[0]
    return float(np.mean(L05[tst] - (beta[0] + beta[1] * x[tst, 0])))


def one_run(F, FD, x, val, cal, tst, af, ad, C, rng, qs, seed=None):
    out = {}
    one_c, one_t = np.ones(len(cal)), np.ones(len(tst))
    out["standard"] = metrics(F, FD, tst, *certify(F, FD, cal, tst, one_c, one_t, af, ad))
    wc, wt = weights.loo_weights(x[cal, 0], x[tst, 0], C=C)
    out["ours"] = metrics(F, FD, tst, *certify(F, FD, cal, tst, wc, wt, af, ad))
    out["ours-split"] = metrics(F, FD, tst, *ours_split(F, FD, x, val, cal, tst, af, ad, C, rng))
    for q in qs:
        out[f"gate{q:.2f}"] = metrics(F, FD, tst, *gate(F, FD, x, cal, tst, af, ad, q))
    m = F.shape[1] // 2
    out["delta"] = {"d_fnr": delta(F[:, m], x, cal, tst), "d_fdr": delta(FD[:, m], x, cal, tst)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--alpha_fnr", type=float, default=0.15)
    ap.add_argument("--alpha_fdr", type=float, default=0.25)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--C_score", type=float, default=1.0)
    ap.add_argument("--n_settings", type=int, default=16, help="for the Bonferroni bound")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/revision")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    qs = [0.80, 0.90, 0.95]
    sweep_alpha = np.round(np.arange(0.05, 0.401, 0.025), 3)
    sweep_q = np.round(np.arange(0.50, 1.001, 0.025), 3)
    rows, sweep_rows = [], []
    for p in sorted({q for pat in a.runs for q in glob.glob(pat)}):
        f = os.path.join(p, "summaries.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        D = {k: z[k] for k in z.files}
        role = D["role"].copy()
        if a.phase != "all" and "phase" in D:
            role[D["phase"] != a.phase] = "excluded"
        val, cal, tst = (np.where(role == r)[0] for r in ("val", "cal", "test"))
        ha, hp = D["hist_all"][:, 0], D["hist_pos"][:, 0]
        F = losses.fnr_curve(ha, hp, D["n_pos"])
        FD = losses.fdr_curve(ha, hp)
        names = list(D["unc_names"])
        x = np.log(D["unc"][:, names.index("entropy_pred")] + 1e-6)[:, None]
        site = str(D["target_site"])
        point = one_run(F, FD, x, val, cal, tst, a.alpha_fnr, a.alpha_fdr, a.C_score, np.random.default_rng(1), qs, seed=0)
        boot = {k: [] for k in point}
        for b in range(a.B):
            cb, tb, vb = rng.choice(cal, len(cal)), rng.choice(tst, len(tst)), rng.choice(val, len(val))
            res = one_run(F, FD, x, vb, cb, tb, a.alpha_fnr, a.alpha_fdr, a.C_score, rng, qs)
            for k, v in res.items():
                boot[k].append(v)
        lev = 1 - 0.05 / a.n_settings
        for k in point:
            bk = pd.DataFrame(boot[k])
            row = {"tag": a.tag, "site": site, "method": k, "n_test": len(tst), "n_cal": len(cal), "n_val": len(val)}
            for col, v in point[k].items():
                row[col] = v
                s = bk[col].dropna()
                row[col + "_lo95"] = s.quantile(0.025) if len(s) else np.nan
                row[col + "_hi95"] = s.quantile(0.975) if len(s) else np.nan
                row[col + "_ub95"] = s.quantile(0.95) if len(s) else np.nan          # one-sided 95% upper bound
                row[col + "_lbBonf"] = s.quantile(1 - lev) if len(s) else np.nan     # Bonferroni one-sided lower
                row[col + "_ubBonf"] = s.quantile(lev) if len(s) else np.nan
            rows.append(row)
        # risk-coverage sweeps (point estimates)
        one_c, one_t = np.ones(len(cal)), np.ones(len(tst))
        wc, wt = weights.loo_weights(x[cal, 0], x[tst, 0], C=a.C_score)
        for al in sweep_alpha:
            ad = al + (a.alpha_fdr - a.alpha_fnr)
            for name, (wcc, wtt) in (("ours", (wc, wt)), ("standard", (one_c, one_t))):
                r = metrics(F, FD, tst, *certify(F, FD, cal, tst, wcc, wtt, al, ad))
                sweep_rows.append({"tag": a.tag, "site": site, "method": name, "param": al, **r, "n_test": len(tst)})
        for q in sweep_q:
            r = metrics(F, FD, tst, *gate(F, FD, x, cal, tst, a.alpha_fnr, a.alpha_fdr, min(q, 1.0)))
            sweep_rows.append({"tag": a.tag, "site": site, "method": "gate", "param": q, **r, "n_test": len(tst)})
        print("done", a.tag, site, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"rev_{a.tag}.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(a.out, f"sweep_{a.tag}.csv"), index=False)


if __name__ == "__main__":
    main()
