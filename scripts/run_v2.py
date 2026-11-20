"""Two-sided certify-or-abstain (A) and hybrid calibration with k labeled target patients (B).

For each target centre and each repetition, k target patients are drawn as "labeled"
(used only by hybrid / target-only methods); all methods are evaluated on the SAME
remaining target patients E. Weights use only unlabeled images (score classifier).

    python scripts/run_v2.py --runs "runs/*" --out results/v2
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import crc, losses, twosided, weights  # noqa: E402


def run_site(D, a, rng):
    site = str(D["target_site"])
    role = D["role"].copy()
    if a.phase != "all" and "phase" in D:          # keep a single cardiac phase (one case per patient)
        role[D["phase"] != a.phase] = "excluded"
    cal_ids = np.where(np.isin(role, a.cal_roles))[0]
    tst_ids = np.where(role == "test")[0]
    ha, hp = D["hist_all"][:, 0], D["hist_pos"][:, 0]
    FNR = losses.fnr_curve(ha, hp, D["n_pos"])
    FDR = losses.fdr_curve(ha, hp)
    DICE = losses.dice_curve(ha, hp, D["n_pos"])
    m = FNR.shape[1]
    names = list(D["unc_names"])
    x = np.log(D["unc"][:, names.index(a.shift_score)] + 1e-6)[:, None]
    # r = q/p from source-cal vs ALL target images (unlabeled), cross-fitted
    r_cal, r_tst, info = weights.estimate_weights(x[cal_ids], x[tst_ids], n_components=None, C=a.C_score, seed=0)
    r = np.empty(len(role)); r[cal_ids] = r_cal; r[tst_ids] = r_tst
    n = len(cal_ids)

    rows = []
    for rep in range(a.reps):
        perm = rng.permutation(tst_ids)
        for k in a.ks:
            if k >= len(tst_ids) - 3:
                continue
            Lab, E = perm[:k], perm[k:]
            for af, ad in a.alphas:
                meth = {}
                ones_c = np.ones(n)
                if k == 0:
                    meth["standard (source)"] = twosided.certify_or_abstain(
                        FNR[cal_ids], FDR[cal_ids], ones_c, np.ones(len(E)), af, ad)
                    meth["score-weighted (source)"] = twosided.certify_or_abstain(
                        FNR[cal_ids], FDR[cal_ids], r[cal_ids], r[E], af, ad)
                else:
                    pool = np.r_[cal_ids, Lab]
                    w_pool = twosided.mixture_weights(r[pool], n, k)
                    w_E = twosided.mixture_weights(r[E], n, k)
                    meth[f"hybrid score-weighted (k={k})"] = twosided.certify_or_abstain(
                        FNR[pool], FDR[pool], w_pool, w_E, af, ad)
                    meth[f"pooled unweighted (k={k})"] = twosided.certify_or_abstain(
                        FNR[pool], FDR[pool], np.ones(len(pool)), np.ones(len(E)), af, ad)
                    meth[f"target only (k={k})"] = twosided.certify_or_abstain(
                        FNR[Lab], FDR[Lab], np.ones(k), np.ones(len(E)), af, ad)
                for name, (idx, cert) in meth.items():
                    idx = np.clip(np.broadcast_to(idx, (len(E),)), 0, m - 1)
                    cert = np.broadcast_to(cert, (len(E),))
                    fnr = FNR[E, idx]; fdr = FDR[E, idx]; dice = DICE[E, idx]
                    rows.append({
                        "site": site, "rep": rep, "k": k, "alpha_fnr": af, "alpha_fdr": ad, "method": name,
                        "joint_fnr": float(np.mean(fnr * cert)), "joint_fdr": float(np.mean(fdr * cert)),
                        "cert_rate": float(cert.mean()),
                        "fnr_cert": float(fnr[cert].mean()) if cert.any() else np.nan,
                        "fdr_cert": float(fdr[cert].mean()) if cert.any() else np.nan,
                        "dice_cert": float(dice[cert].mean()) if cert.any() else np.nan,
                        "dice_rejected_fixed": float(DICE[E, m // 2][~cert].mean()) if (~cert).any() else np.nan,
                    })
    diag = {"site": site, "n_cal": n, "n_test": len(tst_ids), "score_auc": info["domain_auc"],
            "ess": crc.effective_sample_size(r_cal),
            "fnr@0.5 cal": FNR[cal_ids, m // 2].mean(), "fdr@0.5 cal": FDR[cal_ids, m // 2].mean(),
            "fnr@0.5 test": FNR[tst_ids, m // 2].mean(), "fdr@0.5 test": FDR[tst_ids, m // 2].mean(),
            "dice@0.5 test": DICE[tst_ids, m // 2].mean()}
    return rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/v2")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.20, 0.15, 0.25],
                    help="pairs alpha_fnr alpha_fdr")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 3, 5])
    ap.add_argument("--cal_roles", nargs="+", default=["cal"])
    ap.add_argument("--phase", default="all", help="cardiac data: ED, ES or all")
    ap.add_argument("--shift_score", default="entropy_pred")
    ap.add_argument("--C_score", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.alphas = list(zip(a.alphas[::2], a.alphas[1::2]))
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    runs = sorted({p for pat in a.runs for p in glob.glob(pat) if os.path.exists(os.path.join(p, "summaries.npz"))})
    R, G = [], []
    for p in runs:
        z = np.load(os.path.join(p, "summaries.npz"))
        rows, diag = run_site({k: z[k] for k in z.files}, a, rng)
        R += rows; G.append(diag)
    df, dg = pd.DataFrame(R), pd.DataFrame(G)
    df.to_csv(os.path.join(a.out, "twosided.csv"), index=False)
    dg.to_csv(os.path.join(a.out, "diagnostics.csv"), index=False)
    pd.set_option("display.width", 250)
    out = ["== diagnostics", dg.round(3).to_string()]
    for (af, ad), g in df.groupby(["alpha_fnr", "alpha_fdr"]):
        site = g.groupby(["method", "site"])[["joint_fnr", "joint_fdr", "cert_rate", "dice_cert"]].mean()
        summ = pd.concat({
            "worst joint FNR": site["joint_fnr"].groupby("method").max(),
            "worst joint FDR": site["joint_fdr"].groupby("method").max(),
            "mean cert rate": site["cert_rate"].groupby("method").mean(),
            "min cert rate": site["cert_rate"].groupby("method").min(),
            "dice certified": site["dice_cert"].groupby("method").mean(),
        }, axis=1)
        out += [f"\n== alpha_fnr={af}  alpha_fdr={ad}  (worst = worst centre)", summ.round(3).to_string(),
                "\n-- per centre: joint FNR / joint FDR / certified rate / Dice certified",
                site.round(3).unstack(1).to_string()]
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(a.out, "summary.txt"), "w").write(txt)


if __name__ == "__main__":
    main()
