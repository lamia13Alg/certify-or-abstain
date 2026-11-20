"""All conformal experiments on the patient-level summaries (CPU only, seconds).

Inputs : one or several fold directories containing summaries.npz (from infer_summaries.py)
Outputs: <out>/risk_control.csv, <out>/volume.csv, <out>/selective.csv,
         <out>/summary.txt and figures.

For every fold (target centre) and repetition r:
  * the target patients are split in two halves: A (labels used ONLY by the
    oracle) and E (evaluation). Every method is evaluated on E.
  * weights are estimated from embeddings of the calibration patients vs the
    unlabeled target patients (cross-fitted; labels never used).

    python scripts/run_conformal.py --runs /kaggle/working/runs/* --out results/prostate
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import crc, losses, selective, volume, weights  # noqa: E402


def load_fold(path):
    z = np.load(os.path.join(path, "summaries.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def per_patient_at(mat, idx):
    idx = np.broadcast_to(np.asarray(idx), (mat.shape[0],))
    return mat[np.arange(mat.shape[0]), idx]


def run_fold(D, a, rng):
    site = str(D["target_site"])
    role = D["role"]
    cal = np.isin(role, a.cal_roles)
    tst = role == "test"
    bi = int(np.argmin(np.abs(D["betas"] - a.beta)))
    ha, hp = D["hist_all"][:, bi], D["hist_pos"][:, bi]
    n_bins = ha.shape[-1]
    lam = losses.lambda_grid(n_bins)
    FNR = losses.fnr_curve(ha, hp, D["n_pos"])
    DICE = losses.dice_curve(ha, hp, D["n_pos"])
    VOL = losses.volume_curve(ha, D["voxel_volume_ml"])
    vt = D["vol_true_ml"]
    unc_names = list(D["unc_names"])
    u_all = D["unc"][:, unc_names.index(a.unc)]
    k_fixed = n_bins // 2                     # threshold t = 0.5

    # ---- weights (labels never used)
    ws, wt_all, info = weights.estimate_weights(D["emb"][cal], D["emb"][tst],
                                                n_components=a.pca, C=a.C, seed=0)
    # score-conditional weights: covariate = log of a 1-D patient-level uncertainty score
    xs = np.log(D["unc"][:, unc_names.index(a.shift_score)] + 1e-6)[:, None]
    ws_u, wt_u_all, info_u = weights.estimate_weights(xs[cal], xs[tst], n_components=None, C=a.C_score, seed=0)
    cal_ids, tst_ids = np.where(cal)[0], np.where(tst)[0]

    rc_rows, vol_rows, sel_rows = [], [], []
    for r in range(a.reps):
        perm = rng.permutation(len(tst_ids))
        half = len(perm) // 2
        A, E = tst_ids[perm[:half]], tst_ids[perm[half:]]
        wE = wt_all[perm[half:]]
        wE_u = wt_u_all[perm[half:]]
        Lc = FNR[cal_ids]
        for alpha in a.alphas:
            res = {
                "fixed t=0.5": np.full(len(E), k_fixed),
                "standard CRC": crc.crc_index(Lc, alpha),
                "weighted CRC": crc.weighted_crc_index(Lc, ws, wE, alpha),
                "score-weighted CRC": crc.weighted_crc_index(Lc, ws_u, wE_u, alpha),
                "oracle CRC (labeled target)": crc.crc_index(FNR[A], alpha),
            }
            for q in a.trunc:
                m = np.quantile(ws, q)
                res[f"weighted CRC trunc{q}"] = crc.weighted_crc_index(
                    Lc, np.minimum(ws, m), np.minimum(wE, m), alpha)
            for name, idx in res.items():
                idx = np.broadcast_to(np.asarray(idx), (len(E),))
                rc_rows.append({
                    "site": site, "rep": r, "alpha": alpha, "method": name,
                    "risk_fnr": float(per_patient_at(FNR[E], idx).mean()),
                    "dice": float(per_patient_at(DICE[E], idx).mean()),
                    "vol_ratio": float(np.mean(per_patient_at(VOL[E], idx) / np.maximum(vt[E], 1e-6))),
                    "frac_trivial": float(np.mean(idx == len(lam) - 1)),
                    # trivial (full) mask = abstention; risk among certified patients
                    "risk_certified": float(per_patient_at(FNR[E], idx)[idx < len(lam) - 1].mean())
                    if np.any(idx < len(lam) - 1) else np.nan,
                    "dice_certified": float(per_patient_at(DICE[E], idx)[idx < len(lam) - 1].mean())
                    if np.any(idx < len(lam) - 1) else np.nan,
                    "threshold_mean": float(np.mean(1 - lam[idx])),
                })

        # ---- volume intervals (relative residual and CQR on MC volumes)
        vh = D["unc"][:, unc_names.index("vol_mean_ml")]
        lo_mc = np.quantile(D["vol_samples_ml"], 0.1, axis=1)
        hi_mc = np.quantile(D["vol_samples_ml"], 0.9, axis=1)
        for alpha in a.alphas:
            s_rel = volume.relative_scores(vt[cal_ids], vh[cal_ids])
            s_cqr = volume.cqr_scores(vt[cal_ids], lo_mc[cal_ids], hi_mc[cal_ids])
            ivs = {
                "relative / standard": volume.relative_interval(vh[E], volume.conformal_quantile(s_rel, alpha)),
                "relative / weighted": volume.relative_interval(
                    vh[E], volume.weighted_conformal_quantile(s_rel, ws, wE, alpha)),
                "CQR / standard": volume.cqr_interval(lo_mc[E], hi_mc[E], volume.conformal_quantile(s_cqr, alpha)),
                "CQR / weighted": volume.cqr_interval(
                    lo_mc[E], hi_mc[E], volume.weighted_conformal_quantile(s_cqr, ws, wE, alpha)),
                "raw MC 10-90%": (lo_mc[E], hi_mc[E]),
            }
            for name, (lo, hi) in ivs.items():
                cov, width = volume.coverage_width(vt[E], lo, hi)
                rw = (hi - lo) / np.maximum(vt[E], 1e-6)
                rw = np.where(np.isfinite(rw), rw, np.nan)
                vol_rows.append({"site": site, "rep": r, "alpha": alpha, "method": name,
                                 "coverage": cov, "width_ml": width,
                                 "frac_infinite": float(np.mean(~np.isfinite(np.broadcast_to(hi, vt[E].shape)))),
                                 "rel_width": float(np.nanmean(rw)) if np.isfinite(rw).any() else np.nan})

        # ---- selective: accept if u <= tau, loss = FNR at the fixed threshold
        l_cal, l_E = FNR[cal_ids, k_fixed], FNR[E, k_fixed]
        taus = np.unique(np.quantile(u_all[cal_ids], np.linspace(0, 1, 101)))
        for alpha in a.sel_alphas:
            out = {
                "joint / standard": np.full(len(E), selective.joint_risk_tau(l_cal, u_all[cal_ids], alpha, taus)[0]),
                "joint / weighted": selective.joint_risk_tau(l_cal, u_all[cal_ids], alpha, taus, ws, wE),
                "LTT selective (delta=0.1)": np.full(
                    len(E), selective.ltt_selective_tau(l_cal, u_all[cal_ids], alpha, 0.1, taus, min_accept=5)),
            }
            for name, tau in out.items():
                acc = u_all[E] <= tau
                sel_rows.append({"site": site, "rep": r, "alpha": alpha, "method": name,
                                 "accept_rate": float(acc.mean()),
                                 "joint_risk": float(np.mean(l_E * acc)),
                                 "selective_risk": float(l_E[acc].mean()) if acc.any() else np.nan,
                                 "risk_rejected": float(l_E[~acc].mean()) if (~acc).any() else np.nan})

    diag = {"site": site, "n_cal": int(cal.sum()), "n_test": int(tst.sum()),
            "domain_auc": info["domain_auc"], "ess_cal": crc.effective_sample_size(ws),
            "score_auc": info_u["domain_auc"], "ess_score": crc.effective_sample_size(ws_u),
            "dice_fixed_cal": float(DICE[cal_ids, k_fixed].mean()),
            "dice_fixed_test": float(DICE[tst_ids, k_fixed].mean()),
            "fnr_fixed_cal": float(FNR[cal_ids, k_fixed].mean()),
            "fnr_fixed_test": float(FNR[tst_ids, k_fixed].mean())}
    return rc_rows, vol_rows, sel_rows, diag


def plots(rc, out, alpha_main):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = rc[rc.alpha == alpha_main]
    methods = ["fixed t=0.5", "standard CRC", "weighted CRC", "score-weighted CRC", "oracle CRC (labeled target)"]
    methods = [m for m in methods if m in d.method.unique()]
    sites = sorted(d.site.unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(sites) + 3, 4))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        g = d[d.method == m].groupby("site")["risk_fnr"]
        mu, sd = g.mean().reindex(sites), g.std().reindex(sites)
        ax.bar(np.arange(len(sites)) + j * w, mu, w, yerr=sd, label=m, capsize=2)
    ax.axhline(alpha_main, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(sites)) + 0.4 - w / 2, sites)
    ax.set_ylabel("patient-level FNR on unseen centre")
    ax.set_title(f"Risk control under centre shift (alpha = {alpha_main})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "risk_by_site.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/prostate")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--sel_alphas", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--beta", type=float, default=0.0, help="score s = mean - beta * std")
    ap.add_argument("--unc", default="entropy_pred", help="uncertainty feature for abstention")
    ap.add_argument("--cal_roles", nargs="+", default=["cal"])
    ap.add_argument("--trunc", type=float, nargs="*", default=[0.9, 0.95])
    ap.add_argument("--pca", type=int, default=4)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--shift_score", default="entropy_pred", help="1-D score used by score-weighted CRC")
    ap.add_argument("--C_score", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    runs = sorted({p for pat in a.runs for p in glob.glob(pat) if os.path.exists(os.path.join(p, "summaries.npz"))})
    if not runs:
        raise SystemExit("no summaries.npz found")
    RC, VOL, SEL, DIAG = [], [], [], []
    for p in runs:
        rc, vo, se, dg = run_fold(load_fold(p), a, rng)
        RC += rc; VOL += vo; SEL += se; DIAG.append(dg)
        print(f"done {dg['site']}: domain AUC {dg['domain_auc']:.2f}, ESS {dg['ess_cal']:.1f}/{dg['n_cal']}")
    rc, vo, se, dg = map(pd.DataFrame, (RC, VOL, SEL, DIAG))
    rc.to_csv(os.path.join(a.out, "risk_control.csv"), index=False)
    vo.to_csv(os.path.join(a.out, "volume.csv"), index=False)
    se.to_csv(os.path.join(a.out, "selective.csv"), index=False)
    dg.to_csv(os.path.join(a.out, "diagnostics.csv"), index=False)

    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        def show(title, df):
            print("\n== " + title); print(df.to_string()); f.write(f"\n== {title}\n{df.to_string()}\n")
        show("diagnostics", dg.round(3))
        show("risk control (mean over sites and reps)",
             rc.groupby(["alpha", "method"])[["risk_fnr", "dice", "frac_trivial", "risk_certified", "dice_certified"]].mean().round(3))
        show("worst site risk", rc.groupby(["alpha", "method", "site"])["risk_fnr"].mean()
             .groupby(["alpha", "method"]).max().round(3).unstack())
        show("volume intervals", vo.groupby(["alpha", "method"])[["coverage", "width_ml", "rel_width", "frac_infinite"]].mean().round(3))
        show("selective", se.groupby(["alpha", "method"])[
            ["accept_rate", "joint_risk", "selective_risk", "risk_rejected"]].mean().round(3))
    plots(rc, a.out, 0.1 if 0.1 in a.alphas else a.alphas[0])


if __name__ == "__main__":
    main()
