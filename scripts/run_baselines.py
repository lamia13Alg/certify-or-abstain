"""Abstention baselines against score-weighted certify-or-abstain (CPU only).

  standard        two-sided CRC, w = 1, certifies everyone it can
  shift-refuse    centre-level shift detector: cross-fitted domain classifier on bottleneck
                  embeddings (PCA 8, logistic); if its AUC > --auc_thr the whole target centre
                  is referred, otherwise standard two-sided CRC
  unc-threshold   patient-level MC-dropout gate: refer a target patient whose uncertainty
                  score exceeds the --q quantile of source calibration scores; standard
                  two-sided CRC calibrated on the source patients below that quantile
  score-weighted  ours

All thresholds are fixed a priori (auc_thr 0.75, q 0.90). Bootstrap over calibration and
target patients, as in run_bootstrap.py.

    python scripts/run_baselines.py --runs "runs/*" --phase ED --alpha_fnr 0.15 --alpha_fdr 0.25
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import losses, twosided, weights  # noqa: E402

METHODS = ["standard", "shift-refuse", "unc-threshold", "score-weighted"]


def _apply(F, FD, cal, tst, wc, wt, af, ad):
    m = F.shape[1]
    idx, cert = twosided.certify_or_abstain(F[cal], FD[cal], wc, wt, af, ad)
    idx = np.clip(np.broadcast_to(idx, (len(tst),)), 0, m - 1)
    cert = np.broadcast_to(cert, (len(tst),)).copy()
    return idx, cert


def evaluate(F, FD, D1, x, emb, cal, tst, af, ad, C, auc_thr, q, seed):
    out = {}
    one_c, one_t = np.ones(len(cal)), np.ones(len(tst))
    # standard
    out["standard"] = _apply(F, FD, cal, tst, one_c, one_t, af, ad)
    # shift detector + refusal
    _, _, info = weights.estimate_weights(emb[cal], emb[tst], n_components=8, C=0.1, seed=seed)
    idx, cert = out["standard"]
    out["shift-refuse"] = (idx, cert & (info["domain_auc"] <= auc_thr))
    # MC-dropout uncertainty gate
    thr = np.quantile(x[cal, 0], q)
    cal_in = cal[x[cal, 0] <= thr]
    idx, cert = _apply(F, FD, cal_in, tst, np.ones(len(cal_in)), one_t, af, ad)
    out["unc-threshold"] = (idx, cert & (x[tst, 0] <= thr))
    # ours
    r_c, r_t, _ = weights.estimate_weights(x[cal], x[tst], n_components=None, C=C, seed=seed)
    out["score-weighted"] = _apply(F, FD, cal, tst, r_c, r_t, af, ad)
    res = {}
    for k, (idx, cert) in out.items():
        fnr = F[tst, idx] * cert
        fdr = FD[tst, idx] * cert
        dice = D1[tst, idx]
        res[k] = (float(fnr.mean()), float(fdr.mean()), float(cert.mean()),
                  float(dice[cert].mean()) if cert.any() else np.nan)
    return res, info["domain_auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--alpha_fnr", type=float, default=0.15)
    ap.add_argument("--alpha_fdr", type=float, default=0.25)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--C_score", type=float, default=1.0)
    ap.add_argument("--auc_thr", type=float, default=0.75)
    ap.add_argument("--q", type=float, default=0.90)
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
        D1 = losses.dice_curve(ha, hp, D["n_pos"])
        names = list(D["unc_names"])
        x = np.log(D["unc"][:, names.index("entropy_pred")] + 1e-6)[:, None]
        emb = D["emb"]
        point, auc = evaluate(F, FD, D1, x, emb, cal, tst, a.alpha_fnr, a.alpha_fdr, a.C_score,
                              a.auc_thr, a.q, 0)
        boot = {k: [] for k in point}
        for b in range(a.B):
            cb, tb = rng.choice(cal, len(cal)), rng.choice(tst, len(tst))
            res, _ = evaluate(F, FD, D1, x, emb, cb, tb, a.alpha_fnr, a.alpha_fdr, a.C_score,
                              a.auc_thr, a.q, b)
            for k, v in res.items():
                boot[k].append(v)
        for k in METHODS:
            arr = np.array(boot[k])
            rows.append({"site": str(D["target_site"]), "method": k, "n_test": len(tst),
                         "emb_auc": auc,
                         "fnr": point[k][0], "fnr_lo": np.quantile(arr[:, 0], 0.025),
                         "fnr_hi": np.quantile(arr[:, 0], 0.975),
                         "fdr": point[k][1], "fdr_lo": np.quantile(arr[:, 1], 0.025),
                         "fdr_hi": np.quantile(arr[:, 1], 0.975),
                         "cert": point[k][2], "dice_cert": point[k][3]})
        print("done", D["target_site"], flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
