"""Conformal intervals on the LV ejection fraction (cardiac data, patient level).

EF = (EDV - ESV) / EDV, with EDV / ESV the LV volumes at end-diastole / end-systole.
Per patient: predicted EF from the MC-mean predicted volumes, true EF from the labels.
Score s = |EF_true - EF_hat| (in EF points).
  * standard split conformal  : quantile of the source-calibration scores
  * score-weighted conformal  : weights from a domain classifier on the patient's
    log-entropy (mean of ED and ES), unlabeled target images only; an infinite
    quantile means abstention (no certified interval for this patient)
Clinical read-out: fraction of patients whose interval stays inside one EF category
(reduced < 40 %, mildly reduced 40-49 %, preserved >= 50 %).

    python scripts/run_ef.py --runs "runs_mnms_strong/*" --out results_mnms/ef
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import crc, volume, weights  # noqa: E402

CUTS = (40.0, 50.0)


def category(ef):
    return np.digitize(ef, CUTS)


def patient_table(D):
    names = list(D["unc_names"])
    df = pd.DataFrame({
        "patient": D["patient"], "phase": D["phase"], "role": D["role"], "site": D["site"],
        "v_true": D["vol_true_ml"], "v_hat": D["unc"][:, names.index("vol_mean_ml")],
        "ent": D["unc"][:, names.index("entropy_pred")],
    })
    ed = df[df.phase == "ED"].set_index("patient")
    es = df[df.phase == "ES"].set_index("patient")
    p = ed.join(es[["v_true", "v_hat", "ent"]], rsuffix="_es", how="inner")
    p["ef_true"] = 100 * (p.v_true - p.v_true_es) / np.maximum(p.v_true, 1e-6)
    p["ef_hat"] = 100 * (p.v_hat - p.v_hat_es) / np.maximum(p.v_hat, 1e-6)
    p["x"] = np.log(0.5 * (p.ent + p.ent_es) + 1e-6)
    return p.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results_mnms/ef")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--C_score", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=1, help="kept for symmetry; the split is fixed by the fold")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for p in sorted({q for pat in a.runs for q in glob.glob(pat)}):
        f = os.path.join(p, "summaries.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        P = patient_table({k: z[k] for k in z.files})
        site = str(z["target_site"])
        cal, tst = P[P.role == "cal"], P[P.role == "test"]
        s_cal = np.abs(cal.ef_true - cal.ef_hat).values
        ws, wt = weights.loo_weights(cal["x"].values, tst["x"].values, C=a.C_score)
        for alpha in a.alphas:
            q_std = volume.conformal_quantile(s_cal, alpha)
            q_w = volume.weighted_conformal_quantile(s_cal, ws, wt, alpha)
            for name, q in (("standard", np.full(len(tst), q_std)), ("score-weighted", q_w)):
                cert = np.isfinite(q)
                lo, hi = tst.ef_hat.values - q, tst.ef_hat.values + q
                cov = (tst.ef_true.values >= lo) & (tst.ef_true.values <= hi)
                same_cat = category(lo) == category(hi)
                rows.append({
                    "site": site, "alpha": alpha, "method": name, "n_cal": len(cal), "n_test": len(tst),
                    "ess": crc.effective_sample_size(ws), "mae_ef_cal": float(s_cal.mean()),
                    "mae_ef_test": float(np.abs(tst.ef_true - tst.ef_hat).mean()),
                    # abstention counts as "not covered by a certified interval" but not as an error:
                    "coverage_certified": float(cov[cert].mean()) if cert.any() else np.nan,
                    "miscoverage_joint": float(np.mean(~cov & cert)),
                    "cert_rate": float(cert.mean()),
                    "half_width": float(np.median(q[cert])) if cert.any() else np.nan,
                    "one_category": float(same_cat[cert].mean()) if cert.any() else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "ef_intervals.csv"), index=False)
    txt = df.round(3).to_string(index=False)
    print(txt)
    open(os.path.join(a.out, "summary.txt"), "w").write(txt)


if __name__ == "__main__":
    main()
