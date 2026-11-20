"""Synthetic experiment: target risk and efficiency of each method vs shift magnitude.

Runs on CPU in ~1 minute. Produces results/synthetic.csv and results/synthetic.png.

    python scripts/exp_synthetic.py --reps 200 --alpha 0.1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import crc, synthetic, weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_cal", type=int, default=100)
    ap.add_argument("--n_target_unlab", type=int, default=100)
    ap.add_argument("--shifts", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for s in a.shifts:
        for r in range(a.reps):
            P = synthetic.make_problem(n_cal=a.n_cal, n_test=a.n_target_unlab + 500, shift=s, seed=r)
            lam = P["lambdas"]
            Lt = P["L_test"]
            t_cal, t_eval = slice(0, a.n_target_unlab), slice(a.n_target_unlab, None)
            ws, wt, info = weights.estimate_weights(P["x_cal"], P["x_test"], seed=r)
            res = {
                "standard CRC": crc.crc_index(P["L_cal"], a.alpha),
                "weighted CRC (true w)": crc.weighted_crc_index(
                    P["L_cal"], P["w_cal_true"], P["w_test_true"][t_eval], a.alpha),
                "weighted CRC (estimated w)": crc.weighted_crc_index(P["L_cal"], ws, wt[t_eval], a.alpha),
                "oracle (labeled target)": crc.crc_index(Lt[t_cal], a.alpha),
            }
            for q in (0.95, 0.99):
                m = np.quantile(ws, q)
                res[f"weighted CRC (est. w, trunc q={q})"] = crc.weighted_crc_index(
                    P["L_cal"], np.minimum(ws, m), np.minimum(wt[t_eval], m), a.alpha)
            for name, idx in res.items():
                idx_b = np.broadcast_to(np.asarray(idx), (Lt[t_eval].shape[0],))
                rows.append({
                    "shift": s, "rep": r, "method": name,
                    "risk": crc.realized_risk(Lt[t_eval], idx_b),
                    "mean_lambda": float(lam[idx_b].mean()),
                    "frac_trivial": float(np.mean(idx_b == len(lam) - 1)),
                    "domain_auc": info["domain_auc"],
                    "ess_cal": crc.effective_sample_size(ws),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "synthetic.csv"), index=False)
    summ = df.groupby(["shift", "method"])[["risk", "mean_lambda", "frac_trivial", "domain_auc", "ess_cal"]].mean()
    print(summ.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in df.groupby("method"):
        m = g.groupby("shift")[["risk", "frac_trivial"]].mean()
        ax[0].plot(m.index, m["risk"], marker="o", label=name)
        ax[1].plot(m.index, m["frac_trivial"], marker="o", label=name)
    ax[0].axhline(a.alpha, color="k", ls="--", lw=1)
    ax[0].set(xlabel="shift magnitude", ylabel="realised target risk (FNR)", title="Validity")
    ax[1].set(xlabel="shift magnitude", ylabel="fraction of trivial (full) masks", title="Efficiency")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "synthetic.png"), dpi=150)


if __name__ == "__main__":
    main()
