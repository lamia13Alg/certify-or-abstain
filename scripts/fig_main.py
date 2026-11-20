"""Main results figure: standard CRC vs score-weighted certify-or-abstain, per centre,
for both segmentation models. 3 rows (joint FNR, joint FDR, certified rate) x 2 columns.

    python scripts/fig_main.py --base results/base/v2/twosided.csv \
        --strong results/strong/v2/twosided.csv --out results/fig_main.png
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_STD, C_OURS = "#2a78d6", "#eb6834"      # validated categorical slots 1-2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SITES = ["BIDMC", "I2CVB", "BMC", "HK", "RUNMC", "UCL"]   # default order (prostate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="left column csv")
    ap.add_argument("--strong", required=True, help="right column csv")
    ap.add_argument("--titles", nargs=2, default=["Baseline model", "Domain-randomised model"])
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--alphas_right", type=float, nargs=2, default=None,
                    help="alpha_fnr alpha_fdr for the right column if different")
    ap.add_argument("--footer", default="Unseen target centre (left: degraded centres, right: in-distribution-like)")
    ap.add_argument("--out", default="fig_main.png")
    ap.add_argument("--boot", nargs=2, default=None, help="bootstrap csv (left, right) for 95%% whiskers on FNR/FDR")
    ap.add_argument("--alpha_fnr", type=float, default=0.15)
    ap.add_argument("--alpha_fdr", type=float, default=0.25)
    a = ap.parse_args()

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    rows = [("joint_fnr", "False-negative rate", a.alpha_fnr),
            ("joint_fdr", "False-discovery rate", a.alpha_fdr),
            ("cert_rate", "Patients certified", None)]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.4), sharex=True, sharey=False)
    meths = [("standard (source)", "Standard CRC", C_STD), ("score-weighted (source)", "Score-weighted (ours)", C_OURS)]
    SITES_ = a.sites
    for j, (path, title) in enumerate([(a.base, a.titles[0]), (a.strong, a.titles[1])]):
        af, ad = (a.alpha_fnr, a.alpha_fdr) if (j == 0 or a.alphas_right is None) else tuple(a.alphas_right)
        d = pd.read_csv(path)
        d = d[(np.isclose(d.alpha_fnr, af)) & (np.isclose(d.alpha_fdr, ad)) & (d.k == 0)]
        g = d.groupby(["method", "site"])[["joint_fnr", "joint_fdr", "cert_rate"]].mean()
        bt = pd.read_csv(a.boot[j]).set_index(["method", "site"]) if a.boot else None
        x = np.arange(len(SITES_))
        rows_j = [("joint_fnr", rows[0][1], af), ("joint_fdr", rows[1][1], ad), rows[2]]
        for i, (col, lab, alpha) in enumerate(rows_j):
            ax = axes[i, j]
            for k, (m, mlab, c) in enumerate(meths):
                v = g.loc[m][col].reindex(SITES_).values
                ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=c, label=mlab, edgecolor="white", linewidth=1)
                if bt is not None and col in ("joint_fnr", "joint_fdr"):
                    key = m.split(" (")[0]
                    lo_c, hi_c = ("fnr_lo", "fnr_hi") if col == "joint_fnr" else (None, "fdr_hi")
                    for xi, site, vi in zip(x, SITES_, v):
                        if (key, site) in bt.index:
                            r = bt.loc[(key, site)]
                            lo = r[lo_c] if lo_c else vi
                            ax.plot([xi + (k - 0.5) * 0.38] * 2, [min(lo, vi), max(r[hi_c], vi)], color=INK, lw=0.9)
                cr = g.loc[m]["cert_rate"].reindex(SITES_).values
                for xi, cri in zip(x, cr):
                    if cri < 0.01:   # abstained on the whole centre: zero risk by abstention, not by accuracy
                        ax.text(xi + (k - 0.5) * 0.38, 0, "abstains", rotation=90, ha="center", va="bottom",
                                fontsize=7, color=MUTED)
            if alpha is not None:
                ax.axhline(alpha, color=INK, ls="--", lw=1)
                ax.text(1.01, alpha, f"α={alpha}", va="center", ha="left", fontsize=7, color=INK,
                        transform=ax.get_yaxis_transform())
            ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(title, fontsize=10, color=INK)
        axes[2, j].set_xticks(x, SITES_, rotation=0)
        axes[2, j].set_ylim(0, 1.05)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:2], l[:2], frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, a.footer,
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(a.out, dpi=300)
    print("saved", a.out)


if __name__ == "__main__":
    main()
