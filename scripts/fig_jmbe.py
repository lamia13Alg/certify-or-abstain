"""JMBE figures (no titles inside the artwork, Liberation Sans ~ Arial, 174 mm wide, hatching
in addition to colour). Two columns of panels, rows = joint FNR, joint FDR, fraction certified.
Input: run_baselines.py csv files (standard and score-weighted rows, bootstrap 95% intervals).

    python scripts/fig_jmbe.py --csv A.csv B.csv --labels "a  Standard augmentation" "b  Domain-randomised" \
        --alphas 0.15 0.25 0.15 0.25 --sites BIDMC I2CVB BMC HK RUNMC UCL --out Fig2
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_STD, C_OURS = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#55544f", "#e4e3de"
METH = [("standard", "Standard two-sided CRC", C_STD, None),
        ("score-weighted", "Score-weighted certify-or-abstain", C_OURS, "////")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs=2, required=True)
    ap.add_argument("--labels", nargs=2, required=True)
    ap.add_argument("--alphas", type=float, nargs=4, required=True, help="fnr fdr (left) fnr fdr (right)")
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--xlabel", default="Unseen target centre")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "hatch.color": "white", "hatch.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(174 / 25.4, 150 / 25.4), sharex=True)
    x = np.arange(len(a.sites))
    w = 0.38
    for j in range(2):
        d = pd.read_csv(a.csv[j]).set_index(["method", "site"])
        af, ad = a.alphas[2 * j], a.alphas[2 * j + 1]
        for i, (col, lab, alpha) in enumerate([("fnr", "False-negative rate", af),
                                               ("fdr", "False-discovery rate", ad),
                                               ("cert", "Fraction certified", None)]):
            ax = axes[i, j]
            for k, (m, mlab, c, hatch) in enumerate(METH):
                v = np.array([d.loc[(m, s), col] for s in a.sites])
                xs = x + (k - 0.5) * w
                ax.bar(xs, v, w * 0.92, color=c, hatch=hatch, label=mlab, edgecolor=c, linewidth=0)
                if col in ("fnr", "fdr"):
                    lo = np.array([d.loc[(m, s), col + "_lo"] for s in a.sites])
                    hi = np.array([d.loc[(m, s), col + "_hi"] for s in a.sites])
                    cert = np.array([d.loc[(m, s), "cert"] for s in a.sites])
                    keep = cert >= 0.01   # a centre referred in full has no certified error to bound
                    ax.errorbar(xs[keep], v[keep], yerr=[np.clip(v - lo, 0, None)[keep], np.clip(hi - v, 0, None)[keep]],
                                fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8)
                    for xi, ci in zip(xs, cert):
                        if ci < 0.01:
                            ax.text(xi, 0.01, "all referred", rotation=90, ha="center", va="bottom",
                                    fontsize=6.5, color=MUTED)
            if alpha is not None:
                ax.axhline(alpha, color=INK, ls=(0, (4, 2)), lw=0.9)
                ax.text(1.0, alpha, f" α = {alpha:g}", va="center", ha="left", fontsize=7, color=INK,
                        transform=ax.get_yaxis_transform(), clip_on=False)
                ax.set_ylim(0, 0.8 if col == "fdr" else 0.45)
            else:
                ax.set_ylim(0, 1.05)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(lab)
        axes[0, j].text(0.0, 1.06, a.labels[j], transform=axes[0, j].transAxes, fontsize=9,
                        fontweight="bold", color=INK, ha="left", va="bottom")
        axes[2, j].set_xticks(x, a.sites)
        axes[2, j].set_xlabel(a.xlabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:2], l[:2], frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 0.97, 0.95), h_pad=0.8, w_pad=2.0)
    for ext in ("pdf", "eps"):
        fig.savefig(f"{a.out}.{ext}")
    fig.savefig(f"{a.out}.png", dpi=600)
    print("saved", a.out)


if __name__ == "__main__":
    main()
