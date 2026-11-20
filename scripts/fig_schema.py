"""Fig. 1 of the JMBE manuscript: the certify-or-abstain pipeline (no title inside the artwork)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED = "#1f1f1e", "#55544f"
SRC, TGT, DEC, OK, REF = "#e7f1fb", "#f9ece7", "#f0efec", "#e5f4e4", "#fbebeb"
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "font.size": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(figsize=(174 / 25.4, 70 / 25.4))
ax.set_xlim(0, 174)
ax.set_ylim(5, 75)
ax.axis("off")


def box(x, y, w, h, text, fc, bold=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
                                fc=fc, ec=MUTED, lw=0.7))
    if bold:
        ax.text(x + w / 2, y + h - 3.2, bold, ha="center", va="top", fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 8.2, text, ha="center", va="top", color=INK, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, linespacing=1.35)


def arrow(x0, y0, x1, y1, label=None, dy=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=6.5, color=MUTED)


L, H = r"$\lambda_{\mathrm{lo}}$", r"$\lambda_{\mathrm{hi}}$"
# lane 1: source calibration patients (labeled)
box(1, 50, 27, 22, "images + masks\n(calibration)", SRC, "Source centres")
box(35, 50, 33, 22, "score histograms:\nFNR, FDR curves;\nuncertainty U", SRC, "Network, run once")
# lane 2: target centre (unlabeled)
box(1, 8, 27, 22, "images only,\nno masks", TGT, "Target centre")
box(35, 8, 33, 22, "voxel scores;\nuncertainty U", TGT, "Network, run once")
arrow(28, 61, 35, 61)
arrow(28, 19, 35, 19)
# weights
box(78, 29, 30, 22, "classifier on U,\nsource vs target\n" + r"$\rightarrow\ \hat{w}(U)$", DEC, "Score weights")
arrow(68, 55, 78, 46)
ax.text(70, 50.5, "U", fontsize=6.5, color=MUTED)
arrow(68, 25, 78, 34)
ax.text(70, 27.5, "U", fontsize=6.5, color=MUTED)
# decision
box(113, 29, 37, 22, L + r" : weighted FNR $\leq \alpha_{\mathrm{FN}}$" + "\n" + H + r" : weighted FDR $\leq \alpha_{\mathrm{FD}}$",
    DEC, "Each target patient")
arrow(108, 40, 113, 40)
ax.add_patch(FancyArrowPatch((68, 67), (131, 51), connectionstyle="arc3,rad=-0.3", arrowstyle="-|>",
                             mutation_scale=8, color=INK, lw=0.8))
ax.text(100, 70.5, "loss curves of calibration patients", ha="center", fontsize=6.5, color=MUTED)
box(155, 45, 18, 20, L + r" $\leq$ " + H + "\nmask at " + L, OK, "Certify")
box(155, 15, 18, 20, L + " > " + H + "\nrefer the\npatient", REF, "Refer")
arrow(150, 45, 155, 53)
arrow(150, 35, 155, 27)

if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
    print("usage: python scripts/fig_schema.py [OUT_PREFIX]  (writes OUT_PREFIX.pdf/.eps/.png, default Fig1)")
    sys.exit(0)
out = sys.argv[1] if len(sys.argv) > 1 else "Fig1"
for ext in ("pdf", "eps"):
    fig.savefig(f"{out}.{ext}")
fig.savefig(f"{out}.png", dpi=600)
print("saved", out)
