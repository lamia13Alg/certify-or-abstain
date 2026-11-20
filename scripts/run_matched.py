"""Equal-coverage comparison of our rule with the uncertainty gate, pooled over all centre settings.

In every bootstrap replicate (calibration and target patients resampled in each setting), the gate
quantile q is chosen so that its pooled certification rate matches that of our rule as closely as
possible; we then record the difference in pooled risk among certified patients (gate - ours).
Also writes pooled risk-coverage curves (point estimates).

    python scripts/run_matched.py --prostate data/prostate_summaries/strong --mnms runs_mnms --out results/revision
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from run_revision import certify, gate  # noqa: E402
from scrc import losses, weights  # noqa: E402

SETS = [("P", "PROSTATE", ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"], "all", .15, .25),
        ("ED", "MNMS", ["C1", "C2", "C3", "C4", "C5"], "ED", .15, .25),
        ("ES", "MNMS", ["C1", "C2", "C3", "C4", "C5"], "ES", .25, .35)]
QS = np.round(np.arange(0.50, 1.0001, 0.01), 3)


def load(root, site, phase):
    z = np.load(os.path.join(root, site, "summaries.npz"))
    D = {k: z[k] for k in z.files}
    role = D["role"].copy()
    if phase != "all":
        role[D["phase"] != phase] = "excluded"
    ha, hp = D["hist_all"][:, 0], D["hist_pos"][:, 0]
    names = list(D["unc_names"])
    return dict(cal=np.where(role == "cal")[0], tst=np.where(role == "test")[0],
                F=losses.fnr_curve(ha, hp, D["n_pos"]), FD=losses.fdr_curve(ha, hp),
                x=np.log(D["unc"][:, names.index("entropy_pred")] + 1e-6)[:, None])


def sums(S, tst, idx, cert):
    fnr, fdr = S["F"][tst, idx], S["FD"][tst, idx]
    return np.array([cert.sum(), (fnr * cert).sum(), (fdr * cert).sum(), len(tst)])


def replicate(settings, rng, boot):
    ours = np.zeros(4)
    gates = np.zeros((len(QS), 4))
    for S, af, ad in settings:
        cal, tst = S["cal"], S["tst"]
        if boot:
            cal, tst = rng.choice(cal, len(cal)), rng.choice(tst, len(tst))
        wc, wt = weights.loo_weights(S["x"][cal, 0], S["x"][tst, 0], C=1.0)
        ours += sums(S, tst, *certify(S["F"], S["FD"], cal, tst, wc, wt, af, ad))
        for j, q in enumerate(QS):
            gates[j] += sums(S, tst, *gate(S["F"], S["FD"], S["x"], cal, tst, af, ad, q))
    return ours, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--out", default="results/revision")
    ap.add_argument("--prostate", default="data/prostate_summaries/strong")
    ap.add_argument("--mnms", required=True, help="folder with C1..C5/summaries.npz (not distributed)")
    a = ap.parse_args()
    roots = {"PROSTATE": a.prostate, "MNMS": a.mnms}
    settings = [(load(roots[root], s, ph), af, ad) for tag, root, sites, ph, af, ad in SETS for s in sites]
    rng = np.random.default_rng(0)
    o, g = replicate(settings, rng, False)
    curve = pd.DataFrame({"q": QS, "cert": g[:, 0] / g[:, 3], "fnr_c": g[:, 1] / g[:, 0], "fdr_c": g[:, 2] / g[:, 0]})
    curve.to_csv(os.path.join(a.out, "gate_curve_pooled.csv"), index=False)
    print("ours pooled: cert %.3f fnr_c %.3f fdr_c %.3f" % (o[0] / o[3], o[1] / o[0], o[2] / o[0]))
    diffs = []
    for b in range(a.B):
        o, g = replicate(settings, rng, True)
        j = int(np.argmin(np.abs(g[:, 0] / g[:, 3] - o[0] / o[3])))
        diffs.append({"q": QS[j], "cert_ours": o[0] / o[3], "cert_gate": g[j, 0] / g[j, 3],
                      "d_fnr_c": g[j, 1] / g[j, 0] - o[1] / o[0], "d_fdr_c": g[j, 2] / g[j, 0] - o[2] / o[0]})
    d = pd.DataFrame(diffs)
    d.to_csv(os.path.join(a.out, "matched_boot.csv"), index=False)
    print(d.describe(percentiles=[0.025, 0.5, 0.975]).round(4).to_string())


if __name__ == "__main__":
    main()
