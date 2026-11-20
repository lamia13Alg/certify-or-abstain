"""Second-round review analyses (CPU only).

  split   : cross-fitted vs sample-split weights over R independent repetitions (fold seeds / half-splits),
            with the spread of the weights (effective sample size of the calibration weights, max weight)
  nval    : sample-split variant with sub-sampled source validation sets (cardiac), to find the
            sample size needed to estimate the weights
  delta   : uniform (sup) version of the conditional shift: max over thresholds lambda in {0.1,...,0.9}
            and over target-score tertiles of [mean target loss - mean source loss in the same score bin],
            with a bootstrap 95% upper bound
  ncal    : sensitivity to the calibration-set size (25-100% of calibration patients)

    python scripts/run_revision2.py --what split --out results/revision2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from run_revision import certify, metrics, split_weights  # noqa: E402
from scrc import losses, weights  # noqa: E402

SETS = [("P", "PROSTATE", ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"], "all", .15, .25),
        ("ED", "MNMS", ["C1", "C2", "C3", "C4", "C5"], "ED", .15, .25),
        ("ES", "MNMS", ["C1", "C2", "C3", "C4", "C5"], "ES", .25, .35)]


def load(root, site, phase):
    z = np.load(os.path.join(root, site, "summaries.npz"))
    D = {k: z[k] for k in z.files}
    role = D["role"].copy()
    if phase != "all":
        role[D["phase"] != phase] = "excluded"
    ha, hp = D["hist_all"][:, 0], D["hist_pos"][:, 0]
    names = list(D["unc_names"])
    return dict(val=np.where(role == "val")[0], cal=np.where(role == "cal")[0], tst=np.where(role == "test")[0],
                F=losses.fnr_curve(ha, hp, D["n_pos"]), FD=losses.fdr_curve(ha, hp),
                x=np.log(D["unc"][:, names.index("entropy_pred")] + 1e-6)[:, None])


def ess(w):
    w = np.asarray(w, float)
    return w.sum() ** 2 / (w ** 2).sum()


def run_split(S, af, ad, R, rng):
    """LOO cross-fitted (deterministic), 5-fold cross-fitted over R fold assignments (with the
    patient-level decision flip rate) and sample-split over R half-splits."""
    out = []
    F, FD, x, val, cal, tst = S["F"], S["FD"], S["x"], S["val"], S["cal"], S["tst"]
    wc, wt = weights.loo_weights(x[cal, 0], x[tst, 0], C=1.0)
    out.append({"variant": "loo", "rep": 0, **metrics(F, FD, tst, *certify(F, FD, cal, tst, wc, wt, af, ad)),
                "ess_cal": ess(wc), "flip": 0.0})
    decisions = []
    for r in range(R):
        wc, wt, _ = weights.estimate_weights(x[cal], x[tst], n_components=None, C=1.0, seed=r)
        idx, cert = certify(F, FD, cal, tst, wc, wt, af, ad)
        decisions.append(cert)
        out.append({"variant": "5-fold", "rep": r, **metrics(F, FD, tst, idx, cert), "ess_cal": ess(wc)})
    D = np.array(decisions, float)
    maj = D.mean(0) >= 0.5
    flip = float(np.mean(D != maj[None, :]))
    for o in out:
        if o["variant"] == "5-fold":
            o["flip"] = flip
    for r in range(R):
        wt2, wc_list = split_weights(x, val, cal, tst, 1.0, rng)
        idx = np.zeros(len(tst), int)
        cert = np.zeros(len(tst), bool)
        for hb, wcb in wc_list:
            i, c = certify(F, FD, cal, tst[hb], wcb, wt2[hb], af, ad)
            idx[hb], cert[hb] = i, c
        out.append({"variant": "sample-split", "rep": r, **metrics(F, FD, tst, idx, cert),
                    "ess_cal": np.mean([ess(w) for _, w in wc_list]), "flip": np.nan})
    return out


def run_nval(S, af, ad, R, rng, sizes):
    out = []
    F, FD, x, val, cal, tst = S["F"], S["FD"], S["x"], S["val"], S["cal"], S["tst"]
    for n in sizes:
        if n > len(val):
            continue
        for r in range(R):
            v = rng.choice(val, n, replace=False)
            wt2, wc_list = split_weights(x, v, cal, tst, 1.0, rng)
            idx = np.zeros(len(tst), int)
            cert = np.zeros(len(tst), bool)
            for hb, wcb in wc_list:
                i, c = certify(F, FD, cal, tst[hb], wcb, wt2[hb], af, ad)
                idx[hb], cert[hb] = i, c
            out.append({"n_val": n, "rep": r, **metrics(F, FD, tst, idx, cert)})
    return out


def delta_sup(S, B, rng):
    F, FD, x, cal, tst = S["F"], S["FD"], S["x"], S["cal"], S["tst"]
    FDe = np.maximum.accumulate(FD, axis=1)
    m = F.shape[1] - 1
    lam_idx = [int(round(l * m)) for l in np.arange(0.1, 0.91, 0.1)]

    def one(c, t):
        edges = np.quantile(x[t, 0], [0, 1 / 3, 2 / 3, 1])
        edges[0], edges[-1] = -np.inf, np.inf
        best = {"fnr": -np.inf, "fdr": -np.inf}
        for k in range(3):
            ib_t = t[(x[t, 0] > edges[k]) & (x[t, 0] <= edges[k + 1])]
            ib_c = c[(x[c, 0] > edges[k]) & (x[c, 0] <= edges[k + 1])]
            if len(ib_t) < 3 or len(ib_c) < 3:
                continue
            for name, L in (("fnr", F), ("fdr", FDe)):
                d = (L[ib_t][:, lam_idx].mean(0) - L[ib_c][:, lam_idx].mean(0)).max()
                best[name] = max(best[name], d)
        return best
    pt = one(cal, tst)
    bs = [one(rng.choice(cal, len(cal)), rng.choice(tst, len(tst))) for _ in range(B)]
    return {"dsup_fnr": pt["fnr"], "dsup_fnr_ub95": np.quantile([b["fnr"] for b in bs], 0.95),
            "dsup_fdr": pt["fdr"], "dsup_fdr_ub95": np.quantile([b["fdr"] for b in bs], 0.95)}


def run_ncal(S, af, ad, R, rng, fracs):
    out = []
    F, FD, x, cal, tst = S["F"], S["FD"], S["x"], S["cal"], S["tst"]
    for fr in fracs:
        for r in range(R if fr < 1 else 1):
            c = cal if fr == 1 else rng.choice(cal, max(int(fr * len(cal)), 5), replace=False)
            wc, wt = weights.loo_weights(x[c, 0], x[tst, 0], C=1.0)
            for name, (a1, a2) in (("ours", (wc, wt)), ("standard", (np.ones(len(c)), np.ones(len(tst))))):
                out.append({"frac": fr, "n_cal": len(c), "rep": r, "method": name,
                            **metrics(F, FD, tst, *certify(F, FD, c, tst, a1, a2, af, ad))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["split", "nval", "delta", "ncal"])
    ap.add_argument("--prostate", default="data/prostate_summaries/strong")
    ap.add_argument("--mnms", required=True)
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--out", default="results/revision2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    roots = {"PROSTATE": a.prostate, "MNMS": a.mnms}
    rng = np.random.default_rng(0)
    rows = []
    for tag, root, sites, ph, af, ad in SETS:
        if a.what == "nval" and tag == "P":
            continue
        for s in sites:
            S = load(roots[root], s, ph)
            base = {"tag": tag, "site": s, "n_test": len(S["tst"]), "n_val": len(S["val"])}
            if a.what == "split":
                rows += [{**base, **o} for o in run_split(S, af, ad, a.R, rng)]
            elif a.what == "nval":
                rows += [{**base, **o} for o in run_nval(S, af, ad, a.R, rng, [5, 10, 15, 20])]
            elif a.what == "delta":
                rows.append({**base, "af": af, "ad": ad, **delta_sup(S, a.B, rng)})
            else:
                rows += [{**base, **o} for o in run_ncal(S, af, ad, a.R, rng, [0.25, 0.5, 0.75, 1.0])]
            print("done", a.what, tag, s, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{a.what}.csv"), index=False)


if __name__ == "__main__":
    main()
