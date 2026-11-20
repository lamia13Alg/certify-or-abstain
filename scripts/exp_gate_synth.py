"""Synthetic comparison of our rule with the uncertainty gate (Online Resource 2, S7).

Loss curve L(lambda) = (1 - lambda) ** exp(-h): a power of (1 - lambda), hence in [0, 1] and
non-increasing in lambda, for any real hardness h.
Base generator: score U ~ N(0, 1) at the source and N(s, 1) at the target, hardness h = 0.8 U + 0.3 eps.
Scenarios (Assumption 1 holds in the first five, not in the last two):
  gaussian       base generator, s in {0, 0.5, 1, 1.5, 2}
  nonlinear      h = 0.8 sign(U) U^2 / 2 + 0.3 eps
  heavy-tailed   U ~ Student t(3), shifted by s
  bimodal        target U ~ 0.5 N(0, 1) + 0.5 N(2 s, 1)
  low overlap    s = 3
  noisy score    h driven by a latent V; the score is U = V + N(0, 1): the conditional law of h given U
                 changes with the shift
  confident err. 20% of target patients have h increased by 1 with an unchanged score
Rules: standard CRC; uncertainty gate (q = 0.90, and q matched to the certification rate of our rule);
our rule with leave-one-out cross-fitted logistic weights (60 calibration, 40 unlabeled target scores),
a patient being referred when its weighted CRC is infeasible. Alpha = 0.15.

    python scripts/exp_gate_synth.py --reps 300 --out results/gate_synth
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc import crc, weights  # noqa: E402

LAM = np.linspace(0, 1, 401)
QS = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def draw(n, scen, s, target, rng):
    if scen == "heavy-tailed":
        u = rng.standard_t(3, n) + (s if target else 0.0)
    elif scen == "bimodal" and target:
        u = np.where(rng.random(n) < 0.5, rng.normal(0, 1, n), rng.normal(2 * s, 1, n))
    else:
        u = rng.normal(s if target else 0.0, 1.0, n)
    if scen == "noisy score":
        v = u
        u = v + rng.normal(0, 1, n)
        h = 0.8 * v + 0.3 * rng.normal(size=n)
    elif scen == "nonlinear":
        h = 0.8 * np.sign(u) * u ** 2 / 2 + 0.3 * rng.normal(size=n)
    else:
        h = 0.8 * u + 0.3 * rng.normal(size=n)
    if scen == "confident errors" and target:
        h = h + (rng.random(n) < 0.2) * 1.0
    return u, (1.0 - LAM[None, :]) ** np.exp(-h)[:, None]


def summ(Le, idx, cert):
    loss = Le[np.arange(len(idx)), idx] * cert
    return [cert.mean(), loss.sum() / max(cert.sum(), 1) if cert.any() else np.nan]


def diff_stats(d):
    """Per-replicate paired difference gate (matched q) minus ours, over replicates where both certify."""
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"diff_mean": np.nan, "diff_se": np.nan, "diff_q025": np.nan, "diff_q975": np.nan,
                "diff_pos": np.nan, "n_pairs": 0}
    return {"diff_mean": d.mean(), "diff_se": d.std() / np.sqrt(len(d)), "diff_q025": np.quantile(d, 0.025),
            "diff_q975": np.quantile(d, 0.975), "diff_pos": (d > 0).mean(), "n_pairs": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--out", default="results/gate_synth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grid = [("gaussian", s) for s in (0.0, 0.5, 1.0, 1.5, 2.0)]
    grid += [("nonlinear", 1.0), ("heavy-tailed", 1.0), ("bimodal", 1.0), ("low overlap", 3.0),
             ("noisy score", 1.0), ("confident errors", 1.0)]
    rows = []
    for scen, s in grid:
        gen = "gaussian" if scen == "low overlap" else scen
        ours, std, gates = [], [], []
        for r in range(a.reps):
            rng = np.random.default_rng(7919 * r + int(100 * s) + sum(map(ord, scen)))
            uc, Lc = draw(60, gen, s, False, rng)
            ut, _ = draw(40, gen, s, True, rng)
            ue, Le = draw(300, gen, s, True, rng)
            wc, _ = weights.loo_weights(uc, ut, C=1.0)
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(
                ((np.r_[uc, ut] - np.r_[uc, ut].mean()) / np.r_[uc, ut].std())[:, None],
                np.r_[np.zeros(len(uc)), np.ones(len(ut))])
            p = np.clip(clf.predict_proba(((ue - np.r_[uc, ut].mean()) / np.r_[uc, ut].std())[:, None])[:, 1],
                        1e-3, 1 - 1e-3)
            we = (len(uc) / len(ut)) * p / (1 - p)
            idx = np.broadcast_to(np.asarray(crc.weighted_crc_index(Lc, wc, we, a.alpha)), (len(ue),))
            ours.append(summ(Le, idx, idx < len(LAM) - 1))
            j = crc.crc_index(Lc, a.alpha)
            std.append(summ(Le, np.full(len(ue), j), np.ones(len(ue), bool)))
            g = []
            for q in QS:
                t = np.quantile(uc, q)
                k = uc <= t
                jj = crc.crc_index(Lc[k], a.alpha) if k.sum() >= 5 else len(LAM) - 1
                g.append(summ(Le, np.full(len(ue), jj), ue <= t))
            gates.append(g)
        O, S, G = np.array(ours, float), np.array(std, float), np.array(gates, float)
        o_cert, o_risk = O[:, 0].mean(), np.nanmean(O[:, 1])
        g_cert = G[:, :, 0].mean(0)
        jm = int(np.argmin(np.abs(g_cert - o_cert)))
        j90 = int(np.where(np.isclose(QS, 0.90))[0][0])
        rows.append({"scenario": scen, "shift": s, "assumption": scen not in ("noisy score", "confident errors"),
                     "std_risk": np.nanmean(S[:, 1]),
                     "gate90_cert": g_cert[j90], "gate90_risk": np.nanmean(G[:, j90, 1]),
                     "gate_matched_q": QS[jm], "gate_matched_cert": g_cert[jm],
                     "gate_matched_risk": np.nanmean(G[:, jm, 1]),
                     "ours_cert": o_cert, "ours_risk": o_risk,
                     "ours_risk_se": np.nanstd(O[:, 1]) / np.sqrt(np.isfinite(O[:, 1]).sum()),
                     **diff_stats(G[:, jm, 1] - O[:, 1])})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "gate_synth.csv"), index=False)


if __name__ == "__main__":
    main()
