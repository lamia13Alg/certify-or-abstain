import numpy as np
import pytest

from scrc import crc, losses, selective, summary, synthetic, volume, weights


# --------------------------------------------------------------------- summaries
def test_histogram_losses_match_direct_computation():
    rng = np.random.default_rng(0)
    lab = np.zeros((8, 16, 16), bool)
    lab[2:6, 4:12, 4:12] = True
    p = np.clip(lab * 0.7 + rng.normal(0, 0.2, lab.shape), 0, 1)
    s = summary.summarize_case(p[None], lab, voxel_volume_ml=0.001, betas=(0.0,), n_bins=100)
    lam = losses.lambda_grid(100)
    fnr = losses.fnr_curve(s["hist_all"][0], s["hist_pos"][0], s["n_pos"])
    vol = losses.volume_curve(s["hist_all"][0], 0.001)
    for k in [10, 37, 50, 80]:
        t = 1 - lam[k]
        # mask = bins whose lower edge >= t  <=>  floor(p*100)/100 >= t
        b = np.minimum(np.floor(p * 100), 99) / 100
        m = b >= t - 1e-12
        assert np.isclose(fnr[k], 1 - (m & lab).sum() / lab.sum())
        assert np.isclose(vol[k], m.sum() * 0.001)
    assert fnr[0] == 1 and fnr[-1] == 0
    assert np.all(np.diff(fnr) <= 1e-12)


def test_summary_with_samples_and_roi():
    rng = np.random.default_rng(1)
    ps = rng.uniform(size=(4, 6, 10, 10))
    lab = rng.uniform(size=(6, 10, 10)) > 0.7
    roi = np.ones_like(lab)
    roi[:, :2] = False
    s = summary.summarize_case(ps, lab, 0.5, roi=roi)
    assert s["hist_all"].shape == (4, 1000)
    assert s["hist_all"][0].sum() == roi.sum()
    assert s["n_pos"] == (lab & roi).sum()
    assert s["vol_samples_ml"].shape == (4,)


# --------------------------------------------------------------------- CRC
def _mc_risk(method, shift, reps=300, alpha=0.1, seed=0):
    risks = []
    for r in range(reps):
        P = synthetic.make_problem(n_cal=100, n_test=500, shift=shift, seed=seed + r)
        if method == "crc":
            idx = crc.crc_index(P["L_cal"], alpha)
        elif method == "wcrc_true":
            idx = crc.weighted_crc_index(P["L_cal"], P["w_cal_true"], P["w_test_true"], alpha)
        risks.append(crc.realized_risk(P["L_test"], idx))
    return float(np.mean(risks))


def test_crc_valid_without_shift():
    assert _mc_risk("crc", shift=0.0) <= 0.1 + 0.005


def test_crc_breaks_under_shift_and_weighted_crc_repairs_it():
    r_std = _mc_risk("crc", shift=1.0)
    r_w = _mc_risk("wcrc_true", shift=1.0)
    assert r_std > 0.12          # standard CRC violates alpha = 0.1
    assert r_w <= 0.1 + 0.01     # weighted CRC (true weights) controls it


def test_estimated_weights_detect_shift():
    P = synthetic.make_problem(n_cal=150, n_test=150, shift=1.0, seed=3)
    ws, wt, info = weights.estimate_weights(P["x_cal"], P["x_test"])
    assert info["domain_auc"] > 0.65
    assert np.corrcoef(np.log(ws), np.log(P["w_cal_true"]))[0, 1] > 0.5
    P0 = synthetic.make_problem(n_cal=150, n_test=150, shift=0.0, seed=4)
    _, _, info0 = weights.estimate_weights(P0["x_cal"], P0["x_test"])
    assert info0["domain_auc"] < 0.65


def test_truncation_and_ess():
    w = np.r_[np.ones(99), 100.0]
    assert crc.effective_sample_size(w) < 10
    assert crc.truncate_weights(w, max_w=5).max() == 5


# --------------------------------------------------------------------- selective
def test_joint_risk_tau_is_valid():
    alpha, risks = 0.05, []
    for r in range(200):
        P = synthetic.make_problem(n_cal=200, n_test=1000, shift=0.0, seed=100 + r)
        k = 250  # fixed mask threshold index
        l_cal, l_te = P["L_cal"][:, k], P["L_test"][:, k]
        taus = np.quantile(P["u_cal"], np.linspace(0, 1, 101))
        tau = selective.joint_risk_tau(l_cal, P["u_cal"], alpha, taus)[0]
        risks.append(np.mean(l_te * (P["u_test"] <= tau)))
    assert np.mean(risks) <= alpha + 0.005


def test_ltt_selective_high_probability():
    alpha, delta, viol = 0.15, 0.1, 0
    reps = 200
    for r in range(reps):
        P = synthetic.make_problem(n_cal=300, n_test=3000, shift=0.0, seed=500 + r)
        k = 250
        taus = np.quantile(P["u_cal"], np.linspace(0.05, 1, 40))
        tau = selective.ltt_selective_tau(P["L_cal"][:, k], P["u_cal"], alpha, delta, taus)
        acc = P["u_test"] <= tau
        if acc.sum() > 0 and P["L_test"][acc, k].mean() > alpha:
            viol += 1
    assert viol / reps <= delta + 0.03


# --------------------------------------------------------------------- volume
def test_weighted_quantile_reduces_to_standard_with_unit_weights():
    rng = np.random.default_rng(0)
    s = rng.exponential(size=99)
    q = volume.conformal_quantile(s, 0.1)
    qw = volume.weighted_conformal_quantile(s, np.ones(99), 1.0, 0.1)[0]
    assert np.isclose(q, qw)


def test_volume_interval_coverage():
    covs = []
    rng = np.random.default_rng(0)
    for _ in range(300):
        vh = rng.uniform(20, 60, 200)
        vt = vh * (1 + 0.1 * rng.normal(size=200))
        q = volume.conformal_quantile(volume.relative_scores(vt[:100], vh[:100]), 0.1)
        lo, hi = volume.relative_interval(vh[100:], q)
        covs.append(volume.coverage_width(vt[100:], lo, hi)[0])
    assert np.mean(covs) >= 0.9 - 0.01


# --------------------------------------------------------------------- two-sided
def test_upper_index_matches_bruteforce():
    from scrc import twosided
    rng = np.random.default_rng(0)
    L = np.sort(rng.uniform(size=(40, 51)), axis=1)          # non-decreasing losses
    w, wt = rng.uniform(0.5, 2, 40), np.array([0.7, 3.0])
    hi = twosided.weighted_crc_upper_index(L, w, wt, 0.4)
    for j, t in enumerate(wt):
        adj = (w @ L) / (w.sum() + t) + t / (w.sum() + t)
        ok = np.where(adj <= 0.4)[0]
        assert hi[j] == (ok.max() if len(ok) else -1)


def test_certify_or_abstain_valid_on_synthetic_two_sided():
    from scrc import twosided
    alpha_fnr, alpha_fdr, risks = 0.1, 0.2, []
    for r in range(200):
        P = synthetic.make_problem(n_cal=100, n_test=500, shift=0.0, seed=900 + r)
        fdr_cal = 1 - P["L_cal"][:, ::-1] ** 0.5               # synthetic non-decreasing FDR-like loss
        fdr_te = 1 - P["L_test"][:, ::-1] ** 0.5
        idx, cert = twosided.certify_or_abstain(P["L_cal"], fdr_cal, np.ones(100), np.ones(500), alpha_fnr, alpha_fdr)
        idx = np.clip(idx, 0, P["L_cal"].shape[1] - 1)
        risks.append((np.mean(P["L_test"][np.arange(500), idx] * cert),
                      np.mean(fdr_te[np.arange(500), idx] * cert)))
    m = np.mean(risks, 0)
    assert m[0] <= alpha_fnr + 0.005 and m[1] <= alpha_fdr + 0.005
