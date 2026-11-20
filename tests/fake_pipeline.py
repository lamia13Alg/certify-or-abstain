"""End-to-end CPU check of the pipeline without a GPU / PyTorch.

1. writes fake multi-site NIfTI volumes in the raw layout of the prostate dataset
2. runs scripts/preprocess_prostate.py on them
3. replaces the network by a fake predictor whose quality degrades on some sites
   (under-segmentation, so the FNR rises under shift) and writes summaries.npz
   exactly like scripts/infer_summaries.py
4. runs scripts/run_conformal.py

    python tests/fake_pipeline.py --work /tmp/fake
"""
import argparse
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from scrc.summary import stack_summaries, summarize_case  # noqa: E402
from seg.common import load_cases, make_split, save_json  # noqa: E402

SITES = {"SITEA": 0.0, "SITEB": 0.0, "SITEC": 0.3, "SITED": 0.6}   # site -> degradation


def write_raw(raw, n_per_site, rng):
    for s, deg in SITES.items():
        os.makedirs(os.path.join(raw, s), exist_ok=True)
        for i in range(n_per_site):
            shape = (160, 160, 24)                       # x, y, z
            zz, yy, xx = np.meshgrid(np.arange(24), np.arange(160), np.arange(160), indexing="ij")
            r = rng.uniform(14, 22)
            c = 80 + rng.normal(0, 4, 2)
            lab = (((xx - c[0]) / r) ** 2 + ((yy - c[1]) / (r * 0.8)) ** 2 + ((zz - 12) / 6) ** 2) < 1
            img = 100 + 60 * lab + rng.normal(0, 10 + 30 * deg, lab.shape)
            img = gaussian_filter(img, 1)
            aff = np.diag([0.5, 0.5, 3.0, 1])
            to_nib = lambda a: np.transpose(a, (2, 1, 0))  # (z,y,x) -> (x,y,z)
            nib.save(nib.Nifti1Image(to_nib(img).astype(np.float32), aff), os.path.join(raw, s, f"Case{i:02d}.nii.gz"))
            nib.save(nib.Nifti1Image(to_nib(lab).astype(np.uint8), aff),
                     os.path.join(raw, s, f"Case{i:02d}_segmentation.nii.gz"))


def fake_predict(c, deg, rng, K=4):
    lab = c["label"] > 0
    er = binary_erosion(lab, iterations=int(1 + 3 * deg * rng.uniform(0.5, 1.5)))
    base = gaussian_filter(er.astype(np.float32), 1.5)
    samples = []
    for _ in range(K):
        noise = gaussian_filter(rng.normal(0, 0.15 + 0.3 * deg, lab.shape), 2)
        samples.append(np.clip(base + noise, 0, 1))
    return np.stack(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/fake_scrc")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    raw, npz, runs = (os.path.join(a.work, d) for d in ("raw", "npz", "runs"))
    write_raw(raw, a.n, rng)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts/preprocess_prostate.py"),
                    "--raw", raw, "--out", npz], check=True)
    cases = load_cases(npz)
    for target in SITES:
        run = os.path.join(runs, target)
        split = make_split(cases, target)
        save_json(split, os.path.join(run, "split.json"))
        S, M, E = [], [], []
        for c in cases:
            role = split["roles"][c["name"]]
            if role not in ("cal", "test", "val"):
                continue
            deg = SITES[c["site"]]
            ps = fake_predict(c, deg, rng)
            S.append(summarize_case(ps, c["label"], float(np.prod(c["spacing"])) / 1000, roi=c["roi"]))
            M.append({"name": c["name"], "site": c["site"], "role": role})
            img = c["image"]
            E.append(np.r_[img.std(), np.abs(np.diff(img, axis=2)).mean(), deg + 0.1 * rng.normal(size=6)])
        out = stack_summaries(S, M)
        out["emb"] = np.stack(E)
        out["target_site"] = np.array(target)
        np.savez_compressed(os.path.join(run, "summaries.npz"), **out)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts/run_conformal.py"),
                    "--runs", os.path.join(runs, "*"), "--out", os.path.join(a.work, "results"),
                    "--reps", "30", "--pca", "2"], check=True)


if __name__ == "__main__":
    main()
