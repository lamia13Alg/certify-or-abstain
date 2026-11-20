"""Pre-process the Multi-site Prostate MRI dataset (Liu et al., 6 sites) into .npz.

Expected raw layout (as distributed by the authors; names are matched loosely):

    RAW/
      BIDMC/ Case00.nii.gz  Case00_segmentation.nii.gz  ...
      BMC/   ...
      HK/    ...
      I2CVB/ ...
      RUNMC/ ...
      UCL/   ...

Any file whose name contains "seg" (case-insensitive) is a label; the image is
the file with the same name without the "_seg..." suffix.

Steps per case
  1. reorient to RAS (closest canonical)
  2. resample to the target spacing (default 0.6 x 0.6 x 2.0 mm)
  3. intensity: clip to [0.5, 99.5] percentiles, z-score
  4. in-plane centre crop / pad to 192 x 192, pad depth to >= 32
  5. label binarised (whole prostate: any label > 0)
Output: OUT/<SITE>_<case>.npz with image (D,H,W float16), label (uint8),
spacing (D,H,W mm), and OUT/metadata.csv.

    python scripts/preprocess_prostate.py --raw /kaggle/input/prostate-multisite --out /kaggle/working/prostate_npz
"""
import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

SITES = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]


def find_pairs(raw):
    pairs = []
    for site_dir in sorted(glob.glob(os.path.join(raw, "*"))):
        if not os.path.isdir(site_dir):
            continue
        site = os.path.basename(site_dir)
        files = glob.glob(os.path.join(site_dir, "**", "*.nii*"), recursive=True)
        labels = [f for f in files if re.search("seg", os.path.basename(f), re.I)]
        images = set(files) - set(labels)
        by_stem = {re.sub(r"\.nii(\.gz)?$", "", os.path.basename(f)).lower(): f for f in images}
        for lf in sorted(labels):
            stem = re.sub(r"[_\-]?seg\w*\.nii(\.gz)?$", "", os.path.basename(lf), flags=re.I).lower()
            if stem in by_stem:
                pairs.append((site, stem, by_stem[stem], lf))
    return pairs


def load_ras(path):
    img = nib.as_closest_canonical(nib.load(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0]
    sp = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    # nibabel order (x, y, z)  ->  (z, y, x) = (D, H, W)
    return np.transpose(arr, (2, 1, 0)), sp[::-1]


def resample(arr, sp, target, order):
    f = sp / np.asarray(target)
    return zoom(arr, f, order=order, mode="nearest", prefilter=False) if np.any(np.abs(f - 1) > 1e-3) else arr


def crop_pad(arr, size_hw, min_d, value=0):
    D, H, W = arr.shape
    th, tw = size_hw
    out = arr
    # in-plane centre crop
    if H > th:
        s = (H - th) // 2
        out = out[:, s:s + th]
    if W > tw:
        s = (W - tw) // 2
        out = out[:, :, s:s + tw]
    D, H, W = out.shape
    pad = []
    for cur, tgt in ((D, min_d), (H, th), (W, tw)):
        p = max(tgt - cur, 0)
        pad.append((p // 2, p - p // 2))
    return np.pad(out, pad, constant_values=value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, nargs=3, default=[2.0, 0.6, 0.6], help="D H W in mm")
    ap.add_argument("--size", type=int, nargs=2, default=[192, 192])
    ap.add_argument("--min_depth", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pairs = find_pairs(a.raw)
    if not pairs:
        raise SystemExit(f"No image/label pairs found under {a.raw}")
    print(f"{len(pairs)} cases found:", pd.Series([p[0] for p in pairs]).value_counts().to_dict())
    rows = []
    for site, case, fimg, flab in pairs:
        img, sp = load_ras(fimg)
        lab, _ = load_ras(flab)
        img = resample(img, sp, a.spacing, order=1)
        lab = resample((lab > 0).astype(np.float32), sp, a.spacing, order=1) > 0.5
        lo, hi = np.percentile(img, [0.5, 99.5])
        img = np.clip(img, lo, hi)
        img = (img - img.mean()) / (img.std() + 1e-8)
        n_before = int(lab.sum())
        img_c = crop_pad(img, a.size, a.min_depth, value=float(img.min()))
        lab_c = crop_pad(lab.astype(np.uint8), a.size, a.min_depth)
        roi_c = crop_pad(np.ones(lab.shape, np.uint8), a.size, a.min_depth)  # 0 on padding
        kept = lab_c.sum() / max(n_before, 1)
        if kept < 0.99:
            print(f"  WARNING {site}/{case}: only {kept:.1%} of the prostate kept by the crop")
        name = f"{site}_{case}"
        np.savez_compressed(os.path.join(a.out, name + ".npz"), image=img_c.astype(np.float16),
                            label=lab_c.astype(np.uint8), roi=roi_c, spacing=np.array(a.spacing, np.float32),
                            orig_spacing=sp)
        vox_ml = float(np.prod(a.spacing)) / 1000.0
        rows.append({"name": name, "site": site, "case": case, "shape": "x".join(map(str, img_c.shape)),
                     "orig_spacing": "x".join(f"{s:.2f}" for s in sp),
                     "prostate_ml": float(lab_c.sum() * vox_ml), "crop_kept": float(kept)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "metadata.csv"), index=False)
    print(df.groupby("site").agg(n=("name", "size"), vol_ml=("prostate_ml", "mean")).round(1))


if __name__ == "__main__":
    main()
