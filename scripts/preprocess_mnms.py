"""Pre-process the M&Ms cardiac MRI dataset (multi-centre, multi-vendor) into .npz.

Raw layout (as downloaded from MEGA; the three split folders are searched recursively):

    RAW/
      211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
      Training/Labeled/<CODE>/<CODE>_sa.nii.gz, <CODE>_sa_gt.nii.gz
      Validation/<CODE>/...
      Testing/<CODE>/...

The CSV gives, per subject: External code, VendorName, Vendor, Centre, ED, ES, ...
Each subject yields TWO 3D cases (end-diastole and end-systole frames).
The structure to segment is binary (default: LV blood pool, label 1); M&Ms labels are
1 = LV, 2 = myocardium, 3 = RV.  A "site" is a clinical centre (C1..C5); the vendor is kept
in the metadata.  Both phases of a subject share the same "patient" id, so that the
leave-one-centre-out split never separates them.

Steps per case: RAS reorientation, resampling to 10 x 1.25 x 1.25 mm (D,H,W), clip
[0.5, 99.5] percentiles + z-score, in-plane centre crop/pad to 192 x 192, depth padded to
16. The crop does NOT look at the labels (it would leak information at test time); the
fraction of the structure kept by the crop is reported.

    python scripts/preprocess_mnms.py --raw /kaggle/input/mnms --out /kaggle/working/mnms_npz --label lv
"""
import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

from preprocess_prostate import crop_pad, resample

LABELS = {"lv": [1], "myo": [2], "rv": [3], "lv+myo": [1, 2]}


def read_info(raw):
    csvs = glob.glob(os.path.join(raw, "**", "*.csv"), recursive=True)
    if not csvs:
        raise SystemExit("M&Ms information CSV not found under " + raw)
    df = pd.read_csv(csvs[0], sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.lower().startswith("unnamed")]]
    need = {"External code", "Vendor", "Centre", "ED", "ES"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"columns missing in {csvs[0]}: {missing}; found {list(df.columns)}")
    df["External code"] = df["External code"].astype(str).str.strip()
    return df.drop_duplicates("External code").set_index("External code")


def load_4d(path):
    """Native voxel order, NOT reoriented: cardiac short-axis stacks are oblique, and
    nibabel's closest-canonical reorientation can move the slice axis to x or y for some
    subjects (inconsistent network inputs). The slice axis is the one with the largest
    spacing (slice thickness ~ 5-10 mm vs ~1.3 mm in-plane); it is moved to position 2."""
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    if arr.ndim == 3:
        arr = arr[..., None]
    sp = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    k = int(np.argmax(sp))
    if k != 2:
        arr = np.moveaxis(arr, k, 2)
        sp = np.r_[np.delete(sp, k), sp[k]].astype(np.float32)
    return arr, sp            # (in-plane 1, in-plane 2, slice, t), spacing in the same order


def labeled_frames(gt):
    return [t for t in range(gt.shape[-1]) if gt[..., t].any()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", choices=list(LABELS), default="lv")
    ap.add_argument("--spacing", type=float, nargs=3, default=[10.0, 1.25, 1.25], help="D H W in mm")
    ap.add_argument("--size", type=int, nargs=2, default=[192, 192])
    ap.add_argument("--min_depth", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    info = read_info(a.raw)
    gts = sorted(glob.glob(os.path.join(a.raw, "**", "*_sa_gt.nii*"), recursive=True))
    print(f"{len(gts)} annotated subjects found; {len(info)} rows in the information CSV")
    rows, skipped = [], []
    for fgt in gts:
        code = re.sub(r"_sa_gt\.nii(\.gz)?$", "", os.path.basename(fgt))
        fimg = re.sub(r"_sa_gt(\.nii(\.gz)?)$", r"_sa\1", fgt)
        if code not in info.index or not os.path.exists(fimg):
            skipped.append(code)
            continue
        meta = info.loc[code]
        try:                                   # a partially downloaded .nii.gz fails here
            img4, sp = load_4d(fimg)
            gt4, _ = load_4d(fgt)
            # some label files are stored as scaled floats (e.g. 0.99999994): round to integers
            gt4 = np.rint(np.asarray(gt4, dtype=np.float32)).astype(np.int16)
        except Exception as e:
            print(f"  SKIP {code}: unreadable file ({type(e).__name__}) - re-download it")
            skipped.append(code)
            continue
        frames = labeled_frames(gt4)
        ed, es = int(meta["ED"]), int(meta["ES"])
        # CSV frame indices are 0-based; fall back on the annotated frames if inconsistent
        if not (ed in frames and es in frames):
            if len(frames) == 2:
                ed, es = frames[0], frames[1]
            else:
                skipped.append(code)
                continue
        for phase, t in (("ED", ed), ("ES", es)):
            img = np.transpose(img4[..., t].astype(np.float32), (2, 1, 0))      # (z, y, x)
            lab = np.isin(np.transpose(gt4[..., t], (2, 1, 0)), LABELS[a.label])
            if not lab.any():                  # structure not annotated in this frame: unusable
                present = sorted(int(v) for v in np.unique(gt4[..., t]) if v)
                print(f"  SKIP {code} {phase}: label {a.label} absent (labels present: {present})")
                skipped.append(f"{code}_{phase}")
                continue
            spz = sp[::-1]
            img = resample(img, spz, a.spacing, order=1)
            lab = resample(lab.astype(np.float32), spz, a.spacing, order=1) > 0.5
            lo, hi = np.percentile(img, [0.5, 99.5])
            img = np.clip(img, lo, hi)
            img = (img - img.mean()) / (img.std() + 1e-8)
            n_before = int(lab.sum())
            img_c = crop_pad(img, a.size, a.min_depth, value=float(img.min()))
            lab_c = crop_pad(lab.astype(np.uint8), a.size, a.min_depth)
            roi_c = crop_pad(np.ones(lab.shape, np.uint8), a.size, a.min_depth)
            kept = lab_c.sum() / max(n_before, 1)
            site = f"C{int(meta['Centre'])}"
            name = f"{site}_{code}_{phase}"
            np.savez_compressed(os.path.join(a.out, name + ".npz"), image=img_c.astype(np.float16),
                                label=lab_c, roi=roi_c, spacing=np.array(a.spacing, np.float32),
                                orig_spacing=spz)
            vox_ml = float(np.prod(a.spacing)) / 1000.0
            rows.append({"name": name, "site": site, "patient": code, "phase": phase,
                         "vendor": str(meta["Vendor"]), "case": f"{code}_{phase}",
                         "shape": "x".join(map(str, img_c.shape)),
                         "orig_spacing": "x".join(f"{s:.2f}" for s in spz),
                         "volume_ml": float(lab_c.sum() * vox_ml), "crop_kept": float(kept)})
            if kept < 0.99:
                print(f"  WARNING {name}: only {kept:.1%} of the structure kept by the crop")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "metadata.csv"), index=False)
    if skipped:
        print(f"{len(skipped)} subjects skipped (missing in CSV / image / inconsistent frames):", skipped[:10])
    print(df.groupby(["site", "vendor"]).agg(patients=("patient", "nunique"), cases=("name", "size"),
                                            vol_ED=("volume_ml", lambda v: v[df.loc[v.index, "phase"] == "ED"].mean()),
                                            vol_ES=("volume_ml", lambda v: v[df.loc[v.index, "phase"] == "ES"].mean()),
                                            crop_kept_min=("crop_kept", "min")).round(2))


if __name__ == "__main__":
    main()
