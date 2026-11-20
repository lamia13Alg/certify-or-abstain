"""Why were some annotated M&Ms subjects not preprocessed?

    python scripts/diagnose_mnms.py MnMs mnms_npz

Prints, for every annotated subject absent from the preprocessed set, its centre and the reason
(missing in the CSV, image file missing, file unreadable or truncated, ED/ES frames not annotated),
then a count per centre and per reason.
"""
import collections
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


def main():
    import nibabel as nib
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="?", default="MnMs", help="raw M&Ms folder (images, masks, information CSV)")
    ap.add_argument("npz", nargs="?", default="mnms_npz", help="output folder of preprocess_mnms.py")
    a = ap.parse_args()
    raw, npz = a.raw, a.npz
    done = set(pd.read_csv(os.path.join(npz, "metadata.csv"))["patient"].astype(str))
    csvs = glob.glob(os.path.join(raw, "**", "*.csv"), recursive=True)
    info = pd.read_csv(csvs[0], sep=None, engine="python")
    info.columns = [str(c).strip() for c in info.columns]
    info["External code"] = info["External code"].astype(str).str.strip()
    info = info.drop_duplicates("External code").set_index("External code")
    gts = {re.sub(r"_sa_gt\.nii(\.gz)?$", "", os.path.basename(f)): f
           for f in glob.glob(os.path.join(raw, "**", "*_sa_gt.nii*"), recursive=True)}
    per_centre, per_reason = collections.Counter(), collections.Counter()
    missing = sorted(set(gts) - done)
    print(f"{len(gts)} annotated subjects, {len(done)} preprocessed, {len(missing)} not preprocessed")
    for code in missing:
        centre = f"C{int(info.loc[code, 'Centre'])}" if code in info.index else "?"
        fgt = gts.get(code)
        fimg = re.sub(r"_sa_gt(\.nii(\.gz)?)$", r"_sa\1", fgt) if fgt else None
        if code not in info.index:
            reason = "not in the CSV"
        elif not fimg or not os.path.exists(fimg):
            reason = "image file (_sa.nii.gz) missing"
        else:
            try:
                img = np.asarray(nib.load(fimg).dataobj)
                gt = np.rint(np.asarray(nib.load(fgt).dataobj, dtype=np.float32))
                frames = [t for t in range(gt.shape[-1]) if gt[..., t].any()]
                ed, es = int(info.loc[code, "ED"]), int(info.loc[code, "ES"])
                if ed in frames and es in frames:
                    lv = [t for t in (ed, es) if not (gt[..., t] == 1).any()]
                    reason = "readable, ED/ES annotated" + (f"; LV label absent in frame(s) {lv}" if lv else
                                                            " -> should not have been skipped")
                else:
                    reason = f"ED/ES ({ed},{es}) not among annotated frames {frames}"
            except Exception as e:
                size = os.path.getsize(fimg) / 1e6
                reason = f"unreadable ({type(e).__name__}), image {size:.1f} MB -> re-download"
        per_centre[centre] += 1
        per_reason[reason.split(" (")[0].split(";")[0]] += 1
        print(f"{code}  {centre}  {reason}")
    print("\nper centre:", dict(per_centre))
    print("per reason:", dict(per_reason))


if __name__ == "__main__":
    main()
