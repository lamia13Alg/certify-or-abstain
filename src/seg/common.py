"""Shared pieces of the 3D segmentation pipeline (pure PyTorch).

Model: nnU-Net-like residual 3D U-Net (pure PyTorch, seg/nets.py) with anisotropic first stage (z spacing 2 mm vs 0.6 mm
in-plane), dropout in every block so that MC-dropout can be used at inference.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- data


def load_cases(data_dir: str) -> list[dict]:
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    cases = []
    for r in meta.itertuples():
        z = np.load(os.path.join(data_dir, r.name + ".npz"))
        cases.append({
            "name": r.name, "site": r.site,
            # several 3D cases may come from one patient (e.g. cardiac ED / ES phases)
            "patient": str(getattr(r, "patient", r.name)), "phase": str(getattr(r, "phase", "all")),
            "image": z["image"].astype(np.float32),
            "label": z["label"].astype(np.uint8),
            "roi": z["roi"].astype(bool) if "roi" in z else np.ones(z["label"].shape, bool),
            "spacing": z["spacing"].astype(np.float32),
        })
    return cases


def make_split(cases, target_site, seed=0, frac_train=0.6, frac_val=0.1) -> dict:
    """Leave-one-centre-out split.

    target centre        -> role 'test'  (unlabeled for the method; labels only for evaluation)
    each source centre   -> 'train' / 'val' / 'cal' (stratified by centre)
    """
    rng = np.random.default_rng(seed)
    roles = {}
    sites = sorted({c["site"] for c in cases})
    if target_site not in sites:
        raise ValueError(f"{target_site} not in {sites}")
    for s in sites:
        site_cases = [c for c in cases if c["site"] == s]
        if s == target_site:
            roles.update({c["name"]: "test" for c in site_cases})
            continue
        # split PATIENTS (all cases of a patient get the same role); identical to a split by
        # case name when there is one case per patient (prostate)
        patients = sorted({c.get("patient", c["name"]) for c in site_cases})
        patients = list(rng.permutation(patients))
        n = len(patients)
        n_tr = max(1, int(round(frac_train * n)))
        n_va = max(1, int(round(frac_val * n)))
        prole = {p: ("train" if i < n_tr else ("val" if i < n_tr + n_va else "cal")) for i, p in enumerate(patients)}
        for c in site_cases:
            roles[c["name"]] = prole[c.get("patient", c["name"])]
    return {"target_site": target_site, "seed": seed, "roles": roles}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- patches


def pad_to(arr, shape, value=0.0):
    pad = [(0, max(s - a, 0)) for a, s in zip(arr.shape, shape)]
    return np.pad(arr, pad, constant_values=value) if any(p[1] for p in pad) else arr


def random_patch(img, lab, patch, rng, p_fg=0.67):
    img = pad_to(img, patch, float(img.min()))
    lab = pad_to(lab, patch, 0)
    D, H, W = img.shape
    pd_, ph, pw = patch
    if rng.random() < p_fg and lab.any():
        fg = np.argwhere(lab > 0)
        cz, cy, cx = fg[rng.integers(len(fg))]
        z0 = int(np.clip(cz - pd_ // 2 + rng.integers(-pd_ // 4, pd_ // 4 + 1), 0, D - pd_))
        y0 = int(np.clip(cy - ph // 2 + rng.integers(-ph // 4, ph // 4 + 1), 0, H - ph))
        x0 = int(np.clip(cx - pw // 2 + rng.integers(-pw // 4, pw // 4 + 1), 0, W - pw))
    else:
        z0, y0, x0 = (int(rng.integers(0, max(s - p, 0) + 1)) for s, p in zip((D, H, W), patch))
    sl = (slice(z0, z0 + pd_), slice(y0, y0 + ph), slice(x0, x0 + pw))
    return img[sl], lab[sl]


def _bias_field(shape, rng, strength):
    """Smooth multiplicative field: exp of a random 2nd-order polynomial in (z, y, x)."""
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s, dtype=np.float32) for s in shape], indexing="ij")
    terms = [zz, yy, xx, zz * yy, zz * xx, yy * xx, zz ** 2, yy ** 2, xx ** 2]
    c = rng.normal(0, strength, len(terms)).astype(np.float32)
    return np.exp(sum(ci * t for ci, t in zip(c, terms)))


def augment_strong(img, lab, rng):
    """Domain-randomisation augmentations against scanner / coil / protocol shift:
    bias field (endorectal coil), contrast & gamma, blur, simulated low resolution, noise."""
    from scipy.ndimage import gaussian_filter, zoom

    if rng.random() < 0.5:                       # left-right flip (anatomically symmetric)
        img, lab = img[:, :, ::-1], lab[:, :, ::-1]
    img = np.ascontiguousarray(img, np.float32)
    lo = float(img.min())
    if rng.random() < 0.5:                       # bias field on the positive-shifted image
        img = (img - lo) * _bias_field(img.shape, rng, 0.3) + lo
    if rng.random() < 0.5:                       # contrast
        m = img.mean()
        img = (img - m) * rng.uniform(0.65, 1.5) + m
    if rng.random() < 0.3:                       # gamma (both directions, with inversion trick)
        lo, hi = img.min(), img.max()
        x = (img - lo) / (hi - lo + 1e-8)
        g = rng.uniform(0.6, 1.6)
        x = 1 - (1 - x) ** g if rng.random() < 0.3 else x ** g
        img = x * (hi - lo) + lo
    if rng.random() < 0.2:                       # blur
        img = gaussian_filter(img, sigma=(0, rng.uniform(0.5, 1.2), rng.uniform(0.5, 1.2)))
    if rng.random() < 0.2:                       # simulated low in-plane resolution
        f = rng.uniform(0.5, 0.9)
        small = zoom(img, (1, f, f), order=1)
        img = zoom(small, (1, img.shape[1] / small.shape[1], img.shape[2] / small.shape[2]), order=1)
    if rng.random() < 0.3:                       # noise
        img = img + rng.normal(0, rng.uniform(0, 0.15), img.shape).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)   # re-standardise (matches preprocessing)
    return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(lab)


def augment(img, lab, rng, strong: bool = False):
    if strong:
        return augment_strong(img, lab, rng)
    if rng.random() < 0.5:                       # left-right flip
        img, lab = img[:, :, ::-1], lab[:, :, ::-1]
    if rng.random() < 0.3:                       # anterior-posterior flip
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5:                       # intensity scale / shift
        img = img * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
    if rng.random() < 0.3:                       # gamma on min-max normalised intensities
        lo, hi = img.min(), img.max()
        x = (img - lo) / (hi - lo + 1e-8)
        img = x ** rng.uniform(0.7, 1.5) * (hi - lo) + lo
    if rng.random() < 0.2:                       # gaussian noise
        img = img + rng.normal(0, rng.uniform(0, 0.1), img.shape)
    return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(lab)


# ----------------------------------------------------------------------------- model


KERNELS = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
STRIDES = [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
FILTERS = [32, 64, 128, 256, 320]


def build_model(dropout: float = 0.1):
    """Pure-PyTorch residual 3D U-Net (no MONAI dependency)."""
    from seg.nets import ResUNet3D

    return ResUNet3D(1, 2, FILTERS, [tuple(k) for k in KERNELS], [tuple(s) for s in STRIDES], dropout)


def enable_mc_dropout(model):
    """eval mode everywhere except dropout layers (MC-dropout)."""
    import torch.nn as nn

    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout3d, nn.Dropout2d)):
            m.train()


class EmbeddingHook:
    """Collects the global-average-pooled bottleneck features over all sliding windows."""

    def __init__(self, module):
        self.feats = []
        self.h = module.register_forward_hook(self._hook)
        self.active = False

    def _hook(self, mod, inp, out):
        if self.active:
            self.feats.append(out.float().mean(dim=(2, 3, 4)).detach().cpu())

    def pop(self) -> np.ndarray:
        import torch

        f = torch.cat(self.feats, 0).mean(0).numpy() if self.feats else None
        self.feats = []
        return f

    def remove(self):
        self.h.remove()


def find_ckpt(out_dir, name):
    hits = sorted(glob.glob(os.path.join(out_dir, name)))
    return hits[-1] if hits else None
