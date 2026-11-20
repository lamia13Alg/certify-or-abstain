"""Pure-PyTorch 3D network and sliding-window inference (no MONAI dependency).

ResUNet3D: nnU-Net-like residual U-Net with anisotropic first stages
(prostate MRI: 2 mm between slices vs 0.6 mm in-plane), instance norm,
LeakyReLU and element-wise dropout in every block (for MC-dropout).
The deepest stage is exposed as `model.bottleneck` (embedding hook).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad(k):
    return tuple(kk // 2 for kk in k)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, kernel, stride=(1, 1, 1), dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, kernel, stride, _pad(kernel), bias=False)
        self.norm1 = nn.InstanceNorm3d(cout, affine=True)
        self.conv2 = nn.Conv3d(cout, cout, kernel, 1, _pad(kernel), bias=False)
        self.norm2 = nn.InstanceNorm3d(cout, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Identity() if cin == cout and tuple(stride) == (1, 1, 1) else
                     nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.InstanceNorm3d(cout, affine=True)))

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return self.act(y + self.skip(x))


class ResUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=2,
                 filters=(32, 64, 128, 256, 320),
                 kernels=((1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
                 strides=((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
                 dropout=0.1):
        super().__init__()
        n = len(filters)
        self.encoders = nn.ModuleList()
        cin = in_ch
        for i in range(n - 1):
            self.encoders.append(ResBlock(cin, filters[i], kernels[i], strides[i], dropout))
            cin = filters[i]
        self.bottleneck = ResBlock(cin, filters[-1], kernels[-1], strides[-1], dropout)
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for i in range(n - 2, -1, -1):
            s = strides[i + 1]
            self.ups.append(nn.ConvTranspose3d(filters[i + 1], filters[i], s, s, bias=False))
            self.decoders.append(ResBlock(2 * filters[i], filters[i], kernels[i], (1, 1, 1), dropout))
        self.head = nn.Conv3d(filters[0], out_ch, 1)
        # total downsampling factor per axis (input sizes must be multiples of it)
        self.factor = tuple(int(np.prod([s[d] for s in strides])) for d in range(3))

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, dec in zip(self.ups, self.decoders):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)


def gaussian_map(patch, sigma_scale=0.125, device="cpu"):
    grids = []
    for p in patch:
        c = (p - 1) / 2
        g = torch.exp(-0.5 * ((torch.arange(p, device=device) - c) / (sigma_scale * p)) ** 2)
        grids.append(g)
    m = grids[0][:, None, None] * grids[1][None, :, None] * grids[2][None, None, :]
    m = m / m.max()
    return m.clamp_min(1e-3)


def _starts(size, p, overlap):
    if size <= p:
        return [0]
    step = max(1, int(p * (1 - overlap)))
    st = list(range(0, size - p + 1, step))
    if st[-1] != size - p:
        st.append(size - p)
    return st


@torch.no_grad()
def sliding_window(x, patch, model, overlap=0.5, batch=4, amp=True):
    """Gaussian-weighted sliding-window prediction. x: [1, C, D, H, W] -> logits [1, K, D, H, W]."""
    _, _, D, H, W = x.shape
    pads = [max(p - s, 0) for p, s in zip(patch, (D, H, W))]
    xp = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]), mode="replicate") if any(pads) else x
    Dp, Hp, Wp = xp.shape[2:]
    g = gaussian_map(patch, device=x.device)
    out, norm = None, torch.zeros((Dp, Hp, Wp), device=x.device)
    coords = list(itertools.product(_starts(Dp, patch[0], overlap), _starts(Hp, patch[1], overlap),
                                    _starts(Wp, patch[2], overlap)))
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        xb = torch.cat([xp[:, :, z:z + patch[0], y:y + patch[1], w:w + patch[2]] for z, y, w in cs])
        with torch.autocast("cuda", enabled=amp and x.device.type == "cuda"):
            yb = model(xb).float()
        if out is None:
            out = torch.zeros((1, yb.shape[1], Dp, Hp, Wp), device=x.device)
        for (z, y, w), yy in zip(cs, yb):
            out[0, :, z:z + patch[0], y:y + patch[1], w:w + patch[2]] += yy * g
            norm[z:z + patch[0], y:y + patch[1], w:w + patch[2]] += g
    out = out / norm
    return out[:, :, :D, :H, :W]
