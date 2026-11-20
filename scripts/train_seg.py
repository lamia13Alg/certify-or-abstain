"""Train the 3D segmentation model for one leave-one-centre-out fold.

Designed for Kaggle (T4/P100, 16 GB, 12 h sessions): patch training, AMP,
checkpoint every --ckpt_every iterations, automatic resume from last.pt.

    python scripts/train_seg.py --data /kaggle/working/prostate_npz \
        --target_site BIDMC --out /kaggle/working/runs/BIDMC --iters 10000
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from seg.nets import sliding_window  # noqa: E402
from seg.common import (augment, build_model, load_cases, make_split, random_patch,  # noqa: E402
                        save_json)


def dice_ce_loss(logits, target):
    ce = F.cross_entropy(logits, target)
    p = torch.softmax(logits.float(), 1)[:, 1]
    t = (target == 1).float()
    inter = (p * t).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1e-5) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1e-5)
    return ce + dice.mean()


@torch.no_grad()
def validate(model, cases, patch, device):
    model.eval()
    dices = []
    for c in cases:
        x = torch.from_numpy(c["image"])[None, None].to(device)
        logit = sliding_window(x, patch, model, overlap=0.5, batch=4)
        pred = (logit.argmax(1)[0].cpu().numpy() == 1)
        lab = c["label"] > 0
        dices.append(2 * (pred & lab).sum() / (pred.sum() + lab.sum() + 1e-8))
    model.train()
    return float(np.mean(dices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target_site", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, nargs=3, default=[32, 128, 128])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", choices=["default", "strong"], default="default",
                    help="strong = domain randomisation (bias field, contrast, blur, low-res)")
    ap.add_argument("--max_hours", type=float, default=11.0, help="stop cleanly before Kaggle's 12 h limit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = load_cases(a.data)
    split = make_split(cases, a.target_site, seed=a.seed)
    save_json(split, os.path.join(a.out, "split.json"))
    train = [c for c in cases if split["roles"][c["name"]] == "train"]
    val = [c for c in cases if split["roles"][c["name"]] == "val"]
    print(f"target={a.target_site}  train={len(train)}  val={len(val)}  "
          f"cal={sum(r == 'cal' for r in split['roles'].values())}  "
          f"test={sum(r == 'test' for r in split['roles'].values())}  device={device}")

    model = build_model(a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    it, best = 0, -1.0

    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        it, best = ck["it"], ck["best"]
        rng = np.random.default_rng(a.seed + it)
        print(f"resumed from iteration {it} (best val dice {best:.4f})")

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "it": it, "best": best, "args": vars(a)}, path)

    log = open(os.path.join(a.out, "train_log.csv"), "a")
    model.train()
    run_loss = 0.0
    while it < a.iters:
        xs, ys = [], []
        for _ in range(a.batch):
            c = train[rng.integers(len(train))]
            x, y = random_patch(c["image"], c["label"], a.patch, rng)
            x, y = augment(x, y, rng, strong=a.aug == "strong")
            xs.append(x); ys.append(y)
        x = torch.from_numpy(np.stack(xs))[:, None].to(device, non_blocking=True)
        y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
        loss = dice_ce_loss(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(opt); scaler.update(); sched.step()
        it += 1
        run_loss += loss.item()

        if it % 50 == 0:
            print(f"it {it:6d}  loss {run_loss / 50:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            run_loss = 0.0
        if it % a.val_every == 0 or it == a.iters:
            d = validate(model, val, a.patch, device)
            log.write(f"{it},{d:.5f}\n"); log.flush()
            if d > best:
                best = d
                torch.save({"model": model.state_dict(), "it": it, "val_dice": d, "args": vars(a)},
                           os.path.join(a.out, "best.pt"))
            print(f"  val dice {d:.4f}  (best {best:.4f})", flush=True)
        if it % a.ckpt_every == 0:
            save(last)
        if (time.time() - t0) / 3600 > a.max_hours:
            save(last)
            print("time budget reached, checkpoint saved: re-run the same command to resume")
            return
    save(last)
    print(f"done. best val dice {best:.4f}")


if __name__ == "__main__":
    main()
