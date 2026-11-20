"""Inference for one fold: stochastic probability maps -> patient-level summaries.

For every calibration (source) and test (target) patient:
  * a deterministic pass (eval mode) that also records the bottleneck embedding
  * K MC-dropout passes (alternating left-right flip TTA)
  * summary (score histograms for several beta, volumes, uncertainty features)

Everything the conformal layer needs is stored in  <run>/summaries.npz  (a few MB),
so the GPU is used once and all conformal experiments run on CPU.

    python scripts/infer_summaries.py --data /kaggle/working/prostate_npz --run /kaggle/working/runs/BIDMC --mc 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scrc.summary import stack_summaries, summarize_case  # noqa: E402
from seg.nets import sliding_window  # noqa: E402
from seg.common import EmbeddingHook, build_model, enable_mc_dropout, load_cases, load_json  # noqa: E402


@torch.no_grad()
def predict(model, x, patch, device, flip=False):
    if flip:
        x = torch.flip(x, dims=[-1])
    logit = sliding_window(x, patch, model, overlap=0.5, batch=4)
    p = torch.softmax(logit.float(), 1)[:, 1]
    if flip:
        p = torch.flip(p, dims=[-1])
    return p[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="fold directory created by train_seg.py")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--mc", type=int, default=8, help="number of MC-dropout samples")
    ap.add_argument("--roles", nargs="+", default=["cal", "test", "val"])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--save_maps", type=int, default=3, help="probability maps saved per role (figures)")
    ap.add_argument("--out_name", default="summaries.npz", help="e.g. summaries_mc16.npz for the MC ablation")
    a = ap.parse_args()
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_json(os.path.join(a.run, "split.json"))
    ck = torch.load(os.path.join(a.run, a.ckpt), map_location=device, weights_only=False)
    targs = ck["args"]
    patch = tuple(targs["patch"])
    model = build_model(targs["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    hook = EmbeddingHook(model.bottleneck)

    cases = [c for c in load_cases(a.data) if split["roles"][c["name"]] in a.roles]
    summaries, meta, embs = [], [], []
    maps_dir = os.path.join(a.run, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    n_saved = {}
    for i, c in enumerate(cases):
        role = split["roles"][c["name"]]
        x = torch.from_numpy(c["image"])[None, None].to(device)
        model.eval()
        hook.active = True
        p_det = predict(model, x, patch, device)
        hook.active = False
        emb = hook.pop()
        samples = [p_det]
        if a.mc > 0:
            enable_mc_dropout(model)
            for k in range(a.mc):
                samples.append(predict(model, x, patch, device, flip=bool(k % 2)))
            model.eval()
        ps = np.stack(samples).astype(np.float32)
        vox_ml = float(np.prod(c["spacing"])) / 1000.0
        s = summarize_case(ps, c["label"], vox_ml, betas=a.betas, roi=c["roi"])
        summaries.append(s)
        meta.append({"name": c["name"], "site": c["site"], "role": role,
                     "patient": c["patient"], "phase": c["phase"]})
        embs.append(emb)
        if n_saved.get(role, 0) < a.save_maps:
            n_saved[role] = n_saved.get(role, 0) + 1
            np.savez_compressed(os.path.join(maps_dir, c["name"] + ".npz"),
                                image=c["image"].astype(np.float16), label=c["label"],
                                mean=ps.mean(0).astype(np.float16), std=ps.std(0).astype(np.float16))
        dice = 2 * ((ps.mean(0) >= 0.5) & (c["label"] > 0)).sum() / ((ps.mean(0) >= 0.5).sum() + c["label"].sum() + 1e-8)
        print(f"[{i + 1}/{len(cases)}] {c['name']:<20s} {role:<5s} dice@0.5={dice:.3f}  "
              f"vol={s['vol_true_ml']:.1f}ml  {(time.time() - t0) / 60:.1f} min", flush=True)

    out = stack_summaries(summaries, meta)
    out["emb"] = np.stack(embs)
    out["target_site"] = np.array(split["target_site"])
    np.savez_compressed(os.path.join(a.run, a.out_name), **out)
    print("saved", os.path.join(a.run, a.out_name))


if __name__ == "__main__":
    main()
