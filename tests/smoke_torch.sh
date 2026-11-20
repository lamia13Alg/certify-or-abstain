#!/usr/bin/env bash
# Smoke test of the PyTorch part on CPU (a few minutes):
# fake NIfTI -> preprocess -> train 20 iterations -> MC inference -> conformal analysis
set -e
W=${1:-/tmp/scrc_smoke}
HERE=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$W"
python "$HERE/tests/fake_pipeline.py" --work "$W" --n 8 > /dev/null   # creates $W/npz (fake predictor ignored)
rm -rf "$W/runs"
python "$HERE/scripts/train_seg.py" --data "$W/npz" --target_site SITED --out "$W/runs/SITED" \
       --iters 20 --val_every 10 --ckpt_every 10 --batch 1
python "$HERE/scripts/train_seg.py" --data "$W/npz" --target_site SITED --out "$W/runs/SITED" \
       --iters 30 --val_every 10 --ckpt_every 10 --batch 1          # resume test
python "$HERE/scripts/infer_summaries.py" --data "$W/npz" --run "$W/runs/SITED" --mc 2
python "$HERE/scripts/run_conformal.py" --runs "$W/runs/*" --out "$W/results" --reps 5 --pca 2 > /dev/null
echo "SMOKE TEST OK"
