#!/usr/bin/env bash
# Dump first/middle/last frames of W&B-logged rollout videos (v1 + v2 runs) as
# PNG contact sheets so they can be inspected from the dev box (no ffmpeg there).
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
X=$H/experiments/2026-08-27_leo2-unirl-flowgrpo
source $H/env/leo2-venv/bin/activate
export PYTHONPATH=$H/env/leo2-venv/lib/python3.12/site-packages:$PYTHONPATH
OUT=$X/docs/frames; mkdir -p $OUT
python - "$X" "$OUT" <<'PY'
import glob, os, sys
import av, numpy as np
from PIL import Image
X, OUT = sys.argv[1], sys.argv[2]
runs = sorted(glob.glob(f"{X}/wandb/wandb/run-*/files/media/videos/rollout"))
for run in runs:
    tag = run.split("/wandb/wandb/")[1].split("/")[0][-8:]
    vids = sorted(glob.glob(f"{run}/*.mp4") + glob.glob(f"{run}/*.gif"), key=os.path.getmtime)
    picks = vids[:2] + vids[-2:] if len(vids) > 4 else vids
    for v in picks:
        try:
            c = av.open(v); frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]; c.close()
        except Exception as e:
            print("skip", v, e); continue
        if not frames: continue
        idx = [0, len(frames)//2, len(frames)-1]
        sheet = np.concatenate([frames[i] for i in idx], axis=1)
        name = f"{tag}_{os.path.basename(v)[:60]}.png"
        Image.fromarray(sheet).save(f"{OUT}/{name}")
        print("wrote", name, "frames", len(frames), "size", frames[0].shape)
PY
ls -la $OUT | tail -12
