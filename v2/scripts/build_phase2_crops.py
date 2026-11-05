"""Precompute Phase 2 CNN crops once, so training does not re-derive them.

The first version of ``train_phase2_cnn.py`` called ``generate_candidates`` inside
``__getitem__``. That regenerates a 2048^2 connected-component labelling for
*every* sample, ~0.8 s each, so one epoch over 6,166 candidates took ~82 minutes
and the smoke test had to be killed. The candidate map depends only on the file,
not on the sample, so it is computed once here.

Each crop is stored as a uint8 4-channel 256^2 array (4 x 65,536 = 262 KB raw,
~40 KB compressed). 6,166 crops is ~250 MB, which is cheap next to the 1.3 GB of
OOF dumps it is derived from.

Channels:
0. flattened image, normalised as the segmentation model saw it (scaled to uint8)
1. ensemble probability
2. candidate mask
3. r/R, normalised distance from the candidate centroid

    python v2/scripts/build_phase2_crops.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.instances import generate_candidates  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"
P2 = ROOT / "v2/runs/phase2"
CACHE = ROOT / "cache/preprocess/flatten"

MEAN, STD = 0.4254, 0.0809
CROP = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--min-area", type=int, default=128)
    ap.add_argument("--out", default=str(P2 / "crops.npy"))
    args = ap.parse_args()

    z = np.load(P2 / "candidates.npz", allow_pickle=True)
    q_all, groups = z["cand_y"], z["cand_group"]
    paths = sorted(OOF.glob("fold*/*.npz"))

    crops: list[np.ndarray] = []
    qs: list[float] = []
    file_of: list[int] = []
    cids: list[int] = []

    t0 = time.time()
    for i, p in enumerate(paths):
        rows = np.nonzero(groups == i)[0]
        if len(rows) == 0:
            continue
        with np.load(p, allow_pickle=False) as zz:
            prob = zz["prob"].astype(np.float32)
            meta = json.loads(str(zz["meta"]))
        stem = meta["stem"]

        img = np.load(CACHE / f"{stem}.npy").astype(np.float32) / 255.0
        img = (img - MEAN) / STD

        cand = generate_candidates(prob, args.threshold, args.min_area)
        n = int(cand.max())
        assert n == len(rows), (stem, n, len(rows))

        for j, row in enumerate(rows):
            cid = j + 1
            mask = (cand == cid).astype(np.float32)
            ys, xs = np.nonzero(mask)
            cy, cx = float(ys.mean()), float(xs.mean())
            area = float(len(ys))

            side = int(np.clip(3.0 * np.sqrt(max(area, 1.0)), 128, 512))
            half = side // 2
            y0 = int(np.clip(round(cy) - half, 0, 2048 - side))
            x0 = int(np.clip(round(cx) - half, 0, 2048 - side))
            sl = (slice(y0, y0 + side), slice(x0, x0 + side))

            yy, xx = np.mgrid[0:side, 0:side]
            rr = np.sqrt((yy - (cy - y0)) ** 2 + (xx - (cx - x0)) ** 2)
            rnorm = np.clip(rr / max(1.0, np.sqrt(area / np.pi) * 3.0), 0, 1)

            ch = np.stack([img[sl], prob[sl], mask[sl], rnorm.astype(np.float32)])
            if side != CROP:
                ch = np.stack(
                    [cv2.resize(c, (CROP, CROP), interpolation=cv2.INTER_AREA)
                     for c in ch]
                )
            # Channel 0 is standardised (roughly [-5, 5]); map to uint8 for
            # storage and undo it in the dataset. Channels 1-3 are already [0,1].
            ch0 = np.clip((ch[0] + 5.0) / 10.0 * 255.0, 0, 255)
            ch[0] = ch0
            crops.append(np.clip(ch, 0, 255).astype(np.uint8))
            qs.append(float(q_all[row]))
            file_of.append(i)
            cids.append(cid)

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(paths)}  {len(crops)} crops  "
                  f"{el:.0f}s  eta {el / (i + 1) * (len(paths) - i - 1):.0f}s",
                  flush=True)

    arr = np.stack(crops)
    out = Path(args.out)
    np.save(out, arr)
    # ``file`` is a reserved keyword argument of ``savez_compressed``, so the
    # per-crop file index is stored as ``file_idx``.
    np.savez_compressed(
        P2 / "crops_meta.npz",
        q=np.asarray(qs, np.float32),
        file_idx=np.asarray(file_of, np.int32),
        cid=np.asarray(cids, np.int32),
    )
    print(f"wrote {out}  {arr.shape} {arr.dtype}  "
          f"({arr.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
