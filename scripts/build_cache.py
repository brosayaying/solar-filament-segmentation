"""Cache val-set derived arrays (residual, disk, GT labelmap) for fast iteration.

    python scripts/build_cache.py --n 60 --out /tmp/cv_cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cvlib  # noqa: E402
from solar.data import anns_to_labelmap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="/tmp/cv_cache")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--sigma", type=float, default=25.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coco, ann_index, images = cvlib.load_coco_index()
    split = cvlib.load_split(args.split)
    ids = split["val"][: args.n]

    for i, iid in enumerate(tqdm(ids, desc="cache")):
        meta = images[iid]
        img = cvlib.read_image(meta["file_name"])
        res = cvlib.residual(img, args.sigma)
        disk = cvlib.disk_mask(img)
        gt = anns_to_labelmap(ann_index[iid], img.shape[0], img.shape[1])
        np.savez_compressed(
            out / f"{i:04d}.npz",
            res=res.astype(np.float16),
            disk=disk,
            gt=gt.astype(np.int16),
            img=img,
            file_name=meta["file_name"],
        )
    print(f"cached {len(ids)} -> {out}")


if __name__ == "__main__":
    main()
