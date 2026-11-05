"""Evaluate a saved patch classifier with boundary refinement, sweeping post-processing.

    python scripts/eval_patch.py --model /tmp/rf250.joblib --val-images 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import patch_clf as pc  # noqa: E402
import refine as rf  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--val-images", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pc.set_patch_size(args.patch)
    model = joblib.load(args.model)

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    val_ids = split["val"][: args.val_images]
    cache = []
    t0 = time.time()
    for iid in val_ids:
        img = pc.read_image(images[iid]["file_name"])
        disk = pc.disk_mask(img)
        prob = pc.predict_prob(model, img, disk)
        gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
        cache.append((img, prob, gt))
    print(f"inference {len(val_ids)} images in {time.time() - t0:.0f}s")

    results = []
    for alpha in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
            for ma in (200, 512):
                acc = PQAccumulator()
                for img, prob, gt in cache:
                    score = rf.refine(prob, img, alpha=alpha) if alpha < 1.0 else rf.upsample_prob(prob, img.shape)
                    acc.update(rf.to_labelmap(score > thr, ma), gt)
                s = acc.summary()
                s.update(alpha=alpha, threshold=thr, min_area=ma)
                results.append(s)

    results.sort(key=lambda r: -r["pq"])
    print(f"\n{'alpha':>6} {'thr':>5} {'min_area':>9}  {'PQ':>7} {'SQ':>6} {'RQ':>6}  tp   fp   fn")
    for r in results[:15]:
        print(
            f"{r['alpha']:>6.1f} {r['threshold']:>5.1f} {r['min_area']:>9}  "
            f"{r['pq']:>7.4f} {r['sq']:>6.3f} {r['rq']:>6.3f}  "
            f"{r['tp']:>3} {r['fp']:>4} {r['fn']:>4}"
        )

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
