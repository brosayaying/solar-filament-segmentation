"""Visualise patch-classifier predictions against ground truth."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import patch_clf as pc  # noqa: E402
import refine as rf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--out", default="/tmp/pred.png")
    args = ap.parse_args()

    model = joblib.load(args.model)
    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    iid = split["val"][args.index]
    img = pc.read_image(images[iid]["file_name"])
    disk = pc.disk_mask(img)
    prob = pc.predict_prob(model, img, disk)
    gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)

    score = rf.refine(prob, img, alpha=args.alpha)
    pred = rf.to_labelmap(score > args.thr, 512)

    # Overlay: GT in red, prediction in green, overlap yellow.
    ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ov[gt > 0] = (0, 0, 255)
    ov[pred > 0] = (0, 255, 0)
    ov[(gt > 0) & (pred > 0)] = (0, 255, 255)

    cv2.imwrite(args.out, cv2.resize(ov, (1024, 1024), interpolation=cv2.INTER_AREA))
    print(f"{images[iid]['file_name']}  gt={gt.max()} pred={pred.max()} -> {args.out}")

    # Also dump the raw score map for inspection.
    sm = np.clip(score * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(args.out.replace(".png", "_score.png"), cv2.resize(sm, (1024, 1024)))


if __name__ == "__main__":
    main()
