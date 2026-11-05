"""Patch-level diagnostics: precision/recall of the classifier itself.

Separates "the classifier is bad" from "the post-processing is bad".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import patch_clf as pc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", type=int, default=40)
    ap.add_argument("--val-images", type=int, default=20)
    ap.add_argument("--neg-frac", type=float, default=0.03)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--model", default="rf")
    args = ap.parse_args()

    pc.set_patch_size(args.patch)

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    X, y = pc.build_dataset(split["train"], images, ann_index, args.train_images, args.neg_frac)
    print(f"train {X.shape} pos={y.mean():.3f}")
    model = pc.build_model(args.model)
    model.fit(X, y)

    P, Y = [], []
    for iid in split["val"][: args.val_images]:
        img = pc.read_image(images[iid]["file_name"])
        disk = pc.disk_mask(img)
        gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
        Xv, valid = pc.extract_features(img, disk)
        p = model.predict_proba(Xv.reshape(-1, Xv.shape[-1]))[:, 1].reshape(pc.GRID, pc.GRID)
        yv = (pc.patch_fraction(gt) > 0.25).astype(np.uint8)
        P.append(p[valid])
        Y.append(yv[valid])
    P = np.concatenate(P)
    Y = np.concatenate(Y)
    print(f"val patches {len(P)} pos={Y.mean():.4f}")

    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pred = P > thr
        tp = int((pred & (Y == 1)).sum())
        fp = int((pred & (Y == 0)).sum())
        fn = int((~pred & (Y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(
            f"thr={thr:.1f}  P={prec:.3f} R={rec:.3f} F1={2 * prec * rec / max(1e-9, prec + rec):.3f} "
            f"tp={tp} fp={fp} fn={fn}"
        )

    # How separable is it at all?
    print(f"\nprob quantiles  pos: {np.percentile(P[Y == 1], [10, 50, 90]).round(3)}")
    print(f"prob quantiles  neg: {np.percentile(P[Y == 0], [50, 90, 99]).round(3)}")
    print(f"AUC-ish: frac neg > 0.5 = {(P[Y == 0] > 0.5).mean():.4f}")


if __name__ == "__main__":
    main()
