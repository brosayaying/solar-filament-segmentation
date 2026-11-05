"""PQ error decomposition: where exactly are the true positives going?

    python scripts/pq_decompose.py --n 64

PQ = SQ * RQ, and RQ = TP / (TP + 0.5*FP + 0.5*FN). So every point of PQ is
lost either to a *missed* instance (FN), a *spurious* instance (FP), or a
*matched but poorly delineated* instance (SQ).

This script splits the errors into actionable buckets:

FN buckets
  ``missed``      no prediction overlaps the GT instance at all
  ``merged``      the GT instance is covered, but by a prediction that also
                  covers other GT instances (under-segmentation)
  ``low_iou``     a prediction overlaps it, but IoU < 0.5 (boundary/size error)

FP buckets
  ``spurious``    no GT overlap at all
  ``split``       overlaps a GT instance that is already matched elsewhere
                  (over-segmentation)

Also reports the GT instance size distribution, because if many GT instances are
smaller than ``min_area`` they are being deleted by post-processing before PQ
ever sees them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar.data import FilamentDataset, load_coco  # noqa: E402
from solar.metrics import pairwise_iou  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402


def classify(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict:
    """Bucket every FN and FP for one image."""
    n_pred = int(pred.max()) if pred.size else 0
    n_gt = int(gt.max()) if gt.size else 0
    out = dict(
        n_pred=n_pred, n_gt=n_gt,
        tp=0, fn_missed=0, fn_merged=0, fn_low_iou=0,
        fp_spurious=0, fp_split=0, iou_sum=0.0,
    )
    if n_pred == 0 or n_gt == 0:
        out["fn_missed"] = n_gt
        out["fp_spurious"] = n_pred
        return out

    iou = pairwise_iou(pred, gt, n_pred, n_gt)

    # Greedy one-to-one matching, same rule as PQAccumulator.
    order = np.argsort(-iou, axis=None)
    mp = np.zeros(n_pred, bool)
    mg = np.zeros(n_gt, bool)
    for flat in order:
        i, j = divmod(int(flat), n_gt)
        if iou[i, j] < thr or mp[i] or mg[j]:
            continue
        mp[i] = mg[j] = True
        out["tp"] += 1
        out["iou_sum"] += float(iou[i, j])

    # Unmatched GT: did anything overlap it at all?
    for j in np.flatnonzero(~mg):
        col = iou[:, j]
        if col.max() <= 0:
            out["fn_missed"] += 1
        else:
            # Overlapped, but the prediction was taken by another GT instance
            # (or IoU was too low). Distinguish by whether that prediction
            # covers more than one GT instance.
            i = int(np.argmax(col))
            covers = int((iou[i] > 0.1).sum())
            if covers > 1:
                out["fn_merged"] += 1
            else:
                out["fn_low_iou"] += 1

    # Unmatched predictions: spurious, or a split of an already-matched GT?
    for i in np.flatnonzero(~mp):
        row = iou[i]
        if row.max() <= 0:
            out["fp_spurious"] += 1
        else:
            out["fp_split"] += 1

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--ckpt", default="runs/baseline_resnet34/best.pt")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=512)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    root = Path(cfg["data"]["root"])
    coco = load_coco(root / cfg["data"]["ann_file"])
    split = json.load(open(cfg["data"]["split_file"]))
    ds = FilamentDataset(
        root, coco, split["val"], mode="val",
        mean=cfg["data"]["mean"], std=cfg["data"]["std"],
    )

    model = build_model(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    tot = dict(tp=0, fn_missed=0, fn_merged=0, fn_low_iou=0,
               fp_spurious=0, fp_split=0, iou_sum=0.0, n_gt=0, n_pred=0)
    gt_areas = []
    pred_areas = []

    for i in range(min(args.n, len(ds))):
        batch = ds[i]
        gt = batch["label"].numpy()
        with torch.no_grad():
            logits = model(batch["image"].unsqueeze(0).cuda())
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

        pred = prob_to_labelmap(prob, threshold=args.threshold, min_area=args.min_area)
        r = classify(pred, gt)
        for k in tot:
            tot[k] += r[k]

        for lab in range(1, int(gt.max()) + 1):
            gt_areas.append(int((gt == lab).sum()))
        for lab in range(1, int(pred.max()) + 1):
            pred_areas.append(int((pred == lab).sum()))

    tp = tot["tp"]
    fn = tot["fn_missed"] + tot["fn_merged"] + tot["fn_low_iou"]
    fp = tot["fp_spurious"] + tot["fp_split"]
    rq = tp / max(1e-9, tp + 0.5 * fp + 0.5 * fn)
    sq = tot["iou_sum"] / max(1, tp)

    print(f"images={min(args.n, len(ds))}  GT={tot['n_gt']}  pred={tot['n_pred']}")
    print(f"PQ={sq * rq:.4f}  SQ={sq:.4f}  RQ={rq:.4f}   (tp={tp} fp={fp} fn={fn})")
    print()
    print("FALSE NEGATIVES (missed instances)")
    for k, label in [("fn_missed", "no overlap at all"),
                     ("fn_merged", "merged into a multi-GT prediction"),
                     ("fn_low_iou", "overlapped but IoU<0.5")]:
        v = tot[k]
        print(f"  {label:34s} {v:5d}  {100 * v / max(1, fn):5.1f}% of FN  "
              f"{100 * v / max(1, tot['n_gt']):5.1f}% of GT")
    print()
    print("FALSE POSITIVES (spurious instances)")
    for k, label in [("fp_spurious", "no GT overlap"),
                     ("fp_split", "split of an already-matched GT")]:
        v = tot[k]
        print(f"  {label:34s} {v:5d}  {100 * v / max(1, fp):5.1f}% of FP  "
              f"{100 * v / max(1, tot['n_pred']):5.1f}% of pred")
    print()

    ga = np.array(gt_areas)
    pa = np.array(pred_areas)
    print("GT instance area (px):")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} {np.percentile(ga, q):8.0f}")
    print(f"  below min_area({args.min_area}): "
          f"{100 * (ga < args.min_area).mean():.1f}% of GT instances")
    print()
    print("predicted instance area (px):")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} {np.percentile(pa, q):8.0f}")
    print(f"  median ratio pred/gt: {np.median(pa) / max(1, np.median(ga)):.2f}")


if __name__ == "__main__":
    main()
