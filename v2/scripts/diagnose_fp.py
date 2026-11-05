"""Measure where Phase 1's false positives actually come from.

The inherited claim (``docs/PQ_DIAGNOSIS.md``) is that 58.5 % of FPs are
"splits of an already-matched GT". That number was computed under the old
``cv2.fillPoly`` rasteriser and before the pycocotools switch, so it is not
trustworthy. This script re-measures it on the honest OOF dumps.

Each FP is classified by its relationship to the GT instances of the same image:

``fragment``
    Overlaps a GT instance that a *true positive* already matched, and covers a
    meaningful fraction of that GT. This is the case merging would fix.
``duplicate``
    Overlaps a matched GT but is nearly contained in the TP's mask -- a
    near-copy, not a fragment.
``unmatched_gt``
    Overlaps a GT instance that no TP matched. This is a *detection* failure
    (the GT was found but drawn badly), not a fragmentation failure.
``background``
    Overlaps no GT at all. Pure noise.

    python v2/scripts/diagnose_fp.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"


def classify(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> Counter:
    """Classify every predicted instance in one image against one GT copy."""
    n_pred = int(pred.max())
    n_gt = int(gt.max())
    c: Counter = Counter()
    if n_pred == 0:
        return c
    if n_gt == 0:
        c["background"] += n_pred
        return c

    iou = pairwise_iou(pred, gt, n_pred, n_gt)

    # Greedy 1:1 matching, identical to the scorer.
    matched_pred = np.zeros(n_pred, bool)
    matched_gt = np.zeros(n_gt, bool)
    for flat in np.argsort(iou, axis=None)[::-1]:
        i, j = divmod(int(flat), n_gt)
        if iou[i, j] <= iou_thr:
            break
        if matched_pred[i] or matched_gt[j]:
            continue
        matched_pred[i] = True
        matched_gt[j] = True

    # Overlap *fraction of the GT* covered by each prediction, which is what
    # distinguishes a fragment (covers part of a GT) from a duplicate.
    inter = np.zeros((n_pred, n_gt), np.float64)
    for j in range(n_gt):
        g = gt == j + 1
        ga = g.sum()
        if ga == 0:
            continue
        for i in range(n_pred):
            inter[i, j] = np.logical_and(pred == i + 1, g).sum() / ga

    for i in range(n_pred):
        if matched_pred[i]:
            c["tp"] += 1
            continue
        j = int(inter[i].argmax())
        cover = inter[i, j]
        if cover < 0.10:
            c["background"] += 1
        elif matched_gt[j]:
            # Overlaps a GT that a TP already claimed.
            c["fragment" if cover < 0.80 else "duplicate"] += 1
        else:
            c["unmatched_gt"] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=256)
    ap.add_argument("--close-k", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(OOF.glob("fold*/*.npz"))
    if args.limit:
        paths = paths[: args.limit]

    total: Counter = Counter()
    per_file = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
        pred = prob_to_labelmap(
            prob, threshold=args.threshold, min_area=args.min_area,
            close_k=args.close_k,
        )
        fc: Counter = Counter()
        for lab in labels:
            fc.update(classify(pred, lab))
        total.update(fc)
        per_file.append({"stem": p.stem, **fc})

    n_fp = total["fragment"] + total["duplicate"] + total["unmatched_gt"] + total["background"]
    print(f"{len(paths)} files | thr {args.threshold} area {args.min_area} "
          f"close {args.close_k}")
    print(f"  tp {total['tp']}  fp {n_fp}  "
          f"fn {total['unmatched_gt'] + 0}  (fn counted separately below)")
    print()
    print("FP anatomy:")
    for k in ("fragment", "duplicate", "unmatched_gt", "background"):
        v = total[k]
        print(f"  {k:14s} {v:6d}  {v / max(1, n_fp):6.1%}")

    out = ROOT / "v2/runs/phase2/fp_anatomy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"params": vars(args), "total": dict(total), "per_file": per_file},
        open(out, "w"), indent=2,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
