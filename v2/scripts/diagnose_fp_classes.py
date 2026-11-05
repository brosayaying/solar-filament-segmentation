"""Characterise the two dominant false-positive classes.

The FP anatomy (``diagnose_fp.py``) says Phase 1's 3,403 false positives split
into two nearly equal halves:

``background`` (1,608, 47 %)
    Overlaps no GT instance at all. Removing all of them is worth +0.0445.
``unmatched_gt`` (1,735, 51 %)
    Overlaps a GT instance that no true positive matched. Removing all of them
    is worth +0.0484.

These are *different* problems and need different fixes, so this script measures
what distinguishes them from true positives. If a scalar feature separates them
cleanly, a GBDT on scalars is enough and the CNN in 2.3 is unnecessary.

    python v2/scripts/diagnose_fp_classes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.instances import candidate_features, generate_candidates  # noqa: E402
from solar.metrics import pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"


def classify_pred(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5):
    """Return a class label per predicted instance."""
    n_pred = int(pred.max())
    n_gt = int(gt.max())
    if n_pred == 0:
        return []
    if n_gt == 0:
        return ["background"] * n_pred

    iou = pairwise_iou(pred, gt, n_pred, n_gt)
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

    inter = np.zeros((n_pred, n_gt), np.float64)
    for j in range(n_gt):
        g = gt == j + 1
        ga = g.sum()
        if ga == 0:
            continue
        for i in range(n_pred):
            inter[i, j] = np.logical_and(pred == i + 1, g).sum() / ga

    out = []
    for i in range(n_pred):
        if matched_pred[i]:
            out.append("tp")
            continue
        j = int(inter[i].argmax())
        cover = inter[i, j]
        if cover < 0.10:
            out.append("background")
        elif matched_gt[j]:
            out.append("fragment" if cover < 0.80 else "duplicate")
        else:
            out.append("unmatched_gt")
    return out


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

    rows: dict[str, list[dict]] = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
        pred = prob_to_labelmap(
            prob, threshold=args.threshold, min_area=args.min_area,
            close_k=args.close_k,
        )
        if pred.max() == 0:
            continue
        # Features are computed on the *final* prediction, not on low-threshold
        # candidates, so that the classes are directly comparable.
        for lab in labels:
            cls = classify_pred(pred, lab)
            for cid, c in enumerate(cls, start=1):
                rows.setdefault(c, []).append(candidate_features(prob, pred, cid))

    keys = list(rows["tp"][0].keys())
    print(f"{len(paths)} files | thr {args.threshold} area {args.min_area} "
          f"close {args.close_k}")
    print(f"{'class':14s} {'n':>6s}  " + "  ".join(f"{k[:11]:>11s}" for k in keys))
    print("-" * (22 + 13 * len(keys)))
    for c in ("tp", "background", "unmatched_gt", "fragment"):
        if c not in rows:
            continue
        arr = np.asarray([[r[k] for k in keys] for r in rows[c]], np.float64)
        med = np.median(arr, axis=0)
        print(f"{c:14s} {len(arr):6d}  " + "  ".join(f"{v:11.3f}" for v in med))

    out = ROOT / "v2/runs/phase2/fp_classes.json"
    json.dump(
        {c: {k: float(np.median([r[k] for r in v])) for k in keys}
         for c, v in rows.items()},
        open(out, "w"), indent=2,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
