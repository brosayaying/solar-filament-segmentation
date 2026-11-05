"""Phase 3 acceptance evaluation: subset PQ and outer-ring recall.

The plan's acceptance criteria (§7.3) are not just "pooled PQ improves":

| check | threshold |
|---|---|
| pooled OOF PQ | >= Phase 2 + 0.02 |
| 2021-22 subset PQ | improves *more* than the overall PQ |
| outer-ring (r/R > 0.8) recall | improves |

The second and third are the point of the phase — label transfer and the
2021-22 oversampling target the known weak spots, so a gain that is uniform
across years and radii would mean the mechanism is not what we think it is.

This script scores a checkpoint on a fold's val set and reports PQ broken down
by year and by radial band, so the comparison against Phase 1 is like-for-like.

    python v2/scripts/eval_phase3.py --ckpt v2/runs/phase3_fold0/best.pt \
        --split v2/data/phase3/fold0.json --ann v2/data/phase3/annotations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import (  # noqa: E402
    anns_to_labelmap,
    build_ann_index,
    normalise_image,
    parse_preprocess,
    preprocess_image,
)
from solar.gong import disk_geometry  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402
from solar.predict import predict_prob  # noqa: E402


def radial_band(gt: np.ndarray, cy: float, cx: float, radius: float) -> np.ndarray:
    """Mean r/R of each GT instance, indexed 1..N (0 unused)."""
    n = int(gt.max())
    out = np.zeros(n + 1, np.float32)
    if n == 0:
        return out
    ys, xs = np.nonzero(gt)
    ids = gt[ys, xs]
    r = np.hypot(ys - cy, xs - cx) / max(1.0, radius)
    for i in range(1, n + 1):
        m = ids == i
        if m.any():
            out[i] = float(r[m].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-area", type=int, default=256)
    ap.add_argument("--close-k", type=int, default=15)
    ap.add_argument("--device", default="auto",
                    help="auto|cuda|cpu. Use cpu to run alongside a training job "
                         "that already occupies most of the GPU.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"ckpt: epoch {ck.get('epoch')} val pq {ck.get('pq', 0):.4f}")

    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    pp = parse_preprocess(cfg["data"].get("preprocess"))
    mask_off = cfg["data"].get("mask_offdisk", False)
    offdisk = cfg["data"].get("offdisk_value", -3.0)

    coco = json.load(open(args.ann))
    split = json.load(open(args.split))
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}

    root = Path(cfg["data"]["root"])
    img_dir = Path(args.img_dir) if args.img_dir else root / "train/train_images"

    # Accumulators keyed by subset.
    accs: dict[str, PQAccumulator] = defaultdict(lambda: PQAccumulator(threshold=0.5))
    # Per-instance records for outer-ring recall.
    ring_tp = ring_fn = 0
    inner_tp = inner_fn = 0

    val_ids = split["val"]
    print(f"{len(val_ids)} val entries")

    for k, iid in enumerate(val_ids):
        meta = images[iid]
        path = img_dir / meta["file_name"]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  MISSING {path}")
            continue

        img_pp, disk = preprocess_image(img, pp)
        x = normalise_image(img_pp, mean, std, disk=disk,
                            mask_offdisk=mask_off, offdisk_value=offdisk)
        tensor = torch.from_numpy(x).unsqueeze(0).to(dev)
        prob = predict_prob(model, tensor, tta=False)

        pred = prob_to_labelmap(prob, threshold=args.threshold,
                                min_area=args.min_area, close_k=args.close_k)
        gt = anns_to_labelmap(ann_index.get(iid, []), 2048, 2048)

        year = meta["file_name"][:4]
        accs["all"].update(pred, gt)
        accs[f"year_{year}"].update(pred, gt)
        if year in ("2021", "2022"):
            accs["year_2021_22"].update(pred, gt)

        # Outer-ring recall: fraction of GT instances at r/R > 0.8 that are
        # matched by any prediction at IoU > 0.5.
        cy, cx, R = disk_geometry(img)
        bands = radial_band(gt, cy, cx, R)
        n_gt = int(gt.max())
        if n_gt:
            from solar.metrics import pairwise_iou

            n_pred = int(pred.max())
            if n_pred:
                iou = pairwise_iou(pred, gt, n_pred, n_gt)
                matched = (iou > 0.5).any(axis=0)
            else:
                matched = np.zeros(n_gt, bool)
            for i in range(1, n_gt + 1):
                if bands[i] > 0.8:
                    ring_tp += int(matched[i - 1])
                    ring_fn += int(not matched[i - 1])
                else:
                    inner_tp += int(matched[i - 1])
                    inner_fn += int(not matched[i - 1])

        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(val_ids)}", flush=True)

    print()
    print(f"{'subset':16s} {'n':>6s} {'PQ':>7s} {'SQ':>7s} {'RQ':>7s} "
          f"{'tp':>6s} {'fp':>6s} {'fn':>6s}")
    print("-" * 62)
    rows = {}
    for name in sorted(accs, key=lambda s: (s != "all", s)):
        s = accs[name].summary()
        rows[name] = s
        print(f"{name:16s} {s['n_images']:6d} {s['pq']:>7.4f} {s['sq']:>7.4f} "
              f"{s['rq']:>7.4f} {s['tp']:>6d} {s['fp']:>6d} {s['fn']:>6d}")

    print()
    ring_rec = ring_tp / max(1, ring_tp + ring_fn)
    inner_rec = inner_tp / max(1, inner_tp + inner_fn)
    print(f"recall r/R > 0.8 : {ring_rec:.4f}  ({ring_tp}/{ring_tp + ring_fn})")
    print(f"recall r/R <= 0.8: {inner_rec:.4f}  ({inner_tp}/{inner_tp + inner_fn})")
    print(f"ring gap         : {inner_rec - ring_rec:+.4f} "
          f"(negative = limb is worse, which is the known failure)")

    if args.out:
        json.dump({"subsets": rows,
                   "ring_recall": ring_rec, "inner_recall": inner_rec,
                   "ckpt": args.ckpt, "epoch": ck.get("epoch")},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
