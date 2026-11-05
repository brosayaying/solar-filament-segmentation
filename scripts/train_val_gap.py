"""Is the model underfitting, overfitting, or limited by resolution?

    python scripts/train_val_gap.py --n 60

Three measurements that together distinguish the possibilities:

1. **train vs val PQ** on the same checkpoint. Train PQ below val PQ means
   underfitting; far above means overfitting; roughly equal at a low value means
   the model is at capacity.
2. **stride-quantisation ceiling** -- GT mask quantised to a coarse grid, then
   connected components. This is the best any prediction on that grid can do, so
   it bounds how much of the gap resolution can possibly explain.
3. **width breakdown** -- best IoU per GT instance binned by a width proxy
   (area / max extent), to see whether thin structures are the failure mode.

IMPORTANT: training images are sampled **randomly**, not from the head of the
split. The first 60 train images are much denser than the val split (8.85 vs
6.48 instances/image) and PQ is anti-correlated with instance count, so taking
the head produces a spurious "train is worse than val" result. Use
``--sample head`` only to reproduce that mistake.
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
from solar.metrics import PQAccumulator, pairwise_iou  # noqa: E402
from solar.models import build_model  # noqa: E402


def connected_components(binary: np.ndarray, min_area: int = 512) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = np.zeros(binary.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def stride_ceiling(gts: list[np.ndarray], strides=(1, 2, 4, 8, 16), min_area=512):
    """PQ achievable if the only error were quantisation to a coarse grid."""
    rows = []
    for s in strides:
        acc = PQAccumulator(threshold=0.5)
        for gt in gts:
            b = (gt > 0).astype(np.uint8)
            if s > 1:
                h, w = b.shape
                small = cv2.resize(b, (w // s, h // s), interpolation=cv2.INTER_AREA)
                b = cv2.resize((small > 0.5).astype(np.uint8), (w, h), cv2.INTER_NEAREST)
            acc.update(connected_components(b, min_area), gt)
        r = acc.summary()
        rows.append((s, r["pq"], r["sq"], r["rq"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--ckpt", default="runs/baseline_resnet34/best.pt")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--sample", choices=("random", "head"), default="random",
                    help="how to pick train images (random avoids the density bias)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    root = Path(cfg["data"]["root"])
    coco = load_coco(root / cfg["data"]["ann_file"])
    split = json.load(open(cfg["data"]["split_file"]))

    model = build_model(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    def predict(image: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = model(image.unsqueeze(0).cuda())
            if isinstance(out, tuple):
                out = out[0]
            return torch.sigmoid(out).cpu().numpy()[0, 0]

    print(f"checkpoint: {args.ckpt}  (epoch {ck.get('epoch')})\n")

    # ---- 1. train vs val -------------------------------------------------- #
    print("train vs val PQ")
    print(f"{'set':>6s} {'n':>4s} {'PQ':>7s} {'SQ':>7s} {'RQ':>7s} {'tp':>5s} {'fp':>5s} {'fn':>5s}")
    val_gts = []
    val_items = []
    rng = np.random.default_rng(args.seed)

    for tag, ids in (("train", split["train"]), ("val", split["val"])):
        if tag == "train" and args.sample == "random" and len(ids) > args.n:
            # Randomise: the head of the train split is denser than val, and PQ
            # is anti-correlated with instance count.
            pick = rng.choice(len(ids), size=args.n, replace=False)
            ids = [ids[i] for i in sorted(pick)]
        ds = FilamentDataset(
            root, coco, ids, mode="val",
            mean=cfg["data"]["mean"], std=cfg["data"]["std"],
        )
        acc = PQAccumulator(threshold=0.5)
        for i in range(min(args.n, len(ds))):
            batch = ds[i]
            gt = batch["label"].numpy()
            prob = predict(batch["image"])
            acc.update(connected_components(prob > args.threshold, args.min_area), gt)
            if tag == "val":
                val_gts.append(gt)
                val_items.append(batch)
        s = acc.summary()
        print(f"{tag:>6s} {acc.n_images:4d} {s['pq']:7.4f} {s['sq']:7.4f} {s['rq']:7.4f} "
              f"{s['tp']:5d} {s['fp']:5d} {s['fn']:5d}")

    # ---- 2. stride ceiling ------------------------------------------------ #
    print("\nstride-quantisation ceiling (GT mask quantised, then CC)")
    print(f"{'stride':>6s} {'PQ':>7s} {'SQ':>7s} {'RQ':>7s}")
    for s, pq, sq, rq in stride_ceiling(val_gts, min_area=args.min_area):
        print(f"{s:6d} {pq:7.4f} {sq:7.4f} {rq:7.4f}")

    # ---- 3. width breakdown ----------------------------------------------- #
    print("\nbest IoU per GT instance by width proxy (area / max extent)")
    rows = []
    for batch in val_items:
        gt = batch["label"].numpy()
        n_gt = int(gt.max())
        if n_gt == 0:
            continue
        prob = predict(batch["image"])
        pred = connected_components(prob > args.threshold, args.min_area)
        n_pred = int(pred.max())
        iou = (
            pairwise_iou(pred, gt, n_pred, n_gt)
            if n_pred
            else np.zeros((0, n_gt))
        )
        for j in range(1, n_gt + 1):
            ys, xs = np.nonzero(gt == j)
            if len(ys) == 0:
                continue
            area = len(ys)
            extent = max(1, max(ys.max() - ys.min(), xs.max() - xs.min()))
            best = float(iou[:, j - 1].max()) if n_pred else 0.0
            rows.append((area / extent, best, area))

    if rows:
        arr = np.array(rows)
        print(f"  n={len(arr)}  median width proxy {np.median(arr[:, 0]):.1f} px")
        print(f"{'width':>12s} {'n':>5s} {'med IoU':>8s} {'>0.5':>7s}")
        for lo, hi in ((0, 8), (8, 14), (14, 22), (22, 35), (35, 10**6)):
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
            if len(sel):
                label = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
                print(f"{label:>12s} {len(sel):5d} {np.median(sel[:, 1]):8.3f} "
                      f"{100 * (sel[:, 1] > 0.5).mean():6.1f}%")


if __name__ == "__main__":
    main()
