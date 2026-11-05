"""Is the bottleneck semantics or spatial precision?

The fusion audit found fused pixel AUC 0.9968 -- near-perfect *ranking* of
filament vs background -- while pixel IoU is only 0.49. Those are compatible
only if the probability map is a spatially blurred version of the filament: the
model knows where the filament is but not exactly where its edge is.

DeepCrack has no decoder. It takes a side head at each encoder stage and
bilinearly upsamples it to full resolution, so the semantically strongest heads
are also the coarsest:

    side1  stride 1   (2 convs, shallow)
    side2  stride 2
    side3  stride 4
    side4  stride 8
    side5  stride 16

If the deep heads rank well but localise badly, the fix is a decoder with skip
connections, not more encoder layers.

This script measures, per head: AUC (ranking), pixel IoU (localisation), and the
optimal threshold's distance from 0.5 (calibration). It also measures how much
IoU a perfect *oracle* threshold would give, which separates "wrong level" from
"wrong shape".

    python scripts/deepcrack_head_precision.py --weights runs/deepcrack_ft_256_long/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import anns_to_labelmap, build_ann_index, load_coco  # noqa: E402
from solar.deepcrack import DeepCrack  # noqa: E402

cv2.setNumThreads(4)

NAMES = ["side1", "side2", "side3", "side4", "side5", "fused"]
STRIDES = [1, 2, 4, 8, 16, 1]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with tie handling. ``labels`` must be bool."""
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def predict_all(model, img, device, mean, std):
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    outs = model.net(t)
    return [torch.sigmoid(o)[0, 0].float().cpu().numpy() for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=142)
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--std", type=float, default=1.0)
    ap.add_argument("--auc-pixels", type=int, default=200000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    seen: set[str] = set()
    val_ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        val_ids.append(iid)
    val_ids = val_ids[: args.n]

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck))
    model.eval()
    print(f"loaded {args.weights}  ({len(val_ids)} val files)")

    # For each head: AUC, and IoU at a sweep of thresholds (to find the best).
    thrs = np.arange(0.05, 0.96, 0.05)
    auc_sum = {n: 0.0 for n in NAMES}
    inter = {n: np.zeros(len(thrs), np.int64) for n in NAMES}
    union = {n: np.zeros(len(thrs), np.int64) for n in NAMES}
    # Oracle: per-image best IoU over thresholds (upper bound on calibration).
    oracle_iou = {n: 0.0 for n in NAMES}
    rng = np.random.default_rng(0)

    for iid in tqdm(val_ids, desc="heads"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        maps = predict_all(model, img, device, args.mean, args.std)
        g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        gbin = (g > 0).astype(bool)

        flat_g = gbin.ravel()
        sub = rng.choice(len(flat_g), size=min(args.auc_pixels, len(flat_g)), replace=False)
        sub_g = flat_g[sub]

        for name, p in zip(NAMES, maps):
            auc_sum[name] += auc(p.ravel()[sub], sub_g)
            best_iou = 0.0
            for j, t in enumerate(thrs):
                pred = p > t
                i = int((pred & gbin).sum())
                u = int((pred | gbin).sum())
                inter[name][j] += i
                union[name][j] += u
                if u:
                    best_iou = max(best_iou, i / u)
            oracle_iou[name] += best_iou

    n = len(val_ids)
    print("\n=== per-head: ranking vs localisation ===")
    print(f"{'head':>6} {'stride':>7} {'AUC':>7} {'bestIoU':>8} {'@thr':>6} {'oracleIoU':>10}")
    rows = {}
    for name, stride in zip(NAMES, STRIDES):
        ious = inter[name] / np.maximum(union[name], 1)
        j = int(np.argmax(ious))
        rows[name] = {
            "stride": stride,
            "auc": auc_sum[name] / n,
            "best_iou": float(ious[j]),
            "best_threshold": float(thrs[j]),
            "oracle_iou": oracle_iou[name] / n,
        }
        print(
            f"{name:>6} {stride:7d} {rows[name]['auc']:7.4f} {ious[j]:8.4f} "
            f"{thrs[j]:6.2f} {rows[name]['oracle_iou']:10.4f}"
        )

    print("\n=== interpretation ===")
    f = rows["fused"]
    print(f"  fused AUC {f['auc']:.4f} but IoU {f['best_iou']:.4f}")
    print(f"  oracle per-image threshold would give IoU {f['oracle_iou']:.4f}")
    print(
        f"  -> calibration headroom {f['oracle_iou'] - f['best_iou']:+.4f} IoU, "
        f"shape headroom {1.0 - f['oracle_iou']:.4f} IoU"
    )
    deep = rows["side4"]
    print(
        f"  side4 (stride 8) AUC {deep['auc']:.4f} vs side1 (stride 1) "
        f"AUC {rows['side1']['auc']:.4f}"
    )

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"weights": args.weights, "n": n, "heads": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
