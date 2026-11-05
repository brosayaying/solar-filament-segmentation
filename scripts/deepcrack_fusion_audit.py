"""Is the fusion losing information? Score every side head against the fused map.

DeepCrack's ``fuse_conv`` is a bias-free 1x1 conv over the five side logits, so
the fused map is a fixed linear combination of them. That makes a specific
failure mode possible: if the learned weights are wrong, the fused output can be
*worse* than the best individual side head, and the network is throwing away
signal it already computed.

This script answers three questions per checkpoint:

1. **Per-head quality.** AUC and PQ for each of the five side heads and for the
   fused map. If some side beats the fused map, fusion is destroying information.
2. **Is the learned fusion optimal?** Fit the best non-negative least-squares
   combination of the side *probabilities* on the val set and score it. The gap
   between that oracle and the learned fusion is the headroom in ``fuse_conv``.
3. **Is a richer fusion worth it?** Same, but with a logistic regression on the
   five side probabilities (learnable weights + bias + per-head calibration).

Memory note: 6 maps x 142 images x 2048^2 float32 is ~14 GB, so nothing is
retained. Pass 1 accumulates per-head PQ and a pixel subsample for the fusion
fits; pass 2 re-runs inference to score the fitted fusions.

    python scripts/deepcrack_fusion_audit.py --weights runs/deepcrack_ft_256_long/best.pt
    python scripts/deepcrack_fusion_audit.py --weights runs/deepcrack_scratch/best.pt --n 60
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
from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)

NAMES = ["side1", "side2", "side3", "side4", "side5", "fused"]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with tie handling (Mann-Whitney U).

    ``labels`` is cast to bool: a uint8 array would be treated as integer
    indices by ``scores[labels]``, silently returning nonsense.
    """
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
    """Return the five side logits plus the fused logit."""
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    outs = model.net(t)
    return [o[0, 0].float().cpu().numpy() for o in outs]


def balanced_sample(sides, tgt, n, rng):
    """Sample half foreground, half background.

    Uniform sampling is ~99% background here, so a least-squares fit on it just
    learns to predict zero and says nothing about the fusion.
    """
    fg = np.flatnonzero(tgt)
    bg = np.flatnonzero(~tgt.astype(bool))
    half = n // 2
    take_fg = rng.choice(fg, size=min(half, len(fg)), replace=False)
    take_bg = rng.choice(bg, size=min(half, len(bg)), replace=False)
    idx = np.concatenate([take_fg, take_bg])
    return sides[idx], tgt[idx]


def load_val(args):
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    # One annotator variant per file: averaging over variants measures agreement
    # with a particular human, not correctness (docs/METRIC_CEILING.md).
    seen: set[str] = set()
    val_ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        val_ids.append(iid)
    return data_root, images, ann_index, val_ids[: args.n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=142)
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--std", type=float, default=1.0)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.45, 0.6, 0.75])
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--pixels-per-image", type=int, default=20000)
    ap.add_argument(
        "--auc-pixels",
        type=int,
        default=200000,
        help="pixels sampled per image for AUC; full 2048^2 sorts are CPU-bound",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root, images, ann_index, val_ids = load_val(args)

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck))
    model.eval()
    print(f"loaded {args.weights}  ({len(val_ids)} val files)")

    # ---- Pass 1: per-head PQ + AUC, and a pixel subsample for the fusion fits.
    accs = {(n, t): PQAccumulator() for n in NAMES for t in args.thresholds}
    auc_sum = {n: 0.0 for n in NAMES}
    rng = np.random.default_rng(0)
    X_rows, y_rows = [], []

    for iid in tqdm(val_ids, desc="pass1"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        maps = predict_all(model, img, device, args.mean, args.std)
        g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        gbin = (g > 0).astype(np.uint8)

        # AUC on a pixel subsample: sorting 4.2M values x 6 heads x 142 images
        # is CPU-bound and dominates the runtime. 200k pixels per image keeps
        # the standard error well under 0.001.
        flat_g = gbin.ravel()
        sub = rng.choice(len(flat_g), size=min(args.auc_pixels, len(flat_g)), replace=False)
        sub_g = flat_g[sub]
        for name, logit in zip(NAMES, maps):
            prob = 1.0 / (1.0 + np.exp(-logit))
            auc_sum[name] += auc(prob.ravel()[sub], sub_g)
            for t in args.thresholds:
                accs[(name, t)].update(
                    prob_to_labelmap(prob, threshold=t, min_area=args.min_area), g
                )

        sides = np.stack(maps[:5], -1).reshape(-1, 5)
        xs, ys = balanced_sample(sides, flat_g, args.pixels_per_image, rng)
        X_rows.append(xs)
        y_rows.append(ys)
    n = len(val_ids)
    print("\n=== per-head AUC (pixel-level, mean over images) ===")
    aucs = {k: v / n for k, v in auc_sum.items()}
    for name in NAMES:
        print(f"  {name:6s} AUC={aucs[name]:.4f}")

    print(f"\n=== per-head PQ (min_area={args.min_area}) ===")
    per_head = {}
    for name in NAMES:
        rows = [(t, accs[(name, t)].summary()) for t in args.thresholds]
        t, s = max(rows, key=lambda r: r[1]["pq"])
        per_head[name] = {"threshold": t, **s}
        print(
            f"  {name:6s} best PQ={s['pq']:.4f} @thr={t:.2f} "
            f"SQ={s['sq']:.4f} RQ={s['rq']:.4f} TP={s['tp']} FP={s['fp']} FN={s['fn']}"
        )

    X = np.concatenate(X_rows)
    y = np.concatenate(y_rows)
    del X_rows, y_rows

    # ---- Fit the two alternative fusions on the balanced subsample.
    # fuse_conv is a bias-free 1x1 conv over the side *logits*, so the fair
    # comparison is a linear model on logits, not on probabilities.
    from scipy.optimize import nnls

    w, _ = nnls(X, y.astype(np.float32))
    print("\n=== oracle NNLS fusion on logits (fit on val -> optimistic) ===")
    print(f"  weights: {' '.join('%+.3f' % v for v in w)}  (sum {w.sum():+.3f})")

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, y)
    print("\n=== logistic fusion on logits (fit on val) ===")
    print(f"  coef: {' '.join('%+.3f' % v for v in lr.coef_[0])}  bias {lr.intercept_[0]:+.3f}")

    # ---- Pass 2: score the fitted fusions on full maps.
    acc_nnls = {t: PQAccumulator() for t in args.thresholds}
    acc_lr = {t: PQAccumulator() for t in args.thresholds}

    for iid in tqdm(val_ids, desc="pass2"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        maps = predict_all(model, img, device, args.mean, args.std)
        g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        sides = np.stack(maps[:5], -1)

        p_nnls = 1.0 / (1.0 + np.exp(-(sides @ w)))
        flat = sides.reshape(-1, 5)
        p_lr = lr.predict_proba(flat)[:, 1].reshape(sides.shape[:2]).astype(np.float32)

        for t in args.thresholds:
            acc_nnls[t].update(
                prob_to_labelmap(p_nnls, threshold=t, min_area=args.min_area), g
            )
            acc_lr[t].update(prob_to_labelmap(p_lr, threshold=t, min_area=args.min_area), g)

    def best(accs_):
        rows = [(t, a.summary()) for t, a in accs_.items()]
        return max(rows, key=lambda r: r[1]["pq"])

    t, s = best(acc_nnls)
    print(
        f"\n  NNLS-fused     best PQ={s['pq']:.4f} @thr={t:.2f} "
        f"SQ={s['sq']:.4f} RQ={s['rq']:.4f}"
    )
    t2, s2 = best(acc_lr)
    print(
        f"  logistic-fused best PQ={s2['pq']:.4f} @thr={t2:.2f} "
        f"SQ={s2['sq']:.4f} RQ={s2['rq']:.4f}"
    )

    learned = per_head["fused"]
    best_side = max(per_head[k]["pq"] for k in NAMES[:5])
    print(f"\n  learned fuse_conv   PQ={learned['pq']:.4f}")
    print(f"  best single side    PQ={best_side:.4f}")
    print(f"  NNLS oracle         PQ={s['pq']:.4f}  (delta {s['pq'] - learned['pq']:+.4f})")
    print(f"  logistic            PQ={s2['pq']:.4f}  (delta {s2['pq'] - learned['pq']:+.4f})")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "weights": args.weights,
                    "n_images": n,
                    "min_area": args.min_area,
                    "auc": aucs,
                    "per_head": per_head,
                    "nnls_weights": w.tolist(),
                    "nnls_pq": s,
                    "logistic_coef": lr.coef_[0].tolist(),
                    "logistic_bias": float(lr.intercept_[0]),
                    "logistic_pq": s2,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
