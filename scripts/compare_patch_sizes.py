"""Compare patch sizes on an identical train/val split.

Runs the full pipeline -- feature extraction, RandomForest, boundary refinement,
PQ sweep -- at two or more patch sizes, holding the image split and all
post-processing parameters fixed so the only variable is the patch size.

This exists because the first 16x16 run was not comparable to the 8x8 result
(60 train images vs 250). Comparing patch sizes requires identical inputs.

    python scripts/compare_patch_sizes.py --patch 8 16 --train-images 60 --val-images 20
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

cv2.setNumThreads(4)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_one(
    patch: int,
    train_ids,
    val_ids,
    images,
    ann_index,
    neg_frac: float,
    model_kind: str,
    save: str | None,
) -> dict:
    pc.set_patch_size(patch)

    t0 = time.time()
    X, y = pc.build_dataset(train_ids, images, ann_index, len(train_ids), neg_frac)
    build_s = time.time() - t0

    t0 = time.time()
    model = pc.build_model(model_kind)
    model.fit(X, y)
    fit_s = time.time() - t0

    if save:
        joblib.dump(model, save)

    # Patch-level quality, so classifier strength is separable from
    # post-processing.
    P, Y = [], []
    probs, gts = [], []
    t0 = time.time()
    for iid in val_ids:
        img = pc.read_image(images[iid]["file_name"])
        disk = pc.disk_mask(img)
        Xv, valid = pc.extract_features(img, disk)
        p = model.predict_proba(Xv.reshape(-1, Xv.shape[-1]))[:, 1].reshape(pc.GRID, pc.GRID)
        gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
        yv = (pc.patch_fraction(gt) > 0.25).astype(np.uint8)
        P.append(p[valid])
        Y.append(yv[valid])
        probs.append(p)
        gts.append(gt)
    infer_s = time.time() - t0

    P = np.concatenate(P)
    Y = np.concatenate(Y)

    best_f1, best_thr = 0.0, 0.5
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        pred = P > thr
        tp = int((pred & (Y == 1)).sum())
        fp = int((pred & (Y == 0)).sum())
        fn = int((~pred & (Y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    # PQ with boundary refinement, sweeping the same grid as the 8x8 study.
    best = None
    for alpha in (1.0, 0.8, 0.6):
        for thr in (0.5, 0.6, 0.7):
            for ma in (200, 512, 1024):
                acc = PQAccumulator()
                for img, prob, gt in zip(
                    [pc.read_image(images[i]["file_name"]) for i in val_ids], probs, gts
                ):
                    score = (
                        rf.upsample_prob(prob, img.shape)
                        if alpha >= 1.0
                        else rf.refine(prob, img, alpha=alpha)
                    )
                    acc.update(rf.to_labelmap(score > thr, ma), gt)
                s = acc.summary()
                s.update(alpha=alpha, threshold=thr, min_area=ma)
                if best is None or s["pq"] > best["pq"]:
                    best = s

    return {
        "patch": patch,
        "grid": pc.GRID,
        "n_features": int(X.shape[1]),
        "n_train_patches": int(X.shape[0]),
        "train_pos_frac": float(y.mean()),
        "patch_auc": auc(P, Y),
        "patch_f1": best_f1,
        "patch_f1_thr": best_thr,
        "pq": best["pq"],
        "sq": best["sq"],
        "rq": best["rq"],
        "tp": best["tp"],
        "fp": best["fp"],
        "fn": best["fn"],
        "best_alpha": best["alpha"],
        "best_thr": best["threshold"],
        "best_min_area": best["min_area"],
        "build_s": build_s,
        "fit_s": fit_s,
        "infer_s": infer_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--train-images", type=int, default=60)
    ap.add_argument("--val-images", type=int, default=20)
    ap.add_argument("--neg-frac", type=float, default=0.05)
    ap.add_argument("--model", default="rf")
    ap.add_argument("--out", default="docs/EDGE/patch_size_compare.json")
    ap.add_argument("--save-dir", default="/tmp")
    args = ap.parse_args()

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    train_ids = split["train"][: args.train_images]
    val_ids = split["val"][: args.val_images]
    print(f"train {len(train_ids)} images, val {len(val_ids)} images, model={args.model}\n")

    rows = []
    for ps in args.patch:
        print(f"=== PATCH={ps} ===", flush=True)
        save = str(Path(args.save_dir) / f"rf{ps}_cmp.joblib")
        r = run_one(ps, train_ids, val_ids, images, ann_index, args.neg_frac, args.model, save)
        rows.append(r)
        print(
            f"  grid={r['grid']} feats={r['n_features']} patches={r['n_train_patches']:,}\n"
            f"  patch AUC={r['patch_auc']:.4f}  F1={r['patch_f1']:.3f}@{r['patch_f1_thr']}\n"
            f"  PQ={r['pq']:.4f} SQ={r['sq']:.3f} RQ={r['rq']:.3f} "
            f"tp={r['tp']} fp={r['fp']} fn={r['fn']}\n"
            f"  (a={r['best_alpha']} thr={r['best_thr']} ma={r['best_min_area']})  "
            f"build={r['build_s']:.0f}s fit={r['fit_s']:.0f}s infer={r['infer_s']:.0f}s\n",
            flush=True,
        )

    print(f"{'patch':>6} {'grid':>5} {'feats':>6} {'AUC':>7} {'F1':>6} {'PQ':>7} {'SQ':>6} {'RQ':>6}")
    print("-" * 58)
    for r in rows:
        print(
            f"{r['patch']:>6} {r['grid']:>5} {r['n_features']:>6} {r['patch_auc']:>7.4f} "
            f"{r['patch_f1']:>6.3f} {r['pq']:>7.4f} {r['sq']:>6.3f} {r['rq']:>6.3f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
