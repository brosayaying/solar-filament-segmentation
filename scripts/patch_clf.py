"""Patch-level binary classification of filaments with classical ML.

Each 8x8 patch is described by its raw pixels, a locally-background-subtracted
version, and a 4x-downsampled 32x32 context window. A RandomForest / ExtraTrees
/ HistGradientBoosting classifier predicts P(filament). Per-patch probabilities
are upsampled to 2048x2048, thresholded, and split into instances with
connected components.

Why 8x8 and not 32x32: the PQ ceiling for a blocky mask at stride 8 with a 25%
fill rule is ~0.65, versus ~0.28 at stride 16 and ~0.04 at stride 32. Filaments
are only ~17 px wide at the median, so a 32x32 patch cannot resolve them.

    python scripts/patch_clf.py --train-images 120 --val-images 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar.data import anns_to_labelmap, build_ann_index, load_coco  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402

ROOT = Path("MAGFiLO_1.0_Kaggle_2026")

# Patch geometry. Set via ``set_patch_size`` before calling anything else; the
# defaults match the original 8x8 experiment.
PATCH = 8
GRID = 2048 // PATCH  # 256
cv2.setNumThreads(4)


def set_patch_size(ps: int) -> None:
    """Set the patch side length and recompute the derived grid size."""
    global PATCH, GRID
    if 2048 % ps != 0:
        raise ValueError(f"patch size must divide 2048, got {ps}")
    PATCH = ps
    GRID = 2048 // ps


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def disk_mask(img: np.ndarray, thr: int = 25) -> np.ndarray:
    """Binary mask of the solar disk (largest bright blob, holes filled)."""
    m = (img > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones_like(img, np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out[:] = 0
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


BLACKHAT_KS = (25, 41, 65, 81)
RES_SIGMAS = (32, 64, 96)


def feature_maps(img: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel response maps. Filaments are *dark*, so all are positive-on-filament."""
    f = img.astype(np.float32)
    maps: dict[str, np.ndarray] = {"raw": f}
    for k in BLACKHAT_KS:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        maps[f"bh{k}"] = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ker).astype(np.float32)
    for s in RES_SIGMAS:
        maps[f"res{s}"] = cv2.GaussianBlur(f, (0, 0), s) - f
    return maps


def _pool(A: np.ndarray) -> np.ndarray:
    """Pool an (H, W) map to (GRID, GRID, 4): mean, std, min, max per 8x8 patch."""
    W = sliding_window_view(A, (PATCH, PATCH))[::PATCH, ::PATCH]
    return np.stack(
        [W.mean(axis=(-1, -2)), W.std(axis=(-1, -2)), W.min(axis=(-1, -2)), W.max(axis=(-1, -2))],
        axis=-1,
    )


def extract_features(img: np.ndarray, disk: np.ndarray, margin: int = 40):
    """Return ``(X, valid)`` with ``X`` of shape (GRID, GRID, F)."""
    maps = feature_maps(img)

    blocks = [_pool(maps["raw"])]
    for k in BLACKHAT_KS:
        blocks.append(_pool(maps[f"bh{k}"]))
    for s in RES_SIGMAS:
        blocks.append(_pool(maps[f"res{s}"]))

    # Raw 8x8 pixels of the patch itself (texture / shape within the patch).
    blocks.append(sliding_window_view(maps["raw"], (PATCH, PATCH))[::PATCH, ::PATCH].reshape(GRID, GRID, -1))

    # Multi-patch context: a window spanning 4 patch-widths, centred on each
    # patch, taken from the 4x-downsampled image (512x512 for a 2048 frame).
    # At PATCH=8 this is the original 32x32 / stride-2 block; at PATCH=16 it is
    # 64x64 / stride-4, so the context always covers the same number of patches.
    win = PATCH  # window size, in 4x-downsampled pixels
    step = PATCH // 4  # stride, in 4x-downsampled pixels
    need = (GRID - 1) * step + win  # last index touched
    over = need - 512
    pad_before = max(0, over // 2)
    pad_after = max(0, over - pad_before)
    for name in ("raw", "bh65"):
        ctx = cv2.resize(maps[name], (512, 512), interpolation=cv2.INTER_AREA)
        if over > 0:
            ctx = np.pad(ctx, ((pad_before, pad_after), (pad_before, pad_after)), mode="edge")
        idx = np.arange(GRID) * step
        W = sliding_window_view(ctx, (win, win))[idx][:, idx]
        blocks.append(W.reshape(GRID, GRID, -1))

    # Contrast of the patch against its surroundings, scaled with patch size.
    blur_k = 8 * PATCH
    for name in ("bh65", "res64"):
        local = cv2.blur(maps[name], (blur_k, blur_k))
        blocks.append((_pool(maps[name])[..., 0] - local[::PATCH, ::PATCH][:GRID, :GRID])[..., None])

    X = np.concatenate(blocks, axis=-1).astype(np.float32)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    inner = cv2.erode(disk, k)
    valid = inner[::PATCH, ::PATCH][:GRID, :GRID] > 0
    return X, valid


def patch_fraction(gt: np.ndarray) -> np.ndarray:
    """Fraction of each 8x8 patch covered by any filament."""
    p = (gt > 0).astype(np.float32)
    return sliding_window_view(p, (PATCH, PATCH))[::PATCH, ::PATCH].mean(axis=(-1, -2))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def read_image(file_name: str) -> np.ndarray:
    img = cv2.imread(str(ROOT / "train/train_images" / file_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(file_name)
    return img


def build_dataset(ids, images, ann_index, n_images, neg_frac, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for iid in ids[:n_images]:
        img = read_image(images[iid]["file_name"])
        disk = disk_mask(img)
        X, valid = extract_features(img, disk)
        gt = anns_to_labelmap(ann_index[iid], *img.shape)
        y = (patch_fraction(gt) > 0.25).astype(np.uint8)

        pos = np.flatnonzero((valid & (y == 1)).ravel())
        neg = np.flatnonzero((valid & (y == 0)).ravel())
        keep = neg[rng.random(len(neg)) < neg_frac]
        sel = np.concatenate([pos, keep])

        Xs.append(X.reshape(-1, X.shape[-1])[sel])
        ys.append(y.ravel()[sel])
    return np.concatenate(Xs), np.concatenate(ys)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_model(kind: str, seed: int = 0):
    # NOTE: no class_weight. Negatives are already subsampled by --neg-frac, so
    # re-balancing here would double-count and make the model wildly
    # over-predict filaments.
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "et":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=seed
        )
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Inference / evaluation
# --------------------------------------------------------------------------- #


def predict_prob(model, img: np.ndarray, disk: np.ndarray) -> np.ndarray:
    X, valid = extract_features(img, disk)
    p = model.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(GRID, GRID)
    p[~valid] = 0.0
    return p.astype(np.float32)


def to_labelmap(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = np.zeros(binary.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", type=int, default=120)
    ap.add_argument("--val-images", type=int, default=40)
    ap.add_argument("--neg-frac", type=float, default=0.03)
    ap.add_argument("--patch", type=int, default=8, help="patch side length (divides 2048)")
    ap.add_argument("--model", default="rf", choices=["rf", "et", "hgb"])
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    set_patch_size(args.patch)

    coco = load_coco(ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    t0 = time.time()
    X, y = build_dataset(
        split["train"], images, ann_index, args.train_images, args.neg_frac
    )
    print(f"train patches {X.shape} pos={y.mean():.3f} ({time.time() - t0:.0f}s)")

    model = build_model(args.model)
    t0 = time.time()
    model.fit(X, y)
    print(f"fit {args.model} in {time.time() - t0:.0f}s")

    if args.save:
        import joblib

        joblib.dump(model, args.save)
        print(f"saved -> {args.save}")

    # Cache val probabilities once, then sweep post-processing.
    val_ids = split["val"][: args.val_images]
    probs, gts = [], []
    t0 = time.time()
    for iid in val_ids:
        img = read_image(images[iid]["file_name"])
        disk = disk_mask(img)
        probs.append(predict_prob(model, img, disk))
        gts.append(anns_to_labelmap(ann_index[iid], *img.shape))
    print(f"val inference {len(val_ids)} images in {time.time() - t0:.0f}s")

    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        for ma in (200, 512, 1024):
            acc = PQAccumulator()
            for p, gt in zip(probs, gts):
                up = cv2.resize(p, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                acc.update(to_labelmap(up > thr, ma), gt)
            s = acc.summary()
            print(
                f"thr={thr:.1f} min_area={ma:>4}  PQ={s['pq']:.4f} SQ={s['sq']:.3f} "
                f"RQ={s['rq']:.3f} tp={s['tp']} fp={s['fp']} fn={s['fn']}"
            )
            if best is None or s["pq"] > best["pq"]:
                best = dict(s, threshold=thr, min_area=ma)

    print(f"\nBEST PQ={best['pq']:.4f} at thr={best['threshold']} min_area={best['min_area']}")


if __name__ == "__main__":
    main()
