"""Dump every false-negative instance as an image.

Mirror of ``deepcrack_dump_tps.py`` / ``deepcrack_dump_fps.py`` for the missed
side of the greedy IoU>0.5 assignment. Each FN gets its own 3-panel PNG
(input / GT / prediction) plus a row in ``manifest.csv``.

Bucketed by the FN's *best* IoU against any prediction, because the three groups
need completely different fixes:

``missed``
    best IoU == 0. Nothing was predicted anywhere near it. A genuine detection
    failure -- the model does not see the structure at all.
``partial``
    0 < best IoU <= 0.3. Something was predicted but barely overlaps. Usually a
    fragment of the filament, or a prediction that drifted off it.
``near_miss``
    0.3 < best IoU <= 0.5. The model found it and got most of it right, but the
    overlap fell just short of the 0.5 threshold. These are the cheapest to
    recover: they are a segmentation-quality problem, not a detection problem.

    python scripts/deepcrack_dump_fns.py
    python scripts/deepcrack_dump_fns.py --threshold 0.5 --close-k 0

Defaults to the shipped post-processing config (thr=0.45, close_k=15) so the
dump is directly comparable to the TP and FP dumps.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import build_ann_index, load_coco  # noqa: E402
from solar.metrics import pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


def labelmap(prob, thr, min_area, close_k):
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx((prob > thr).astype(np.uint8), cv2.MORPH_CLOSE, k)
        return prob_to_labelmap(binary.astype(np.float32), thr, min_area)
    return prob_to_labelmap(prob, thr, min_area)


def match(iou, n_pred, n_gt, thr=0.5):
    """Greedy one-to-one matching; returns per-side flags."""
    mp = np.zeros(n_pred, bool)
    mg = np.zeros(n_gt, bool)
    for flat in np.argsort(iou, axis=None)[::-1]:
        i, j = divmod(int(flat), n_gt)
        if iou[i, j] <= thr:
            break
        if mp[i] or mg[j]:
            continue
        mp[i] = True
        mg[j] = True
    return mp, mg


def panel(img, gt, pred, pad=56):
    """3-panel strip: input | GT overlay | prediction overlay."""
    ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        return None
    h, w = img.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)

    crop = img[y0:y1, x0:x1]
    g = gt[y0:y1, x0:x1]
    p = pred[y0:y1, x0:x1]

    # BGR, not RGB: cv2.imwrite expects BGR, so building the overlay in RGB
    # silently swaps red and blue and the prediction renders blue.
    base = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    ov_gt = base.copy()
    ov_gt[g > 0] = (0.45 * ov_gt[g > 0] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    ov_pr = base.copy()
    ov_pr[p > 0] = (0.35 * ov_pr[p > 0] + 0.65 * np.array([40, 40, 255])).astype(np.uint8)

    sep = np.full((crop.shape[0], 3, 3), 255, np.uint8)
    return np.hstack([base, sep, ov_gt, sep, ov_pr])


def contact_sheet(paths, out_stem, cols=2, cell_w=1000, cell_h=340, per_sheet=12):
    """Paginated contact sheets; wide short cells suit the 3:1 strips."""
    if not paths:
        return []
    written = []
    for page, start in enumerate(range(0, len(paths), per_sheet), 1):
        chunk = paths[start : start + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * cell_h, cols * cell_w, 3), 30, np.uint8)
        for i, p in enumerate(chunk):
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            s = min(cell_w / w, cell_h / h)
            im = cv2.resize(
                im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
            )
            r, c = divmod(i, cols)
            y = r * cell_h + (cell_h - im.shape[0]) // 2
            x = c * cell_w + (cell_w - im.shape[1]) // 2
            sheet[y : y + im.shape[0], x : x + im.shape[1]] = im
        out = out_stem.parent / f"{out_stem.name}_p{page:02d}.png"
        cv2.imwrite(str(out), sheet)
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/deepcrack_ft_256_long/val_probs")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--close-k", type=int, default=15)
    ap.add_argument("--pad", type=int, default=56)
    ap.add_argument("--out", default="docs/DEEPCRACK_FINETUNE/fn_dump")
    args = ap.parse_args()

    cache = ROOT / args.cache
    files = sorted(glob.glob(str(cache / "*_prob.npy")))
    if not files:
        raise FileNotFoundError(f"no cached probs in {cache}; run deepcrack_cache_probs.py")

    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / "splits/date_seed42.json").read_text())

    # Rebuild the same 142-file dedup'd val order the cache was written in.
    seen: set[str] = set()
    ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[: len(files)]

    out = ROOT / args.out
    buckets = ("missed", "partial", "near_miss")
    dirs = {b: out / b for b in buckets}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()

    rows = []
    counts = {b: 0 for b in buckets}
    n_tp = n_fp = n_fn = 0

    for idx, (pf, iid) in enumerate(zip(files, ids)):
        prob = np.load(pf).astype(np.float32)
        gt = np.load(pf.replace("_prob.npy", "_label.npy"))
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        if img is None:
            continue

        pred = labelmap(prob, args.threshold, args.min_area, args.close_k)
        n_pred, n_gt = int(pred.max()), int(gt.max())
        if n_gt == 0:
            n_fp += n_pred
            continue
        if n_pred == 0:
            iou = np.zeros((0, n_gt))
            mg = np.zeros(n_gt, bool)
        else:
            iou = pairwise_iou(pred, gt, n_pred, n_gt)
            mp, mg = match(iou, n_pred, n_gt)
            n_tp += int(mp.sum())
            n_fp += int((~mp).sum())
        n_fn += int((~mg).sum())

        stem = Path(im["file_name"]).stem
        for gid in range(n_gt):
            if mg[gid]:
                continue
            best = float(iou[:, gid].max()) if n_pred else 0.0
            if best == 0.0:
                bucket = "missed"
            elif best <= 0.3:
                bucket = "partial"
            else:
                bucket = "near_miss"

            m = gt == gid + 1
            strip = panel(img, m, pred, args.pad)
            if strip is None:
                continue

            name = f"{stem}__fn_gt{gid + 1:02d}_best{iou_val_str(best)}_a{int(m.sum())}.png"
            cv2.imwrite(str(dirs[bucket] / name), strip)

            rows.append(
                {
                    "file": stem,
                    "image_index": idx,
                    "gt_id": gid + 1,
                    "bucket": bucket,
                    "best_iou": round(best, 4),
                    "gt_area": int(m.sum()),
                    "n_pred": n_pred,
                    "n_gt": n_gt,
                    "image": name,
                }
            )
            counts[bucket] += 1

    rows.sort(key=lambda r: (buckets.index(r["bucket"]), -r["best_iou"]))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)

    for b in buckets:
        contact_sheet(sorted(dirs[b].glob("*.png")), out / f"sheet_{b}")

    print(f"config: thr={args.threshold} min_area={args.min_area} close_k={args.close_k}")
    for b in buckets:
        print(f"{b:>10} FNs: {counts[b]:4d}  -> {dirs[b]}")
    print(f"{'total':>10} FNs: {n_fn:4d}   (TP={n_tp}, FP={n_fp})")
    print(f"manifest: {out / 'manifest.csv'}")
    print(f"sheets:   {out}/sheet_missed_p*.png, sheet_partial_p*.png, sheet_near_miss_p*.png")


def iou_val_str(v: float) -> str:
    return f"{v:.2f}"


if __name__ == "__main__":
    main()
