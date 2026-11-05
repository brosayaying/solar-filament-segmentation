"""Dump every false-positive instance as an image.

Two categories, matching the analysis in docs/DEEPCRACK_FINETUNE.md §10:

``partial``
    The FP overlaps some GT (best IoU > 0.1). These are fragments of a filament
    that was already detected -- the dominant failure mode (fragmentation
    outnumbers merging 17:1, see ``deepcrack_split_diagnosis.py``).
``full``
    The FP has no GT overlap at all. Either a real but unannotated structure, or
    noise the model over-fires on.

Each instance gets its own 3-panel PNG (input / GT / prediction) plus a row in
``manifest.csv``. Contact sheets are written for quick scanning.

    python scripts/deepcrack_dump_fps.py
    python scripts/deepcrack_dump_fps.py --threshold 0.5 --close-k 0

Defaults to the best post-processing config found in the sweep
(thr=0.45, close_k=15), so the dump reflects the model as it would be shipped.
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
    """Greedy one-to-one matching with separate flags per side."""
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
    ys, xs = np.nonzero(pred)
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
    """Paginated contact sheets.

    Each instance image is a 3-panel horizontal strip (~3:1), so square
    thumbnails shrink it to an unreadable sliver. Cells are therefore wide and
    short, and pages hold few instances.

    Writes ``<stem>_p01.png``, ``<stem>_p02.png``, ...
    """
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
    ap.add_argument("--out", default="docs/DEEPCRACK_FINETUNE/fp_dump")
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
    dir_partial = out / "partial"
    dir_full = out / "full"
    for d in (dir_partial, dir_full):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()

    rows = []
    n_partial = n_full = 0

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
        if n_pred == 0:
            continue

        if n_gt == 0:
            iou = np.zeros((n_pred, 0))
            mp = np.zeros(n_pred, bool)
        else:
            iou = pairwise_iou(pred, gt, n_pred, n_gt)
            mp, _ = match(iou, n_pred, n_gt)

        stem = Path(im["file_name"]).stem
        for pid in range(n_pred):
            if mp[pid]:
                continue
            best = float(iou[pid].max()) if n_gt else 0.0
            kind = "partial" if best > 0.1 else "full"
            m = pred == pid + 1
            strip = panel(img, gt, m, args.pad)
            if strip is None:
                continue

            name = f"{stem}__fp{pid + 1:02d}_{kind}_iou{best:.2f}_a{int(m.sum())}.png"
            target = dir_partial if kind == "partial" else dir_full
            cv2.imwrite(str(target / name), strip)

            rows.append(
                {
                    "file": stem,
                    "image_index": idx,
                    "pred_id": pid + 1,
                    "kind": kind,
                    "best_iou": round(best, 4),
                    "area": int(m.sum()),
                    "n_pred": n_pred,
                    "n_gt": n_gt,
                    "image": name,
                }
            )
            if kind == "partial":
                n_partial += 1
            else:
                n_full += 1

    rows.sort(key=lambda r: (r["kind"], -r["area"]))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)

    contact_sheet(sorted(dir_partial.glob("*.png")), out / "sheet_partial")
    contact_sheet(sorted(dir_full.glob("*.png")), out / "sheet_full")

    print(f"config: thr={args.threshold} min_area={args.min_area} close_k={args.close_k}")
    print(f"partial FPs (overlap GT): {n_partial}  -> {dir_partial}")
    print(f"full FPs (no GT overlap): {n_full}  -> {dir_full}")
    print(f"total: {n_partial + n_full}")
    print(f"manifest: {out / 'manifest.csv'}")
    print(f"sheets:   {out}/sheet_partial_p*.png, {out}/sheet_full_p*.png")


if __name__ == "__main__":
    main()
