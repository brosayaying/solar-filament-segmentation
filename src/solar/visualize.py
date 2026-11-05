"""Render annotated samples for visual inspection.

    python -m solar.visualize --n 6 --out docs/samples.png
    python -m solar.visualize --n 4 --split val --out docs/val_samples.png
    python -m solar.visualize --n 4 --crop 1024 --out docs/crops.png

Each panel shows the H-alpha image with, per filament:
  * segmentation polygon outline (green)
  * spine / centerline (red)
  * bounding box (thin yellow)
  * category label (Left / Right / Unidentifiable / Ambiguous)

The point is to see what the model is actually being asked to predict: thin,
low-contrast dark structures against a noisy disk, often several per image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CATEGORY_COLORS = {
    1: (80, 220, 80),    # Left        - green
    2: (80, 160, 255),   # Right       - blue
    3: (200, 200, 200),  # Unidentifiable - grey
    4: (255, 120, 255),  # Ambiguous   - magenta
}
CATEGORY_NAMES = {1: "Left", 2: "Right", 3: "Unident.", 4: "Ambig."}


def draw_annotations(
    img: np.ndarray,
    anns: list[dict],
    offset: tuple[int, int] = (0, 0),
    show_bbox: bool = True,
    show_spine: bool = True,
    show_poly: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay annotations on a grayscale image, returning BGR."""
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ox, oy = offset

    for ann in anns:
        color = CATEGORY_COLORS.get(ann["category_id"], (255, 255, 255))

        if show_poly:
            for poly in ann["segmentation"]:
                pts = np.asarray(poly, np.float32).reshape(-1, 2).copy()
                pts[:, 0] -= ox
                pts[:, 1] -= oy
                cv2.polylines(
                    canvas, [np.round(pts).astype(np.int32)], True, color, thickness
                )

        if show_spine and ann.get("spine"):
            pts = np.asarray(ann["spine"], np.float32).reshape(-1, 2).copy()
            pts[:, 0] -= ox
            pts[:, 1] -= oy
            cv2.polylines(
                canvas, [np.round(pts).astype(np.int32)], False, (60, 60, 255), thickness
            )

        if show_bbox:
            x, y, w, h = ann["bbox"]
            cv2.rectangle(
                canvas,
                (int(x - ox), int(y - oy)),
                (int(x + w - ox), int(y + h - oy)),
                (0, 220, 220),
                1,
            )

    return canvas


def label_panel(canvas: np.ndarray, text: str) -> np.ndarray:
    """Draw a dark caption bar at the top of a panel."""
    bar = np.zeros((26, canvas.shape[1], 3), np.uint8)
    cv2.putText(
        bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([bar, canvas])


def make_grid(panels: list[np.ndarray], cols: int, pad: int = 6) -> np.ndarray:
    """Tile panels into a grid with padding, resizing to a common width."""
    w = min(p.shape[1] for p in panels)
    resized = []
    for p in panels:
        if p.shape[1] != w:
            scale = w / p.shape[1]
            p = cv2.resize(p, (w, int(round(p.shape[0] * scale))))
        resized.append(p)

    h = max(p.shape[0] for p in resized)
    rows = (len(resized) + cols - 1) // cols
    grid = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 30, np.uint8)

    for i, p in enumerate(resized):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        grid[y : y + p.shape[0], x : x + p.shape[1]] = p
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--ann-file", default="train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"])
    ap.add_argument("--split-file", default="splits/date_seed42.json")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--crop", type=int, default=0, help="crop size; 0 = full image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/samples.png")
    ap.add_argument("--no-bbox", action="store_true")
    ap.add_argument("--no-spine", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    with open(root / args.ann_file) as fh:
        coco = json.load(fh)

    ann_index: dict[str, list[dict]] = {}
    for a in coco["annotations"]:
        ann_index.setdefault(a["image_id"], []).append(a)
    images = {i["id"]: i for i in coco["images"]}

    ids = list(images)
    if args.split != "all":
        with open(args.split_file) as fh:
            ids = json.load(fh)[args.split]

    rng = np.random.default_rng(args.seed)
    # Prefer images with a representative number of filaments.
    counts = np.array([len(ann_index.get(i, [])) for i in ids])
    order = np.argsort(np.abs(counts - np.median(counts)))
    pool = [ids[i] for i in order[: max(args.n * 4, 20)]]
    chosen = rng.choice(pool, size=min(args.n, len(pool)), replace=False)

    panels = []
    for image_id in chosen:
        meta = images[image_id]
        img = cv2.imread(str(root / "train/train_images" / meta["file_name"]), cv2.IMREAD_GRAYSCALE)
        anns = ann_index.get(image_id, [])

        offset = (0, 0)
        if args.crop:
            # Centre the crop on the densest cluster of filaments.
            if anns:
                cx = int(np.mean([a["bbox"][0] + a["bbox"][2] / 2 for a in anns]))
                cy = int(np.mean([a["bbox"][1] + a["bbox"][3] / 2 for a in anns]))
            else:
                cx = cy = img.shape[0] // 2
            half = args.crop // 2
            x0 = int(np.clip(cx - half, 0, img.shape[1] - args.crop))
            y0 = int(np.clip(cy - half, 0, img.shape[0] - args.crop))
            img = img[y0 : y0 + args.crop, x0 : x0 + args.crop]
            offset = (x0, y0)

        canvas = draw_annotations(
            img,
            anns,
            offset=offset,
            show_bbox=not args.no_bbox,
            show_spine=not args.no_spine,
        )
        cats = sorted({CATEGORY_NAMES.get(a["category_id"], "?") for a in anns})
        caption = f"{meta['file_name'][:14]}  n={len(anns)}  {','.join(cats)}"
        panels.append(label_panel(canvas, caption))

    grid = make_grid(panels, args.cols)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"wrote {out}  ({grid.shape[1]}x{grid.shape[0]}, {len(panels)} panels)")


if __name__ == "__main__":
    main()
