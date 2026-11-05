"""Visualise the Jiang et al. (2007) morphological edge detector on filaments.

Runs the five pipeline stages and shows each one, plus the ground truth, so it
is visible *where* the detector succeeds and where it fails on this data.

    python scripts/viz_morph_edge.py --n 3 --out docs/EDGE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import morph_edge as me  # noqa: E402
import patch_clf as pc  # noqa: E402

cv2.setNumThreads(4)


def crop_to_filaments(img: np.ndarray, gt: np.ndarray, pad: int = 120) -> tuple[int, int, int, int]:
    """Bounding box around the annotations, padded, clamped to the image."""
    ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        h, w = img.shape
        return 0, 0, w, h
    h, w = img.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)
    return x0, y0, x1 - x0, y1 - y0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--se-size", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--min-side", type=int, default=128)
    ap.add_argument("--out", default="docs/EDGE")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    for k, iid in enumerate(split["val"][: args.n]):
        img = pc.read_image(images[iid]["file_name"])
        disk = pc.disk_mask(img)
        gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
        inner = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))

        r = me.detect(
            img,
            threshold=args.threshold,
            min_side=args.min_side,
            se_size=args.se_size,
            use_masks=False,
            disk=disk,
        )

        # Crop to the annotated region: at full frame a handful of thin
        # filaments is invisible, and the failure mode is only legible when
        # zoomed in.
        cx, cy, cw, ch = crop_to_filaments(img, gt)
        inner_c = inner[cy : cy + ch, cx : cx + cw] > 0
        edges = (r["ff"] > 0)[cy : cy + ch, cx : cx + cw]
        fq = r["fq"][cy : cy + ch, cx : cx + cw]
        img_c = img[cy : cy + ch, cx : cx + cw]
        gt_c = gt[cy : cy + ch, cx : cx + cw]

        fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

        axes[0, 0].imshow(img_c, cmap="gray", vmin=60, vmax=160)
        axes[0, 0].set_title(f"(a) input  {images[iid]['file_name']}")

        axes[0, 1].imshow(np.clip(fq, 0, np.percentile(fq, 99)), cmap="inferno")
        axes[0, 1].set_title(
            f"(b) Fq (quad-enhanced)\nmean={fq[inner_c].mean():.1f} "
            f"std={fq[inner_c].std():.1f} vs paper's m<2, d<2"
        )

        ov = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
        ov[edges] = (0, 255, 0)
        axes[0, 2].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"(c) detected edges (T={args.threshold:g}, SE={args.se_size})\n{int(edges.sum()):,} px in crop")

        axes[1, 0].imshow(gt_c > 0, cmap="gray")
        axes[1, 0].set_title(f"(d) ground truth ({int(gt_c.max())} filaments)")

        # Edge coverage of a 9-px band around the GT boundary.
        band = cv2.dilate((gt_c > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        hit = (edges & band).sum()
        axes[1, 1].imshow(band, cmap="gray")
        axes[1, 1].set_title(
            f"(e) 9-px band around GT\nedges on band: {hit}/{max(1, edges.sum())} "
            f"({100 * hit / max(1, edges.sum()):.1f}%)"
        )

        ov2 = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
        ov2[gt_c > 0] = (0, 0, 255)
        ov2[edges] = (0, 255, 0)
        ov2[(gt_c > 0) & edges] = (0, 255, 255)
        axes[1, 2].imshow(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("(f) GT red / edges green / overlap yellow")

        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        p = out / f"stage_panels_{k}.png"
        fig.savefig(p, dpi=115)
        plt.close(fig)
        print(f"wrote {p}")

        # Summary numbers for the doc.
        print(
            f"  {images[iid]['file_name']}: edges={int(r['ff'].sum()):,}  "
            f"on_band={hit}  band={int(band.sum()):,}  "
            f"Fq mean={r['fq'][inner > 0].mean():.1f} std={r['fq'][inner > 0].std():.1f}"
        )

    # Whole-disk view of one image, to show why the quad-statistics gate fails.
    iid = split["val"][0]
    img = pc.read_image(images[iid]["file_name"])
    disk = pc.disk_mask(img)
    r = me.detect(img, args.threshold, args.min_side, args.se_size, use_masks=False, disk=disk)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("input (full frame)")
    axes[1].imshow(np.clip(r["fq"], 0, np.percentile(r["fq"], 99.5)), cmap="inferno")
    axes[1].set_title("Fq, full frame: response is\nuniform noise, not structure")
    axes[2].imshow(r["ff"], cmap="gray")
    axes[2].set_title(f"thresholded (T={args.threshold:g})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    p = out / "full_disk.png"
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
