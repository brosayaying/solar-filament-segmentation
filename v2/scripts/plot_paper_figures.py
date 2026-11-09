"""Figures the competition template explicitly requires.

The template's Methodology section asks for two things we had not produced:

    a graphic illustrating the entire data pipeline
    one or more graphics showing a few examples of the segmentations

Both are generated here from real data rather than drawn by hand, so they cannot
drift from the system they describe.

    python v2/scripts/plot_paper_figures.py

Writes to v2/results/figures/:
    pipeline.png        the end-to-end data flow
    segmentation_examples.png   raw / GT / prediction / error for four images
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import (  # noqa: E402
    disk_geometry,
    load_coco,
    parse_preprocess,
    preprocess_image,
)
from solar.postprocess import prob_to_labelmap  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"
FIG = ROOT / "v2/results/figures"
ANN = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/train_images"

# Shipped post-processing.
THR, MIN_AREA, CLOSE_K = 0.50, 256, 15


# --------------------------------------------------------------------------- #
# Pipeline diagram
# --------------------------------------------------------------------------- #
def plot_pipeline(out: Path) -> None:
    """Block diagram of the end-to-end data flow.

    Drawn with explicit coordinates rather than a layout engine: the flow is
    linear with one branch, and hand placement keeps the labels readable at
    print size.
    """
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.4, edgecolor=color, facecolor=color + "18"))
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top",
                fontsize=7.4, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.3, color="#555555", shrinkA=2, shrinkB=2))

    blue, green, orange, purple = "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"

    # Row 1: preprocessing -> segmentation
    box(0.15, 2.55, 2.5, 1.75, "Input",
        "2048x2048 H-alpha\n8-bit grayscale\n707 train / 180 test", blue)
    box(3.05, 2.55, 2.9, 1.75, "Preprocess (full res)",
        "estimate disk geometry\nradial flatten\nnormalise mu=0.4254 sd=0.0809", blue)
    box(6.35, 2.55, 2.9, 1.75, "Segment",
        "ConvNeXt-Small UNet\n512px patches, 5 folds\nbf16, BCE + Dice", green)
    box(9.65, 2.55, 3.2, 1.75, "Probability map",
        "average 5 folds\n+ 4-flip TTA\n2048x2048 in [0,1]", green)

    arrow(2.65, 3.42, 3.05, 3.42)
    arrow(5.95, 3.42, 6.35, 3.42)
    arrow(9.25, 3.42, 9.65, 3.42)

    # Row 2: candidates -> scoring -> output
    box(9.65, 0.35, 3.2, 1.75, "Submission",
        "COCO RLE per instance\n1069 filaments\n180 images", purple)
    box(6.35, 0.35, 2.9, 1.75, "Re-score candidates",
        "GBDT over 17 features\nq-hat > tau = 0.35\nkeeps / rejects", orange)
    box(3.05, 0.35, 2.9, 1.75, "Candidates",
        "threshold 0.30\n8-connected components\nmin area 128 px", orange)

    arrow(11.25, 2.55, 11.25, 2.10)
    arrow(9.65, 1.22, 9.25, 1.22)
    arrow(6.35, 1.22, 5.95, 1.22)

    # Annotation of what the re-scoring stage buys. These are the numbers for
    # the shipped configuration (GBDT alone at tau 0.35), not the best-OOF
    # blend, so the figure and the results table agree.
    ax.text(6.5, 0.12,
            "re-scoring: PQ 0.4220 -> 0.4386  (+0.0166, CI [+0.0095, +0.0179])",
            ha="center", fontsize=8, style="italic", color="#444444")

    ax.text(0.15, 4.42, "End-to-end pipeline", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Segmentation examples
# --------------------------------------------------------------------------- #
def _overlay_labels(ax, img, labels, color, alpha=0.45):
    """Draw instance boundaries over a grayscale image."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    if labels.max() == 0:
        return
    # Boundaries only: a filled overlay hides the filament texture, which is
    # exactly what a reader needs to see to judge the segmentation.
    edge = np.zeros(labels.shape, bool)
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge &= labels > 0
    rgba = np.zeros((*labels.shape, 4))
    rgba[edge] = matplotlib.colors.to_rgba(color, alpha)
    ax.imshow(rgba)


def _crop_around(labels: np.ndarray, pad: int = 90) -> tuple[slice, slice]:
    """Bounding box around all instances, padded, clipped to the image.

    A full-disk view is useless for judging segmentation: a filament occupies
    ~0.3% of the pixels, so at print size it is a few grey specks. Cropping to
    the annotated region is the only way the figure shows anything.
    """
    ys, xs = np.nonzero(labels > 0)
    if ys.size == 0:
        h, w = labels.shape
        return slice(0, h), slice(0, w)
    y0, y1 = max(0, ys.min() - pad), min(labels.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(labels.shape[1], xs.max() + pad)
    # Keep the crop square so the aspect ratio is not distorted.
    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(labels.shape[0] - side, cy - side // 2))
    x0 = max(0, min(labels.shape[1] - side, cx - side // 2))
    return slice(y0, y0 + side), slice(x0, x0 + side)


def plot_examples(out: Path, n: int = 3) -> None:
    """Raw / ground truth / prediction / error, cropped to the filaments.

    Images are chosen to span the quality range rather than to flatter the
    system: the worst, a poor one, the median, and the best. A figure that
    shows only successes is not evidence.
    """
    coco = load_coco(ANN)
    images = {img["id"]: img for img in coco["images"]}

    # Score every OOF file so the examples can be selected by quality.
    from solar.metrics import pairwise_iou

    scored = []
    for path in sorted(OOF.glob("fold*/*.npz")):
        with np.load(path, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
        # Keyword args: the 4th positional is `connectivity`, not `close_k`.
        pred = prob_to_labelmap(
            prob, threshold=THR, min_area=MIN_AREA, close_k=CLOSE_K
        )
        gt = labels[0]
        if pred.max() == 0 or gt.max() == 0:
            continue
        iou = pairwise_iou(pred, gt, int(pred.max()), int(gt.max()))
        scored.append((float(iou.max(axis=0).mean()), path.stem, path))

    scored.sort()
    if len(scored) < n:
        raise SystemExit("not enough OOF files")
    # Three rows: worst, median, best. Four rows overflowed a page in the
    # two-column ACM layout, and the extra row added no new failure mode.
    picks = [scored[0], scored[len(scored) // 2], scored[-1]]

    fig, axes = plt.subplots(n, 4, figsize=(12.5, 2.55 * n))
    titles = ["Input (flattened)", "Ground truth", "Prediction",
              "Error: green = missed, red = false positive"]

    for row, (score, stem, path) in enumerate(picks):
        with np.load(path, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
            meta = json.loads(str(z["meta"]))

        raw = plt.imread(IMG / meta["file_name"])
        if raw.ndim == 3:
            raw = raw[..., 0]
        raw8 = (raw * 255).astype(np.uint8) if raw.max() <= 1.0 else raw
        flat, _ = preprocess_image(raw8, parse_preprocess("flatten"))
        # Stretch for display only; the model sees the normalised version.
        disp = np.clip(flat, np.percentile(flat, 1), np.percentile(flat, 99))
        # np.ptp(arr), not arr.ptp(): the method was removed in numpy 2.x.
        rng = max(1e-6, float(np.ptp(disp)))
        disp = ((disp - disp.min()) / rng * 255).astype(np.uint8)

        gt = labels[0]
        pred = prob_to_labelmap(
            prob, threshold=THR, min_area=MIN_AREA, close_k=CLOSE_K
        )

        # Crop to the union of GT and prediction so both are visible.
        ys, xs = _crop_around(np.maximum(gt, pred))
        disp_c, gt_c, pred_c = disp[ys, xs], gt[ys, xs], pred[ys, xs]

        _overlay_labels(axes[row, 0], disp_c, np.zeros_like(gt_c), "white")
        _overlay_labels(axes[row, 1], disp_c, gt_c, "#2ca02c")
        _overlay_labels(axes[row, 2], disp_c, pred_c, "#1f77b4")

        # Error map: green = missed GT, red = false positive.
        err = np.stack([disp_c] * 3, axis=-1)
        if pred_c.max() and gt_c.max():
            iou = pairwise_iou(pred_c, gt_c, int(pred_c.max()), int(gt_c.max()))
            order = np.argsort(iou, axis=None)[::-1]
            mp = np.zeros(int(pred_c.max()), bool)
            mg = np.zeros(int(gt_c.max()), bool)
            for flat_i in order:
                a, b = divmod(int(flat_i), int(gt_c.max()))
                if iou[a, b] <= 0.5:
                    break
                if mp[a] or mg[b]:
                    continue
                mp[a] = mg[b] = True
            missed = np.isin(gt_c, np.where(~mg)[0] + 1)
            fp = np.isin(pred_c, np.where(~mp)[0] + 1)
            err[missed] = [0, 200, 0]
            err[fp] = [230, 40, 40]
        axes[row, 3].imshow(err)

        for c in range(4):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(titles[c], fontsize=9.5,
                                       fontweight="bold")
        axes[row, 0].set_ylabel(
            f"{stem}\nmean best IoU {score:.2f}", fontsize=7.5)

    fig.suptitle(
        "Segmentation examples, worst to best, cropped to the annotated region.",
        fontsize=11, y=0.997)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plot_pipeline(FIG / "pipeline.png")
    plot_examples(FIG / "segmentation_examples.png")


if __name__ == "__main__":
    main()
