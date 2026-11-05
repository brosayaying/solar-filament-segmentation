"""Plot the classifier's per-patch score map for one image.

Answers "is the classifier really working?" by showing the raw probability
field, not just aggregate metrics. A high AUC with a poor PQ usually means the
score map is *diffuse*: correct on average, but with no sharp decision boundary.

    python scripts/viz_scoremap.py --model /tmp/rf8_cmp.joblib --patch 8 --index 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import patch_clf as pc  # noqa: E402

cv2.setNumThreads(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="docs/EDGE/scoremap.png")
    args = ap.parse_args()

    pc.set_patch_size(args.patch)
    model = joblib.load(args.model)

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    iid = split["val"][args.index]
    img = pc.read_image(images[iid]["file_name"])
    disk = pc.disk_mask(img)
    gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)

    prob = pc.predict_prob(model, img, disk)
    y = (pc.patch_fraction(gt) > 0.25).astype(np.uint8)

    # Restrict to the inner disk, matching how the model was trained/scored.
    inner = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    valid = inner[:: args.patch, :: args.patch][: pc.GRID, : pc.GRID] > 0

    pv, yv = prob[valid], y[valid]
    print(f"{images[iid]['file_name']}  patch={args.patch} grid={pc.GRID}")
    print(f"  valid patches {len(pv):,}  positives {int(yv.sum()):,} ({100 * yv.mean():.3f}%)")
    print(f"  prob on positives: {np.percentile(pv[yv == 1], [10, 50, 90]).round(3)}")
    print(f"  prob on negatives: {np.percentile(pv[yv == 0], [50, 90, 99]).round(3)}")
    print(f"  prob overall: min={pv.min():.3f} max={pv.max():.3f} mean={pv.mean():.3f}")

    # How many patches clear a usable threshold, and how many are ambiguous?
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        n = int((pv > thr).sum())
        tp = int(((pv > thr) & (yv == 1)).sum())
        print(f"  thr={thr:.1f}: {n:>7,} patches above ({100 * n / len(pv):5.2f}%), {tp:>5,} true")

    # ---- figure ----------------------------------------------------------- #
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title(f"(a) input {images[iid]['file_name']}")

    im = axes[0, 1].imshow(prob, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title(f"(b) P(filament) per {args.patch}x{args.patch} patch")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(y, cmap="gray")
    axes[0, 2].set_title(f"(c) patch labels (GT frac > 0.25)\n{int(y.sum())} positive patches")

    # Histogram: the key diagnostic. Overlapping distributions = diffuse map.
    axes[1, 0].hist(pv[yv == 0], bins=60, range=(0, 1), alpha=0.7, label="background", color="steelblue")
    axes[1, 0].hist(pv[yv == 1], bins=60, range=(0, 1), alpha=0.7, label="filament", color="crimson")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("P(filament)")
    axes[1, 0].set_ylabel("patch count (log)")
    axes[1, 0].set_title("(d) score distribution")
    axes[1, 0].legend()

    # Zoom on a filament region: does the score localise?
    ys, xs = np.nonzero(gt)
    if len(ys):
        pad = 100
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        # Keep the crop square-ish and bounded.
        if y1 - y0 > 700:
            y1 = y0 + 700
        if x1 - x0 > 700:
            x1 = x0 + 700
        axes[1, 1].imshow(img[y0:y1, x0:x1], cmap="gray", vmin=60, vmax=160)
        axes[1, 1].set_title("(e) zoom: input")

        # Overlay the probability as a translucent heatmap on the zoom.
        pz = prob[
            y0 // args.patch : y1 // args.patch, x0 // args.patch : x1 // args.patch
        ]
        gz = gt[y0:y1, x0:x1]
        axes[1, 2].imshow(img[y0:y1, x0:x1], cmap="gray", vmin=60, vmax=160)
        axes[1, 2].imshow(
            cv2.resize(pz, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST),
            cmap="inferno",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        # GT boundary in cyan so it is visible against the heatmap.
        edge = cv2.morphologyEx((gz > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        axes[1, 2].contour(edge, levels=[0.5], colors="cyan", linewidths=1.2)
        axes[1, 2].set_title("(f) zoom: score overlay, cyan = GT boundary")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
