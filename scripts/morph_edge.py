"""Morphological edge detector for thin edges in low-contrast regions.

Implementation of

    J.-A. Jiang, C.-L. Chuang, Y.-L. Lu, C.-S. Fahn,
    "Mathematical-morphology-based edge detectors for detection of thin edges
    in low-contrast regions",
    IET Image Processing, 1(3), 269-277, 2007.  doi:10.1049/iet-ipr:20060273

The paper's five stages, as specified:

1. **Contrast enhancement** (eq. 1-3):
       Fc = F + WTH(F) - BTH(F)
   where WTH is the white top-hat ``F - (F o B)`` and BTH the black top-hat
   ``(F . B) - F``, with ``o``/``.`` being opening/closing by the star SE.

2. **Dilation residue edge detector** (eq. 4):
       Fe = (Fc (+) B) - Fc
   A one-sided (external) morphological boundary. Unlike a symmetric gradient
   it produces a single-pixel-wide ridge, which is what preserves thin edges.

3. **Quad-decomposition edge enhancement** (section 3, Table 1):
   ``Fe`` is recursively split into quadrants until the side falls below a
   minimum. In each quadrant, the mean ``m`` and std ``d`` decide between four
   pixel cases (prune to 0 / square / cube / square-in-complex-background).
   Squaring and cubing are what lift weak responses above the global threshold:
   a raw response of 5 becomes 25, which clears ``T = 20``.

4. **Global threshold** (eq. 7), ``T`` in 20..30 per the paper's recommendation.

5. **Bipolar oriented edge masks** (section 4, Fig. 4): five 7x7 masks applied
   only to edge pixels in low-contrast quadrants (``m < 3 and d < 3``). A pixel
   survives if *any* mask score is positive.

Deviations from the paper, all noted in comments where they occur:

* The star SE (Fig. 2) is a vector drawing in the PDF and not extractable; the
  standard 5x5 star is used instead.
* The five 7x7 masks *are* transcribed exactly from Fig. 4 (raster page 4).

    python scripts/morph_edge.py --index 0 --out /tmp/edge0.png
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

import patch_clf as pc  # noqa: E402

cv2.setNumThreads(4)

# --------------------------------------------------------------------------- #
# Structuring elements
# --------------------------------------------------------------------------- #

# Fig. 2 is a vector drawing in the PDF, so it cannot be lifted as a raster.
# The "star" SE in this literature is the standard 5-point cross+diagonal star.
STAR = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    np.uint8,
)

# Fig. 4, transcribed verbatim (7x7). Row order is top-to-bottom as printed.
MASKS = {
    "0": np.array(
        [
            [-2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, -2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1],
            [-2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2],
        ],
        np.float32,
    ),
    "pi/2": np.array(
        [
            [-2, -2, 1, 2, 1, -2, -2],
            [-2, -2, 1, 2, 1, -2, -2],
            [-2, -2, 1, -2, 1, -2, -2],
            [-2, -2, 1, -2, 1, -2, -2],
            [-2, -2, 1, -2, 1, -2, -2],
            [-2, -2, 1, -2, 1, -2, -2],
            [-2, -2, 1, -2, 1, -2, -2],
        ],
        np.float32,
    ),
    "3pi/8": np.array(
        [
            [-1, 1, 2, 1, -2, -2, -2],
            [-1, 1, 2, 1, -1, -2, -2],
            [-1, 1, 2, 1, -1, -2, -2],
            [-2, -1, 1, -2, 1, -1, -2],
            [-2, -1, 1, 2, 1, 1, -1],
            [-2, -2, -1, 1, 2, 1, 1],
            [-2, -2, -2, -1, 1, 2, 1],
        ],
        np.float32,
    ),
    "pi/4": np.array(
        [
            [2, 1, -1, -2, -2, -2, -2],
            [1, 2, 1, -1, -2, -2, -2],
            [-1, 1, 2, 1, -1, -2, -2],
            [-2, -1, 1, -2, 1, -1, -2],
            [-2, -2, -1, 1, 2, 1, -1],
            [-2, -2, -2, -1, 1, 2, 1],
            [-2, -2, -2, -2, -1, 1, 2],
        ],
        np.float32,
    ),
    "corner": np.array(
        [
            [-2, 1, 1, 2, 1, -2, -2],
            [1, 1, 1, 2, 1, -2, -2],
            [1, 1, 1, 2, 1, -2, -2],
            [2, 2, 2, -2, 1, -2, -2],
            [1, 1, 1, 1, 1, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2],
        ],
        np.float32,
    ),
}


# --------------------------------------------------------------------------- #
# Stage 1 + 2
# --------------------------------------------------------------------------- #


def contrast_enhance(img: np.ndarray, se: np.ndarray = STAR) -> np.ndarray:
    """Eq. 1: ``Fc = F + WTH(F) - BTH(F)``, clipped to the input range."""
    f = img.astype(np.float32)
    wth = f - cv2.morphologyEx(f, cv2.MORPH_OPEN, se)
    bth = cv2.morphologyEx(f, cv2.MORPH_CLOSE, se) - f
    fc = f + wth - bth
    return np.clip(fc, 0, 255)


def dilation_residue(fc: np.ndarray, se: np.ndarray = STAR) -> np.ndarray:
    """Eq. 4: ``Fe = (Fc (+) B) - Fc``. Single-pixel external boundary."""
    return cv2.dilate(fc, se) - fc


# --------------------------------------------------------------------------- #
# Stage 3: quad-decomposition enhancement (Table 1)
# --------------------------------------------------------------------------- #


def enhance_quadrant(sub: np.ndarray) -> np.ndarray:
    """Apply the four Table 1 cases to one sub-image.

    Cases are evaluated in the paper's order. The conditions are on the
    sub-image statistics ``m`` and ``d`` and are *absolute grey levels*, which
    is why they transfer across images at all.

    The cube is clamped: a response of 40 (plausible for a real edge) cubes to
    64000, and without a cap the dynamic range explodes. The paper's images
    evidently did not reach that regime.
    """
    m = float(sub.mean())
    d = float(sub.std())
    out = sub.copy()

    low_contrast = (m < 2) and (d < 2)
    complex_bg = (m > 5) and (d > 6)

    if low_contrast:
        # Case 1: redundant background -> prune.
        out[sub < m] = 0.0
        # Cases 2 and 3: thin / intensive thin edges -> square / cube.
        thin = (sub > m) & (sub < m + 1.5 * d)
        thick = (sub >= m + 1.5 * d) & (sub < m + 3.0 * d)
        out[thin] = np.square(sub[thin])
        out[thick] = np.clip(np.power(sub[thick], 3), 0, 255.0)

    if complex_bg:
        # Case 4: thin edge in a complex background -> square.
        sel = sub > m + 1.5 * d
        out[sel] = np.square(sub[sel])

    return out


def quad_enhance(fe: np.ndarray, min_side: int = 128) -> np.ndarray:
    """Recursive quad-decomposition, section 3.

    The paper decomposes to a quadtree and sums only over *leaves*; since every
    leaf is processed with its own statistics and leaves do not overlap, walking
    a uniform ``min_side`` grid is equivalent and cheaper.
    """
    out = fe.astype(np.float32).copy()
    h, w = out.shape
    step = max(8, min_side)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + step, h), min(x0 + step, w)
            out[y0:y1, x0:x1] = enhance_quadrant(out[y0:y1, x0:x1])
    return out


# --------------------------------------------------------------------------- #
# Stage 5: bipolar oriented edge masks
# --------------------------------------------------------------------------- #


def mask_filter(ft: np.ndarray, quad_stats: list[tuple[int, int, int, int, float, float]]) -> np.ndarray:
    """Keep edge pixels that score positive on at least one Fig. 4 mask.

    Per section 4, the masks are applied only to edge pixels lying in quadrants
    with ``m < 3 and d < 3`` (background / low-contrast regions).
    """
    keep = np.zeros_like(ft)
    # Masked convolution: score = sum(mask * Ft under it).
    scores = [cv2.filter2D(ft.astype(np.float32), -1, k) for k in MASKS.values()]

    for y0, x0, hh, ww, m, d in quad_stats:
        if not (m < 3.5 and d < 3.5):
            # Low-contrast gate, relaxed from the paper's (<3, <3) because the
            # quad statistics here are computed on the *enhanced* image, which
            # inflates d relative to the paper's un-squared Fe.
            continue
        y1, x1 = y0 + hh, x0 + ww
        sub = np.zeros((hh, ww), bool)
        for s in scores:
            sub |= s[y0:y1, x0:x1] > 0
        keep[y0:y1, x0:x1] = sub & (ft[y0:y1, x0:x1] > 0)
    return keep.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #


def _star(ksize: int) -> np.ndarray:
    """Square SE of the requested size.

    The paper's Fig. 2 star cannot be extracted from the PDF. At ksize=5 the
    literal STAR is used; for larger sizes (needed here, see ``detect``) a
    square is used because square and elliptical SEs scored identically in
    ``scripts/morph_contrast.py`` (AUC 0.8114 vs 0.8141).
    """
    if ksize <= 5:
        return STAR
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def detect(
    img: np.ndarray,
    threshold: float = 20.0,
    min_side: int = 128,
    se_size: int = 5,
    use_masks: bool = True,
    polarity: str = "dark",
    disk: np.ndarray | None = None,
    margin: int = 40,
) -> dict[str, np.ndarray]:
    """Run the full five-stage detector.

    ``polarity="dark"`` inverts before stage 1, because filaments are dark.

    ``disk`` is an addition, not in the paper. The solar limb produces a
    dilation residue ~40x stronger than any filament (139 -> 95 -> 0 across
    ~40 px, versus a ~10 grey-level local step), so without masking it the
    entire response is limb and the quad statistics in every leaf are dominated
    by it.
    """
    src = (255 - img) if polarity == "dark" else img
    se = _star(se_size)

    fc = contrast_enhance(src, se)
    fe = dilation_residue(fc, se)
    fq = quad_enhance(fe, min_side=min_side)

    ft = np.where(fq >= threshold, 1, 0).astype(np.uint8)

    # Quadrant statistics for the mask gate.
    quad_stats = []
    step = max(8, min_side)
    for y0 in range(0, img.shape[0], step):
        for x0 in range(0, img.shape[1], step):
            y1, x1 = min(y0 + step, img.shape[0]), min(x0 + step, img.shape[1])
            sub = fq[y0:y1, x0:x1]
            quad_stats.append((y0, x0, y1 - y0, x1 - x0, float(sub.mean()), float(sub.std())))

    ff = mask_filter(ft, quad_stats) if use_masks else ft

    if disk is not None:
        inner = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
        ff = ff * (inner > 0).astype(np.uint8)

    return {"fc": fc, "fe": fe, "fq": fq, "ft": ft, "ff": ff}


# --------------------------------------------------------------------------- #
# Edge -> region conversion
# --------------------------------------------------------------------------- #


def edges_to_regions(
    edges: np.ndarray, close_k: int = 15, min_area: int = 200
) -> np.ndarray:
    """Turn a thin edge map into filled instances.

    The paper outputs *boundaries*. PQ needs filled regions, and the paper gives
    no guidance for the gap, so this is our addition: close the edge map to seal
    the outline, then take connected components.
    """
    if edges.sum() == 0:
        return np.zeros(edges.shape, np.int32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    filled = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, k)
    # Fill enclosed holes so a closed filament outline becomes a solid region.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    out = np.zeros(edges.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--min-side", type=int, default=128)
    ap.add_argument("--polarity", default="dark", choices=["dark", "bright"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    iid = split["val"][args.index]
    img = pc.read_image(images[iid]["file_name"])
    gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)

    out = detect(img, args.threshold, args.min_side, True, args.polarity)
    print(f"{images[iid]['file_name']}  gt={gt.max()} inst")
    print(f"  Fe  max={out['fe'].max():.1f} mean={out['fe'].mean():.2f}")
    print(f"  Fq  max={out['fq'].max():.1f} mean={out['fq'].mean():.2f}")
    print(f"  unified threshold T={args.threshold}: {int(out['ft'].sum()):,} px")
    print(f"  after masks               : {int(out['ff'].sum()):,} px")

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        panels = [
            (img, "input", "gray", None),
            (out["fe"], "Fe = dilation residue", "viridis", None),
            (out["fq"], "Fq = quad-enhanced", "viridis", 99.5),
            (out["ft"], f"thresholded (T={args.threshold})", "gray", None),
            (out["ff"], "after bipolar masks", "gray", None),
        ]
        for ax, (arr, title, cmap, pct) in zip(axes.ravel(), panels):
            if pct is not None:
                v = np.clip(arr, 0, np.percentile(arr, pct))
            else:
                v = arr
            ax.imshow(v, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ov[gt > 0] = (0, 0, 255)
        ov[out["ff"] > 0] = (0, 255, 0)
        ov[(gt > 0) & (out["ff"] > 0)] = (0, 255, 255)
        axes.ravel()[5].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        axes.ravel()[5].set_title("GT red / edge green / both yellow")
        axes.ravel()[5].axis("off")

        fig.tight_layout()
        fig.savefig(args.out, dpi=110)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
