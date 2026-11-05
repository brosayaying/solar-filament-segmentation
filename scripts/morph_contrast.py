"""Which morphological operator best separates filaments from background?

Filaments are *dark*, elongated (~17 px wide at the median), and low-contrast
(+9.5 grey levels against a background std of 12.5). This script scores a range
of morphological operators on two criteria:

* **AUC** -- rank separation of filament vs background 8x8 patches.
* **Contrast SNR** -- ``(mean_fg - mean_bg) / std_bg``, i.e. how many background
  standard deviations the filament signal stands out by. This is the number that
  matters for thresholding; AUC alone can look good while the absolute margin
  stays too small to threshold reliably.

Operators tested fall into four families:

1. **Isotropic black-hat** -- closing minus image. The standard dark-feature
   detector. SE size must exceed the feature width or the response saturates.
2. **Directional black-hat** -- linear SEs at several orientations. A filament
   is locally a ridge, so an SE aligned with it should respond more strongly
   than an isotropic one of the same length.
3. **Reconstruction-based** -- opening/closing by reconstruction. These remove
   structures by *shape* rather than by size, so they preserve filament edges
   exactly instead of rounding them off the way a plain opening does.
4. **Multi-scale combinations** -- max/mean over scales, and DoG as a
   non-morphological reference.

    python scripts/morph_contrast.py --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_clf as pc  # noqa: E402

cv2.setNumThreads(4)
PATCH = 8

def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U)."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def snr(score: np.ndarray, y: np.ndarray) -> float:
    """(mean_fg - mean_bg) / std_bg, in background standard deviations."""
    fg = score[y == 1]
    bg = score[y == 0]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    s = bg.std()
    return float((fg.mean() - bg.mean()) / s) if s > 0 else 0.0


# --------------------------------------------------------------------------- #
# Operators. All return a map that is POSITIVE on filaments (dark features).
# --------------------------------------------------------------------------- #


def ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def rect(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def line(length: int, angle_deg: float) -> np.ndarray:
    """A 1-px-wide line SE of the given length and orientation."""
    k = max(3, length | 1)  # odd
    se = np.zeros((k, k), np.uint8)
    c = k // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    for t in np.linspace(-c, c, 4 * k):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < k and 0 <= y < k:
            se[y, x] = 1
    return se


def blackhat(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se).astype(np.float32)


def tophat(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se).astype(np.float32)


def close_recon_blackhat(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Closing by reconstruction, minus the image.

    This is the correct reconstruction-based dark-feature detector. A plain
    closing fills dark structures *and* dilates the bright regions around them;
    closing by reconstruction fills the dark structures then shrinks the result
    back so the bright regions keep their exact original extent. The difference
    therefore isolates dark features without displacing their edges.

    (Opening by reconstruction is the mirror operator for *bright* features and
    is not useful here -- it returns ~0 on this data, which is why it is absent.)
    """
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    recon = closed.copy()
    for _ in range(30):
        nxt = cv2.erode(recon, ellipse(3))
        nxt = np.maximum(nxt, img)
        if np.array_equal(nxt, recon):
            break
        recon = nxt
    return recon.astype(np.float32) - img.astype(np.float32)


def morph_gradient(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se).astype(np.float32)


def dog(img: np.ndarray, s1: float, s2: float) -> np.ndarray:
    f = img.astype(np.float32)
    return cv2.GaussianBlur(f, (0, 0), s2) - cv2.GaussianBlur(f, (0, 0), s1)


def build_operators() -> dict[str, callable]:
    ops: dict[str, callable] = {}

    # 1. Isotropic black-hat, both SE shapes.
    for k in (25, 41, 65, 81, 101):
        ops[f"bh_ellipse_{k}"] = lambda im, k=k: blackhat(im, ellipse(k))
    for k in (41, 65, 81):
        ops[f"bh_rect_{k}"] = lambda im, k=k: blackhat(im, rect(k))

    # 2. Directional black-hat: linear SEs at 4 orientations.
    for L in (41, 65, 81):
        for ang in (0, 45, 90, 135):
            ops[f"bh_line_{L}_{ang}"] = lambda im, L=L, a=ang: blackhat(im, line(L, a))
        # Orientation-max: a filament at any angle should light up.
        ops[f"bh_line_max_{L}"] = lambda im, L=L: np.maximum.reduce(
            [blackhat(im, line(L, a)) for a in (0, 45, 90, 135)]
        )

    # 3. Reconstruction-based. Closing by reconstruction is the dark-feature
    #    operator; opening by reconstruction targets bright features and is
    #    omitted (it returns ~0 here).
    for k in (41, 65):
        ops[f"closerecon_{k}"] = lambda im, k=k: close_recon_blackhat(im, ellipse(k))

    # 4. Multi-scale and references.
    ops["bh_multiscale_max"] = lambda im: np.maximum.reduce(
        [blackhat(im, ellipse(k)) for k in (41, 65, 81)]
    )
    ops["bh_multiscale_mean"] = lambda im: np.mean(
        [blackhat(im, ellipse(k)) for k in (41, 65, 81)], axis=0
    )
    ops["bh65_median3"] = lambda im: cv2.medianBlur(
        blackhat(im, ellipse(65)).astype(np.uint8), 3
    ).astype(np.float32)
    ops["dog_32_64"] = lambda im: dog(im, 32, 64)
    ops["tophat_ellipse_65"] = lambda im: tophat(im, ellipse(65))
    ops["morph_gradient_9"] = lambda im: morph_gradient(im, ellipse(9))

    # 5. Combinations of the two best families. The directional max and the
    #    isotropic black-hat respond to slightly different structure, so their
    #    geometric mean (a soft AND) should suppress responses only one finds.
    ops["bh65_times_linemax65"] = lambda im: (
        blackhat(im, ellipse(65))
        * np.maximum.reduce([blackhat(im, line(65, a)) for a in (0, 45, 90, 135)])
    ) ** 0.5
    ops["bh65_min_linemax65"] = lambda im: np.minimum(
        blackhat(im, ellipse(65)),
        np.maximum.reduce([blackhat(im, line(65, a)) for a in (0, 45, 90, 135)]),
    )
    ops["closerecon65_x_bh65"] = lambda im: (
        close_recon_blackhat(im, ellipse(65)) * blackhat(im, ellipse(65))
    ) ** 0.5

    return ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="docs/HEATMAP/morph_contrast.json")
    args = ap.parse_args()

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    ops = build_operators()
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in ops}
    ys: list[np.ndarray] = []

    for iid in split["val"][: args.n]:
        img = pc.read_image(images[iid]["file_name"])
        disk = pc.disk_mask(img)
        gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
        y = (pc.patch_fraction(gt) > 0.25).astype(np.uint8)

        # Same inner-disk restriction the classifier uses, so the numbers are
        # directly comparable to the feature AUCs measured earlier.
        valid = (
            cv2.erode(disk, ellipse(81))[::PATCH, ::PATCH][: pc.GRID, : pc.GRID] > 0
        )

        for name, fn in ops.items():
            m = fn(img)
            pooled[name].append(
                sliding_window_view(m, (PATCH, PATCH))[::PATCH, ::PATCH]
                .mean(axis=(-1, -2))[valid]
            )
        ys.append(y[valid])
        print(f"  {iid}", flush=True)

    Y = np.concatenate(ys)
    print(f"\n{len(Y):,} patches, positive fraction {Y.mean():.4f}\n")

    rows = []
    for name, vals in pooled.items():
        v = np.concatenate(vals)
        rows.append({"operator": name, "auc": auc(v, Y), "snr": snr(v, Y)})

    rows.sort(key=lambda r: -r["auc"])
    print(f"{'operator':>24}  {'AUC':>6}  {'SNR':>6}")
    print("-" * 42)
    for r in rows:
        print(f"{r['operator']:>24}  {r['auc']:>6.4f}  {r['snr']:>6.3f}")

    best = rows[0]
    print(f"\nbest by AUC: {best['operator']} (AUC={best['auc']:.4f}, SNR={best['snr']:.3f})")
    best_snr = max(rows, key=lambda r: r["snr"])
    print(f"best by SNR: {best_snr['operator']} (AUC={best_snr['auc']:.4f}, SNR={best_snr['snr']:.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
