"""Dataset-wide intensity statistics.

Computes mean/variance of the H-alpha training images, both globally and
conditioned on the annotated filament mask. Used to set the normalisation
constants in ``configs/*.json`` (``data.mean`` / ``data.std``) and to size the
residual thresholds in the classical pipeline.

Statistics are accumulated from exact per-image sums, so the pooled mean and
variance are exact rather than an average of per-image means.

    python scripts/img_stats.py --n all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar.data import anns_to_mask, build_ann_index, load_coco  # noqa: E402

ROOT = Path("MAGFiLO_1.0_Kaggle_2026")
cv2.setNumThreads(4)


def disk_mask(img: np.ndarray, thr: int = 25) -> np.ndarray:
    """Binary solar-disk mask: largest bright blob, holes filled."""
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


class Moments:
    """Exact pooled mean/variance from running count, sum and sum of squares."""

    def __init__(self) -> None:
        self.n = 0
        self.s = 0.0
        self.sq = 0.0

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64).ravel()
        self.n += x.size
        self.s += x.sum()
        self.sq += np.square(x).sum()

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else 0.0

    @property
    def var(self) -> float:
        if not self.n:
            return 0.0
        return max(0.0, self.sq / self.n - self.mean**2)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def summary(self) -> dict[str, float]:
        return {"n": self.n, "mean": self.mean, "var": self.var, "std": self.std}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="all", help="number of image files, or 'all'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    coco = load_coco(ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}

    # One image entry per unique file: annotator duplicates share the pixels,
    # so including them would only reweight the dates that were double-annotated.
    files = sorted({img["file_name"] for img in coco["images"]})
    if args.n != "all":
        files = files[: int(args.n)]

    # Map file_name -> all image ids, to find annotations for the unique file.
    by_fn: dict[str, list[str]] = {}
    for iid, meta in images.items():
        by_fn.setdefault(meta["file_name"], []).append(iid)

    all_px = Moments()
    fg_px = Moments()
    bg_px = Moments()
    disk_px = Moments()
    disk_bg_px = Moments()
    rim_px = Moments()
    fg_res = Moments()
    disk_bg_res = Moments()
    per_image: list[dict] = []
    per_image_contrast: list[float] = []

    for k, fn in enumerate(files):
        img = cv2.imread(str(ROOT / "train/train_images" / fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(fn)
        h, w = img.shape

        # Union of every annotator's mask for this file.
        mask = np.zeros((h, w), np.uint8)
        for iid in by_fn[fn]:
            mask |= anns_to_mask(ann_index.get(iid, []), h, w)

        disk = disk_mask(img)
        # The limb ring is ~40 px of steeply falling intensity; it is neither
        # filament nor representative on-disk background, so it is reported
        # separately rather than pooled.
        inner = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
        rim = (disk > 0) & (inner == 0)

        # Filaments are dark relative to their *local* surroundings, not to the
        # disk average: the center-to-limb intensity falloff (~139 -> ~95)
        # dominates the raw image. A large-sigma blur removes that trend.
        f = img.astype(np.float32)
        res = cv2.GaussianBlur(f, (0, 0), 64) - f  # positive on dark features

        all_px.update(img)
        if mask.any():
            fg_px.update(img[mask > 0])
            fg_res.update(res[mask > 0])
        # "Background" = off-disk, i.e. the black surround. Kept for reference.
        offdisk = disk == 0
        if offdisk.any():
            bg_px.update(img[offdisk])
        if inner.any():
            disk_px.update(img[inner > 0])
        ondisk_bg = (inner > 0) & (mask == 0)
        if ondisk_bg.any():
            disk_bg_px.update(img[ondisk_bg])
            disk_bg_res.update(res[ondisk_bg])
        if rim.any():
            rim_px.update(img[rim])

        if mask.any() and ondisk_bg.any():
            per_image_contrast.append(float(res[mask > 0].mean() - res[ondisk_bg].mean()))

        per_image.append(
            {
                "file_name": fn,
                "mean": float(img.mean()),
                "std": float(img.std()),
                "disk_mean": float(img[inner > 0].mean()) if inner.any() else float("nan"),
                "disk_std": float(img[inner > 0].std()) if inner.any() else float("nan"),
                "fg_frac": float((mask > 0).mean()),
            }
        )
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(files)}", flush=True)

    result = {
        "n_files": len(files),
        "all_pixels": all_px.summary(),
        "filament_pixels": fg_px.summary(),
        "offdisk_pixels": bg_px.summary(),
        "disk_pixels": disk_px.summary(),
        "disk_background_pixels": disk_bg_px.summary(),
        "limb_ring_pixels": rim_px.summary(),
        "filament_local_contrast": fg_res.summary(),
        "disk_background_local_contrast": disk_bg_res.summary(),
        "per_image_fg_minus_bg_contrast": {
            "median": float(np.median(per_image_contrast)) if per_image_contrast else 0.0,
            "p05": float(np.percentile(per_image_contrast, 5)) if per_image_contrast else 0.0,
            "p95": float(np.percentile(per_image_contrast, 95)) if per_image_contrast else 0.0,
        },
        "per_image_mean": {
            "median": float(np.median([p["mean"] for p in per_image])),
            "p05": float(np.percentile([p["mean"] for p in per_image], 5)),
            "p95": float(np.percentile([p["mean"] for p in per_image], 95)),
        },
        "per_image_std": {
            "median": float(np.median([p["std"] for p in per_image])),
            "p05": float(np.percentile([p["std"] for p in per_image], 5)),
            "p95": float(np.percentile([p["std"] for p in per_image], 95)),
        },
        "per_image_disk_mean": {
            "median": float(np.nanmedian([p["disk_mean"] for p in per_image])),
            "p05": float(np.nanpercentile([p["disk_mean"] for p in per_image], 5)),
            "p95": float(np.nanpercentile([p["disk_mean"] for p in per_image], 95)),
        },
        "per_image_disk_std": {
            "median": float(np.nanmedian([p["disk_std"] for p in per_image])),
            "p05": float(np.nanpercentile([p["disk_std"] for p in per_image], 5)),
            "p95": float(np.nanpercentile([p["disk_std"] for p in per_image], 95)),
        },
        "fg_frac_median": float(np.median([p["fg_frac"] for p in per_image])),
    }

    print(f"\n{len(files)} unique training files\n")
    for key in (
        "all_pixels",
        "filament_pixels",
        "offdisk_pixels",
        "disk_pixels",
        "disk_background_pixels",
        "limb_ring_pixels",
        "filament_local_contrast",
        "disk_background_local_contrast",
    ):
        s = result[key]
        print(f"{key:>32}: mean={s['mean']:8.3f}  var={s['var']:9.2f}  std={s['std']:7.3f}  n={s['n']:,}")
    c = result["per_image_fg_minus_bg_contrast"]
    print(f"\nfilament-vs-bg local contrast: median={c['median']:.3f} [{c['p05']:.3f}, {c['p95']:.3f}]")
    print(
        f"\nper-image mean: median={result['per_image_mean']['median']:.2f} "
        f"[{result['per_image_mean']['p05']:.2f}, {result['per_image_mean']['p95']:.2f}]"
    )
    print(
        f"per-image std : median={result['per_image_std']['median']:.2f} "
        f"[{result['per_image_std']['p05']:.2f}, {result['per_image_std']['p95']:.2f}]"
    )
    print(
        f"per-image disk mean: median={result['per_image_disk_mean']['median']:.2f} "
        f"[{result['per_image_disk_mean']['p05']:.2f}, {result['per_image_disk_mean']['p95']:.2f}]"
    )
    print(
        f"per-image disk std : median={result['per_image_disk_std']['median']:.2f} "
        f"[{result['per_image_disk_std']['p05']:.2f}, {result['per_image_disk_std']['p95']:.2f}]"
    )
    print(f"filament pixel fraction: median={result['fg_frac_median']:.5f}")

    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
