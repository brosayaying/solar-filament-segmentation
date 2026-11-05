"""Measure normalisation statistics over the *disk only*, before and after
radial flattening.

    python scripts/disk_stats.py --n 200

Why this exists
---------------
`configs/base.json` uses ``mean=0.5, std=0.25``. Those numbers are wrong for
this data in two independent ways:

1. They are computed over the whole frame, but 27% of pixels are off-disk black
   (mean 3.2). That drags the mean down and inflates the variance.
2. Even restricted to the disk, the true std is ~0.11, not 0.25. Dividing by
   0.25 compresses the real signal into a narrow band and throws away ~2.3x of
   the dynamic range the network could be using.

This script reports the honest numbers so the normalisation can be set from
measurement rather than convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.preprocess import disk_geometry, radial_flatten  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of val files to scan")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--out", default="docs/PREPROCESSING/disk_stats.json")
    args = ap.parse_args()

    coco = json.loads(
        (ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text()
    )
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    # Dedup by file so multi-annotator duplicates do not weight some images more.
    seen: set[str] = set()
    ids: list[str] = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[: args.n]

    img_dir = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/train_images"

    # Accumulate sums so the pooled statistics are exact, not an average of
    # per-image statistics (which would be biased by disk area differences).
    acc = {
        "global": [0.0, 0.0, 0],
        "disk": [0.0, 0.0, 0],
        "flat_disk": [0.0, 0.0, 0],
    }
    per_image_disk_mean: list[float] = []
    per_image_disk_std: list[float] = []
    disk_frac: list[float] = []

    for iid in tqdm(ids, desc="stats"):
        img = cv2.imread(str(img_dir / images[iid]["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x = img.astype(np.float64) / 255.0

        cx, cy, r = disk_geometry(img)
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        disk = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0)) <= 1.0

        for key, vals in (("global", x.ravel()), ("disk", x[disk])):
            acc[key][0] += float(vals.sum())
            acc[key][1] += float((vals**2).sum())
            acc[key][2] += int(vals.size)

        per_image_disk_mean.append(float(x[disk].mean()))
        per_image_disk_std.append(float(x[disk].std()))
        disk_frac.append(float(disk.mean()))

        # After radial flattening the disk is rescaled to its own median, so the
        # statistics change; measure them rather than assuming.
        flat = radial_flatten(img.astype(np.float32), cx, cy, r) / 255.0
        fv = flat[disk]
        acc["flat_disk"][0] += float(fv.sum())
        acc["flat_disk"][1] += float((fv**2).sum())
        acc["flat_disk"][2] += int(fv.size)

    out: dict = {"n_files": len(ids), "regions": {}}
    for key, (s, s2, n) in acc.items():
        mean = s / n
        var = s2 / n - mean**2
        out["regions"][key] = {
            "n_pixels": n,
            "mean": mean,
            "std": float(np.sqrt(max(var, 0.0))),
        }

    out["per_image_disk"] = {
        "mean_of_means": float(np.mean(per_image_disk_mean)),
        "std_of_means": float(np.std(per_image_disk_mean)),
        "mean_of_stds": float(np.mean(per_image_disk_std)),
        "std_of_stds": float(np.std(per_image_disk_std)),
        "min_mean": float(np.min(per_image_disk_mean)),
        "max_mean": float(np.max(per_image_disk_mean)),
    }
    out["disk_area_fraction"] = {
        "mean": float(np.mean(disk_frac)),
        "min": float(np.min(disk_frac)),
        "max": float(np.max(disk_frac)),
    }

    print(f"\n=== normalisation statistics (n={len(ids)} files) ===")
    print(f"{'region':>12} {'mean':>8} {'std':>8} {'n_pixels':>14}")
    for key, v in out["regions"].items():
        print(f"{key:>12} {v['mean']:8.4f} {v['std']:8.4f} {v['n_pixels']:14d}")

    print(f"\ndisk area fraction: {out['disk_area_fraction']['mean']:.3f} "
          f"(range {out['disk_area_fraction']['min']:.3f}-{out['disk_area_fraction']['max']:.3f})")
    pi = out["per_image_disk"]
    print(f"per-image disk mean: {pi['mean_of_means']:.4f} +/- {pi['std_of_means']:.4f} "
          f"(range {pi['min_mean']:.3f}-{pi['max_mean']:.3f})")
    print(f"per-image disk std:  {pi['mean_of_stds']:.4f} +/- {pi['std_of_stds']:.4f}")

    print("\n=== comparison with configs/base.json (mean=0.5, std=0.25) ===")
    d = out["regions"]["disk"]
    print(f"  disk mean {d['mean']:.4f} vs 0.5   -> offset {d['mean'] - 0.5:+.4f}")
    print(f"  disk std  {d['std']:.4f} vs 0.25  -> ratio  {0.25 / d['std']:.2f}x too large")

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
