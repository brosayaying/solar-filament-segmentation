"""Precompute preprocessed images to disk so training does not pay for them.

    python scripts/build_preprocess_cache.py --preprocess flatten

Radial flattening costs ~0.5 s per 2048^2 image. With 565 train files and 160
epochs that is ~12 hours of pure preprocessing, all of it recomputing the same
deterministic function of the same file. This script pays it once.

The cache is keyed by ``(file_name, preprocess spec)`` and stores float16, which
is lossless enough for 8-bit source data (the flattening output is a smooth
rescale of the input, so float16's 11-bit mantissa comfortably covers the 8-bit
input range) and halves the disk cost.

Layout::

    <cache_dir>/<spec>/<file_stem>.npy     float16, full resolution
    <cache_dir>/<spec>/<file_stem>.disk.npy  bool disk mask

``FilamentDataset`` picks these up automatically when ``data.preprocess_cache``
is set; it falls back to computing on the fly if a file is missing.
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

from solar.data import parse_preprocess, preprocess_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", default="flatten", help="e.g. flatten, flatten+blur+clahe")
    ap.add_argument("--out", default=None, help="default: cache/preprocess/<spec>")
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--splits", nargs="*", default=["train", "test"], help="which image dirs")
    args = ap.parse_args()

    steps = parse_preprocess(args.preprocess)
    if not steps:
        raise SystemExit("--preprocess is empty; nothing to cache")

    spec = "+".join(sorted(steps))
    out = Path(args.out) if args.out else ROOT / "cache/preprocess" / spec
    out.mkdir(parents=True, exist_ok=True)

    root = ROOT / args.root
    dirs: list[Path] = []
    if "train" in args.splits:
        dirs.append(root / "train/train_images")
    if "test" in args.splits:
        dirs.append(root / "test/test_images")

    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png")))

    print(f"preprocess={spec}  ->  {out}")
    print(f"{len(files)} images from {[str(d.relative_to(ROOT)) for d in dirs if d.exists()]}")

    done = skipped = 0
    for path in tqdm(files, desc="cache"):
        img_path = out / f"{path.stem}.npy"
        disk_path = out / f"{path.stem}.disk.npy"
        if img_path.exists() and disk_path.exists():
            skipped += 1
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: could not read {path}")
            continue
        pp, disk = preprocess_image(img, steps)
        np.save(img_path, pp.astype(np.float16))
        np.save(disk_path, disk)
        done += 1

    meta = {
        "preprocess": spec,
        "n_images": len(files),
        "n_computed": done,
        "n_skipped": skipped,
        "dtype": "float16",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"computed {done}, skipped {skipped} (already cached) -> {out}")


if __name__ == "__main__":
    main()
