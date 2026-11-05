"""Phase 3.1: build a label-transfer training set from the GONG archive.

For each of the 707 training files, download the same-station GONG frame at
±2/5/10 minutes and reuse the annotation unchanged. Solar rotation over 10 min is
~1.6 px at disk centre (see ``solar/gong.py``), and the frames are verified
aligned to zero shift, so the labels transfer without realignment.

**Why JPEGs and not FITS.** The competition images *are* the GONG ``hag`` JPEGs:
median absolute difference 1, correlation 0.997 at zero shift. The ``haf``
``.fits.fz`` product is a different reduction (correlation 0.86 after a best-fit
stretch), so using it would inject a domain shift the JPEG route avoids.

**Why this is not leakage.** The transferred frames are different observations of
the same region at a different time. They are additional *training* data; the
validation split is unchanged, so OOF PQ remains honest. What is *not* allowed is
transferring a label onto a frame within ±1 day of a test timestamp, which is
excluded here.

Output is a COCO-shaped annotation file plus a directory of JPEGs, so the existing
``FilamentDataset`` can consume it with no changes.

    python v2/scripts/build_label_transfer.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.gong import download_frame, load_frame, nearby_timestamps, parse_stem  # noqa: E402

ANN = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT = ROOT / "v2/data/label_transfer"
TEST_DIR = ROOT / "MAGFiLO_1.0_Kaggle_2026/test/test_images"


def test_dates() -> list[datetime]:
    """Timestamps of the test images, for the ±1 day exclusion."""
    out = []
    for p in TEST_DIR.iterdir():
        if p.suffix.lower() not in (".jpeg", ".jpg", ".png"):
            continue
        try:
            out.append(parse_stem(p.stem)[0])
        except ValueError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", type=int, nargs="*", default=[-10, -5, -2, 2, 5, 10])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    coco = json.load(open(ANN))
    images = coco["images"]
    anns = coco["annotations"]

    # Group annotations by image_id so a transferred frame inherits all of them.
    by_image: dict[str, list[dict]] = {}
    for a in anns:
        by_image.setdefault(a["image_id"], []).append(a)

    # One entry per unique file, keeping every annotator copy's annotations.
    files: dict[str, list[dict]] = {}
    for im in images:
        stem = Path(im["file_name"]).stem
        files.setdefault(stem, []).append(im)

    if args.limit:
        files = dict(list(files.items())[: args.limit])
    print(f"{len(files)} unique training files, {len(args.offsets)} offsets each")

    tdates = test_dates()
    print(f"{len(tdates)} test timestamps for the ±1 day exclusion")

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Build the download list, applying the test-proximity exclusion.
    jobs: list[tuple[str, str, int]] = []  # (new_stem, source_stem, offset)
    excluded = 0
    for stem in files:
        try:
            dt, _ = parse_stem(stem)
        except ValueError:
            continue
        for new_stem, off in nearby_timestamps(stem, tuple(args.offsets)):
            ndt = dt + timedelta(minutes=off)
            if any(abs((ndt - t).total_seconds()) < 86400 for t in tdates):
                excluded += 1
                continue
            jobs.append((new_stem, stem, off))

    print(f"{len(jobs)} frames to fetch, {excluded} excluded for test proximity")

    ok, missing = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_frame, s, img_dir): (s, src, off)
                for s, src, off in jobs}
        for i, f in enumerate(as_completed(futs)):
            s, src, off = futs[f]
            p = f.result()
            if p is None:
                missing.append((s, src, off))
            else:
                ok.append((s, src, off))
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(jobs)}  ok {len(ok)} missing {len(missing)}  "
                      f"{el:.0f}s  eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s",
                      flush=True)

    print(f"\ndownloaded {len(ok)}, missing {len(missing)} "
          f"({len(missing) / max(1, len(jobs)):.1%})")

    # Verify every downloaded frame actually decodes at 2048^2.
    good = []
    for s, src, off in ok:
        img = load_frame(img_dir / f"{s}.jpg")
        if img is not None:
            good.append((s, src, off))
    print(f"{len(good)} frames decode at 2048^2")

    # Build the COCO file. Each transferred frame gets a new image_id and copies
    # every annotation of its source file, with fresh annotation ids.
    new_images, new_anns = [], []
    next_ann = 1
    for s, src, off in good:
        src_images = files[src]
        # Use the first annotator copy's id as the template; all copies of a file
        # share the same underlying image, and we want every annotation.
        new_id = f"lt-{s}"
        new_images.append({
            "license": 1,
            "file_name": f"{s}.jpeg",
            "height": 2048,
            "width": 2048,
            "date_captured": datetime.strptime(s[:14], "%Y%m%d%H%M%S").isoformat(),
            "id": new_id,
            "source_stem": src,
            "offset_min": off,
        })
        for im in src_images:
            for a in by_image.get(im["id"], []):
                b = dict(a)
                b["id"] = f"lt{next_ann}"
                b["image_id"] = new_id
                next_ann += 1
                new_anns.append(b)

    out_coco = {
        "info": {"description": "GONG label transfer, Phase 3.1"},
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_anns,
    }
    ann_path = out_dir / "annotations.json"
    ann_path.write_text(json.dumps(out_coco))
    print(f"\nwrote {ann_path}")
    print(f"  {len(new_images)} images, {len(new_anns)} annotations")

    (out_dir / "meta.json").write_text(json.dumps({
        "offsets": args.offsets,
        "n_source_files": len(files),
        "n_frames": len(new_images),
        "n_annotations": len(new_anns),
        "n_missing": len(missing),
        "n_excluded_test_proximity": excluded,
        "missing_sample": [m[0] for m in missing[:20]],
    }, indent=2))


if __name__ == "__main__":
    main()
