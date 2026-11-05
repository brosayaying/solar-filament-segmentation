"""Inter-annotator agreement: what PQ do two humans score against each other?

    python scripts/annotator_agreement.py

The COCO file has 1154 image entries but only 707 unique files. Files annotated
more than once appear under different ``image_id``s with a numeric prefix
(``050101-…``, ``050102-…``, ``050103-…``), i.e. separate human attempts at the
same image.

This script rasterises every variant and scores each pair with the competition
metric, giving the human-vs-human reference line. Any model scoring near or above
it is at the practical ceiling of this metric, because the ground truth itself is
ambiguous at that level.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar.metrics import PQAccumulator  # noqa: E402


def rasterize(anns: list[dict], h: int = 2048, w: int = 2048) -> np.ndarray:
    lab = np.zeros((h, w), np.int32)
    for k, ann in enumerate(anns, 1):
        for poly in ann["segmentation"]:
            pts = np.round(np.asarray(poly, np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(lab, [pts], k)
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--ann-file", default="train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()

    coco = json.load(open(Path(args.root) / args.ann_file))
    anns_by_id = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns_by_id[a["image_id"]].append(a)

    files = collections.defaultdict(list)
    for img in coco["images"]:
        files[img["file_name"]].append(img["id"])

    print(f"{len(coco['images'])} image entries, {len(files)} unique files, "
          f"{len(coco['annotations'])} annotations")
    multi = {f: ids for f, ids in files.items() if len(ids) > 1}
    print(f"files with more than one annotation variant: {len(multi)}\n")

    # ---- all pairs -------------------------------------------------------- #
    rows = []
    for fname, ids in multi.items():
        rasters = [rasterize(anns_by_id[i], args.size, args.size) for i in ids]
        for a in range(len(rasters)):
            for b in range(a + 1, len(rasters)):
                acc = PQAccumulator(threshold=0.5)
                acc.update(rasters[a], rasters[b])
                s = acc.summary()
                rows.append((fname, s["pq"], s["sq"], s["rq"]))

    r = np.array([[x[1], x[2], x[3]] for x in rows])
    print(f"INTER-ANNOTATOR AGREEMENT ({len(rows)} pairs over {len(multi)} files)")
    print(f"  PQ: mean {r[:, 0].mean():.3f}  median {np.median(r[:, 0]):.3f}  "
          f"p10 {np.percentile(r[:, 0], 10):.3f}  p90 {np.percentile(r[:, 0], 90):.3f}")
    print(f"  SQ: mean {r[:, 1].mean():.3f}  median {np.median(r[:, 1]):.3f}  "
          f"p10 {np.percentile(r[:, 1], 10):.3f}  p90 {np.percentile(r[:, 1], 90):.3f}")
    print(f"  RQ: mean {r[:, 2].mean():.3f}  median {np.median(r[:, 2]):.3f}")

    # ---- restricted to the val split -------------------------------------- #
    split_path = Path(args.split)
    if split_path.exists():
        split = json.load(open(split_path))
        valset = set(split["val"])

        # How much of val is duplicated?
        n_val = len(split["val"])
        by_file = collections.Counter()
        img_meta = {img["id"]: img for img in coco["images"]}
        for i in split["val"]:
            by_file[img_meta[i]["file_name"]] += 1
        dup = sum(v for v in by_file.values() if v > 1)
        print(f"\nVAL SPLIT: {n_val} entries over {len(by_file)} unique files "
              f"({dup} entries, {100 * dup / n_val:.0f}%, are duplicates)")

        acc = PQAccumulator(threshold=0.5)
        pairs = 0
        for fname, ids in files.items():
            vin = [i for i in ids if i in valset]
            if len(vin) < 2:
                continue
            acc.update(
                rasterize(anns_by_id[vin[0]], args.size, args.size),
                rasterize(anns_by_id[vin[1]], args.size, args.size),
            )
            pairs += 1
        s = acc.summary()
        print(f"HUMAN BASELINE on val ({pairs} file-pairs): "
              f"PQ={s['pq']:.4f} SQ={s['sq']:.4f} RQ={s['rq']:.4f}")
        print("Compare any model score against this line, not against 1.0.")


if __name__ == "__main__":
    main()
