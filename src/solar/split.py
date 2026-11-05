"""Create and inspect the train/val split.

    python -m solar.split --group-by date --val-frac 0.2
    python -m solar.split --inspect
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import group_key, load_coco, make_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--ann-file", default="train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--group-by", default="date", choices=["date", "month", "year", "station", "file_name"])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    coco = load_coco(root / args.ann_file)

    if args.inspect:
        path = Path(args.out or f"splits/{args.group_by}_seed{args.seed}.json")
        with open(path) as fh:
            split = json.load(fh)
        print(f"split file: {path}")
        print(json.dumps(split["meta"], indent=2))
        return

    split = make_split(coco, args.group_by, args.val_frac, args.seed)

    out = Path(args.out or f"splits/{args.group_by}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(split, fh, indent=2)

    # Report leakage checks.
    images = {img["id"]: img for img in coco["images"]}
    ann_index: Counter[str] = Counter()
    for ann in coco["annotations"]:
        ann_index[ann["image_id"]] += 1

    def stats(ids: list[str]) -> dict:
        files = {images[i]["file_name"] for i in ids}
        dates = {group_key(images[i]["file_name"], "date") for i in ids}
        n_anns = sum(ann_index[i] for i in ids)
        return {
            "image_entries": len(ids),
            "unique_files": len(files),
            "unique_dates": len(dates),
            "annotations": n_anns,
            "anns_per_image": round(n_anns / max(1, len(ids)), 2),
        }

    tr, va = stats(split["train"]), stats(split["val"])
    tr_files = {images[i]["file_name"] for i in split["train"]}
    va_files = {images[i]["file_name"] for i in split["val"]}
    tr_dates = {group_key(images[i]["file_name"], "date") for i in split["train"]}
    va_dates = {group_key(images[i]["file_name"], "date") for i in split["val"]}

    print(f"wrote {out}")
    print(f"  train: {tr}")
    print(f"  val:   {va}")
    print(f"  file overlap: {len(tr_files & va_files)} (must be 0)")
    print(f"  date overlap: {len(tr_dates & va_dates)} (must be 0 when group_by=date)")


if __name__ == "__main__":
    main()
