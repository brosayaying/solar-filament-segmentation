"""Phase 3.1: merge original + label-transferred data into one training set.

The transferred frames are extra *training* data. The validation split is
unchanged, so OOF PQ stays honest — but only if the merge respects one rule:

**A transferred frame derived from a validation file must not enter training.**
The transferred frame is a different observation of the same region minutes
later, so it carries essentially the same filaments. Training on it while
validating on the original would leak.

This script therefore takes the fold's ``train``/``val`` file lists and admits a
transferred frame only if its ``source_stem`` is in ``train``.

Output is a COCO file whose ``images`` are the union, plus a split file in the
same format ``train.py`` already reads, so no training code changes are needed.

    python v2/scripts/build_phase3_dataset.py --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ORIG = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
LT = ROOT / "v2/data/label_transfer"
SPLITS = ROOT / "v2/splits/week5_seed42"
OUT = ROOT / "v2/data/phase3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    orig = json.load(open(ORIG))
    lt = json.load(open(LT / "annotations.json"))
    split = json.load(open(SPLITS / f"fold{args.fold}.json"))

    train_ids = set(split["train"])
    val_ids = set(split["val"])

    # Map image_id -> file stem for the original data.
    stem_of = {im["id"]: Path(im["file_name"]).stem for im in orig["images"]}
    train_stems = {stem_of[i] for i in train_ids if i in stem_of}
    val_stems = {stem_of[i] for i in val_ids if i in stem_of}
    print(f"fold {args.fold}: {len(train_stems)} train files, "
          f"{len(val_stems)} val files")

    # Admit a transferred frame only if its source is a training file.
    admitted, rejected = [], 0
    for im in lt["images"]:
        src = im["source_stem"]
        if src in train_stems:
            admitted.append(im)
        elif src in val_stems:
            rejected += 1
    print(f"transferred frames: {len(admitted)} admitted, "
          f"{rejected} rejected (source is a val file)")

    admitted_ids = {im["id"] for im in admitted}
    lt_anns = [a for a in lt["annotations"] if a["image_id"] in admitted_ids]

    # Union COCO. Original images keep their ids; transferred ones are prefixed
    # "lt-" and cannot collide.
    images = list(orig["images"]) + admitted
    anns = list(orig["annotations"]) + lt_anns
    coco = {
        "info": {"description": "Phase 3.1 original + GONG label transfer"},
        "licenses": orig.get("licenses", []),
        "categories": orig["categories"],
        "images": images,
        "annotations": anns,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_dir / "annotations.json"
    ann_path.write_text(json.dumps(coco))
    print(f"\nwrote {ann_path}")
    print(f"  {len(images)} images ({len(orig['images'])} original + "
          f"{len(admitted)} transferred)")
    print(f"  {len(anns)} annotations")

    # Split file: train = original train ids + admitted transferred ids.
    new_split = {
        "train": sorted(train_ids) + sorted(admitted_ids),
        "val": sorted(val_ids),
        "meta": {
            **split.get("meta", {}),
            "phase3": True,
            "n_transferred": len(admitted),
            "n_rejected_val_leak": rejected,
        },
    }
    split_path = out_dir / f"fold{args.fold}.json"
    split_path.write_text(json.dumps(new_split, indent=2))
    print(f"wrote {split_path}")
    print(f"  train {len(new_split['train'])} "
          f"({len(train_ids)} original + {len(admitted_ids)} transferred)")
    print(f"  val   {len(new_split['val'])} (unchanged)")

    # Leakage assertion: no transferred frame may share a source with val.
    bad = [im["id"] for im in admitted if im["source_stem"] in val_stems]
    assert not bad, f"LEAK: {len(bad)} transferred frames from val sources"
    print("\nleakage check: PASS (no transferred frame derives from a val file)")


if __name__ == "__main__":
    main()
