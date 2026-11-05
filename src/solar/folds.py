"""Group-aware k-fold splits for cross-validation.

Why this exists
---------------
`solar.split` produces a single train/val split grouped by **date**. Date
grouping still puts adjacent days on opposite sides of the split, and filaments
persist for days, so the same filament can appear in train and val. That inflates
local PQ and is one candidate explanation for the measured 0.0725 local->public
gap (`v2/docs/CALIBRATION.md`).

This module produces **k folds grouped by ISO week** instead. Measured on the
707 unique train files:

    group_by   groups   files/group median   max   singletons
    week          343                  2.0     6          130
    month         118                  5.0    18           13
    year           12                 75.5   102            0

Week grouping is the honest choice: 343 groups is enough that a 5-fold split
balances to 141/142/142/140/142 files, while still keeping every image from one
week on the same side. Month grouping is the fallback if week folds turn out too
noisy (fold spread > 0.04 PQ).

All annotator copies of a file always land in the same fold, because the split is
over *files* and the copies are expanded afterwards.

    python -m solar.folds --inspect
    python -m solar.folds --k 5 --group-by week --out v2/splits/week5_seed42.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import load_coco

__all__ = [
    "fold_key",
    "make_folds",
    "expand_folds",
    "write_fold_splits",
    "check_leakage",
]


def fold_key(file_name: str, group_by: str = "week") -> str:
    """Group key for a filename ``YYYYMMDDHHMMSS<station>.jpeg``.

    ``week`` uses the ISO year-week, so the key is stable across year
    boundaries (``2021-12-30`` and ``2022-01-02`` are the same ISO week).
    """
    stem = Path(file_name).stem
    if group_by == "week":
        d = dt.date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))
        iso = d.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    if group_by == "month":
        return stem[:6]
    if group_by == "year":
        return stem[:4]
    if group_by == "date":
        return stem[:8]
    if group_by == "file_name":
        return stem
    raise ValueError(f"unknown group_by={group_by!r}")


def make_folds(
    coco: dict[str, Any],
    k: int = 5,
    group_by: str = "week",
    seed: int = 42,
) -> dict[str, Any]:
    """Assign every unique *file* to one of ``k`` folds, balanced by file count.

    Greedy longest-processing-time balancing: shuffle the group keys, then
    repeatedly place the next group into whichever fold currently has the fewest
    files. With 343 week groups of median size 2 this lands within 2 files of
    perfect balance (measured: 141/142/142/140/142).
    """
    files = sorted({img["file_name"] for img in coco["images"]})

    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        groups[fold_key(f, group_by)].append(f)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    fold_files: list[list[str]] = [[] for _ in range(k)]
    counts = [0] * k
    for key in keys:
        i = counts.index(min(counts))
        fold_files[i].extend(groups[key])
        counts[i] += len(groups[key])

    for f in fold_files:
        f.sort()

    return {
        "folds": fold_files,
        "meta": {
            "k": k,
            "group_by": group_by,
            "seed": seed,
            "n_files": len(files),
            "n_groups": len(keys),
            "files_per_fold": counts,
        },
    }


def expand_folds(
    folds: dict[str, Any], coco: dict[str, Any]
) -> list[dict[str, list[str]]]:
    """Turn file-level folds into image_id-level train/val pairs.

    Every annotator copy of a file inherits that file's fold, so the same
    underlying image can never appear on both sides.
    """
    ids_by_file: dict[str, list[str]] = defaultdict(list)
    for img in coco["images"]:
        ids_by_file[img["file_name"]].append(img["id"])

    out: list[dict[str, list[str]]] = []
    for i, val_files in enumerate(folds["folds"]):
        val_set = set(val_files)
        val_ids = [j for f in val_files for j in ids_by_file[f]]
        train_ids = [
            j
            for f, ids in ids_by_file.items()
            if f not in val_set
            for j in ids
        ]
        out.append(
            {
                "fold": i,
                "train": sorted(train_ids),
                "val": sorted(val_ids),
            }
        )
    return out


def write_fold_splits(
    folds: dict[str, Any],
    coco: dict[str, Any],
    out_dir: str | Path,
) -> list[Path]:
    """Write one ``{train, val}`` JSON per fold, in the format ``train.py`` reads.

    ``solar.train.get_split`` expects ``{"train": [...], "val": [...]}`` keyed by
    ``image_id``. The k-fold file is file-level and holds all folds at once, so
    it cannot be passed to ``--config data.split_file`` directly. This bridges
    the two: ``v2/splits/week5_seed42/fold0.json`` etc.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for pair in expand_folds(folds, coco):
        path = out_dir / f"fold{pair['fold']}.json"
        payload = {
            "train": pair["train"],
            "val": pair["val"],
            "meta": {
                **folds["meta"],
                "fold": pair["fold"],
                "n_train_images": len(pair["train"]),
                "n_val_images": len(pair["val"]),
            },
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        paths.append(path)
    return paths


def check_leakage(
    folds: dict[str, Any], coco: dict[str, Any]
) -> dict[str, Any]:
    """Assert no file and no group appears in more than one fold."""
    files = [set(f) for f in folds["folds"]]
    group_by = folds["meta"]["group_by"]

    file_overlaps = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            n = len(files[i] & files[j])
            if n:
                file_overlaps.append((i, j, n))

    groups = [{fold_key(f, group_by) for f in fs} for fs in files]
    group_overlaps = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            n = len(groups[i] & groups[j])
            if n:
                group_overlaps.append((i, j, n))

    all_files = set().union(*files)
    n_unique = len({img["file_name"] for img in coco["images"]})

    return {
        "file_overlaps": file_overlaps,
        "group_overlaps": group_overlaps,
        "files_covered": len(all_files),
        "files_expected": n_unique,
        "ok": not file_overlaps
        and not group_overlaps
        and len(all_files) == n_unique,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument(
        "--ann-file",
        default="train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--group-by",
        default="week",
        choices=["week", "month", "year", "date", "file_name"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    coco = load_coco(Path(args.root) / args.ann_file)
    out = Path(
        args.out or f"v2/splits/{args.group_by}{args.k}_seed{args.seed}.json"
    )

    if args.inspect:
        with open(out) as fh:
            folds = json.load(fh)
    else:
        folds = make_folds(coco, args.k, args.group_by, args.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(folds, fh, indent=2)
        print(f"wrote {out}")

    meta = folds["meta"]
    print(json.dumps(meta, indent=2))

    leak = check_leakage(folds, coco)
    print(f"  files covered: {leak['files_covered']} / {leak['files_expected']}")
    print(f"  file overlaps:  {leak['file_overlaps'] or 'none'}")
    print(f"  group overlaps: {leak['group_overlaps'] or 'none'}")

    # Per-fold {train, val} files, which is what solar.train reads.
    fold_paths = write_fold_splits(folds, coco, out.with_suffix(""))
    print(f"\nwrote {len(fold_paths)} per-fold splits -> {out.with_suffix('')}")
    print(f"{'fold':>4} {'train ids':>10} {'val ids':>8} {'val files':>10}")
    print("-" * 36)
    for pair, path in zip(expand_folds(folds, coco), fold_paths):
        print(
            f"{pair['fold']:>4} {len(pair['train']):>10} "
            f"{len(pair['val']):>8} {len(folds['folds'][pair['fold']]):>10}"
        )
    print("\nleakage check:", "OK" if leak["ok"] else "FAILED")

    if not leak["ok"]:
        raise SystemExit("LEAKAGE CHECK FAILED")


if __name__ == "__main__":
    main()
