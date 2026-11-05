"""Out-of-fold probability dumps with the metadata Phase 2 needs.

Why this exists
---------------
`solar.tune.build_cache` writes ``NNNN_prob.npy`` / ``NNNN_label.npy`` with **no
filename mapping**. That is enough to sweep a threshold, but it makes it
impossible to join a predicted candidate back to the annotations that produced
it — and Phase 2 (fragment merging + per-instance ``q̂``) is built entirely on
that join. A candidate's label is "the fraction of annotator copies in which
this blob has IoU > 0.5", which requires knowing *which file* and *which copies*.

This module writes one ``.npz`` per unique file, containing:

    prob      float16 (2048, 2048)   ensemble-averaged probability
    labels    int32   (k, 2048, 2048) one instance labelmap per annotator copy
    spines    uint8   (k, 2048, 2048) one centreline raster per annotator copy
    meta      json string            file_name, image_id, annotators, fold, ...

``k`` is the number of annotator copies of that file (1-3). The probability map
is stored once because every copy is the same pixels.

Layout
------
    v2/runs/<exp>/oof/fold<f>/<file_stem>.npz

Every train file appears in exactly one fold directory — the fold where it was
in validation — so the union over folds is a complete OOF set.

    python -m solar.oof --ckpt v2/runs/seg_v2/fold0/best.pt --fold 0
    python -m solar.oof --verify v2/runs/seg_v2/oof
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    FilamentDataset,
    anns_to_labelmap,
    anns_to_spine_mask,
    collate_val,
    load_coco,
)
from .models import build_model
from .predict import predict_prob

__all__ = ["dump_fold", "load_oof", "verify_oof"]


def _unique_files(coco: dict[str, Any], image_ids: list[str]) -> dict[str, list[str]]:
    """Map file stem -> all image_ids (annotator copies) for those ids."""
    images = {img["id"]: img for img in coco["images"]}
    out: dict[str, list[str]] = defaultdict(list)
    for iid in image_ids:
        out[Path(images[iid]["file_name"]).stem].append(iid)
    return {k: sorted(v) for k, v in out.items()}


def _fold_val_ids(coco: dict[str, Any], folds: dict[str, Any], fold: int) -> list[str]:
    """Expand a file-level fold into the image_ids of its annotator copies.

    ``solar.folds`` stores *file names* in ``folds["folds"]``; the dataset and
    the annotation index are keyed by ``image_id`` (``<annotator>-<stem>``), so
    the two must be bridged here.
    """
    from .folds import expand_folds

    return expand_folds(folds, coco)[fold]["val"]


def check_ckpt_was_not_trained_on(
    ckpt: dict[str, Any], val_files: set[str]
) -> dict[str, Any]:
    """Refuse to dump a checkpoint on files it was trained on.

    This is the guard for a mistake that was actually made: dumping
    ``runs/norm_disk_raw/best.pt`` over the week-5 folds put **80 % of the
    "OOF" dumps in-sample**, because that checkpoint was trained on 565 of the
    707 files under the old date split. Those dumps are worthless for Phase 2,
    which trains a candidate scorer on them.

    A checkpoint records the split it was trained with in ``ckpt["cfg"]``. If
    that split file still exists, its train side is the set of files the model
    has seen. Any overlap with the fold's validation files is leakage.
    """
    cfg = ckpt.get("cfg", {})
    split_file = cfg.get("data", {}).get("split_file")
    if not split_file or not Path(split_file).exists():
        return {"checked": False, "reason": f"split file {split_file!r} not found"}

    with open(split_file) as fh:
        trained_split = json.load(fh)

    root = Path(cfg["data"]["root"])
    coco = load_coco(root / cfg["data"]["ann_file"])
    images = {img["id"]: img for img in coco["images"]}

    trained_files = {
        Path(images[i]["file_name"]).stem
        for i in trained_split.get("train", [])
        if i in images
    }
    overlap = sorted(trained_files & val_files)

    return {
        "checked": True,
        "split_file": split_file,
        "n_trained_files": len(trained_files),
        "n_val_files": len(val_files),
        "n_overlap": len(overlap),
        "overlap_frac": len(overlap) / max(1, len(val_files)),
        "examples": overlap[:5],
        "ok": not overlap,
    }


@torch.no_grad()
def dump_fold(
    ckpt_path: str | Path,
    fold: int,
    split_path: str | Path,
    out_dir: str | Path,
    device: str = "cuda",
    tta: bool = True,
    num_workers: int = 6,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run inference on one fold's validation files and write the OOF dumps."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    coco = load_coco(Path(cfg["data"]["root"]) / cfg["data"]["ann_file"])
    folds = json.load(open(split_path))
    val_ids = _fold_val_ids(coco, folds, fold)

    out_dir = Path(out_dir) / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _unique_files(coco, val_ids)

    # Refuse to dump a checkpoint on files it was trained on. Without this the
    # dumps look fine (right shape, right fold, right labels) but are in-sample,
    # and Phase 2 would train its candidate scorer on leaked probabilities.
    guard = check_ckpt_was_not_trained_on(ckpt, set(files))
    if guard.get("checked") and not guard["ok"]:
        raise SystemExit(
            f"REFUSING to dump fold {fold}: {guard['n_overlap']}/"
            f"{guard['n_val_files']} val files "
            f"({100 * guard['overlap_frac']:.0f}%) were in this checkpoint's "
            f"training set ({guard['split_file']}).\n"
            f"  examples: {guard['examples']}\n"
            f"  A checkpoint must be trained on the *other* folds to produce "
            f"honest OOF probabilities. Train per-fold first:\n"
            f"    python -m solar.train --config <cfg> "
            f"--set data.split_file=v2/splits/week5_seed42/fold{fold}.json"
        )

    todo = [
        s for s in files if overwrite or not (out_dir / f"{s}.npz").exists()
    ]
    if not todo:
        print(f"fold {fold}: all {len(files)} files already dumped")
        return {"fold": fold, "n_files": len(files), "n_written": 0}

    # One dataset entry per *file* (first annotator copy) so the model runs once
    # per image rather than once per copy.
    first_ids = [files[s][0] for s in todo]
    ds = FilamentDataset(
        cfg["data"]["root"],
        coco,
        first_ids,
        mode="val",
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        preprocess=cfg["data"].get("preprocess"),
        mask_offdisk=cfg["data"].get("mask_offdisk", False),
        offdisk_value=cfg["data"].get("offdisk_value", -3.0),
        preprocess_cache=cfg["data"].get("preprocess_cache"),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ann_index = defaultdict(list)
    for ann in coco["annotations"]:
        ann_index[ann["image_id"]].append(ann)

    written = 0
    for batch in tqdm(loader, desc=f"fold{fold} oof"):
        image = batch["image"][0].to(device)
        stem = Path(batch["file_name"][0]).stem

        prob = predict_prob(model, image, tta=tta)

        labels, spines = [], []
        for iid in files[stem]:
            anns = ann_index.get(iid, [])
            labels.append(anns_to_labelmap(anns, 2048, 2048))
            spines.append(anns_to_spine_mask(anns, 2048, 2048))

        meta = {
            "file_name": batch["file_name"][0],
            "stem": stem,
            "image_ids": files[stem],
            "annotators": [i.split("-", 1)[0] for i in files[stem]],
            "n_copies": len(files[stem]),
            "fold": fold,
            "tta": tta,
            "ckpt": str(ckpt_path),
            "threshold_note": "prob is a raw sigmoid; thresholding is downstream",
        }

        np.savez_compressed(
            out_dir / f"{stem}.npz",
            prob=prob.astype(np.float16),
            labels=np.stack(labels).astype(np.int32),
            spines=np.stack(spines).astype(np.uint8),
            meta=np.array(json.dumps(meta)),
        )
        written += 1

    print(f"fold {fold}: wrote {written} files -> {out_dir}")
    return {"fold": fold, "n_files": len(files), "n_written": written}


def load_oof(oof_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load every dumped file, keyed by stem."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(Path(oof_dir).glob("fold*/*.npz")):
        with np.load(p, allow_pickle=False) as z:
            out[p.stem] = {
                "prob": z["prob"].astype(np.float32),
                "labels": z["labels"],
                "spines": z["spines"],
                "meta": json.loads(str(z["meta"])),
                "path": p,
            }
    return out


def verify_oof(
    oof_dir: str | Path,
    split_path: str | Path = "v2/splits/week5_seed42.json",
    root: str = "MAGFiLO_1.0_Kaggle_2026",
    ann_file: str = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    strict: bool = True,
) -> dict[str, Any]:
    """Check that every train file appears exactly once, in the right fold.

    ``strict=False`` allows a partial dump (e.g. only fold 0 written so far):
    missing files are reported but do not fail the check. Everything that *is*
    present must still be in the right fold with the right labels.
    """
    coco = load_coco(Path(root) / ann_file)
    folds = json.load(open(split_path))

    expected: dict[str, int] = {}
    for f, files in enumerate(folds["folds"]):
        for s in files:
            # ``folds["folds"]`` holds file names (``...jpeg``); the dumps are
            # keyed by stem. Normalise both to the stem.
            expected[Path(s).stem] = f

    found: dict[str, list[int]] = defaultdict(list)
    for p in sorted(Path(oof_dir).glob("fold*/*.npz")):
        fold = int(p.parent.name.replace("fold", ""))
        found[p.stem].append(fold)

    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    dupes = {s: f for s, f in found.items() if len(f) > 1}
    wrong_fold = {
        s: (expected[s], found[s][0])
        for s in found
        if s in expected and found[s][0] != expected[s]
    }

    # Spot-check that the stored labels match a fresh rasterisation.
    bad_labels = []
    ann_index = defaultdict(list)
    for ann in coco["annotations"]:
        ann_index[ann["image_id"]].append(ann)
    for s in list(found)[:20]:
        p = Path(oof_dir) / f"fold{found[s][0]}" / f"{s}.npz"
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            labels = z["labels"]
        for i, iid in enumerate(meta["image_ids"]):
            fresh = anns_to_labelmap(ann_index.get(iid, []), 2048, 2048)
            if not np.array_equal(labels[i], fresh):
                bad_labels.append((s, iid))

    return {
        "n_expected": len(expected),
        "n_found": len(found),
        "missing": missing,
        "extra": extra,
        "duplicates": dupes,
        "wrong_fold": wrong_fold,
        "bad_labels": bad_labels,
        "ok": (not missing or not strict)
        and not extra
        and not dupes
        and not wrong_fold
        and not bad_labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", default="v2/splits/week5_seed42.json")
    ap.add_argument("--out", default=None, help="default: <ckpt>/../../oof")
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", default=None, help="verify an existing oof dir")
    ap.add_argument(
        "--partial",
        action="store_true",
        help="with --verify: allow missing files (a partially written dump)",
    )
    args = ap.parse_args()

    if args.verify:
        res = verify_oof(args.verify, args.split, strict=not args.partial)
        print(json.dumps({k: v for k, v in res.items() if k != "ok"}, indent=2))
        print("OK" if res["ok"] else "FAILED")
        if not res["ok"]:
            raise SystemExit(1)
        return

    if not args.ckpt or args.fold is None:
        raise SystemExit("pass --ckpt and --fold, or --verify <dir>")

    out = args.out or str(Path(args.ckpt).parent.parent / "oof")
    dump_fold(
        args.ckpt,
        args.fold,
        args.split,
        out,
        tta=args.tta,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
