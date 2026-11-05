"""Compare frozen-backbone token embeddings (DINOv2 vs SAM) on N val images.

    python scripts/compare_embeddings.py --n 20 --out docs/embed_compare

For each image and each backbone we measure four things that matter for a
two-stage detect-then-segment pipeline:

1. ``auc``      -- 5-fold CV AUC of a linear probe on L2-normalised tokens.
                   Can a *linear* read-out find filaments at all?
2. ``knn_fg``   -- for each filament token, does its nearest neighbour also lie
                   on a filament? Chance is the filament prevalence, so this
                   says whether the representation clusters filaments together.
3. ``recall``   -- fraction of true filament pixels covered by filament-labelled
                   tokens (each token treated as a patch-sized square). This is
                   the hard ceiling on any downstream segmentation head.
4. ``precision``-- fraction of positive-token area that is truly filament.

Recall/precision are the decisive numbers: a backbone can have a great AUC and
still be useless if its tokens are too coarse to resolve a 30 px wide filament.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar.embed import (  # noqa: E402
    build_adapter,
    knn_filament_agreement,
    linear_auc,
    load_gray,
    rasterize,
    token_coverage,
    token_labels,
)


def load_coco(root: Path, ann_file: str):
    with open(root / ann_file) as fh:
        coco = json.load(fh)
    images = {img["id"]: img for img in coco["images"]}
    ann_index: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        ann_index.setdefault(ann["image_id"], []).append(ann)
    return images, ann_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of val images")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--ann-file", default="train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--backbones", default="dinov2,sam")
    ap.add_argument("--out", default="docs/embed_compare")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    images, ann_index = load_coco(root, args.ann_file)
    with open(args.split) as fh:
        split = json.load(fh)

    # Deterministic, spread-out sample of the val split.
    rng = np.random.default_rng(args.seed)
    val_ids = list(split["val"])
    pick = rng.choice(len(val_ids), size=min(args.n, len(val_ids)), replace=False)
    chosen = [val_ids[i] for i in sorted(pick)]

    kinds = [k.strip() for k in args.backbones.split(",") if k.strip()]
    adapters = {k: build_adapter(k) for k in kinds}
    for k, a in adapters.items():
        print(f"{k}: patch={a.patch} dim={a.dim}")

    rows = []
    for n, image_id in enumerate(chosen, 1):
        meta = images[image_id]
        gray = load_gray(root / "train/train_images" / meta["file_name"])
        anns = ann_index.get(image_id, [])
        h, w = gray.shape
        mask = rasterize(anns, h, w)
        area = int(mask.sum())

        for kind, adapter in adapters.items():
            t0 = time.time()
            emb = adapter.embed(gray)
            dt = time.time() - t0
            labels = token_labels(emb, mask)
            n_pos = int(labels.sum())

            if n_pos == 0 or n_pos == len(labels):
                auc, auc_std = float("nan"), float("nan")
                knn_all, knn_fg = float("nan"), float("nan")
                rec, prec = float("nan"), float("nan")
            else:
                auc, auc_std = linear_auc(emb.tokens, labels, seed=args.seed)
                knn_all, knn_fg = knn_filament_agreement(emb.tokens, labels)
                rec, prec = token_coverage(emb, labels, mask)

            rows.append(
                dict(
                    image=meta["file_name"],
                    backbone=kind,
                    n_tokens=len(emb),
                    n_pos=n_pos,
                    prevalence=n_pos / len(emb),
                    auc=auc,
                    auc_std=auc_std,
                    knn_all=knn_all,
                    knn_fg=knn_fg,
                    recall=rec,
                    precision=prec,
                    filament_px=area,
                    seconds=dt,
                )
            )
            print(
                f"[{n}/{len(chosen)}] {kind:6s} {meta['file_name'][:28]:28s} "
                f"tok={len(emb):6d} pos={n_pos:5d} "
                f"auc={auc:.3f} knn_fg={knn_fg:.3f} "
                f"rec={rec:.3f} prec={prec:.3f} ({dt:.1f}s)"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    # ---- aggregate table -------------------------------------------------- #
    print("\n" + "=" * 78)
    print(f"{'backbone':10s} {'auc':>14s} {'knn_fg':>8s} {'recall':>8s} {'prec':>8s} {'tok/img':>9s}")
    print("-" * 78)
    summary = {}
    for kind in kinds:
        sub = [r for r in rows if r["backbone"] == kind]
        aucs = np.array([r["auc"] for r in sub], np.float64)
        knn = np.array([r["knn_fg"] for r in sub], np.float64)
        rec = np.array([r["recall"] for r in sub], np.float64)
        prec = np.array([r["precision"] for r in sub], np.float64)
        ntok = np.array([r["n_tokens"] for r in sub], np.float64)
        summary[kind] = dict(
            auc_mean=float(np.nanmean(aucs)),
            auc_std=float(np.nanstd(aucs)),
            knn_fg_mean=float(np.nanmean(knn)),
            recall_mean=float(np.nanmean(rec)),
            precision_mean=float(np.nanmean(prec)),
            tokens_per_image=float(np.nanmean(ntok)),
            seconds_per_image=float(np.mean([r["seconds"] for r in sub])),
        )
        s = summary[kind]
        print(
            f"{kind:10s} {s['auc_mean']:.3f}+-{s['auc_std']:.3f} "
            f"{s['knn_fg_mean']:8.3f} {s['recall_mean']:8.3f} "
            f"{s['precision_mean']:8.3f} {s['tokens_per_image']:9.0f}"
        )
    print("=" * 78)

    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}/metrics.json and {out}/summary.json")


if __name__ == "__main__":
    main()
