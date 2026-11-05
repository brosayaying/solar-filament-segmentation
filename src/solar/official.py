"""A local scorer that mirrors the organisers' evaluation notebook.

Why this exists
---------------
`solar.metrics.PQAccumulator` is fast and correct for *relative* comparisons, but
it is our own implementation. The organisers score with their own notebook, and
the two can diverge in ways that matter:

* GT is keyed ``<annotator>-<image>`` and **one prediction is scored against
  every annotator copy** of that image. A prediction is therefore evaluated k
  times for a file with k annotators.
* GT is decoded from RLE with pycocotools, not rasterised from polygons.
* Matching is greedy 1:1 at IoU **strictly greater than** 0.5.
* PQ is pooled globally over all (prediction, annotator-copy) pairs, not
  averaged per image.

This module reproduces that protocol and is the final check before every
submission. It is deliberately written against the *description* in
`docs/COMPETITION.md` §3-4 rather than against `metrics.py`, so that a bug in one
does not hide in the other.

Validation
----------
`python -m solar.official --selftest` scores the same predictions through both
this module and `PQAccumulator` and asserts they agree to 4 dp. If they do not,
one of them is wrong and it must be found before any further training.

    python -m solar.official --selftest
    python -m solar.official --pred submissions/x.csv --split v2/splits/week5_seed42.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "decode_rle",
    "encode_mask",
    "gt_by_image",
    "score_predictions",
    "bootstrap_ci",
]

IOU_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# RLE helpers
# --------------------------------------------------------------------------- #


def decode_rle(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode a submission RLE *counts* string to a boolean mask."""
    from pycocotools import mask as mask_utils

    m = mask_utils.decode({"counts": counts, "size": [height, width]})
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def encode_mask(mask: np.ndarray) -> str:
    """Encode a boolean mask as a submission RLE counts string."""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def gt_by_image(
    coco: dict[str, Any], image_ids: Iterable[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Group GT annotations by **bare image stem**, keeping annotator copies.

    The COCO ``image_id`` is ``<annotator>-<stem>``. The organisers match a
    prediction for ``<stem>`` against every annotator copy of that stem, so the
    grouping key here is the stem, and the value is a list of per-copy
    annotation lists.
    """
    keep = set(image_ids) if image_ids is not None else None

    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for img in coco["images"]:
        if keep is not None and img["id"] not in keep:
            continue
        stem = Path(img["file_name"]).stem
        annotator = img["id"].split("-", 1)[0]
        out[stem].append(
            {
                "image_id": img["id"],
                "annotator": annotator,
                "anns": anns_by_image.get(img["id"], []),
            }
        )
    return dict(out)


def _rasterise_instances(anns: list[dict], h: int, w: int) -> np.ndarray:
    """Rasterise annotations to an int32 instance labelmap via pycocotools."""
    from pycocotools import mask as mask_utils

    out = np.zeros((h, w), np.int32)
    for i, ann in enumerate(anns, start=1):
        seg = ann.get("segmentation")
        if not seg:
            continue
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        out[m.astype(bool)] = i
    return out


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def _pairwise_iou(
    pred: np.ndarray, gt: np.ndarray, n_pred: int, n_gt: int
) -> np.ndarray:
    combined = pred.astype(np.int64) * (n_gt + 1) + gt.astype(np.int64)
    counts = np.bincount(combined.ravel(), minlength=(n_pred + 1) * (n_gt + 1))
    counts = counts.reshape(n_pred + 1, n_gt + 1)

    inter = counts[1:, 1:].astype(np.float64)
    pred_area = counts[1:, :].sum(axis=1).astype(np.float64)
    gt_area = counts[:, 1:].sum(axis=0).astype(np.float64)
    union = pred_area[:, None] + gt_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _match(iou: np.ndarray, threshold: float = IOU_THRESHOLD) -> tuple[int, float]:
    """Greedy 1:1 matching at IoU **strictly greater than** ``threshold``."""
    if iou.size == 0:
        return 0, 0.0

    n_pred, n_gt = iou.shape
    used_p = np.zeros(n_pred, bool)
    used_g = np.zeros(n_gt, bool)

    tp = 0
    iou_sum = 0.0
    for flat in np.argsort(iou, axis=None)[::-1]:
        i, j = divmod(int(flat), n_gt)
        if iou[i, j] <= threshold:
            break
        if used_p[i] or used_g[j]:
            continue
        used_p[i] = True
        used_g[j] = True
        tp += 1
        iou_sum += float(iou[i, j])
    return tp, iou_sum


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_predictions(
    preds: dict[str, list[np.ndarray]],
    gt: dict[str, list[dict[str, Any]]],
    height: int = 2048,
    width: int = 2048,
    record: bool = False,
) -> dict[str, Any]:
    """Score predictions against every annotator copy of every image.

    ``preds`` maps a bare image stem to a list of boolean masks. ``gt`` is the
    output of :func:`gt_by_image`.

    Returns pooled PQ/SQ/RQ plus TP/FP/FN, and per-image stats when
    ``record=True`` (needed for the bootstrap).
    """
    tp = fp = fn = 0
    iou_sum = 0.0
    per_image: list[dict[str, float]] = []
    n_pairs = 0

    for stem, copies in gt.items():
        masks = preds.get(stem, [])
        n_pred = len(masks)

        if n_pred:
            pred_map = np.zeros((height, width), np.int32)
            for i, m in enumerate(masks, start=1):
                pred_map[m] = i
        else:
            pred_map = np.zeros((height, width), np.int32)

        img_tp = img_fp = img_fn = 0
        img_iou = 0.0

        for copy in copies:
            gt_map = _rasterise_instances(copy["anns"], height, width)
            n_gt = int(gt_map.max()) if gt_map.size else 0

            if n_pred == 0 and n_gt == 0:
                c_tp, c_iou = 0, 0.0
            elif n_pred == 0:
                c_tp, c_iou = 0, 0.0
            elif n_gt == 0:
                c_tp, c_iou = 0, 0.0
            else:
                iou = _pairwise_iou(pred_map, gt_map, n_pred, n_gt)
                c_tp, c_iou = _match(iou)

            img_tp += c_tp
            img_fp += n_pred - c_tp
            img_fn += n_gt - c_tp
            img_iou += c_iou
            n_pairs += 1

        tp += img_tp
        fp += img_fp
        fn += img_fn
        iou_sum += img_iou

        if record:
            per_image.append(
                {
                    "tp": img_tp,
                    "fp": img_fp,
                    "fn": img_fn,
                    "iou_sum": img_iou,
                }
            )

    denom = tp + 0.5 * fp + 0.5 * fn
    out = {
        "pq": iou_sum / denom if denom > 0 else 0.0,
        "sq": iou_sum / tp if tp > 0 else 0.0,
        "rq": tp / denom if denom > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_images": len(gt),
        "n_pairs": n_pairs,
    }
    if record:
        out["per_image"] = per_image
    return out


def bootstrap_ci(
    per_image: list[dict[str, float]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Per-image bootstrap CI, resampling images and recomputing the aggregate."""
    rng = np.random.default_rng(seed)
    n = len(per_image)
    tp = np.array([s["tp"] for s in per_image], np.float64)
    fp = np.array([s["fp"] for s in per_image], np.float64)
    fn = np.array([s["fn"] for s in per_image], np.float64)
    iou = np.array([s["iou_sum"] for s in per_image], np.float64)

    out: dict[str, tuple[float, float]] = {}
    for key in ("pq", "sq", "rq"):
        vals = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            t, f, m, i = tp[idx].sum(), fp[idx].sum(), fn[idx].sum(), iou[idx].sum()
            denom = t + 0.5 * f + 0.5 * m
            if key == "pq":
                vals[b] = i / denom if denom > 0 else 0.0
            elif key == "rq":
                vals[b] = t / denom if denom > 0 else 0.0
            else:
                vals[b] = i / t if t > 0 else 0.0
        out[key] = (
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))),
        )
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_submission(path: Path) -> dict[str, list[np.ndarray]]:
    preds: dict[str, list[np.ndarray]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            stem = row["filament_id"].rsplit("_", 1)[0]
            preds[stem].append(decode_rle(row["segmentation_rle"]))
    return dict(preds)


def _selftest() -> None:
    """Score cached val probabilities through both scorers and compare."""
    import torch

    from .data import load_coco
    from .metrics import PQAccumulator
    from .postprocess import prob_to_labelmap
    from .tune import load_cache

    ckpt = Path("runs/norm_disk_raw/best.pt")
    cache = ckpt.parent / "val_probs"
    if not cache.exists():
        raise SystemExit(f"no cached probabilities at {cache}")

    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
    split = json.load(open("splits/date_seed42.json"))
    coco = load_coco(Path(cfg["data"]["root"]) / cfg["data"]["ann_file"])

    probs, labels = load_cache(cache)
    val_ids = split["val"]
    assert len(probs) == len(val_ids), (len(probs), len(val_ids))

    thr, area, close_k = 0.65, 256, 15

    # --- our scorer ------------------------------------------------------- #
    acc = PQAccumulator(threshold=0.5)
    for p, lab in zip(probs, labels):
        acc.update(
            prob_to_labelmap(p, threshold=thr, min_area=area, close_k=close_k),
            lab,
            record=True,
        )
    ours = acc.summary()

    # --- official scorer -------------------------------------------------- #
    # The official scorer keys by bare stem and pools annotator copies, so the
    # cached per-entry predictions must be re-keyed by stem. Every copy of a
    # file yields the same prediction (same pixels, deterministic model), so
    # take the masks from the first copy seen and do not duplicate them.
    id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in coco["images"]}
    by_stem: dict[str, list[np.ndarray]] = {}
    for p, image_id in zip(probs, val_ids):
        stem = id_to_stem[image_id]
        if stem in by_stem:
            continue
        lm = prob_to_labelmap(p, threshold=thr, min_area=area, close_k=close_k)
        by_stem[stem] = [lm == k for k in range(1, int(lm.max()) + 1)]

    gt = gt_by_image(coco, image_ids=val_ids)
    official = score_predictions(by_stem, gt, record=True)

    print(f"{'metric':>6} {'PQAccumulator':>15} {'official':>12} {'delta':>10}")
    print("-" * 46)
    for k in ("pq", "sq", "rq"):
        d = official[k] - ours[k]
        print(f"{k:>6} {ours[k]:>15.4f} {official[k]:>12.4f} {d:>10.4f}")
    print(
        f"\ntp/fp/fn  ours {ours['tp']}/{ours['fp']}/{ours['fn']}  "
        f"official {official['tp']}/{official['fp']}/{official['fn']}"
    )
    print(f"images {ours['n_images']}  pairs {official['n_pairs']}")

    ci = bootstrap_ci(official["per_image"])
    print(f"\nofficial PQ 95% CI: [{ci['pq'][0]:.4f}, {ci['pq'][1]:.4f}]")

    if abs(official["pq"] - ours["pq"]) > 5e-5:
        raise SystemExit(
            "SCORERS DISAGREE — one of them is wrong. Do not train further "
            "until this is resolved."
        )
    print("\nscorers agree to 4 dp: OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pred", default=None, help="submission CSV to score")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument(
        "--ann-file",
        default="train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.pred:
        raise SystemExit("pass --selftest or --pred <csv>")

    coco = load_coco(Path(args.root) / args.ann_file)
    split = json.load(open(args.split))
    gt = gt_by_image(coco, image_ids=split["val"])
    preds = _load_submission(Path(args.pred))

    res = score_predictions(preds, gt, record=True)
    ci = bootstrap_ci(res.pop("per_image"))
    print(json.dumps(res, indent=2))
    print(f"PQ 95% CI: [{ci['pq'][0]:.4f}, {ci['pq'][1]:.4f}]")


if __name__ == "__main__":
    main()
