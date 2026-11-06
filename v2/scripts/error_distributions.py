"""Dice/IoU distributions and the one-to-many / many-to-one histogram.

Why this exists
---------------
The competition rubric names four quantitative criteria, not just PQ:

    Panoptic Quality (primary)
    Distribution of Dice scores
    Distribution of IoU scores
    Distribution of one-to-many / many-to-one relations between GT and predictions

The last three were never computed. PQ is a single scalar and it hides the shape
of the error: two systems can reach the same PQ with very different failure
modes, and the rubric asks for that shape explicitly.

What is measured
----------------
For every out-of-fold file, every annotator copy, and the shipped
post-processing (thr 0.50 / min_area 256 / close 15):

**IoU and Dice distributions**, reported three ways, because they answer
different questions:

  matched      IoU/Dice of the greedy-matched pairs only. This is the SQ
               distribution -- how good the boundaries are *when the instance
               was found at all*.
  best-per-gt  best IoU/Dice for every ground-truth instance, with unmatched
               instances scoring 0. This is the honest per-object view: it
               includes the objects that were missed entirely.
  best-per-pred best IoU/Dice for every predicted instance. Includes false
               positives, which score 0.

**One-to-many / many-to-one**, counted at several overlap thresholds:

  one-to-many   one GT instance overlapped by k > 1 predictions
                (fragmentation -- the metric punishes this hard)
  many-to-one   one prediction overlapping k > 1 GT instances
                (merging -- also punished)

The counts are reported at IoU > 0.5 (the metric's own matching threshold) and
at looser thresholds, because at 0.5 the counts are small and the interesting
structure lives in the partial overlaps.

Streaming
---------
One file at a time. Holding 707 files x (float32 2048^2 prob + int32 labels) is
~12 GB and has OOM-killed this box before.

    python v2/scripts/error_distributions.py
    python v2/scripts/error_distributions.py --limit 100   # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import match_instances, pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"
OUT = ROOT / "v2/results/error_distributions.json"

# The shipped post-processing. Must match the config that produced the
# submission, or these distributions describe a system we did not submit.
THR = 0.50
MIN_AREA = 256
CLOSE_K = 15

# Overlap thresholds for the relation histogram. 0.5 is the metric's matching
# threshold; the lower values expose partial overlaps that 0.5 hides.
REL_THRESHOLDS = (0.1, 0.3, 0.5, 0.7)


def dice_from_iou(iou: np.ndarray) -> np.ndarray:
    """Dice from IoU. Exact, not an approximation.

    For two sets, Dice = 2|A n B| / (|A| + |B|) and IoU = |A n B| / |A u B|.
    With |A u B| = |A| + |B| - |A n B|, substituting gives Dice = 2I/(1+I).
    """
    return 2.0 * iou / (1.0 + iou)


def quantiles(values: np.ndarray) -> dict:
    """Summary of a distribution, including the tails that matter here."""
    if values.size == 0:
        return {"n": 0}
    q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(values.max()),
        # Fraction of instances that are essentially correct. This is the
        # number a reader actually wants: "how often is it right?"
        "frac_ge_0.75": float((values >= 0.75).mean()),
        "frac_ge_0.50": float((values >= 0.50).mean()),
        "frac_lt_0.10": float((values < 0.10).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N files (0 = all)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    paths = sorted(OOF.glob("fold*/*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no OOF dumps under {OOF}")

    print(f"{len(paths)} OOF files | thr {THR} / area {MIN_AREA} / close {CLOSE_K}")

    # Matched-pair distributions (the SQ view).
    matched_iou: list[np.ndarray] = []
    # Per-object distributions (includes misses and false positives as 0).
    best_gt_iou: list[np.ndarray] = []
    best_pred_iou: list[np.ndarray] = []

    # Relation histogram: rel[thr][k] = number of instances with exactly k
    # overlaps. Separate counters for the GT side and the prediction side.
    one_to_many = {t: {} for t in REL_THRESHOLDS}
    many_to_one = {t: {} for t in REL_THRESHOLDS}

    n_gt_total = 0
    n_pred_total = 0
    n_tp_total = 0

    for i, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]

        pred = prob_to_labelmap(
            prob, threshold=THR, min_area=MIN_AREA, close_k=CLOSE_K
        )
        n_pred = int(pred.max())

        for gt in labels:
            n_gt = int(gt.max())
            n_gt_total += n_gt
            n_pred_total += n_pred

            if n_pred == 0 or n_gt == 0:
                # No pairs to match. Every GT is a miss, every prediction a FP.
                if n_gt:
                    best_gt_iou.append(np.zeros(n_gt))
                if n_pred:
                    best_pred_iou.append(np.zeros(n_pred))
                for t in REL_THRESHOLDS:
                    if n_gt:
                        one_to_many[t][0] = one_to_many[t].get(0, 0) + n_gt
                    if n_pred:
                        many_to_one[t][0] = many_to_one[t].get(0, 0) + n_pred
                continue

            iou = pairwise_iou(pred, gt, n_pred, n_gt)

            # --- matched pairs: the SQ distribution -----------------------
            tp, _ = match_instances(iou, 0.5)
            n_tp_total += tp
            if tp:
                # Recover which pairs were matched by replaying the same greedy
                # order. match_instances returns only counts, and duplicating
                # the loop here is cheaper than changing a function the metric
                # depends on.
                order = np.argsort(iou, axis=None)[::-1]
                mp = np.zeros(n_pred, bool)
                mg = np.zeros(n_gt, bool)
                picked = []
                for flat in order:
                    a, b = divmod(int(flat), n_gt)
                    if iou[a, b] <= 0.5:
                        break
                    if mp[a] or mg[b]:
                        continue
                    mp[a] = mg[b] = True
                    picked.append(iou[a, b])
                matched_iou.append(np.asarray(picked))

            # --- per-object best overlap ---------------------------------
            best_gt_iou.append(iou.max(axis=0))
            best_pred_iou.append(iou.max(axis=1))

            # --- relation histogram --------------------------------------
            for t in REL_THRESHOLDS:
                # How many predictions overlap each GT above t?
                gt_counts = (iou > t).sum(axis=0)
                for k in gt_counts:
                    one_to_many[t][int(k)] = one_to_many[t].get(int(k), 0) + 1
                # How many GT does each prediction overlap above t?
                pred_counts = (iou > t).sum(axis=1)
                for k in pred_counts:
                    many_to_one[t][int(k)] = many_to_one[t].get(int(k), 0) + 1

        if i % 100 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} files")

    # ------------------------------------------------------------------ #
    # Assemble
    # ------------------------------------------------------------------ #
    mi = np.concatenate(matched_iou) if matched_iou else np.zeros(0)
    bg = np.concatenate(best_gt_iou) if best_gt_iou else np.zeros(0)
    bp = np.concatenate(best_pred_iou) if best_pred_iou else np.zeros(0)

    # Raw values are kept so the plotting script can draw true histograms
    # rather than reconstructing a distribution from quantiles, which is not
    # possible without inventing data. ~22k floats total, so this is cheap.
    raw_path = Path(args.out).with_suffix(".npz")
    np.savez_compressed(
        raw_path,
        iou_matched=mi,
        iou_best_per_gt=bg,
        iou_best_per_pred=bp,
        dice_matched=dice_from_iou(mi),
        dice_best_per_gt=dice_from_iou(bg),
        dice_best_per_pred=dice_from_iou(bp),
    )
    print(f"wrote {raw_path} (raw values for plotting)")

    def rel_summary(counter: dict) -> dict:
        total = sum(counter.values())
        if not total:
            return {}
        out = {}
        for k in sorted(counter):
            out[str(k)] = {
                "count": counter[k],
                "fraction": counter[k] / total,
            }
        # The headline number: what fraction of instances are involved in a
        # non-1:1 relation at all.
        out["_summary"] = {
            "total": total,
            "frac_zero": counter.get(0, 0) / total,
            "frac_one": counter.get(1, 0) / total,
            "frac_multi": sum(v for k, v in counter.items() if k > 1) / total,
        }
        return out

    result = {
        "params": {
            "threshold": THR,
            "min_area": MIN_AREA,
            "close_k": CLOSE_K,
            "n_files": len(paths),
            "relation_thresholds": list(REL_THRESHOLDS),
        },
        "totals": {
            "n_gt": n_gt_total,
            "n_pred": n_pred_total,
            "n_tp": n_tp_total,
            "recall": n_tp_total / max(1, n_gt_total),
            "precision": n_tp_total / max(1, n_pred_total),
        },
        "iou": {
            "matched": quantiles(mi),
            "best_per_gt": quantiles(bg),
            "best_per_pred": quantiles(bp),
        },
        "dice": {
            "matched": quantiles(dice_from_iou(mi)),
            "best_per_gt": quantiles(dice_from_iou(bg)),
            "best_per_pred": quantiles(dice_from_iou(bp)),
        },
        "one_to_many": {str(t): rel_summary(one_to_many[t])
                        for t in REL_THRESHOLDS},
        "many_to_one": {str(t): rel_summary(many_to_one[t])
                        for t in REL_THRESHOLDS},
    }

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print(f"\nGT {n_gt_total}  pred {n_pred_total}  TP {n_tp_total}  "
          f"recall {result['totals']['recall']:.3f}  "
          f"precision {result['totals']['precision']:.3f}")

    print(f"\n{'':<16}{'n':>7}{'mean':>8}{'p05':>8}{'p50':>8}{'p95':>8}"
          f"{'>=.75':>8}{'<.10':>8}")
    print("-" * 71)
    for metric in ("iou", "dice"):
        for view in ("matched", "best_per_gt", "best_per_pred"):
            s = result[metric][view]
            if not s.get("n"):
                continue
            print(f"{metric + '/' + view:<16}{s['n']:>7}{s['mean']:>8.3f}"
                  f"{s['p05']:>8.3f}{s['p50']:>8.3f}{s['p95']:>8.3f}"
                  f"{s['frac_ge_0.75']:>8.3f}{s['frac_lt_0.10']:>8.3f}")

    print("\none-to-many (GT overlapped by k predictions):")
    for t in REL_THRESHOLDS:
        s = result["one_to_many"][str(t)].get("_summary", {})
        if s:
            print(f"  IoU>{t}: 0 -> {s['frac_zero']:.3f}  "
                  f"1 -> {s['frac_one']:.3f}  >1 -> {s['frac_multi']:.3f}")

    print("\nmany-to-one (prediction overlapping k GT):")
    for t in REL_THRESHOLDS:
        s = result["many_to_one"][str(t)].get("_summary", {})
        if s:
            print(f"  IoU>{t}: 0 -> {s['frac_zero']:.3f}  "
                  f"1 -> {s['frac_one']:.3f}  >1 -> {s['frac_multi']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
