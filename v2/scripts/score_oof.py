"""Score the pooled out-of-fold predictions from Phase 1.

Streams one file at a time. Loading all 707 dumps at once needs ~12 GB (each
holds a float32 2048^2 probability plus up to three int32 2048^2 label maps) and
was OOM-killed on a 46 GB box that already had other work resident.

Pooled OOF is the honest Phase 1 number: every file is scored by the fold that
did *not* train on it, and each prediction is scored against every annotator copy
of that file, which is what the official scorer does.

    python v2/scripts/score_oof.py
    python v2/scripts/score_oof.py --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"


def score(thr: float, area: int, close_k: int, record: bool = False) -> dict:
    """Pooled PQ over every OOF file, scored against every annotator copy."""
    acc = PQAccumulator(threshold=0.5)
    for path in sorted(OOF.glob("fold*/*.npz")):
        with np.load(path, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
        pred = prob_to_labelmap(
            prob, threshold=thr, min_area=area, close_k=close_k
        )
        for lab in labels:
            acc.update(pred, lab, record=record)
    return acc, acc.summary()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--close-k", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    if args.sweep:
        grid = [
            (t, a, c)
            for t in (0.45, 0.5, 0.55, 0.65)
            for a in (256, 512)
            for c in (0, 15)
        ]
    else:
        grid = [(args.threshold, args.min_area, args.close_k)]

    results = []
    print(f"{'thr':>5} {'area':>6} {'close':>6} {'PQ':>7} {'SQ':>7} {'RQ':>7} "
          f"{'tp':>6} {'fp':>6} {'fn':>6}")
    print("-" * 62)
    for thr, area, ck in grid:
        acc, s = score(thr, area, ck, record=args.bootstrap)
        row = {"threshold": thr, "min_area": area, "close_k": ck, **s}
        if args.bootstrap:
            ci = acc.bootstrap_ci(n_boot=args.n_boot)
            for k, (lo, hi) in ci.items():
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        results.append(row)
        print(f"{thr:>5.2f} {area:>6} {ck:>6} {s['pq']:>7.4f} {s['sq']:>7.4f} "
              f"{s['rq']:>7.4f} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6}")

    results.sort(key=lambda r: r["pq"], reverse=True)
    best = results[0]
    print(f"\nbest: thr {best['threshold']} area {best['min_area']} "
          f"close {best['close_k']} -> PQ {best['pq']:.4f}")
    if args.bootstrap:
        print(f"  95% CI [{best['pq_lo']:.4f}, {best['pq_hi']:.4f}]  "
              f"(SQ [{best['sq_lo']:.4f}, {best['sq_hi']:.4f}], "
              f"RQ [{best['rq_lo']:.4f}, {best['rq_hi']:.4f}])")

    out = ROOT / "v2/runs/seg_v2/oof_results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
