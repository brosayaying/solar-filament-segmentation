"""Sweep post-processing on cached DeepCrack val probabilities.

Compares connected components against distance-transform watershed, and sweeps
threshold / min_area / closing. Writes results to JSON so the numbers survive
terminal buffer issues.

    python scripts/deepcrack_postprocess_sweep.py --method cc
    python scripts/deepcrack_postprocess_sweep.py --method watershed

The baseline to beat is PQ 0.3531 (cc, thr=0.5, min_area=512, no closing) on the
142-file dedup'd val set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap, prob_to_labelmap_watershed  # noqa: E402

cv2.setNumThreads(4)


def load_cache(cache_dir: Path):
    probs, labels = [], []
    for p in sorted(cache_dir.glob("*_prob.npy")):
        probs.append(np.load(p).astype(np.float32))
        labels.append(np.load(p.with_name(p.name.replace("_prob", "_label"))))
    if not probs:
        raise FileNotFoundError(cache_dir)
    return probs, labels


def labelmap(prob, thr, min_area, close_k, method, h_maxima, min_distance):
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx((prob > thr).astype(np.uint8), cv2.MORPH_CLOSE, k)
        p = binary.astype(np.float32)
    else:
        p = prob

    if method == "watershed":
        return prob_to_labelmap_watershed(
            p, thr, min_area, h_maxima=h_maxima, min_distance=min_distance
        )
    return prob_to_labelmap(p, thr, min_area)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/deepcrack_ft_256_long/val_probs")
    ap.add_argument("--method", default="cc", choices=["cc", "watershed"])
    ap.add_argument("--thresholds", default="0.4,0.5,0.6")
    ap.add_argument("--min-areas", default="256,512,1024,2048")
    ap.add_argument("--close-ks", default="0")
    ap.add_argument("--h-maximas", default="8,16,32")
    ap.add_argument("--min-distances", default="12,24")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = ROOT / args.cache
    probs, labels = load_cache(cache)
    print(f"{len(probs)} cached images from {cache}")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    min_areas = [int(x) for x in args.min_areas.split(",")]
    close_ks = [int(x) for x in args.close_ks.split(",")]

    if args.method == "watershed":
        h_maximas = [int(x) for x in args.h_maximas.split(",")]
        min_distances = [int(x) for x in args.min_distances.split(",")]
        grid = list(itertools.product(thresholds, min_areas, close_ks, h_maximas, min_distances))
    else:
        h_maximas, min_distances = [0], [0]
        grid = [(t, a, c, 0, 0) for t, a, c in itertools.product(thresholds, min_areas, close_ks)]

    results = []
    t_start = time.time()
    for n, (thr, area, ck, hm, md) in enumerate(grid, 1):
        acc = PQAccumulator(threshold=0.5)
        for prob, label in zip(probs, labels):
            acc.update(labelmap(prob, thr, area, ck, args.method, hm, md), label)
        s = acc.summary()
        results.append(
            {
                "method": args.method,
                "threshold": thr,
                "min_area": area,
                "close_k": ck,
                "h_maxima": hm,
                "min_distance": md,
                **s,
            }
        )
        print(
            f"[{n}/{len(grid)}] thr={thr:.2f} area={area:5d} close={ck} "
            f"hmax={hm:3d} mdis={md:3d}  PQ={s['pq']:.4f} "
            f"SQ={s['sq']:.4f} RQ={s['rq']:.4f} tp={s['tp']} fp={s['fp']} fn={s['fn']}",
            flush=True,
        )

    results.sort(key=lambda r: -r["pq"])
    print(f"\nsweep took {time.time() - t_start:.0f}s")
    print(f"{'thr':>5} {'area':>6} {'close':>6} {'hmax':>5} {'mdis':>5} "
          f"{'PQ':>7} {'SQ':>7} {'RQ':>7} {'tp':>5} {'fp':>5} {'fn':>5}")
    for r in results[:12]:
        print(
            f"{r['threshold']:>5.2f} {r['min_area']:>6} {r['close_k']:>6} "
            f"{r['h_maxima']:>5} {r['min_distance']:>5} "
            f"{r['pq']:>7.4f} {r['sq']:>7.4f} {r['rq']:>7.4f} "
            f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5}"
        )

    out = ROOT / (args.out or f"docs/DEEPCRACK_FINETUNE/sweep_{args.method}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nbest PQ={results[0]['pq']:.4f} -> {out}")


if __name__ == "__main__":
    main()
