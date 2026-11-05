"""Fast experiment harness for classical semantic segmentation of filaments.

Caches images/GT once, then evaluates a scoring function + threshold + CC
post-processing with PQ. Uses multiprocessing over images.
"""
from __future__ import annotations

import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar.metrics import PQAccumulator  # noqa: E402

CACHE = Path("/tmp/cv_cache")
cv2.setNumThreads(1)


def load_all(n: int | None = None) -> list[dict]:
    files = sorted(glob.glob(str(CACHE / "*.npz")))
    if n:
        files = files[:n]
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        out.append(
            dict(
                img=d["img"],
                gt=d["gt"].astype(np.int32),
                disk=d["disk"],
                file_name=str(d["file_name"]),
            )
        )
    return out


def inner_disk(disk: np.ndarray, margin: int = 40) -> np.ndarray:
    """Disk eroded by ``margin`` px: removes the bright limb ring."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.erode(disk, k)


def blackhat(img: np.ndarray, ksize: int) -> np.ndarray:
    """Dark structures smaller than ksize on a brighter background."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k).astype(np.float32)


def to_labelmap(score: np.ndarray, valid: np.ndarray, thr: float, min_area: int) -> np.ndarray:
    b = ((score > thr) & (valid > 0)).astype(np.uint8)
    if b.sum() == 0:
        return np.zeros(score.shape, np.int32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros(score.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def _eval_one(args):
    idx, score, valid, gt, thr, min_area = args
    pred = to_labelmap(score, valid, thr, min_area)
    return idx, pred, gt


def evaluate(
    data: list[dict],
    score_fn,
    thresholds: list[float],
    min_areas: list[int],
    margin: int = 40,
    workers: int = 10,
) -> list[dict]:
    scores = [score_fn(d) for d in data]
    valids = [inner_disk(d["disk"], margin) for d in data]
    results = []
    for thr in thresholds:
        for ma in min_areas:
            acc = PQAccumulator()
            for d, s, v in zip(data, scores, valids):
                acc.update(to_labelmap(s, v, thr, ma), d["gt"])
            r = acc.summary()
            r.update(threshold=thr, min_area=ma)
            results.append(r)
    return results


def report(results: list[dict], key: str = "pq") -> None:
    for r in sorted(results, key=lambda x: -x[key])[:12]:
        print(
            f"thr={r['threshold']:>6} min_area={r['min_area']:>5}  "
            f"PQ={r['pq']:.4f} SQ={r['sq']:.3f} RQ={r['rq']:.3f} "
            f"tp={r['tp']} fp={r['fp']} fn={r['fn']}"
        )
