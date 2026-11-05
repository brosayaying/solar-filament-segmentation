"""Ceiling analysis: what PQ can a 32x32 blocky mask possibly achieve?

Uses the *ground-truth* mask to decide patch labels, then upsamples to full
resolution and runs connected components. This is the best any patch classifier
could do if it were perfect.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar.metrics import PQAccumulator  # noqa: E402

CACHE = Path("/tmp/cv_cache")


def blocky_mask(gt: np.ndarray, stride: int, frac: float) -> np.ndarray:
    """Upsample a patch-level positive/negative decision to full resolution."""
    H, W = gt.shape
    p = (gt > 0).astype(np.float32)
    ph, pw = H // stride, W // stride
    blocks = p[: ph * stride, : pw * stride].reshape(ph, stride, pw, stride).mean(axis=(1, 3))
    pos = (blocks > frac).astype(np.uint8)
    return cv2.resize(pos, (W, H), interpolation=cv2.INTER_NEAREST)


def to_labelmap(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros(binary.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def main() -> None:
    files = sorted(glob.glob(str(CACHE / "*.npz")))[:40]
    gts = [np.load(f, allow_pickle=True)["gt"].astype(np.int32) for f in files]
    print(f"{len(gts)} images, {sum(int(g.max()) for g in gts)} GT instances\n")

    for stride in (32, 16, 8):
        for frac in (0.0, 0.05, 0.1, 0.25, 0.5):
            for min_area in (200, 512):
                acc = PQAccumulator()
                for gt in gts:
                    acc.update(to_labelmap(blocky_mask(gt, stride, frac), min_area), gt)
                s = acc.summary()
                print(
                    f"stride={stride:>3} frac>{frac:<5} min_area={min_area:>4}  "
                    f"PQ={s['pq']:.4f} SQ={s['sq']:.3f} RQ={s['rq']:.3f} "
                    f"tp={s['tp']} fp={s['fp']} fn={s['fn']}"
                )
        print()


if __name__ == "__main__":
    main()
