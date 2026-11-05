"""Paired bootstrap: does closing actually beat plain connected components?

The sweep picked thr=0.45, close_k=15 (PQ 0.3647) over thr=0.5, close_k=0
(PQ 0.3527). Both are evaluated on the same 142 images, so a paired bootstrap
over images is the right test -- the per-config CIs overlap heavily and would
suggest "no difference" even when the paired difference is consistent.

    python scripts/deepcrack_postprocess_ci.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)

CONFIGS = {
    "baseline (thr=0.5, close=0)": (0.50, 512, 0),
    "closing  (thr=0.45, close=15)": (0.45, 512, 15),
    "closing  (thr=0.48, close=17)": (0.48, 512, 17),
}


def labelmap(prob, thr, min_area, close_k):
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx((prob > thr).astype(np.uint8), cv2.MORPH_CLOSE, k)
        return prob_to_labelmap(binary.astype(np.float32), thr, min_area)
    return prob_to_labelmap(prob, thr, min_area)


def main() -> None:
    cache = ROOT / "runs/deepcrack_ft_256_long/val_probs"
    files = sorted(glob.glob(str(cache / "*_prob.npy")))
    probs = [np.load(f).astype(np.float32) for f in files]
    labels = [np.load(f.replace("_prob.npy", "_label.npy")) for f in files]
    print(f"{len(probs)} images")

    per_image: dict[str, list[dict]] = {}
    for name, (thr, area, ck) in CONFIGS.items():
        acc = PQAccumulator(threshold=0.5)
        for p, g in zip(probs, labels):
            acc.update(labelmap(p, thr, area, ck), g, record=True)
        per_image[name] = acc.per_image
        print(f"{name:32s} PQ={acc.pq:.4f} SQ={acc.sq:.4f} RQ={acc.rq:.4f} "
              f"tp={acc.tp} fp={acc.fp} fn={acc.fn}")

    def agg(stats, idx):
        tp = np.array([s["tp"] for s in stats])[idx].sum()
        fp = np.array([s["fp"] for s in stats])[idx].sum()
        fn = np.array([s["fn"] for s in stats])[idx].sum()
        io = np.array([s["iou_sum"] for s in stats])[idx].sum()
        d = tp + 0.5 * fp + 0.5 * fn
        return io / d if d > 0 else 0.0

    rng = np.random.default_rng(0)
    n = len(probs)
    base = per_image["baseline (thr=0.5, close=0)"]
    print()
    for name in CONFIGS:
        if name.startswith("baseline"):
            continue
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            diffs.append(agg(per_image[name], idx) - agg(base, idx))
        d = np.array(diffs)
        print(
            f"{name} vs baseline: mean={d.mean():+.4f} "
            f"95% CI=[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] "
            f"P(>0)={np.mean(d > 0):.3f}"
        )

    out = ROOT / "docs/DEEPCRACK_FINETUNE/postprocess_ci.json"
    out.write_text(json.dumps({k: v for k, v in per_image.items()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
