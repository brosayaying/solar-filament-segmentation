"""Diagnose which candidate components are TP vs FP, and what distinguishes them."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cvlib  # noqa: E402
from solar.metrics import pairwise_iou  # noqa: E402

CACHE = Path("/tmp/cv_cache")


def load(i: int):
    d = np.load(CACHE / f"{i:04d}.npz", allow_pickle=True)
    return (
        d["res"].astype(np.float32),
        d["disk"],
        d["gt"].astype(np.int32),
        d["img"],
        str(d["file_name"]),
    )


def main() -> None:
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    min_area = int(sys.argv[2]) if len(sys.argv) > 2 else 128

    rows = []
    for i in range(len(list(CACHE.glob("*.npz")))):
        res, disk, gt, img, fn = load(i)
        binary = ((res < -thr) & (disk > 0)).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
        n_gt = int(gt.max())
        if n <= 1:
            continue
        # IoU of each component against GT instances
        comp_map = lab.astype(np.int32)
        iou = pairwise_iou(comp_map, gt, n - 1, n_gt) if n_gt else np.zeros((n - 1, 0))
        for c in range(1, n):
            area = stats[c, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            best = iou[c - 1].max() if n_gt else 0.0
            m = lab == c
            vals = res[m]
            rows.append(
                dict(
                    img=i,
                    area=area,
                    best_iou=best,
                    mean_res=float(vals.mean()),
                    min_res=float(vals.min()),
                    p10=float(np.percentile(vals, 10)),
                    std_res=float(vals.std()),
                    w=stats[c, cv2.CC_STAT_WIDTH],
                    h=stats[c, cv2.CC_STAT_HEIGHT],
                    elong=max(stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT])
                    / max(1, min(stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT])),
                    fill=area / max(1, stats[c, cv2.CC_STAT_WIDTH] * stats[c, cv2.CC_STAT_HEIGHT]),
                )
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    df["tp"] = df.best_iou >= 0.5
    print(f"components={len(df)} tp={df.tp.sum()} fp={(~df.tp).sum()}")
    print("\n-- medians by class --")
    print(df.groupby("tp")[["area", "mean_res", "min_res", "p10", "std_res", "elong", "fill"]].median().round(2))
    print("\n-- quantiles --")
    for col in ["area", "mean_res", "min_res", "p10", "std_res", "elong", "fill"]:
        print(col, "TP", np.percentile(df[df.tp][col], [10, 50, 90]).round(2),
              "FP", np.percentile(df[~df.tp][col], [10, 50, 90]).round(2))
    df.to_csv("/tmp/comp_stats.csv", index=False)


if __name__ == "__main__":
    main()
