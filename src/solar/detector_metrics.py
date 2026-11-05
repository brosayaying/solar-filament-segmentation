"""Stage-1 evaluation metrics.

Two different questions, and they are not the same:

``point_f1``
    Did a predicted point land within ``radius`` px of a ground-truth spine
    midpoint? This is the conventional detection metric, but the radius is
    arbitrary and it under-reports: the stride-4 model had a 27.7 px median
    localisation error, so a 24 px radius scored it at 0.43 while 68% of
    filaments actually had a point inside them.

``mask_hit_recall``
    Did at least one predicted point land *inside the filament's own mask*?
    This is what stage 2 needs -- a seed anywhere on the filament is usable,
    regardless of how far it is from the midpoint. It is also scale-free: no
    radius to tune.

Report both. Optimise ``mask_hit_recall``.
"""

from __future__ import annotations

import cv2
import numpy as np


def match_points(pred: np.ndarray, gt: np.ndarray, radius: float) -> tuple[int, int, int]:
    """Greedy one-to-one point matching by Euclidean distance."""
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0

    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    used_p = np.zeros(len(pred), bool)
    used_g = np.zeros(len(gt), bool)
    tp = 0
    for flat in np.argsort(d, axis=None):
        i, j = divmod(int(flat), len(gt))
        if d[i, j] > radius:
            break
        if used_p[i] or used_g[j]:
            continue
        used_p[i] = used_g[j] = True
        tp += 1
    return tp, int(len(pred) - tp), int(len(gt) - tp)


def mask_hit_stats(
    pred: np.ndarray,
    anns: list[dict],
    height: int,
    width: int,
) -> dict[str, int]:
    """Count how many ground-truth filaments contain at least one predicted point.

    Also counts predictions that fall on no filament at all, which is the
    stage-1 false-positive rate that matters downstream: a seed on background
    would grow a spurious region in stage 2.
    """
    if len(pred) == 0:
        return {"hit": 0, "miss": len(anns), "pred_on_filament": 0, "pred_off": 0}

    xs = np.clip(np.round(pred[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(pred[:, 1]).astype(int), 0, height - 1)

    hit = 0
    on_filament = np.zeros(len(pred), bool)
    for ann in anns:
        mask = np.zeros((height, width), np.uint8)
        for poly in ann["segmentation"]:
            pts = np.round(np.asarray(poly, np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        inside = mask[ys, xs] > 0
        if inside.any():
            hit += 1
        on_filament |= inside

    return {
        "hit": hit,
        "miss": len(anns) - hit,
        "pred_on_filament": int(on_filament.sum()),
        "pred_off": int((~on_filament).sum()),
    }
