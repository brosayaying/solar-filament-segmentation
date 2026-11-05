"""Hysteresis thresholding for the patch-classifier score map.

The classifier reliably finds the *darkest core* of each filament but not its
full extent, so a single global threshold either misses most of the filament
(high threshold) or floods the disk with noise (low threshold).

Hysteresis solves exactly this: keep only low-threshold pixels that are
connected to a high-threshold seed. Filament bodies are connected, so the weak
parts of a filament survive while isolated weak noise does not.
"""
from __future__ import annotations

import cv2
import numpy as np


def hysteresis(
    score: np.ndarray,
    high: float,
    low: float,
    min_area: int = 512,
    close_k: int = 0,
) -> np.ndarray:
    """Binary mask of low-threshold regions connected to high-threshold seeds."""
    strong = (score > high).astype(np.uint8)
    weak = (score > low).astype(np.uint8)
    if strong.sum() == 0:
        return np.zeros(score.shape, np.uint8)

    # Label the weak mask, then keep components that contain a strong pixel.
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    if n <= 1:
        return np.zeros(score.shape, np.uint8)

    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong > 0])] = True
    keep[0] = False

    out = keep[lab].astype(np.uint8)
    if close_k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)
    return out


def to_labelmap(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = np.zeros(binary.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out
