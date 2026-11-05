"""Refine a patch-level probability map to pixel-level boundaries.

The classifier only knows 8x8 blocks, which caps SQ at ~0.67. But the underlying
black-hat response is available at full resolution, so the blocky decision can be
used as a *region* and the black-hat as the *boundary*:

    final = alpha * upsample(patch_prob) + (1 - alpha) * normalise(blackhat)

Both terms are positive-on-filament. The patch term supplies the semantic
decision (is this region a filament at all), the black-hat term supplies the
sub-patch edge. Thresholding the blend recovers thin barbs that a pure 8x8 mask
would round off.
"""
from __future__ import annotations

import cv2
import numpy as np


def upsample_prob(prob: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample of the (GRID, GRID) patch probability to full resolution."""
    return cv2.resize(prob.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def normalise(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Robust min-max scaling to [0, 1] using percentiles."""
    a, b = np.percentile(x, [lo, hi])
    if b <= a:
        return np.zeros_like(x, np.float32)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32)


def refine(
    prob: np.ndarray,
    img: np.ndarray,
    alpha: float = 0.6,
    bh_k: int = 41,
    blur: int = 3,
) -> np.ndarray:
    """Blend the upsampled patch probability with a full-resolution black-hat."""
    up = upsample_prob(prob, img.shape)
    if blur > 1:
        up = cv2.GaussianBlur(up, (blur, blur), 0)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_k, bh_k))
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ker).astype(np.float32)
    bh = normalise(bh)

    return alpha * up + (1.0 - alpha) * bh


def to_labelmap(binary: np.ndarray, min_area: int, close_k: int = 0) -> np.ndarray:
    """Connected components with optional closing and a minimum area."""
    b = binary.astype(np.uint8)
    if close_k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros(binary.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out
