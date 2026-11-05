"""Phase 2: instance-level candidate generation, merging and scoring.

The Phase 1 pipeline makes one global decision -- threshold the probability map
and label connected components -- which has no notion of "is this blob a
filament an annotator would draw". The measured error anatomy says 58.5 % of our
false positives are *splits* of an already-matched GT instance and 57.1 % of our
false negatives are found but drawn at IoU < 0.5. Both are instance-identity
failures, not pixel-classification failures, which is why SQ has been flat at
~0.678 across three architectures.

This module provides the pieces Phase 2 needs:

``generate_candidates``
    Low-threshold connected components, deliberately over-producing for recall.
``candidate_features``
    Per-candidate scalars (area, elongation, mean/max prob, contrast vs annulus).
``pair_features``
    Per-pair merge evidence (gap, orientation alignment, bridge probability).
``label_candidates``
    ``q`` = fraction of annotator copies in which the candidate has IoU > 0.5.
``label_pairs``
    Whether two candidates match the *same* GT instance in the majority of copies.

**No spine feature.** The ``spines`` array in the OOF dumps is the ground-truth
centreline rasterised from the annotation, not a model prediction. Using it as a
merge feature would be leakage -- it is unavailable on the test set. The bridge
evidence here is read from the probability map, which exists at inference time.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .metrics import pairwise_iou


# --------------------------------------------------------------------------- #
# 2.1 Candidate generation
# --------------------------------------------------------------------------- #


def generate_candidates(
    prob: np.ndarray,
    threshold: float = 0.30,
    min_area: int = 128,
    connectivity: int = 8,
) -> np.ndarray:
    """Threshold low and label connected components.

    Deliberately over-produces: a candidate that is rejected later costs nothing,
    a filament that never becomes a candidate is unrecoverable. ``min_area`` is
    well below the Phase 1 value (256) for the same reason.
    """
    binary = (prob > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros(prob.shape, np.int32)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    out = np.zeros(prob.shape, np.int32)
    next_id = 1
    for comp in range(1, n):
        if stats[comp, cv2.CC_STAT_AREA] < min_area:
            continue
        out[labels == comp] = next_id
        next_id += 1
    return out


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _principal_axis(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float, float]:
    """Return ``(angle, elongation, length)`` of a point cloud's principal axis.

    ``angle`` is in radians, axial (mod pi) -- a filament has no head or tail.
    ``elongation`` is sqrt(l1/l2) of the coordinate covariance, so 1.0 is a
    blob and 10 is a thin line.
    """
    if len(ys) < 3:
        return 0.0, 1.0, 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-9)
    major = evecs[:, int(np.argmax(evals))]
    angle = float(np.arctan2(major[1], major[0]))
    elongation = float(np.sqrt(evals.max() / evals.min()))
    length = float(4.0 * np.sqrt(evals.max()))  # ~2 sigma each side
    return angle, elongation, length


def _axial_alignment(a1: float, a2: float) -> float:
    """|cos| of the angle between two axial orientations, in [0, 1]."""
    return float(abs(np.cos(a1 - a2)))


def _edt_of_component(
    cand: np.ndarray, cid: int, margin: int = 60
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Distance transform of everything *except* component ``cid``, on a crop.

    Returns ``(crop_slice, edt, nearest_index)``. ``nearest_index`` is the
    ``(2, h, w)`` array of indices of the nearest component pixel, which gives
    the closest point on ``cid`` for any query pixel for free.
    """
    ys, xs = np.nonzero(cand == cid)
    h, w = cand.shape
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)
    sub = cand[y0:y1, x0:x1] == cid
    edt, idx = ndi.distance_transform_edt(~sub, return_indices=True)
    return (y0, y1, x0, x1), edt.astype(np.float32), idx


# --------------------------------------------------------------------------- #
# Per-candidate features
# --------------------------------------------------------------------------- #


def candidate_features(prob: np.ndarray, cand: np.ndarray, cid: int) -> dict:
    """Scalar features for one candidate.

    ``contrast`` is the mean probability in a 10 px annulus around the candidate
    subtracted from the candidate's own mean. A real filament sits on a dark
    background; a threshold artefact does not.
    """
    mask = cand == cid
    ys, xs = np.nonzero(mask)
    area = int(len(ys))
    p = prob[mask]

    angle, elongation, length = _principal_axis(ys, xs)

    # Annulus contrast.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dil = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    ring = dil & ~mask
    ring_mean = float(prob[ring].mean()) if ring.any() else 0.0

    h, w = prob.shape
    return {
        "area": area,
        "log_area": float(np.log1p(area)),
        "elongation": elongation,
        "length": length,
        "angle": angle,
        "mean_prob": float(p.mean()),
        "max_prob": float(p.max()),
        "p90_prob": float(np.percentile(p, 90)),
        "frac_above_05": float((p > 0.5).mean()),
        "contrast": float(p.mean()) - ring_mean,
        "ring_mean": ring_mean,
        "bbox_h": int(ys.max() - ys.min() + 1),
        "bbox_w": int(xs.max() - xs.min() + 1),
        "fill_ratio": area / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)),
        "touch_border": int(
            ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
        ),
        "cy": float(ys.mean()),
        "cx": float(xs.mean()),
    }


# --------------------------------------------------------------------------- #
# Per-pair merge features
# --------------------------------------------------------------------------- #


def pair_features(
    prob: np.ndarray,
    cand: np.ndarray,
    a: int,
    b: int,
    cache: dict[int, tuple],
    n_bridge: int = 24,
) -> dict | None:
    """Merge evidence for candidates ``a`` and ``b``.

    Returns ``None`` when the two components are further apart than the crop
    margin allows, i.e. when they are not merge candidates at all.

    The bridge is the straight segment between the closest pair of pixels. Its
    probability profile is the main signal: two fragments of one filament are
    joined by a stretch of moderately-high probability, whereas two genuinely
    separate filaments are joined by background.
    """
    if a not in cache:
        cache[a] = _edt_of_component(cand, a)
    (y0, y1, x0, x1), edt, idx = cache[a]

    ys_b, xs_b = np.nonzero(cand == b)
    inside = (ys_b >= y0) & (ys_b < y1) & (xs_b >= x0) & (xs_b < x1)
    if not inside.any():
        return None

    ys_in = ys_b[inside]
    xs_in = xs_b[inside]
    d = edt[ys_in - y0, xs_in - x0]
    j = int(np.argmin(d))
    gap = float(d[j])

    # Closest point on b, and the nearest point on a (free from the EDT indices).
    py_b, px_b = int(ys_in[j]), int(xs_in[j])
    py_a = int(idx[0, py_b - y0, px_b - x0]) + y0
    px_a = int(idx[1, py_b - y0, px_b - x0]) + x0

    # Bridge probability profile.
    ts = np.linspace(0.0, 1.0, n_bridge)
    by = np.clip(np.round(py_a + ts * (py_b - py_a)).astype(int), 0, prob.shape[0] - 1)
    bx = np.clip(np.round(px_a + ts * (px_b - px_a)).astype(int), 0, prob.shape[1] - 1)
    bp = prob[by, bx].astype(np.float32)

    fa = candidate_features(prob, cand, a)
    fb = candidate_features(prob, cand, b)

    # Direction of the bridge, and how well each fragment's own axis continues
    # into it. A filament that is split continues along its own axis.
    bridge_ang = float(np.arctan2(py_b - py_a, px_b - px_a))
    align_a = _axial_alignment(fa["angle"], bridge_ang)
    align_b = _axial_alignment(fb["angle"], bridge_ang)

    area_ratio = min(fa["area"], fb["area"]) / max(1, max(fa["area"], fb["area"]))
    centroid_dist = float(np.hypot(fa["cy"] - fb["cy"], fa["cx"] - fb["cx"]))

    return {
        "gap": gap,
        "log_gap": float(np.log1p(gap)),
        "bridge_mean": float(bp.mean()),
        "bridge_min": float(bp.min()),
        "bridge_max": float(bp.max()),
        "bridge_p25": float(np.percentile(bp, 25)),
        "bridge_frac_above_03": float((bp > 0.3).mean()),
        "bridge_frac_above_05": float((bp > 0.5).mean()),
        "bridge_dip": float(min(fa["mean_prob"], fb["mean_prob"]) - bp.min()),
        "axis_align": _axial_alignment(fa["angle"], fb["angle"]),
        "bridge_align_a": align_a,
        "bridge_align_b": align_b,
        "bridge_align_min": min(align_a, align_b),
        "area_ratio": area_ratio,
        "log_area_ratio": float(np.log1p(area_ratio)),
        "centroid_dist": centroid_dist,
        "elong_a": fa["elongation"],
        "elong_b": fb["elongation"],
        "elong_min": min(fa["elongation"], fb["elongation"]),
        "mean_prob_a": fa["mean_prob"],
        "mean_prob_b": fb["mean_prob"],
        "mean_prob_min": min(fa["mean_prob"], fb["mean_prob"]),
        "contrast_min": min(fa["contrast"], fb["contrast"]),
        "area_a": fa["area"],
        "area_b": fb["area"],
    }


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def label_candidates(
    cand: np.ndarray, gt_labels: np.ndarray, iou_thr: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate ``q`` and the GT instance it matches in each copy.

    ``q`` is the fraction of annotator copies in which the candidate has
    IoU > ``iou_thr`` with some GT instance. This is the quantity Phase 2.3
    regresses: a candidate matched by a fraction ``q`` of copies at mean IoU
    ``s`` raises PQ iff ``2 q s > PQ``, so at PQ 0.45 and s 0.7 anything with
    ``q > 0.32`` is worth keeping.

    Returns ``(q, best_iou, gt_id_per_copy)`` where ``gt_id_per_copy`` is
    ``(n_cand, n_copies)`` and holds the matched GT id or 0 for no match.
    """
    n_cand = int(cand.max())
    n_copies = gt_labels.shape[0]
    if n_cand == 0:
        return np.zeros(0), np.zeros((0, n_copies)), np.zeros((0, n_copies), np.int32)

    q = np.zeros(n_cand, np.float32)
    best = np.zeros((n_cand, n_copies), np.float32)
    gid = np.zeros((n_cand, n_copies), np.int32)

    for k in range(n_copies):
        gt = gt_labels[k]
        n_gt = int(gt.max())
        if n_gt == 0:
            continue
        iou = pairwise_iou(cand, gt, n_cand, n_gt)  # (n_cand, n_gt)
        j = iou.argmax(axis=1)
        v = iou[np.arange(n_cand), j]
        best[:, k] = v
        hit = v > iou_thr
        gid[hit, k] = j[hit] + 1
        q += hit

    q /= max(1, n_copies)
    return q, best, gid


def label_pairs(
    gid: np.ndarray, pairs: np.ndarray, min_frac: float = 0.5
) -> np.ndarray:
    """Whether each pair should be merged.

    A pair should merge when, in the majority of annotator copies, both
    candidates match the *same* GT instance. ``gid`` is ``(n_cand, n_copies)``
    from :func:`label_candidates`; ``pairs`` is ``(n_pairs, 2)`` of 1-based ids.
    """
    if len(pairs) == 0:
        return np.zeros(0, np.float32)
    a = pairs[:, 0] - 1
    b = pairs[:, 1] - 1
    ga = gid[a]
    gb = gid[b]
    same = (ga == gb) & (ga > 0)
    return (same.mean(axis=1) > min_frac).astype(np.float32)


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


class UnionFind:
    """Disjoint-set with path compression, for merging candidate ids."""

    def __init__(self, n: int):
        self.parent = list(range(n + 1))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def apply_merges(
    cand: np.ndarray, pairs: np.ndarray, scores: np.ndarray, threshold: float
) -> np.ndarray:
    """Merge candidate pairs whose score exceeds ``threshold``.

    Pairs are applied in decreasing score order so that the most confident
    merges win when a candidate has several suitors.
    """
    n = int(cand.max())
    uf = UnionFind(n)
    order = np.argsort(-scores)
    for i in order:
        if scores[i] < threshold:
            break
        uf.union(int(pairs[i, 0]), int(pairs[i, 1]))

    lut = np.zeros(n + 1, np.int32)
    next_id = 1
    for cid in range(1, n + 1):
        r = uf.find(cid)
        if lut[r] == 0:
            lut[r] = next_id
            next_id += 1
        lut[cid] = lut[r]
    return lut[cand]
