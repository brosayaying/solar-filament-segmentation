"""CenterNet target encoding and losses for the stage-1 box detector.

Targets are built on the fly from COCO bboxes. Each filament contributes:
  * a Gaussian splat on the heatmap centred at its box centre
  * log(w), log(h) at that centre cell
  * the sub-pixel offset between the centre and its cell

Only the single centre cell of each object is positive, so the size/offset losses
are masked to those cells. The heatmap uses CornerNet's focal loss, which is
designed for the extreme imbalance here (filaments are ~0.36% of pixels).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

STRIDE = 4


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """CornerNet's radius rule: how far a shifted centre can move and still keep
    IoU >= min_overlap with the true box."""
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(max(0.0, b1**2 - 4 * a1 * c1))) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 - np.sqrt(max(0.0, b2**2 - 4 * a2 * c2))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(max(0.0, b3**2 - 4 * a3 * c3))) / 2
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Splat a 2D Gaussian in place, taking the elementwise max."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (2 * sigma * sigma))
    g[g < 1e-6 * g.max()] = 0

    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    masked = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    mg = g[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(masked, mg, out=masked)


def encode_targets(
    boxes: list[np.ndarray],
    height: int,
    width: int,
    stride: int = STRIDE,
    centers: list[np.ndarray] | None = None,
    with_size: bool = True,
) -> dict[str, torch.Tensor]:
    """Build heatmap/offset (and optionally size) targets for one image.

    ``boxes`` is a list of ``[x, y, w, h]`` in input pixels.

    ``centers`` optionally overrides the heatmap centre for each box. This
    matters a lot here: filaments are thin and curved, so the *bounding-box*
    centre lands on the filament only ~51% of the time (median miss 6 px).
    CenterNet's focal loss treats the centre cell as the sole positive, so half
    the objects would be supervised at a background pixel. Passing the spine
    midpoint instead puts the positive on the filament 99.9% of the time.

    ``with_size=False`` skips the size targets entirely (spine mode).
    """
    hm_h, hm_w = height // stride, width // stride
    heatmap = np.zeros((hm_h, hm_w), np.float32)
    offset = np.zeros((2, hm_h, hm_w), np.float32)
    mask = np.zeros((hm_h, hm_w), np.float32)
    size = np.zeros((2, hm_h, hm_w), np.float32) if with_size else None

    for i, box in enumerate(boxes):
        x, y, bw, bh = [float(v) for v in box]
        if bw <= 0 or bh <= 0:
            continue
        if centers is not None and i < len(centers):
            px, py = float(centers[i][0]), float(centers[i][1])
        else:
            px, py = x + bw / 2, y + bh / 2
        cx, cy = px / stride, py / stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < hm_w and 0 <= iy < hm_h):
            continue

        radius = max(1, int(gaussian_radius(bh / stride, bw / stride)))
        draw_gaussian(heatmap, ix, iy, radius)

        if size is not None:
            size[0, iy, ix] = np.log(max(bw, 1.0))
            size[1, iy, ix] = np.log(max(bh, 1.0))
        offset[0, iy, ix] = cx - ix
        offset[1, iy, ix] = cy - iy
        mask[iy, ix] = 1.0

    out = {
        "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
        "offset": torch.from_numpy(offset),
        "mask": torch.from_numpy(mask).unsqueeze(0),
    }
    if size is not None:
        out["size"] = torch.from_numpy(size)
    return out


def focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CornerNet penalty-reduced focal loss on raw logits."""
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, beta)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg

    n_pos = pos.sum()
    if n_pos == 0:
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def detector_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    size_weight: float = 0.1,
    offset_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total detector loss plus a dict of detached components for logging.

    The size term is skipped when the model has no size head (spine mode).
    """
    hm = focal_loss(pred["heatmap"], target["heatmap"])
    mask = target["mask"]

    n_pos = mask.sum().clamp(min=1.0)
    off_l1 = (F.l1_loss(pred["offset"], target["offset"], reduction="none") * mask).sum() / n_pos

    total = hm + offset_weight * off_l1
    comps = {"heatmap": float(hm.detach()), "offset": float(off_l1.detach())}

    if "size" in pred and "size" in target:
        size_l1 = (F.l1_loss(pred["size"], target["size"], reduction="none") * mask).sum() / n_pos
        total = total + size_weight * size_l1
        comps["size"] = float(size_l1.detach())

    return total, comps
