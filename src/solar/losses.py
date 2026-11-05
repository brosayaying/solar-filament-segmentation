"""Segmentation losses.

The PQ diagnosis (``docs/PQ_DIAGNOSIS.md``) found the errors are *boundary
precision*, not detection: median best IoU per GT instance is 0.604, 34% of
instances land in 0.5-0.7 and **0% exceed 0.9**. Plain BCE + soft Dice barely
penalises a contour that is 5 px off on a 30 px wide filament, because the loss
is dominated by the thousands of interior and background pixels.

``boundary_band_loss`` adds a Dice term restricted to a narrow band around the
ground-truth contour, so contour disagreement is no longer diluted by pixels
whose label is not in doubt.

Both terms are pure ``max_pool2d`` morphology, so they are cheap, differentiable,
and safe under AMP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss over the batch."""
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(2, 3)) + eps
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce_weight * bce + dice_weight * dice_loss(logits, target)


# --------------------------------------------------------------------------- #
# Morphology helpers (differentiable, AMP-safe)
# --------------------------------------------------------------------------- #


def dilate(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


def erode(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


def boundary_band(target: torch.Tensor, radius: int, k: int = 3) -> torch.Tensor:
    """Band of half-width ``radius`` around the target contour.

    Dilate minus erode gives the contour; dilating that by ``radius`` gives the
    band. For a thin structure the band covers most of the object, which is
    exactly right -- a 30 px filament has no interior far from its edge.
    """
    edge = dilate(target, k) - erode(target, k)
    return dilate(edge, 2 * radius + 1)


def boundary_band_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    radius: int = 3,
    eps: float = 1.0,
) -> torch.Tensor:
    """Dice restricted to a band around the GT contour."""
    prob = torch.sigmoid(logits)
    band = boundary_band(target, radius).clamp(0.0, 1.0)
    num = 2 * (prob * target * band).sum(dim=(2, 3)) + eps
    den = (prob * band).sum(dim=(2, 3)) + (target * band).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def boundary_weighted_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    alpha: float = 1.0,
    radius: int = 3,
) -> torch.Tensor:
    """BCE + Dice with extra weight on pixels near the GT contour.

    ``alpha`` controls how much the band is emphasised (band weight = 1+alpha,
    interior = 1). Softer than ``boundary_band_loss``: keeps the global IoU
    pressure of plain Dice while pulling harder on the contour.
    """
    band = boundary_band(target, radius).clamp(0.0, 1.0)
    weight = 1.0 + alpha * band

    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight
    ).mean()

    prob = torch.sigmoid(logits)
    num = 2 * (prob * target * weight).sum(dim=(2, 3)) + 1.0
    den = (prob * weight).sum(dim=(2, 3)) + (target * weight).sum(dim=(2, 3)) + 1.0
    dice = 1 - (num / den).mean()

    return bce_weight * bce + dice_weight * dice


# --------------------------------------------------------------------------- #
# Crop-border handling
# --------------------------------------------------------------------------- #


def border_weight(
    target: torch.Tensor,
    margin: int = 16,
    floor: float = 0.0,
) -> torch.Tensor:
    """Per-pixel weight that fades to ``floor`` within ``margin`` px of the crop edge.

    Filaments are long and thin, so a 256 px crop almost always truncates one.
    Without this, the model is taught that filaments terminate abruptly at the
    crop boundary, which shows up as an SQ regression -- and SQ is already where
    the error concentrates (median best IoU 0.604, see ``docs/PQ_DIAGNOSIS.md``).

    The ramp is linear from ``floor`` at the edge to 1.0 at ``margin`` px in.
    ``floor=0`` fully ignores the border band; a small floor (e.g. 0.1) keeps a
    weak signal so the model still learns that the region is not filament.
    """
    if margin <= 0:
        return torch.ones_like(target)
    h, w = target.shape[-2:]
    m = min(margin, h // 2, w // 2)
    if m <= 0:
        return torch.ones_like(target)

    ramp = torch.linspace(0.0, 1.0, m + 1, device=target.device, dtype=target.dtype)[1:]
    wy = torch.ones(h, device=target.device, dtype=target.dtype)
    wx = torch.ones(w, device=target.device, dtype=target.dtype)
    wy[:m] = ramp
    wy[h - m :] = ramp.flip(0)
    wx[:m] = ramp
    wx[w - m :] = ramp.flip(0)

    w2 = wy[:, None] * wx[None, :]
    return floor + (1.0 - floor) * w2


def border_aware_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    margin: int = 16,
    floor: float = 0.0,
) -> torch.Tensor:
    """BCE + Dice with pixels near the crop border down-weighted.

    Same shape as ``bce_dice_loss`` but every pixel is scaled by
    ``border_weight``, so truncated filaments at the crop edge do not dominate
    the gradient.
    """
    wgt = border_weight(target, margin=margin, floor=floor)

    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * wgt
    ).sum() / wgt.sum().clamp_min(1.0)

    prob = torch.sigmoid(logits)
    num = 2 * (prob * target * wgt).sum(dim=(2, 3)) + 1.0
    den = (prob * wgt).sum(dim=(2, 3)) + (target * wgt).sum(dim=(2, 3)) + 1.0
    dice = 1 - (num / den).mean()

    return bce_weight * bce + dice_weight * dice


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def build_loss(cfg: dict):
    """Return ``f(logits, target) -> loss`` for the configured loss name.

    Config keys (all optional except ``loss``):
      loss             "bce_dice" (default) | "boundary" | "boundary_weighted"
      bce_weight       default 0.5
      dice_weight      default 0.5
      boundary_weight  weight of the band term for "boundary" (default 0.5)
      boundary_radius  band half-width in px (default 3)
      boundary_alpha   band emphasis for "boundary_weighted" (default 1.0)
    """
    t = cfg["train"]
    name = t.get("loss", "bce_dice")
    radius = int(t.get("boundary_radius", 3))

    if name == "bce_dice":
        return lambda logits, target: bce_dice_loss(
            logits, target, t.get("bce_weight", 0.5), t.get("dice_weight", 0.5)
        )

    if name == "boundary":
        bw = float(t.get("boundary_weight", 0.5))
        base_w = 1.0 - bw

        def loss_fn(logits, target):
            base = bce_dice_loss(
                logits, target, t.get("bce_weight", 0.5), t.get("dice_weight", 0.5)
            )
            band = boundary_band_loss(logits, target, radius=radius)
            return base_w * base + bw * band

        return loss_fn

    if name == "boundary_weighted":
        return lambda logits, target: boundary_weighted_loss(
            logits,
            target,
            t.get("bce_weight", 0.5),
            t.get("dice_weight", 0.5),
            float(t.get("boundary_alpha", 1.0)),
            radius,
        )

    if name == "border_aware":
        margin = int(t.get("border_margin", 16))
        floor = float(t.get("border_floor", 0.0))
        return lambda logits, target: border_aware_loss(
            logits,
            target,
            t.get("bce_weight", 0.5),
            t.get("dice_weight", 0.5),
            margin,
            floor,
        )

    raise ValueError(f"unknown loss {name!r}")
