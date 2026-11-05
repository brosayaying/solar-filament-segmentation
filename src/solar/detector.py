"""Stage 1: anchor-free bounding-box detector for filaments.

Why a detector at all
---------------------
The single-stage semantic model has no notion of instance identity, which is why
PQ's recognition quality (RQ) is the ceiling: connected components merge touching
filaments and watershed over-splits them. A detector gives explicit instance
proposals, so stage 2 only has to answer "which pixels inside this box belong to
this filament" -- a much easier question than "how many filaments are here".

Design: CenterNet-style
-----------------------
Filaments are thin and elongated, so anchor-based detectors (YOLO/RetinaNet)
would need many aspect-ratio anchors and still match poorly. CenterNet predicts a
single center point per object plus its width/height, which is aspect-ratio
agnostic and needs no NMS.

Outputs at stride 4:
    heatmap  (1, H/4, W/4)  -- Gaussian splat at each box center
    size     (2, H/4, W/4)  -- log(width), log(height) in input pixels
    offset   (2, H/4, W/4)  -- sub-pixel center correction

Losses:
    heatmap -- CornerNet focal loss (handles the extreme foreground/background
               imbalance: filaments occupy ~0.36% of pixels)
    size    -- L1, applied only at positive locations
    offset  -- L1, applied only at positive locations

Crack-detection inspiration
---------------------------
Thin-structure detectors (DeepCrack, CrackFormer) keep resolution high and use
dilated convolutions to grow the receptive field without downsampling. The
encoder here stops at stride 16 and the decoder uses dilated bottleneck blocks,
so a 1-2 px filament is never lost to pooling.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stride 4 is the right default. Stride 2 was tried and is clearly worse: it
# quadruples the heatmap cell count, pushing the negative/positive ratio from
# 7,282 to 29,127, and the focal loss collapses (mask-hit recall 0.10 vs 0.59).
# The extra resolution is not worth the imbalance for a single-point target.
STRIDE = 4


class DilatedBlock(nn.Module):
    """Residual block with dilated 3x3 convs: wide receptive field, no downsampling."""

    def __init__(self, ch: int, dilation: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class FilamentDetector(nn.Module):
    """CenterNet-style detector for filament instances.

    Two modes:

    ``with_size=True`` (box mode)
        heatmap + size + offset. Predicts a bounding box per filament.

    ``with_size=False`` (spine mode)
        heatmap + offset only. Predicts a single point per filament, placed at
        the spine midpoint. This is the recommended mode: a box around a thin
        curved filament is ~73% background even when perfect, so the size head
        has almost nothing to learn from and systematically undershoots. A
        point has no such failure mode, and the spine midpoint lands on the
        filament 99.9% of the time versus 50.8% for the bbox centre.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_chans: int = 1,
        decoder_dim: int = 96,
        dilations: tuple[int, ...] = (2, 4),
        with_size: bool = True,
        out_stride: int = STRIDE,
    ) -> None:
        super().__init__()
        self.with_size = with_size
        self.out_stride = out_stride
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, in_chans=in_chans
        )
        enc_ch = self.encoder.feature_info.channels()
        self._enc_ch = list(enc_ch)
        n = len(enc_ch)

        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_dim, 1) for c in enc_ch])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_ch[-1], decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            *[DilatedBlock(decoder_dim, d) for d in dilations],
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(decoder_dim * 2, decoder_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n - 1)
            ]
        )

        # Upsample to stride 4 (the encoder's finest level is usually stride 4).
        self.up = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(decoder_dim, 1, 1)
        self.offset = nn.Conv2d(decoder_dim, 2, 1)
        self.size = nn.Conv2d(decoder_dim, 2, 1) if with_size else None

        self.heatmap.bias.data.fill_(-4.6)  # prior p ~= 0.01, matches sparsity

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        feats = [
            f.permute(0, 3, 1, 2) if f.shape[1] != self._enc_ch[i] else f
            for i, f in enumerate(feats)
        ]

        y = self.bottleneck(feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            f = feats[i]
            y = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False)
            y = self.blocks[i](torch.cat([y, self.lateral[i](f)], 1))

        # The decoder already walks up to the finest encoder level (stride 2 for
        # resnet). Predicting there instead of at stride 4 doubles localisation
        # resolution, but it also quadruples the number of background cells and
        # the focal loss collapses -- see the STRIDE comment above. Keep 4.
        target = (x.shape[-2] // self.out_stride, x.shape[-1] // self.out_stride)
        if y.shape[-2:] != target:
            y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
        y = self.up(y)

        out = {"heatmap": self.heatmap(y), "offset": self.offset(y)}
        if self.size is not None:
            out["size"] = self.size(y)
        return out


def build_detector(cfg: dict[str, Any]) -> nn.Module:
    d = cfg.get("detector", {})
    return FilamentDetector(
        backbone=d.get("backbone", "resnet34"),
        pretrained=d.get("pretrained", True),
        in_chans=d.get("in_chans", 1),
        decoder_dim=d.get("decoder_dim", 96),
        dilations=tuple(d.get("dilations", (2, 4))),
        with_size=d.get("with_size", True),
        out_stride=d.get("out_stride", STRIDE),
    )
