"""Backbone-agnostic segmentation model.

Any ``timm`` model that supports ``features_only=True`` can be dropped in as an
encoder, which makes backbone sweeps a one-line config change::

    model.backbone=resnet34
    model.backbone=convnext_tiny
    model.backbone=efficientnet_b0
    model.backbone=mit_b0
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TimmUNet(nn.Module):
    """UNet decoder over an arbitrary ``timm`` feature pyramid.

    The encoder returns features from high resolution (stride 2-4) to low
    resolution. The decoder walks back up, concatenating lateral projections.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_chans: int = 1,
        decoder_dim: int = 128,
        out_ch: int = 1,
        dropout: float = 0.0,
        img_size: int | None = None,
        aux_spine: bool = False,
        stem_ch: int = 0,
    ) -> None:
        super().__init__()
        # Windowed transformers (swin, ...) hard-assert that the input matches
        # the size the model was built for, and their features_only path ignores
        # dynamic_img_size. Passing img_size at creation is the only reliable
        # fix. Conv backbones accept and ignore it.
        kwargs: dict[str, Any] = {"in_chans": in_chans}
        if img_size is not None:
            kwargs["img_size"] = img_size
        try:
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained, features_only=True, **kwargs
            )
        except TypeError:
            kwargs.pop("img_size", None)
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained, features_only=True, **kwargs
            )
        enc_channels = self.encoder.feature_info.channels()
        self._enc_ch = list(enc_channels)
        self.n_levels = len(enc_channels)

        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_dim, 1) for c in enc_channels])
        self.bottleneck = ConvBlock(enc_channels[-1], decoder_dim)
        self.blocks = nn.ModuleList(
            [ConvBlock(decoder_dim * 2, decoder_dim) for _ in range(self.n_levels - 1)]
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Stride-1 stem path.
        #
        # The finest feature a timm encoder exposes is stride 2 (resnet) or
        # stride 4 (convnext, efficientnet, pvt). A median filament is 11 px
        # wide, so reconstructing it from stride-4 features alone throws away
        # most of the boundary signal -- this is the same failure that capped
        # DeepCrack at PQ 0.35 (no decoder, best head at stride 8) versus the
        # UNet's 0.44. A cheap conv on the *input* gives the decoder a
        # full-resolution path, so the fine detail is refined rather than
        # hallucinated. stem_ch=0 disables it, keeping old checkpoints loadable.
        self.stem_ch = int(stem_ch)
        if self.stem_ch > 0:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, self.stem_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.stem_ch),
                nn.SiLU(inplace=True),
            )
            head_in = decoder_dim + self.stem_ch
        else:
            self.stem = None
            head_in = decoder_dim

        self.head = nn.Conv2d(head_in, out_ch, 1)
        # Optional auxiliary spine head. Predicting the centerline forces the
        # encoder to learn filament identity, which the semantic mask alone does
        # not require. Disabled by default so existing checkpoints still load.
        self.aux_spine = bool(aux_spine)
        self.spine_head = nn.Conv2d(head_in, 1, 1) if self.aux_spine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = self.encoder(x)

        # Some backbones (swin and other windowed transformers) emit NHWC
        # feature maps; conv backbones emit NCHW. Normalise to NCHW.
        feats = [f.permute(0, 3, 1, 2) if f.shape[1] != self._enc_ch[i] else f
                 for i, f in enumerate(feats)]

        y = self.bottleneck(feats[-1])
        for i in range(self.n_levels - 2, -1, -1):
            f = feats[i]
            y = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False)
            y = self.blocks[i](torch.cat([y, self.lateral[i](f)], 1))

        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        if self.stem is not None:
            y = torch.cat([y, self.stem(x)], 1)
        y = self.dropout(y)
        mask_logits = self.head(y)
        if self.spine_head is not None:
            return mask_logits, self.spine_head(y)
        return mask_logits


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoUNet(nn.Module):
    """Frozen DINOv2 encoder with a multi-scale conv decoder.

    A plain ``TimmUNet`` over DINOv2 does not work: ``features_only`` returns
    three levels *all at stride 14*, so there is no fine scale to skip-connect
    to and the decoder is just a 14x bilinear upsample. This model instead:

    * taps **different blocks** (default 2/5/8/11) for a pseudo-pyramid of
      semantic depth at a single resolution,
    * builds the fine scales from a small **conv stem on the raw image**, which
      is where sub-token detail has to come from -- a filament is ~30 px wide,
      i.e. only ~2 tokens across at stride 14,
    * decodes 14 -> 7 -> 4 -> 2 -> 1, concatenating the stem at each step.

    The encoder is frozen by default. That is deliberate: this is the cheap
    ablation that answers "are the pretrained features worth anything here"
    before paying for full fine-tuning.
    """

    def __init__(
        self,
        backbone: str = "vit_small_patch14_dinov2",
        pretrained: bool = True,
        decoder_dim: int = 128,
        out_ch: int = 1,
        dropout: float = 0.0,
        freeze_encoder: bool = True,
        layers: tuple[int, ...] = (2, 5, 8, 11),
        stem_dim: int = 64,
        aux_spine: bool = False,
        data_mean: float = 0.5,
        data_std: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            dynamic_img_size=True,
        )
        self.patch = int(self.encoder.patch_embed.patch_size[0])
        enc_dim = int(self.encoder.embed_dim)
        self.layers = tuple(layers)
        self.freeze_encoder = bool(freeze_encoder)
        self.data_mean = float(data_mean)
        self.data_std = float(data_std)

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.register_buffer(
            "img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        # Fuse the tapped blocks into one stride-14 feature map.
        self.fuse = nn.Sequential(
            nn.Conv2d(enc_dim * len(self.layers), decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

        # Conv stem on the raw image supplies the fine scales.
        self.stem2 = nn.Sequential(
            nn.Conv2d(1, stem_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True),
        )
        self.stem4 = nn.Sequential(
            nn.Conv2d(stem_dim, stem_dim * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim * 2),
            nn.ReLU(inplace=True),
        )

        # 14 -> 7 -> 4 -> 2 -> 1. Widths taper as resolution grows: a full
        # 128-channel ConvBlock at 1024x1024 costs ~1 GB per tensor per image,
        # which is what made the first version OOM at batch 4.
        self.up7 = ConvBlock(decoder_dim, decoder_dim)
        self.up4 = ConvBlock(decoder_dim + stem_dim * 2, decoder_dim)
        self.up2 = ConvBlock(decoder_dim + stem_dim, decoder_dim // 2)
        self.up1 = ConvBlock(decoder_dim // 2 + 1, decoder_dim // 4)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(decoder_dim // 4, out_ch, 1)
        self.aux_spine = bool(aux_spine)
        self.spine_head = nn.Conv2d(decoder_dim // 4, 1, 1) if self.aux_spine else None

        self._captured: dict[int, torch.Tensor] = {}
        for idx in self.layers:
            self.encoder.blocks[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            self._captured[idx] = output

        return hook

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode regardless of the outer mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        """Run the ViT and return the tapped blocks as NCHW maps."""
        self._captured.clear()
        self.encoder.forward_features(rgb)
        gh = rgb.shape[-2] // self.patch
        gw = rgb.shape[-1] // self.patch
        n_prefix = int(getattr(self.encoder, "num_prefix_tokens", 0))
        maps = []
        for idx in self.layers:
            t = self._captured[idx]  # (B, n_prefix + gh*gw, C)
            t = t[:, n_prefix:]
            b, n, c = t.shape
            if n != gh * gw:
                raise RuntimeError(f"block {idx}: {n} tokens != {gh}x{gw}")
            maps.append(t.reshape(b, gh, gw, c).permute(0, 3, 1, 2).contiguous())
        return maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Invert the dataset normalisation to recover gray in [0, 1].
        g = (x * self.data_std + self.data_mean).clamp(0, 1)

        # The ViT asserts divisibility by the patch size.
        ph = (-h) % self.patch
        pw = (-w) % self.patch
        gp = F.pad(g, (0, pw, 0, ph), mode="replicate") if (ph or pw) else g

        rgb = gp.repeat(1, 3, 1, 1)
        rgb = (rgb - self.img_mean) / self.img_std

        if self.freeze_encoder:
            with torch.no_grad():
                maps = self._encode(rgb)
        else:
            maps = self._encode(rgb)

        d = self.fuse(torch.cat(maps, dim=1))

        s2 = self.stem2(g)
        s4 = self.stem4(s2)

        d = self.up7(F.interpolate(d, size=(d.shape[-2] * 2, d.shape[-1] * 2), mode="bilinear", align_corners=False))
        d = self.up4(torch.cat([F.interpolate(d, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], 1))
        d = self.up2(torch.cat([F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        d = self.up1(torch.cat([F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False), g], 1))

        d = self.dropout(d)
        mask_logits = self.head(d)
        if self.spine_head is not None:
            return mask_logits, self.spine_head(d)
        return mask_logits


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Instantiate the model described by ``cfg["model"]``."""
    m = cfg["model"]
    if m.get("arch", "unet") == "deepcrack":
        from .deepcrack import DeepCrack

        return DeepCrack(
            in_chans=m.get("in_chans", 1),
            ngf=m.get("ngf", 64),
            pretrained=m.get("pretrained", True),
            weights=m.get("pretrained_weights"),
            deep_supervision=m.get("deep_supervision", False),
            decoder=m.get("decoder", False),
            pretrained_encoder=m.get("pretrained_encoder", False),
        )
    if m.get("arch", "unet") == "dino_unet":
        return DinoUNet(
            backbone=m.get("backbone", "vit_small_patch14_dinov2"),
            pretrained=m.get("pretrained", True),
            decoder_dim=m.get("decoder_dim", 128),
            out_ch=m.get("out_ch", 1),
            dropout=m.get("dropout", 0.0),
            freeze_encoder=m.get("freeze_encoder", True),
            layers=tuple(m.get("layers", (2, 5, 8, 11))),
            stem_dim=m.get("stem_dim", 64),
            aux_spine=m.get("aux_spine", False),
            data_mean=cfg["data"].get("mean", 0.5),
            data_std=cfg["data"].get("std", 0.25),
        )
    return TimmUNet(
        backbone=m.get("backbone", "resnet34"),
        pretrained=m.get("pretrained", True),
        in_chans=m.get("in_chans", 1),
        decoder_dim=m.get("decoder_dim", 128),
        out_ch=m.get("out_ch", 1),
        dropout=m.get("dropout", 0.0),
        img_size=m.get("img_size"),
        aux_spine=m.get("aux_spine", False),
        stem_ch=m.get("stem_ch", 0),
    )
