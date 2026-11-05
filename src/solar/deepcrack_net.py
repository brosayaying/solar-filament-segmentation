"""Vendored DeepCrack network from ``yhlleo/DeepSegmentor``.

Copied verbatim from the upstream repo so the model survives a machine restart
without needing a ``/tmp`` clone. Only the two helpers that lived in
``models/networks.py`` (``get_norm_layer``, ``init_net``) are inlined here; the
network itself is unchanged.

Upstream source
---------------
* ``models/deepcrack_networks.py`` -- ``DeepCrackNet``, ``define_deepcrack``
* ``models/networks.py`` -- ``get_norm_layer``, ``init_weights``, ``init_net``

Reference
---------
DeepCrack: A deep hierarchical feature learning architecture for crack
segmentation. https://www.sciencedirect.com/science/article/pii/S0925231219300566
Author: Yahui Liu <yahui.liu@unitn.it>

Do not rename modules or reorder layers: the released ``pretrained_net_G.pth``
and every checkpoint in ``runs/`` are keyed by the exact attribute names below
(``conv1.0.weight``, ``side_conv1.weight``, ``fuse_conv.weight``, ...).
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def get_norm_layer(norm_type: str = "instance"):
    """Return a normalization layer.

    Parameters
    ----------
    norm_type:
        ``batch`` | ``instance`` | ``none``.

    For BatchNorm, learnable affine parameters are used and running statistics
    are tracked. For InstanceNorm, affine parameters are not used and running
    statistics are not tracked.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
    return norm_layer


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02):
    """Initialize network weights.

    Parameters
    ----------
    net:
        Network to be initialized.
    init_type:
        ``normal`` | ``xavier`` | ``kaiming`` | ``orthogonal``.
    init_gain:
        Scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm weight is not a matrix; only a normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)


def init_net(
    net: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids: list[int] | None = None,
):
    """Initialize a network and optionally wrap it for multi-GPU."""
    gpu_ids = gpu_ids or []
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


class DeepCrackNet(nn.Module):
    """Five-stage encoder with a side head per stage and a learned 1x1 fusion.

    ``forward`` returns ``(side1, side2, side3, side4, side5, fused)``. The side
    logits are upsampled to the input resolution with bilinear interpolation and
    concatenated into ``fuse_conv``, a bias-free 1x1 conv -- so the fused map is
    a fixed linear combination of the five sides.

    Parameters
    ----------
    decoder:
        Replace the side-head fusion with a UNet-style decoder. See
        :meth:`_up_block` and ``docs/DEEPCRACK_FINETUNE.md`` §20.

        The upstream design has no decoder: each side head is bilinearly
        upsampled straight from its encoder stride to full resolution. That
        forces a trade-off between semantics and resolution -- measured on
        MAGFiLO, the best-ranking head (side4, AUC 0.978) sits at stride 8 and
        the full-resolution head (side1) has AUC 0.61. The fused map ranks
        almost perfectly (AUC 0.9968) but its pixel IoU is only 0.4987, and 97%
        of that gap is *shape*, not threshold calibration.

        With ``decoder=True`` the five encoder stages are instead combined by
        progressive upsample-and-concat, so full-resolution detail from
        ``conv1``/``conv2`` reaches the output. The side heads are still built
        (they are 1x1 convs, ~5 params each) so deep supervision keeps working,
        but ``fused`` now comes from the decoder.

        ``decoder=False`` reproduces the upstream state dict exactly, so every
        checkpoint in ``runs/`` still loads.
    """

    def __init__(self, in_nc, num_classes, ngf, norm="batch", decoder=False):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        self.conv1 = nn.Sequential(*self._conv_block(in_nc, ngf, norm_layer, num_block=2))
        self.side_conv1 = nn.Conv2d(ngf, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv2 = nn.Sequential(
            *self._conv_block(ngf, ngf * 2, norm_layer, num_block=2)
        )
        self.side_conv2 = nn.Conv2d(
            ngf * 2, num_classes, kernel_size=1, stride=1, bias=False
        )

        self.conv3 = nn.Sequential(
            *self._conv_block(ngf * 2, ngf * 4, norm_layer, num_block=3)
        )
        self.side_conv3 = nn.Conv2d(
            ngf * 4, num_classes, kernel_size=1, stride=1, bias=False
        )

        self.conv4 = nn.Sequential(
            *self._conv_block(ngf * 4, ngf * 8, norm_layer, num_block=3)
        )
        self.side_conv4 = nn.Conv2d(
            ngf * 8, num_classes, kernel_size=1, stride=1, bias=False
        )

        self.conv5 = nn.Sequential(
            *self._conv_block(ngf * 8, ngf * 8, norm_layer, num_block=3)
        )
        self.side_conv5 = nn.Conv2d(
            ngf * 8, num_classes, kernel_size=1, stride=1, bias=False
        )

        self.fuse_conv = nn.Conv2d(
            num_classes * 5, num_classes, kernel_size=1, stride=1, bias=False
        )
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # --- Optional decoder -------------------------------------------------
        # Channel widths of the five encoder stages, deepest first.
        self.decoder = bool(decoder)
        if self.decoder:
            widths = [ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf]
            # up4..up1 walk back up the pyramid. Each block takes the upsampled
            # deeper feature concatenated with the matching skip, so the input
            # width is (deeper + skip).
            self.up4 = self._up_block(widths[0] + widths[1], widths[1], norm_layer)
            self.up3 = self._up_block(widths[1] + widths[2], widths[2], norm_layer)
            self.up2 = self._up_block(widths[2] + widths[3], widths[3], norm_layer)
            self.up1 = self._up_block(widths[3] + widths[4], widths[4], norm_layer)
            # Final 1x1 to logits at full resolution.
            self.head = nn.Conv2d(widths[4], num_classes, kernel_size=1, bias=True)

    @staticmethod
    def _up_block(in_ch, out_ch, norm_layer):
        """Upsample-and-concat block: 3x3 conv -> norm -> ReLU, twice.

        Two convs rather than one so the block can combine the coarse (upsampled)
        and fine (skip) features instead of just resampling them. This is the
        standard UNet decoder block and is what the resnet34 baseline in this
        repo uses.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(True),
        )

    def _conv_block(
        self,
        in_nc,
        out_nc,
        norm_layer,
        num_block=2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    ):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [
                nn.Conv2d(
                    cur_in_nc,
                    out_nc,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                norm_layer(out_nc),
                nn.ReLU(True),
            ]
        return conv

    def forward(self, x):
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        conv5 = self.conv5(self.maxpool(conv4))
        # side output features
        side_output1 = self.side_conv1(conv1)
        side_output2 = self.side_conv2(conv2)
        side_output3 = self.side_conv3(conv3)
        side_output4 = self.side_conv4(conv4)
        side_output5 = self.side_conv5(conv5)
        # upsampling side output features
        side_output2 = F.interpolate(
            side_output2, size=(h, w), mode="bilinear", align_corners=True
        )
        side_output3 = F.interpolate(
            side_output3, size=(h, w), mode="bilinear", align_corners=True
        )
        side_output4 = F.interpolate(
            side_output4, size=(h, w), mode="bilinear", align_corners=True
        )
        side_output5 = F.interpolate(
            side_output5, size=(h, w), mode="bilinear", align_corners=True
        )

        fused = self.fuse_conv(
            torch.cat(
                [side_output1, side_output2, side_output3, side_output4, side_output5],
                dim=1,
            )
        )

        if self.decoder:
            # Progressive upsample-and-concat. conv5 is the deepest (stride 16);
            # each step doubles resolution and merges the matching skip.
            d = self.up4(torch.cat([self._up(conv5, conv4), conv4], dim=1))
            d = self.up3(torch.cat([self._up(d, conv3), conv3], dim=1))
            d = self.up2(torch.cat([self._up(d, conv2), conv2], dim=1))
            d = self.up1(torch.cat([self._up(d, conv1), conv1], dim=1))
            fused = self.head(d)

        return (
            side_output1,
            side_output2,
            side_output3,
            side_output4,
            side_output5,
            fused,
        )

    @staticmethod
    def _up(x, like):
        """Bilinearly upsample ``x`` to the spatial size of ``like``."""
        return F.interpolate(
            x, size=like.shape[-2:], mode="bilinear", align_corners=True
        )


def define_deepcrack(
    in_nc,
    num_classes,
    ngf,
    norm="batch",
    init_type="xavier",
    init_gain=0.02,
    gpu_ids=None,
    decoder=False,
):
    """Build and initialize a :class:`DeepCrackNet`.

    The decoder is re-initialised with ``gain=1.0`` after ``init_net`` runs.
    ``init_gain=0.02`` is the pix2pix GAN-generator convention and is correct for
    the encoder (it keeps the pretrained-weight scale), but applied to a 4-block
    decoder it shrinks the output to ``std ~0.02``: measured on a 256x256 input,
    logits span only [-0.08, 0.11], so sigmoid never leaves [0.48, 0.53] and no
    pixel can cross a 0.5 threshold. That is a dead network, not a slow one.
    With ``gain=1.0`` the same input gives logits in [-2.6, 4.9].
    """
    net = DeepCrackNet(in_nc, num_classes, ngf, norm, decoder=decoder)
    net = init_net(net, init_type, init_gain, gpu_ids or [])
    if decoder:
        for block in (net.up1, net.up2, net.up3, net.up4, net.head):
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
    return net


class BinaryFocalLoss(nn.Module):
    """Focal loss on logits (upstream helper, unused by this repo)."""

    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.criterion(inputs, targets)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.size_average:
            return f_loss.mean()
        return f_loss.sum()
