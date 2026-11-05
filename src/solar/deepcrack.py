"""DeepCrack (``yhlleo/DeepSegmentor``) as a drop-in model for this repo.

The zero-shot probe (``scripts/deepcrack_infer.py``, ``docs/DEEPCRACK/``) showed
the pretrained network fires on the solar limb and almost nowhere else: PQ 0.0,
TP 0, FP 921, FN 265, and in-disk AUC 0.47 under the upstream ``[-1, 1]``
normalisation. That is a distribution mismatch, not a capacity limit, so the
weights are worth finetuning rather than discarding.

This wrapper exists so the network can be trained by the repo's normal
``solar.train`` loop instead of a bespoke script. Two deliberate choices:

* **Grayscale stem.** The dataset emits grayscale, and the released weights have
  a 3-channel ``conv1`` because DeepCrack was trained on RGB asphalt. Rather than
  repeating the image to RGB, the three input-channel slices of ``conv1`` are
  summed into one -- an exact transformation (see :class:`DeepCrack`). This is
  only worth doing because from-scratch training matched the pretrained run
  (0.3594 vs 0.3647, CI includes 0), so the RGB stem was buying nothing.
* **Fused output only.** ``DeepCrackNet.forward`` returns five side outputs plus
  a fused map. Returning the tuple would collide with ``train.py``'s
  ``(logits, spine_logits)`` unpacking, so only the fused logits are exposed.
  The side outputs are still computed (they feed ``fuse_conv``); deep
  supervision on them is opt-in via ``deep_supervision=True``.

The network itself is vendored in :mod:`solar.deepcrack_net`, so no external
clone is needed.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .deepcrack_net import define_deepcrack

DEFAULT_WEIGHTS = Path(
    "/media/yasir/HDD/deepcrack_weights/extracted/deepcrack/pretrained_net_G.pth"
)


class DeepCrack(nn.Module):
    """DeepCrack with the repo's ``(B, 1, H, W)`` in / ``(B, 1, H, W)`` out contract.

    Parameters
    ----------
    in_chans:
        Channels of the *input tensor*. 1 (the repo default) builds a genuine
        single-channel ``conv1``; 3 keeps the upstream RGB stem.
    ngf:
        Base width. Must be 64 to match the released ``pretrained_net_G.pth``.
    pretrained:
        Load the released weights. Ignored if ``weights`` is None.
    weights:
        Path to ``pretrained_net_G.pth``. Defaults to the extracted download.

    Notes
    -----
    The released weights have a 3-channel ``conv1`` because DeepCrack was trained
    on RGB asphalt images. Feeding grayscale by repeating it to 3 channels is
    mathematically wasteful: for a repeated input ``x_rgb = [x, x, x]``,

        conv3(x_rgb) = Σ_c W[:, c] * x = (Σ_c W[:, c]) * x = conv1(x)

    so the three input-channel slices can be summed into one. ``_fold_conv1``
    does exactly that, which is verified to reproduce the repeat path to 8e-7
    (float32 noise). The result is a true 1-channel stem: 576 weights instead of
    1728, and no 3x memory/compute on the first layer.

    Folding is applied automatically whenever a 3-channel ``conv1`` is loaded
    into a 1-channel model, so old checkpoints (including every run before this
    change) still load.
    """

    def __init__(
        self,
        in_chans: int = 1,
        ngf: int = 64,
        pretrained: bool = True,
        weights: str | Path | None = None,
        deep_supervision: bool = False,
        decoder: bool = False,
        pretrained_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.ngf = int(ngf)
        self.deep_supervision = bool(deep_supervision)
        self.decoder = bool(decoder)
        self.pretrained_encoder = bool(pretrained_encoder)

        # Build the stem with the real input width. For in_chans=1 this is a
        # genuine grayscale conv; pretrained RGB weights are folded on load.
        self.net = define_deepcrack(
            in_nc=self.in_chans,
            num_classes=1,
            ngf=self.ngf,
            decoder=self.decoder,
        )
        self.weights_path: Path | None = None
        if pretrained:
            if self.decoder:
                raise ValueError(
                    "pretrained=True is not supported with decoder=True: the "
                    "released weights have no decoder parameters. Use "
                    "pretrained_encoder=True to load just the encoder, or "
                    "pretrained=False to train from scratch."
                )
            path = Path(weights) if weights is not None else DEFAULT_WEIGHTS
            if not path.exists():
                raise FileNotFoundError(
                    f"DeepCrack weights not found at {path}. Download "
                    f"pretrained_net_G.pth (Google Drive id "
                    f"1MAwIlWw27feU8EEkP8pgbia7kcuxBwEG) or pass "
                    f"model.pretrained_weights=<path>."
                )
            state = torch.load(path, map_location="cpu")
            # Route through load_state_dict so the RGB->gray fold applies.
            self.load_state_dict({"net." + k: v for k, v in state.items()})
            self.weights_path = path

        if pretrained_encoder:
            if not self.decoder:
                raise ValueError(
                    "pretrained_encoder=True only makes sense with decoder=True; "
                    "use pretrained=True for the upstream network."
                )
            path = Path(weights) if weights is not None else DEFAULT_WEIGHTS
            if not path.exists():
                raise FileNotFoundError(f"DeepCrack weights not found at {path}.")
            state = torch.load(path, map_location="cpu")
            # Keep only the encoder + side heads; the decoder is randomly
            # initialised. This is the configuration that matters: §20 showed
            # the encoder is fine and only the decoder is missing, so throwing
            # away the pretrained encoder would confound the comparison.
            #
            # Fold conv1 *before* the shape filter: the checkpoint's conv1 is
            # (64,3,3,3) and the model's is (64,1,3,3), so a naive shape match
            # would silently drop the stem and leave it at random init.
            prefixed = {"net." + k: v for k, v in state.items()}
            prefixed = self._fold_conv1(prefixed)
            own = self.state_dict()
            keep = {
                k: v for k, v in prefixed.items() if k in own and own[k].shape == v.shape
            }
            missing, unexpected = self.load_state_dict(keep, strict=False)
            loaded = len(keep)
            print(
                f"pretrained_encoder: loaded {loaded} encoder tensors, "
                f"{len(missing)} decoder tensors left at init"
            )
            self.weights_path = path

    @staticmethod
    def _fold_conv1(state: dict) -> dict:
        """Sum a 3-channel ``conv1.0.weight`` over its input-channel dim.

        Only acts when the incoming weight is 3-channel and the model's stem is
        1-channel; otherwise the state is returned untouched.
        """
        key = "net.conv1.0.weight"
        w = state.get(key)
        if w is None or w.ndim != 4 or w.shape[1] != 3:
            return state
        out = dict(state)
        out[key] = w.sum(dim=1, keepdim=True)
        return out

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights, folding an RGB ``conv1`` into the grayscale stem.

        Needed because every checkpoint produced before the grayscale change has
        a ``(64, 3, 3, 3)`` ``conv1.0.weight``, which would otherwise raise a
        shape mismatch against the ``(64, 1, 3, 3)`` stem.
        """
        if self.in_chans == 1:
            state_dict = self._fold_conv1(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor):
        """Return the fused logits, or a dict when deep supervision is enabled.

        The upstream network emits five side logits plus a fused map, but only
        the fused map is ever supervised -- the side heads are shaped purely by
        gradient flowing back through ``fuse_conv``. Since fusion is a fixed
        linear sum with no bias and no nonlinearity (verified: ``fused ==
        sum(w_i * side_i)`` to 6e-5), a side head that disagrees with the others
        can only dilute the result. Deep supervision gives each side head its own
        target so they become individually meaningful.

        Returns
        -------
        Tensor
            ``(B, 1, H, W)`` fused logits, when ``deep_supervision=False``.
        dict
            ``{"logits": fused, "sides": [side1..side5]}`` when enabled. A dict
            rather than a tuple so it cannot be confused with the
            ``(logits, spine_logits)`` convention used by the timm models.
        """
        if self.in_chans == 3 and x.shape[1] == 1:
            # Only reachable if a 3-channel stem is requested with grayscale
            # input; the default 1-channel path needs no repeat at all.
            x = x.repeat(1, 3, 1, 1)
        outs = self.net(x)
        fused = outs[-1]
        if not self.deep_supervision:
            return fused
        return {"logits": fused, "sides": list(outs[:5])}

    def bn_modules(self) -> list[nn.modules.batchnorm._BatchNorm]:
        """All BatchNorm layers, for the recalibration pass."""
        return [
            m
            for m in self.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))
        ]
