"""Config loading with dotted CLI overrides.

Usage::

    python -m solar.train --config configs/base.json --set train.epochs=5 model.backbone=convnext_tiny
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "data": {
        "root": "MAGFiLO_1.0_Kaggle_2026",
        "ann_file": "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        "split_file": "splits/date_seed42.json",
        "group_by": "date",
        "val_frac": 0.2,
        "seed": 42,
        "patch_size": 512,
        "samples_per_image": 2,
        # tiled=True replaces random cropping with a fixed non-overlapping grid
        # of patch_size tiles (2048/1024 = 4 quadrants). Every pixel is seen
        # once per epoch and no filament is truncated at a random offset.
        "tiled": False,
        "spine_thickness": 3,
        # Normalisation constants. These are measured over the *disk*, not the
        # whole frame: 28% of each 2048^2 image is off-disk black (mean 3.2/255),
        # which drags the global mean to 0.327 and inflates the std to 0.223.
        # Disk-only statistics are mean 0.4463, std 0.1313 (n=142 val files,
        # scripts/disk_stats.py). The old 0.5/0.25 pair was 1.9x too wide, which
        # compressed the real signal into a narrow band.
        "mean": 0.4463,
        "std": 0.1313,
        # Full-resolution preprocessing, applied before cropping because radial
        # flattening needs the whole disk. Subset of {flatten, blur, clahe},
        # e.g. "flatten+blur". None disables. Must be identical at train, val
        # and test time -- applying it only at eval is a distribution shift.
        "preprocess": None,
        # Replace off-disk pixels with a fixed normalised value.
        #
        # Measured to be pointless with disk-derived statistics: the sky already
        # normalises to -3.30 sigma (raw) or -3.40 sigma (flattened) against the
        # disk mean, and offdisk_value=-3.0 would move it *toward* the disk by
        # 0.30 sigma. Meanwhile it costs a full-resolution mask build per
        # sample, which tripled epoch time (38 s -> 120 s) on 9 of 10 slices
        # that contain nothing but sky. Left available but off by default.
        "mask_offdisk": False,
        "offdisk_value": -3.0,
        # On-disk cache of preprocessed images, built by
        # scripts/build_preprocess_cache.py. Strongly recommended when
        # preprocess is set: flattening costs ~0.5 s/image, which is ~12 h of
        # redundant work across a 160-epoch run.
        "preprocess_cache": None,
        "num_workers": 8,
        # sampler="positive_crop" swaps FilamentDataset for PositiveCropDataset,
        # which mixes positive-centred, limb-centred and uniform crops. Needed
        # for DeepCrack finetuning: filaments are 2.6% of pixels and the limb is
        # never annotated, so uniform cropping teaches neither.
        "sampler": "uniform",
        "p_positive": 0.5,
        "p_limb": 0.2,
        "limb_width": 24,
        # soft_labels=True trains on the mean of every annotator's mask for a
        # file instead of one annotator's. The official scorer evaluates against
        # every variant, so the consensus is the target it actually rewards.
        "soft_labels": False,
    },
    "model": {
        "backbone": "resnet34",
        "pretrained": True,
        "in_chans": 1,
        "decoder_dim": 128,
        "out_ch": 1,
        "dropout": 0.0,
        # Required for fixed-size backbones (swin); ignored by conv nets.
        "img_size": None,
        # Auxiliary centerline head; forces instance-identity learning.
        "aux_spine": False,
        # Stride-1 stem path: a conv on the input, concatenated into the final
        # decoder block. The finest timm feature is stride 2 (resnet) or stride 4
        # (convnext/efficientnet/pvt), and a median filament is 11 px wide, so
        # without this the boundary is reconstructed from stride-4 features.
        # 0 disables it (keeps old checkpoints loadable).
        "stem_ch": 0,
        # arch="deepcrack" selects the pretrained DeepCrack network instead of
        # a timm encoder + UNet decoder. ngf must be 64 to match the released
        # weights; pretrained_weights overrides the default download path.
        "arch": "unet",
        "ngf": 64,
        "pretrained_weights": None,
        # deep_supervision=True (deepcrack only) also supervises the five side
        # heads, not just the fused output. Fusion is a fixed linear sum, so
        # unsupervised side heads can only dilute it.
        "deep_supervision": False,
        # decoder=True (deepcrack only) replaces the side-head fusion with a
        # UNet-style upsample-and-concat decoder. Upstream DeepCrack has no
        # decoder, which is why its fused map ranks well (AUC 0.997) but
        # localises badly (IoU 0.50) -- see docs/DEEPCRACK_FINETUNE.md §20.
        # Requires pretrained=False: the released weights have no decoder.
        "decoder": False,
        # pretrained_encoder=True (deepcrack + decoder only) loads the released
        # encoder weights into a decoder model, leaving the decoder at init.
        # This is the configuration §20 points at: the encoder is fine, only the
        # decoder is missing, so discarding the pretrained encoder would
        # confound the comparison.
        "pretrained_encoder": False,
    },
    "train": {
        "epochs": 20,
        "batch_size": 8,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 1,
        "amp": True,
        # AMP dtype: "bf16" or "fp16".
        #
        # bf16 is the default because fp16 diverged to NaN on ConvNeXt-Small:
        # both fold-0 A/B arms went from a healthy loss of 0.17 to NaN at
        # ~2.9e-4 LR (epochs 23 and 17), despite GradScaler. ConvNeXt's
        # LayerNorms are the usual suspect -- fp16's 5-bit exponent cannot
        # represent their activation range, and once one layer overflows the
        # scaler backs off but the weights are already poisoned. bf16 has fp32's
        # exponent range, needs no scaler, and is the standard choice for
        # modern conv/transformer backbones.
        "amp_dtype": "bf16",
        "grad_clip": 1.0,
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        # Weight of the auxiliary spine (centerline) loss. 0 disables it.
        "spine_weight": 0.0,
        # Border-aware loss (loss="border_aware"): pixels within border_margin
        # px of the crop edge are scaled down to border_floor.
        "border_margin": 16,
        "border_floor": 0.0,
        # Deep supervision (deepcrack only): weight of the mean loss over the
        # five side heads, added to the fused loss. 0 disables it.
        "side_weight": 0.0,
        "val_every": 1,
        "val_images": 64,
        # Early stopping on **validation PQ**, not train loss.
        #
        # Train loss is the wrong signal here: on runs/norm_disk_raw it was flat
        # (sd 0.006 over the last 40 epochs) while val PQ was still climbing and
        # peaked at epoch 100 of 106. Stopping on loss would have quit ~60 epochs
        # early. Val PQ is noisy (sd 0.007), so patience must be generous.
        #
        # patience=0 disables. min_delta is the improvement that counts as
        # progress; anything below it is treated as a plateau.
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.002,
        "out_dir": "runs",
        "run_name": None,
        # Warm start: load weights only (schedule restarts). Used to continue a
        # run after a target/metric fix without paying for the full schedule.
        "init_ckpt": None,
    },
    "eval": {
        # 0.5 is the competition's binarisation threshold, so it is fixed rather
        # than tuned. Tuning it on val (0.55, or 0.70 as the ablation sweep
        # preferred) inflates the local number by fitting a knob the leaderboard
        # does not expose, and the gain does not transfer.
        "threshold": 0.5,
        # min_area=512 is worth +0.05 PQ over the naive 64, because every
        # spurious component costs 0.5 FP. This one *is* ours to choose.
        "min_area": 512,
        "batch_size": 1,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    """Parse a CLI string into a Python value."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path:
        with open(path) as fh:
            cfg = _deep_merge(cfg, json.load(fh))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce(val.strip()))
    return cfg
