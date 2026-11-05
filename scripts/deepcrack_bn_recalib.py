"""Recalibrate DeepCrack's BatchNorm statistics on MAGFiLO before finetuning.

Why this is step 1
------------------
The zero-shot probe was sensitive to input scaling in a way that points at BN,
not at the weights: in-disk AUC was 0.47 under the upstream ``[-1, 1]``
normalisation and 0.64 under ``[0, 1]``. A network whose output ranking flips
that much when the input is rescaled is being read out of distribution -- its
running mean/var no longer describe the activations it is actually seeing.

DeepCrack was trained on 256x256 asphalt crops with ``x/255*2 - 1``. MAGFiLO
differs in both intensity distribution (a bright disk on a black surround, mean
82.5 / std 58.1 over the whole image) and content. Refreshing the BN running
statistics is the cheapest possible adaptation: no gradients, no weight changes,
one pass over a few hundred crops.

What it does
------------
Runs the network in ``train()`` mode with ``torch.no_grad()`` so BN updates its
running estimates from the new data while nothing else changes. Momentum is set
to ``None`` (cumulative moving average) so the result is the exact mean/var over
the pass rather than an exponential blend with the old asphalt statistics.

    python scripts/deepcrack_bn_recalib.py --n-images 200 --crops-per-image 4

Writes a checkpoint with the recalibrated weights to ``--out``, which can then
be used as ``train.init_ckpt`` or ``model.pretrained_weights``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.crop_data import PositiveCropDataset  # noqa: E402
from solar.data import collate_train, load_coco  # noqa: E402
from solar.deepcrack import DeepCrack  # noqa: E402

cv2.setNumThreads(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=200, help="train images to use")
    ap.add_argument("--crops-per-image", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--p-positive", type=float, default=0.5)
    ap.add_argument("--p-limb", type=float, default=0.2)
    ap.add_argument("--limb-width", type=int, default=24)
    ap.add_argument("--mean", type=float, default=0.0, help="input normalisation mean")
    ap.add_argument("--std", type=float, default=1.0, help="input normalisation std")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--out", default="runs/deepcrack_bn_recalib.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    import json

    split = json.loads((ROOT / args.split).read_text())
    train_ids = split["train"][: args.n_images]

    ds = PositiveCropDataset(
        data_root,
        coco,
        train_ids,
        patch_size=args.patch_size,
        mean=args.mean,
        std=args.std,
        samples_per_image=args.crops_per_image,
        p_positive=args.p_positive,
        p_limb=args.p_limb,
        limb_width=args.limb_width,
        augment=False,  # statistics should describe the data, not the augmentation
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_train,
        pin_memory=True,
    )
    print(f"recalibrating on {len(ds)} crops from {len(train_ids)} images")

    model = DeepCrack(in_chans=1, ngf=64, pretrained=True).to(device)

    # Cumulative moving average: the result is the exact mean/var over this pass,
    # not an exponential blend with the original asphalt statistics.
    bn_layers = model.bn_modules()
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
    print(f"{len(bn_layers)} BatchNorm layers reset to cumulative averaging")

    # train() so BN updates its running stats; no_grad so no weights change.
    model.train()
    with torch.no_grad():
        for batch in tqdm(loader, desc="bn-recalib"):
            model(batch["image"].to(device, non_blocking=True))

    # Sanity check: every BN layer should have seen data and have finite stats.
    bad = [
        (i, m.running_mean.abs().max().item(), m.running_var.min().item())
        for i, m in enumerate(bn_layers)
        if m.running_mean is None
        or not torch.isfinite(m.running_mean).all()
        or not torch.isfinite(m.running_var).all()
        or m.running_var.min() <= 0
    ]
    if bad:
        raise RuntimeError(f"BN recalibration produced invalid stats: {bad[:5]}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "arch": "deepcrack",
                "ngf": 64,
                "in_chans": 1,
                "mean": args.mean,
                "std": args.std,
            },
            "bn_recalib": {
                "n_images": len(train_ids),
                "n_crops": len(ds),
                "patch_size": args.patch_size,
                "p_positive": args.p_positive,
                "p_limb": args.p_limb,
                "limb_width": args.limb_width,
                "mean": args.mean,
                "std": args.std,
            },
        },
        out,
    )
    print(f"saved recalibrated weights -> {out}")


if __name__ == "__main__":
    main()
