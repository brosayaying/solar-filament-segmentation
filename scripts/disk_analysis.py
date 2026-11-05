"""Solar-disk analysis: off-disk leakage and radial recall/contrast.

    python scripts/disk_analysis.py --n 24

Answers two questions:

1. Does the pipeline leak predictions into the black surround? (No.)
2. Does performance depend on distance from disk centre? (Yes -- recall
   collapses in the outer ring because limb darkening inverts the
   filament/background contrast.)

The disk mask is the same one ``scripts/patch_clf.py`` uses, imported from
``scripts/img_stats.py`` so there is a single implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from img_stats import disk_mask  # noqa: E402
from solar.data import FilamentDataset, load_coco  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

NBINS = 5


def to_uint8(normalised: torch.Tensor, mean: float, std: float) -> np.ndarray:
    """Undo the dataset normalisation to recover the original 0-255 image."""
    return np.clip((normalised.numpy() * std + mean) * 255, 0, 255).astype(np.uint8)


def disk_geometry(disk: np.ndarray):
    ys, xs = np.nonzero(disk)
    if len(ys) == 0:
        return None
    return ys.mean(), xs.mean(), np.sqrt(len(ys) / np.pi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--ckpt", default="runs/baseline_resnet34/best.pt")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=512)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    root = Path(cfg["data"]["root"])
    coco = load_coco(root / cfg["data"]["ann_file"])
    split = json.load(open(cfg["data"]["split_file"]))
    ds = FilamentDataset(
        root,
        coco,
        split["val"],
        mode="val",
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
    )

    model = build_model(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    disk_frac, rim_frac = [], []
    gt_off = gt_tot = 0
    pred_off = pred_tot = 0
    bins = np.zeros(NBINS)
    hits = np.zeros(NBINS)
    fg_sum = np.zeros(NBINS)
    bg_sum = np.zeros(NBINS)
    n_img = np.zeros(NBINS)

    for i in range(min(args.n, len(ds))):
        batch = ds[i]
        label = batch["label"].numpy()
        img = to_uint8(batch["image"][0], cfg["data"]["mean"], cfg["data"]["std"])

        with torch.no_grad():
            logits = model(batch["image"].unsqueeze(0).cuda())
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

        disk = disk_mask(img)
        inner = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
        disk_frac.append(disk.mean())
        rim_frac.append((disk > 0).mean() - (inner > 0).mean())

        pred = prob_to_labelmap(prob, threshold=args.threshold, min_area=args.min_area)
        off = disk == 0
        gt_off += int(((label > 0) & off).sum())
        gt_tot += int((label > 0).sum())
        pred_off += int(((pred > 0) & off).sum())
        pred_tot += int((pred > 0).sum())

        geom = disk_geometry(disk)
        if geom is None:
            continue
        cy, cx, radius = geom
        yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius
        idx = np.clip((r * NBINS).astype(int), 0, NBINS - 1)

        fg = (label > 0) & (disk > 0)
        bg = (label == 0) & (disk > 0)
        if not fg.any():
            continue
        hit = prob[fg] > args.threshold
        fg_r = idx[fg]
        for k in range(NBINS):
            sel = fg_r == k
            bins[k] += sel.sum()
            hits[k] += (hit & sel).sum()
            s = idx == k
            if (fg & s).any():
                fg_sum[k] += img[fg & s].mean()
                bg_sum[k] += img[bg & s].mean()
                n_img[k] += 1

    print(f"disk covers {100 * np.mean(disk_frac):.1f}% of image "
          f"({100 * min(disk_frac):.1f}-{100 * max(disk_frac):.1f}%)")
    print(f"off-disk surround: {100 * (1 - np.mean(disk_frac)):.1f}% of pixels")
    print(f"40px rim band: {100 * np.mean(rim_frac):.1f}% of pixels")
    print()
    print(f"GT filament px off-disk:   {gt_off} / {gt_tot} "
          f"({100 * gt_off / max(1, gt_tot):.2f}%)")
    print(f"predicted area off-disk:   {pred_off} / {pred_tot} "
          f"({100 * pred_off / max(1, pred_tot):.2f}%)")
    print()
    print("recall and contrast by radius (r/R):")
    print(f"{'bin':>10s} {'recall':>8s} {'filament':>9s} {'bg':>7s} {'contrast':>9s} {'n_px':>8s}")
    for k in range(NBINS):
        lo, hi = k / NBINS, (k + 1) / NBINS
        rec = hits[k] / max(1, bins[k])
        if n_img[k]:
            f = fg_sum[k] / n_img[k]
            g = bg_sum[k] / n_img[k]
            print(f"{lo:.1f}-{hi:.1f} {rec:8.3f} {f:9.1f} {g:7.1f} {g - f:+9.1f} {int(bins[k]):8d}")
        else:
            print(f"{lo:.1f}-{hi:.1f} {rec:8.3f} {'-':>9s} {'-':>7s} {'-':>9s} {int(bins[k]):8d}")


if __name__ == "__main__":
    main()
