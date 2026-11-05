"""Fully-convolutional pixel-level classifier for filaments.

The patch classifier throws away 8/9 of what it sees: a 48x48 crop produces one
label for the centre 16x16 patch. This script instead trains a small FCN on
large crops and predicts a **dense mask** at the same resolution as the input,
so every pixel gets a decision and the whole crop contributes to the loss.

Architecture: a small U-Net (3 down / 3 up, skip connections). No pretrained
weights -- this is deliberately a "small network" to compare against the
hand-engineered patch pipeline, not a competitor to the DL baseline in
``src/solar``.

Training crops are sampled to be class-balanced: half centred on a filament,
half on background. That is the "pick both classes in a balance" idea, applied
at the crop level rather than the patch level.

    python scripts/pixel_cnn.py --crop 256 --train-images 60 --val-images 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import patch_clf as pc  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402

cv2.setNumThreads(4)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    """3-level U-Net. ~0.5 M params at width 24."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        w = width
        self.d1 = DoubleConv(1, w)
        self.d2 = DoubleConv(w, w * 2)
        self.d3 = DoubleConv(w * 2, w * 4)
        self.bottleneck = DoubleConv(w * 4, w * 8)
        self.u3 = DoubleConv(w * 8 + w * 4, w * 4)
        self.u2 = DoubleConv(w * 4 + w * 2, w * 2)
        self.u1 = DoubleConv(w * 2 + w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.d1(x)
        s2 = self.d2(F.max_pool2d(s1, 2))
        s3 = self.d3(F.max_pool2d(s2, 2))
        b = self.bottleneck(F.max_pool2d(s3, 2))
        u3 = self.u3(torch.cat([F.interpolate(b, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
        return self.out(u1).squeeze(1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_image(iid, images, ann_index):
    img = pc.read_image(images[iid]["file_name"])
    gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
    disk = pc.disk_mask(img)
    inner = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    return img, gt, inner


def sample_crops(
    img: np.ndarray,
    gt: np.ndarray,
    inner: np.ndarray,
    crop: int,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced crop sampling: ``n_pos`` centred on filaments, ``n_neg`` on background.

    Centring on a filament pixel guarantees the crop contains signal; sampling
    uniformly would give ~99% empty crops and the net would learn to output zero.
    """
    h, w = img.shape
    half = crop // 2
    fg_y, fg_x = np.nonzero(gt > 0)
    bg_y, bg_x = np.nonzero((gt == 0) & (inner > 0))

    def take(ys, xs, n):
        if len(ys) == 0:
            return np.zeros((0, crop, crop), np.uint8), np.zeros((0, crop, crop), np.uint8)
        sel = rng.integers(0, len(ys), n)
        cy = np.clip(ys[sel], half, h - half - 1)
        cx = np.clip(xs[sel], half, w - half - 1)
        X = np.empty((n, crop, crop), np.uint8)
        Y = np.empty((n, crop, crop), np.uint8)
        for i, (a, b) in enumerate(zip(cy, cx)):
            X[i] = img[a - half : a + half, b - half : b + half]
            Y[i] = (gt[a - half : a + half, b - half : b + half] > 0).astype(np.uint8)
        return X, Y

    Xp, Yp = take(fg_y, fg_x, n_pos)
    Xn, Yn = take(bg_y, bg_x, n_neg)
    return np.concatenate([Xp, Xn]), np.concatenate([Yp, Yn])


def augment_pair(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator):
    """Dihedral augmentation, applied identically to image and mask."""
    outX, outY = X.copy(), Y.copy()
    n = len(X)
    f = rng.random(n) < 0.5
    outX[f] = outX[f, :, ::-1]
    outY[f] = outY[f, :, ::-1]
    f = rng.random(n) < 0.5
    outX[f] = outX[f, ::-1, :]
    outY[f] = outY[f, ::-1, :]
    k = rng.integers(0, 4, n)
    for r in (1, 2, 3):
        s = k == r
        if s.any():
            outX[s] = np.rot90(outX[s], r, axes=(1, 2))
            outY[s] = np.rot90(outY[s], r, axes=(1, 2))
    return np.ascontiguousarray(outX), np.ascontiguousarray(outY)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2)) + 1.0
    den = p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1.0
    return 1 - (num / den).mean()


@torch.no_grad()
def predict_full(model, img: np.ndarray, mean: float, std: float, device, tile: int = 512):
    """Dense prediction over a full 2048x2048 image, tiled to bound memory."""
    model.eval()
    h, w = img.shape
    out = np.zeros((h, w), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = img[y0:y1, x0:x1].astype(np.float32)
            x = torch.from_numpy((patch - mean) / std).unsqueeze(0).unsqueeze(0).to(device)
            # Pad to a multiple of 8 so the 3 pooling levels divide evenly.
            ph, pw = x.shape[-2], x.shape[-1]
            nh, nw = (8 - ph % 8) % 8, (8 - pw % 8) % 8
            if nh or nw:
                x = F.pad(x, (0, nw, 0, nh), mode="reflect")
            p = torch.sigmoid(model(x))[0, :ph, :pw].cpu().numpy()
            out[y0:y1, x0:x1] = p
    return out


def to_labelmap(prob: np.ndarray, thr: float, min_area: int) -> np.ndarray:
    b = (prob > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros(prob.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--train-images", type=int, default=60)
    ap.add_argument("--val-images", type=int, default=20)
    ap.add_argument("--pos-per-image", type=int, default=24)
    ap.add_argument("--neg-per-image", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--out", default="docs/EDGE/pixel_cnn.json")
    ap.add_argument("--save", default=None, help="path to save model weights")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} crop={args.crop} width={args.width}")

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    train_ids = split["train"][: args.train_images]
    val_ids = split["val"][: args.val_images]

    # Global normalisation from the training images.
    t0 = time.time()
    stats = []
    for iid in train_ids[:20]:
        img, _, _ = load_image(iid, images, ann_index)
        stats.append((float(img.mean()), float(img.std())))
    mean = float(np.mean([s[0] for s in stats]))
    std = float(np.mean([s[1] for s in stats]))
    print(f"normalisation mean={mean:.2f} std={std:.2f}")

    # Cache training crops once; augmentation is applied per epoch.
    rng = np.random.default_rng(0)
    Xs, Ys = [], []
    for iid in train_ids:
        img, gt, inner = load_image(iid, images, ann_index)
        X, Y = sample_crops(img, gt, inner, args.crop, args.pos_per_image, args.neg_per_image, rng)
        Xs.append(X)
        Ys.append(Y)
    Xtr = np.concatenate(Xs)
    Ytr = np.concatenate(Ys)
    print(f"train crops {Xtr.shape} pos_px={Ytr.mean():.4f} ({time.time() - t0:.0f}s)")

    model = SmallUNet(width=args.width).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    # Validation images cached once.
    val_cache = [load_image(iid, images, ann_index) for iid in val_ids]

    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        Xa, Ya = augment_pair(Xtr, Ytr, rng)
        Xn = ((Xa.astype(np.float32) - mean) / std)
        perm = rng.permutation(len(Xn))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            xb = torch.from_numpy(Xn[idx]).unsqueeze(1).to(device)
            yb = torch.from_numpy(Ya[idx].astype(np.float32)).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb) + dice_loss(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        # PQ on a few val images each epoch (full-image inference is slow).
        acc = PQAccumulator()
        for img, gt, _ in val_cache[:4]:
            p = predict_full(model, img, mean, std, device)
            acc.update(to_labelmap(p, 0.5, 512), gt)
        s = acc.summary()
        hist.append({"epoch": ep + 1, "loss": tot / len(perm), "val_pq": s["pq"]})
        print(f"  ep{ep + 1:>3} loss={tot / len(perm):.4f} val_PQ={s['pq']:.4f} "
              f"SQ={s['sq']:.3f} RQ={s['rq']:.3f}", flush=True)

    train_s = time.time() - t0

    if args.save:
        torch.save(
            {
                "model": model.state_dict(),
                "width": args.width,
                "crop": args.crop,
                "mean": mean,
                "std": std,
            },
            args.save,
        )
        print(f"saved -> {args.save}")

    # Full sweep on all val images. The range spans both extremes because the
    # Dice loss makes the output strongly bimodal (p99=0.002, p99.9=0.997), so
    # the usable threshold band sits high and is not predictable a priori.
    probs = [predict_full(model, img, mean, std, device) for img, _, _ in val_cache]
    gts = [gt for _, gt, _ in val_cache]
    best = None
    sweep = []
    for thr in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        for ma in (200, 512, 1024):
            acc = PQAccumulator()
            for p, gt in zip(probs, gts):
                acc.update(to_labelmap(p, thr, ma), gt)
            s = acc.summary()
            s.update(threshold=thr, min_area=ma)
            sweep.append(s)
            if best is None or s["pq"] > best["pq"]:
                best = s

    # Print the threshold curve at the best min_area so the shape is visible.
    ma_best = best["min_area"]
    print(f"\nthreshold sweep at min_area={ma_best}:")
    for s in sweep:
        if s["min_area"] == ma_best:
            mark = " <-" if s["threshold"] == best["threshold"] else ""
            print(f"  thr={s['threshold']:.2f}  PQ={s['pq']:.4f} SQ={s['sq']:.3f} "
                  f"RQ={s['rq']:.3f} tp={s['tp']:>3} fp={s['fp']:>4} fn={s['fn']:>3}{mark}")

    print(f"\nBEST PQ={best['pq']:.4f} SQ={best['sq']:.3f} RQ={best['rq']:.3f} "
          f"tp={best['tp']} fp={best['fp']} fn={best['fn']} "
          f"(thr={best['threshold']} ma={best['min_area']})  {train_s:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "crop": args.crop,
            "width": args.width,
            "params": n_par,
            "n_train_crops": int(len(Xtr)),
            "best": best,
            "sweep": sweep,
            "train_s": train_s,
            "history": hist,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
