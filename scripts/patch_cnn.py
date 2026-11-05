"""Small CNN on 16x16 patches, compared against the RandomForest.

The RF gets 802 hand-engineered features (multi-scale black-hat, residuals,
pooled stats, context). A CNN can learn its own filters, so the fair comparison
gives it the same *information* -- a context window around the patch -- and lets
it decide what to compute.

Input: a ``CTX``-pixel crop centred on the patch (default 48 = 3x3 patches of
16 px), raw intensity, normalised per image.
Output: P(filament) for the centre 16x16 patch.

Evaluated on the same patch-F1 metric and the same train/val image split as
``scripts/compare_patch_sizes.py``, so the numbers are directly comparable.

    python scripts/patch_cnn.py --patch 16 --train-images 60 --val-images 20
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import patch_clf as pc  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402

cv2.setNumThreads(4)


# --------------------------------------------------------------------------- #
# Patch extraction
# --------------------------------------------------------------------------- #


def extract_patches(
    img: np.ndarray,
    gt: np.ndarray,
    disk: np.ndarray,
    patch: int,
    ctx: int,
    neg_frac: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(crops, labels, grid_yx, valid)`` for one image.

    ``crops`` is (N, ctx, ctx) uint8, centred on each selected patch.
    """
    grid = img.shape[0] // patch
    y = (pc.patch_fraction(gt) > 0.25).astype(np.uint8)

    inner = cv2.erode(disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    valid = inner[::patch, ::patch][:grid, :grid] > 0

    # Pad so every patch centre has a full ctx window.
    half = ctx // 2
    pad = half + patch // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    ys, xs = np.nonzero(valid)
    if rng is not None:
        pos = y[ys, xs] == 1
        keep = pos | (rng.random(len(ys)) < neg_frac)
        ys, xs = ys[keep], xs[keep]

    crops = np.empty((len(ys), ctx, ctx), np.uint8)
    for i, (gy, gx) in enumerate(zip(ys, xs)):
        cy = gy * patch + patch // 2 + pad
        cx = gx * patch + patch // 2 + pad
        crops[i] = padded[cy - half : cy - half + ctx, cx - half : cx - half + ctx]

    return crops, y[ys, xs], np.stack([ys, xs], 1), valid


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class PatchCNN(nn.Module):
    """Small VGG-ish net. ~4 conv layers is plenty for 48x48 input."""

    def __init__(self, ctx: int = 48, width: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, width),
            block(width, width * 2),
            block(width * 2, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #


def normalise(crops: np.ndarray, mean: float | None = None, std: float | None = None):
    """Global standardisation using statistics from the *training* set.

    Per-crop standardisation is wrong here: it forces every patch to mean 0 and
    std 1, which erases exactly the contrast that separates a dark filament from
    flat background. Both would look identical to the network. Global constants
    preserve relative brightness.
    """
    x = crops.astype(np.float32)
    if mean is None:
        mean = float(x.mean())
        std = float(x.std())
    return ((x - mean) / (std + 1e-6)).astype(np.float32), mean, std


def augment(crops: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flips and 90-degree rotations.

    Filaments have no canonical orientation and the sun has no preferred
    handedness, so all 8 dihedral transforms are label-preserving. With only 60
    training images this is the main defence against overfitting.
    """
    out = crops.copy()
    n = len(out)
    flip = rng.random(n) < 0.5
    out[flip] = out[flip, :, ::-1]
    flip = rng.random(n) < 0.5
    out[flip] = out[flip, ::-1, :]
    k = rng.integers(0, 4, n)
    for rot in (1, 2, 3):
        sel = k == rot
        if sel.any():
            out[sel] = np.rot90(out[sel], rot, axes=(1, 2))
    return np.ascontiguousarray(out)


def predict(
    model: PatchCNN, X: np.ndarray, device: torch.device, batch: int = 4096
) -> np.ndarray:
    """Batched inference. The val set is ~220k patches, which will not fit in
    one forward pass on a shared GPU."""
    model.eval()
    out = np.empty(len(X), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).unsqueeze(1).to(device)
            out[i : i + batch] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def train_cnn(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    epochs: int,
    batch: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
) -> tuple[PatchCNN, list[dict]]:
    """Train on ``(N, ctx, ctx)`` float arrays; returns the model and history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PatchCNN(ctx=Xtr.shape[1]).to(device)

    # Class imbalance is handled by pos_weight rather than resampling, so the
    # model sees every negative we extracted.
    n_pos = max(1, int(ytr.sum()))
    n_neg = len(ytr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = []
    for ep in range(epochs):
        model.train()
        # Re-augment every epoch: with 60 images the same patch is seen many
        # times, so fresh dihedral transforms each pass are what stop the net
        # from memorising.
        Xa = augment(Xtr, rng)
        ds = TensorDataset(
            torch.from_numpy(Xa).unsqueeze(1),
            torch.from_numpy(ytr.astype(np.float32)),
        )
        dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)

        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        sched.step()

        pv = predict(model, Xva, device)
        f1, thr = best_f1(pv, yva)
        hist.append({"epoch": ep + 1, "loss": tot / len(ds), "val_f1": f1, "val_thr": thr})
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep + 1:>3} loss={tot / len(ds):.4f} val_F1={f1:.4f}@{thr:.2f}", flush=True)

    return model, hist


def best_f1(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best, bt = 0.0, 0.5
    for thr in np.arange(0.2, 0.91, 0.05):
        pred = p > thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best:
            best, bt = f1, float(thr)
    return best, bt


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=48)
    ap.add_argument("--train-images", type=int, default=60)
    ap.add_argument("--val-images", type=int, default=20)
    ap.add_argument("--neg-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", default="docs/EDGE/patch_cnn.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} patch={args.patch} ctx={args.ctx}")

    # patch_fraction() reads the module-global PATCH, so this must be set before
    # any label is computed or the labels and crops disagree on patch size.
    pc.set_patch_size(args.patch)

    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    rng = np.random.default_rng(0)

    def build(ids, neg_frac):
        Xs, ys = [], []
        for iid in ids:
            img = pc.read_image(images[iid]["file_name"])
            gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
            disk = pc.disk_mask(img)
            c, y, _, _ = extract_patches(img, gt, disk, args.patch, args.ctx, neg_frac, rng)
            Xs.append(c)
            ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    t0 = time.time()
    Xtr, ytr = build(split["train"][: args.train_images], args.neg_frac)
    Xva, yva = build(split["val"][: args.val_images], 1.0)  # all val patches
    print(f"train {Xtr.shape} pos={ytr.mean():.3f} | val {Xva.shape} pos={yva.mean():.4f} "
          f"({time.time() - t0:.0f}s)")

    Xtr_n, mu, sd = normalise(Xtr)
    Xva_n, _, _ = normalise(Xva, mu, sd)
    print(f"normalisation: mean={mu:.2f} std={sd:.2f}")

    t0 = time.time()
    model, hist = train_cnn(
        Xtr_n, ytr, Xva_n, yva, args.epochs, args.batch, args.lr, device
    )
    train_s = time.time() - t0

    model.eval()
    pv = predict(model, Xva_n, device)

    f1, thr = best_f1(pv, yva)
    a = auc(pv, yva)
    print(f"\nCNN: patch AUC={a:.4f}  F1={f1:.4f}@{thr:.2f}  ({train_s:.0f}s)")

    # Reference: the RF numbers from compare_patch_sizes.py at the same split.
    ref = {"patch_auc": 0.9532, "patch_f1": 0.526}
    print(f"RF : patch AUC={ref['patch_auc']:.4f}  F1={ref['patch_f1']:.4f}  (from compare_patch_sizes)")
    print(f"delta F1 = {f1 - ref['patch_f1']:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "patch": args.patch,
            "ctx": args.ctx,
            "epochs": args.epochs,
            "width": args.width,
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
            "cnn_auc": a,
            "cnn_f1": f1,
            "cnn_thr": thr,
            "rf_auc": ref["patch_auc"],
            "rf_f1": ref["patch_f1"],
            "train_s": train_s,
            "history": hist,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
