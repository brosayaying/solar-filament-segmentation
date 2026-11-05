"""Phase 2.3: CNN candidate scorer, trained on precomputed crops.

The GBDT on scalar features reaches ``corr(q_hat, q) = 0.575`` and buys +0.0166,
against a ceiling of +0.0808 for the same candidate set (``ceiling_phase2.py``).
So the candidate set is fine and the scorer is the bottleneck -- which is what
justifies a CNN rather than more scalar features.

Crops come from ``build_phase2_crops.py`` (4 channels, 256^2, uint8). The first
version of this script derived them inside ``__getitem__``, which regenerated a
2048^2 connected-component labelling per sample and took ~82 min/epoch.

Cross-validation is grouped by **file**, not by row: candidates from one image
share illumination, filaments and model errors, so a random split leaks. This
project has been burned four times by numbers that looked valid and were not.

    python v2/scripts/train_phase2_cnn.py --epochs 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

P2 = ROOT / "v2/runs/phase2"


class CropDataset(Dataset):
    """In-memory crops with light augmentation.

    Channel 0 is stored as uint8 over [-5, 5] and is restored here; channels 1-3
    are already in [0, 1].
    """

    def __init__(self, crops: np.ndarray, q: np.ndarray, idx: np.ndarray, train: bool):
        self.crops = crops
        self.q = q
        self.idx = idx
        self.train = train

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        ch = np.asarray(self.crops[j], np.float32) / 255.0
        ch[0] = ch[0] * 10.0 - 5.0

        if self.train:
            if np.random.rand() < 0.5:
                ch = ch[:, :, ::-1]
            k = np.random.randint(4)
            if k:
                ch = np.rot90(ch, k, axes=(1, 2))
            ch = np.ascontiguousarray(ch)

        return torch.from_numpy(ch), torch.tensor(self.q[j], dtype=torch.float32)


def make_model(in_ch: int = 4) -> nn.Module:
    import timm

    return timm.create_model(
        "resnet18.a1_in1k", pretrained=True, num_classes=1, in_chans=in_ch
    )


def run_epoch(model, loader, opt, sched, train: bool, dev) -> float:
    model.train(train)
    tot, n = 0.0, 0
    lossf = nn.SmoothL1Loss(beta=0.1)
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(x).squeeze(1)
            loss = lossf(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        tot += float(loss) * len(y)
        n += len(y)
    return tot / max(1, n)


@torch.no_grad()
def predict(model, loader, dev) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(dev)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(P2 / "cnn_qhat.npz"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    crops = np.load(P2 / "crops.npy", mmap_mode="r")
    meta = np.load(P2 / "crops_meta.npz")
    q, file_of = meta["q"], meta["file_idx"]
    print(f"{len(q)} crops from {len(np.unique(file_of))} files", flush=True)

    uniq = np.unique(file_of)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    fold_of_file = {int(f): k % args.folds for k, f in enumerate(uniq)}
    fold_of_row = np.asarray([fold_of_file[int(f)] for f in file_of], np.int32)

    q_hat = np.zeros(len(q), np.float32)
    t0 = time.time()
    for k in range(args.folds):
        tr = np.nonzero(fold_of_row != k)[0]
        te = np.nonzero(fold_of_row == k)[0]
        if len(te) == 0:
            continue
        dl_tr = DataLoader(
            CropDataset(crops, q, tr, True), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, drop_last=True,
        )
        dl_te = DataLoader(
            CropDataset(crops, q, te, False), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers,
        )
        model = make_model().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(dl_tr))
        )
        for ep in range(args.epochs):
            tl = run_epoch(model, dl_tr, opt, sched, True, dev)
            vl = run_epoch(model, dl_te, opt, sched, False, dev)
            print(f"  fold {k} ep {ep + 1}/{args.epochs} "
                  f"train {tl:.4f} val {vl:.4f} [{time.time() - t0:.0f}s]",
                  flush=True)
        q_hat[te] = predict(model, dl_te, dev)
        torch.save(model.state_dict(), P2 / f"cnn_fold{k}.pt")

    corr = float(np.corrcoef(q_hat, q)[0, 1])
    print(f"\nCNN q_hat corr {corr:.4f}  (GBDT was 0.5749)")
    # ``file`` is a reserved keyword argument of ``savez_compressed``.
    np.savez_compressed(args.out, q_hat=q_hat, y=q, file_idx=file_of)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
