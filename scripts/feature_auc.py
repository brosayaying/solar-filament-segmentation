"""Which single features separate filament patches from background?

Computes AUC for candidate scalar features over a sample of val patches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_clf as pc  # noqa: E402


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney)."""
    order = np.argsort(score)
    ranks = np.empty(len(score), np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    coco = pc.load_coco(pc.ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = pc.build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    split = json.load(open("splits/date_seed42.json"))

    feats: dict[str, list[np.ndarray]] = {}
    ys: list[np.ndarray] = []

    for iid in split["val"][:20]:
        img = pc.read_image(images[iid]["file_name"])
        disk = pc.disk_mask(img)
        gt = pc.anns_to_labelmap(ann_index[iid], *img.shape)
        y = (pc.patch_fraction(gt) > 0.25).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
        valid = cv2.erode(disk, k)[::8, ::8][:256, :256] > 0

        f = img.astype(np.float32)
        cand: dict[str, np.ndarray] = {"raw_mean": f}
        for s in (4, 8, 16, 32, 64):
            cand[f"res_s{s}"] = f - cv2.GaussianBlur(f, (0, 0), s)
        for kk in (15, 25, 41, 65):
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
            cand[f"blackhat_{kk}"] = -cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ker).astype(np.float32)
        cand["grad_mag"] = -cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) ** 2

        for name, arr in cand.items():
            m = sliding_window_view(arr, (8, 8))[::8, ::8].mean(axis=(-1, -2))
            feats.setdefault(name, []).append(m[valid])
        ys.append(y[valid])

    Y = np.concatenate(ys)
    print(f"{len(Y)} patches, pos={Y.mean():.4f}\n")
    rows = []
    for name, vals in feats.items():
        v = np.concatenate(vals)
        rows.append((auc(v, Y), name))
    for a, name in sorted(rows, reverse=True):
        print(f"AUC={a:.4f}  {name}")


if __name__ == "__main__":
    main()
