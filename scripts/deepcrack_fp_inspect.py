"""Render specific pure-FP instances with the disk edge drawn, to settle whether
they are limb artefacts or interior false positives.

    python scripts/deepcrack_fp_inspect.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import anns_to_labelmap, build_ann_index, load_coco  # noqa: E402
from solar.deepcrack import DeepCrack  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)

CASES = [
    ("20110617063134Lh", 6826),
    ("20140501195814Mh", 5303),
    ("20110617063134Lh", 3457),
]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    by_file = {im["file_name"]: im for im in coco["images"]}

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    model.load_state_dict(
        torch.load(
            ROOT / "runs/deepcrack_ft_256_long/best.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]
    )
    model.eval()

    fig, axes = plt.subplots(len(CASES), 3, figsize=(11, 3.7 * len(CASES)))
    for r, (stem, target_area) in enumerate(CASES):
        im = by_file[stem + ".jpeg"]
        iid = im["id"]
        h, w = im["height"], im["width"]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        pred = prob_to_labelmap(prob, 0.5, 512)
        gt = anns_to_labelmap(ann_index[iid], h, w)

        # disk geometry
        mm = (img > 25).astype(np.uint8)
        n, _lab, st, _ = cv2.connectedComponentsWithStats(mm, 8)
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cx = st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] / 2
        cy = st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] / 2
        rad_disk = float(np.sqrt(st[k, cv2.CC_STAT_AREA] / np.pi))

        # pick the instance whose area is closest to the requested one
        areas = {pid: int((pred == pid).sum()) for pid in range(1, int(pred.max()) + 1)}
        pid = min(areas, key=lambda p: abs(areas[p] - target_area))
        m = pred == pid
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 60), min(h, ys.max() + 60)
        x0, x1 = max(0, xs.min() - 60), min(w, xs.max() + 60)

        yy, xx = np.mgrid[0:h, 0:w]
        rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        med_r = float(np.median(rad[m]))

        # full image with disk edge + instance highlighted
        full = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.circle(full, (int(cx), int(cy)), int(rad_disk), (0, 255, 255), 3)
        full[m] = (255, 40, 40)
        axes[r, 0].imshow(full)
        axes[r, 0].set_title(
            f"{stem}\ndisk r={rad_disk:.0f}, FP median r={med_r:.0f} "
            f"(inset {rad_disk - med_r:.0f} px)",
            fontsize=10,
        )

        # zoom on the instance, with the disk edge drawn if it passes through
        crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB)
        cv2.circle(crop, (int(cx) - x0, int(cy) - y0), int(rad_disk), (0, 255, 255), 2)
        crop[m[y0:y1, x0:x1]] = (255, 40, 40)
        axes[r, 1].imshow(crop)
        axes[r, 1].set_title(f"zoom (disk edge = yellow)", fontsize=10)

        axes[r, 2].imshow(prob[y0:y1, x0:x1], cmap="inferno", vmin=0, vmax=1)
        axes[r, 2].set_title(f"prob, area={areas[pid]}", fontsize=10)

        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("Pure false positives: interior, not limb", fontsize=13)
    fig.tight_layout()
    out = ROOT / "docs/DEEPCRACK_FINETUNE/tp_fp/fp_inspect.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
