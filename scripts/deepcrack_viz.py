"""Figure: raw pretrained vs finetuned DeepCrack on val images.

    python scripts/deepcrack_viz.py --n 4 --out docs/DEEPCRACK_FINETUNE

Writes one panel per image plus a combined grid.
"""

from __future__ import annotations

import argparse
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

cv2.setNumThreads(4)


@torch.no_grad()
def predict(model, img, device, mean=0.0, std=1.0):
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    return torch.sigmoid(model(t))[0, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--weights", default="runs/deepcrack_ft_256/best.pt")
    ap.add_argument("--out", default="docs/DEEPCRACK_FINETUNE")
    ap.add_argument("--split", default="splits/date_seed42.json")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    seen: set[str] = set()
    ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[: args.n]

    raw = DeepCrack(in_chans=1, ngf=64, pretrained=True).to(device).eval()
    ft = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    ft.load_state_dict(torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)["model"])
    ft.eval()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for iid in ids:
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        gt = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        p_raw = predict(raw, img, device)
        p_ft = predict(ft, img, device)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"{Path(im['file_name']).stem}\ninput")

        im1 = axes[1].imshow(p_raw, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title(f"raw pretrained\nmax={p_raw.max():.3f}")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(p_ft, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title(f"finetuned 256²\nmax={p_ft.max():.3f}")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay[gt > 0] = (0.45 * overlay[gt > 0] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        axes[3].imshow(overlay)
        axes[3].set_title(f"GT (green)\n{int(gt.max())} instances")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"{Path(im['file_name']).stem}.png", dpi=110)
        plt.close(fig)
        print(f"  {im['file_name']}  raw max={p_raw.max():.3f}  ft max={p_ft.max():.3f}")

    print(f"wrote {len(ids)} panels -> {out}")


if __name__ == "__main__":
    main()
