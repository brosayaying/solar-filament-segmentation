"""DINOv2 ViT-S/14 token-level embeddings for one image, with a t-SNE plot.

    python scripts/embed_dinov2.py --index 0 --out docs/embed_tsne.png

Notes on the setup
------------------
* 2048 is not divisible by 14 (2048/14 = 146.29), so the image is padded to
  2058 (147 x 14) with 5 px on each side. Token (i, j) then covers padded pixels
  [i*14, (i+1)*14), which maps back to original pixels by subtracting the pad.
* The image is grayscale, so it is replicated to 3 channels. DINOv2 expects
  ImageNet statistics.
* Only patch tokens are kept; the CLS token is dropped.
* t-SNE on all 21,609 tokens is slow and dominated by background, so the plot
  uses every filament token plus a random background sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

PATCH = 14
PAD = 5  # (2048 + 2*5) = 2058 = 147 * 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_model_input(img: np.ndarray) -> torch.Tensor:
    """Grayscale HxW uint8 -> (1, 3, H', W') float tensor, padded to a multiple of 14."""
    h, w = img.shape
    padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)
    rgb = np.repeat(padded[:, :, None].astype(np.float32) / 255.0, 3, axis=2)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def extract_tokens(model, x: torch.Tensor, device) -> np.ndarray:
    """Return (N, C) patch tokens, CLS dropped."""
    feats = model.forward_features(x.to(device))
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    tokens = feats[:, n_prefix:, :]
    return tokens[0].float().cpu().numpy()


def token_labels(anns: list[dict], grid: int, height: int, width: int) -> np.ndarray:
    """1 if the token's centre pixel lies inside any filament mask, else 0."""
    mask = np.zeros((height, width), np.uint8)
    for ann in anns:
        for poly in ann["segmentation"]:
            pts = np.round(np.asarray(poly, np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

    labels = np.zeros(grid * grid, np.int8)
    for i in range(grid):
        for j in range(grid):
            # Token centre in padded coords, then back to original coords.
            cy = i * PATCH + PATCH // 2 - PAD
            cx = j * PATCH + PATCH // 2 - PAD
            if 0 <= cy < height and 0 <= cx < width and mask[cy, cx]:
                labels[i * grid + j] = 1
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="index into the val split")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--ann-file", default="train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--model", default="vit_small_patch14_dinov2")
    ap.add_argument("--bg-sample", type=int, default=4000)
    ap.add_argument("--out", default="docs/embed_tsne.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    root = Path(args.root)
    with open(root / args.ann_file) as fh:
        coco = json.load(fh)
    with open(args.split) as fh:
        split = json.load(fh)

    images = {img["id"]: img for img in coco["images"]}
    ann_index: dict[str, list[dict]] = {}
    for ann in coco["annotations"]:
        ann_index.setdefault(ann["image_id"], []).append(ann)

    image_id = split["val"][args.index]
    meta = images[image_id]
    img = load_image(root / "train/train_images" / meta["file_name"])
    anns = ann_index.get(image_id, [])
    h, w = img.shape

    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DINOv2 asserts a fixed input size (518) unless dynamic_img_size is set.
    model = (
        timm.create_model(
            args.model, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        .to(device)
        .eval()
    )

    x = to_model_input(img)
    tokens = extract_tokens(model, x, device)
    grid = int(round(tokens.shape[0] ** 0.5))
    print(f"image {meta['file_name']}  tokens={tokens.shape}  grid={grid}x{grid}")

    labels = token_labels(anns, grid, h, w)
    n_pos = int(labels.sum())
    print(f"filament tokens: {n_pos} / {len(labels)} ({100*n_pos/len(labels):.2f}%)")

    # Subsample: keep every filament token, sample background.
    rng = np.random.default_rng(args.seed)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(neg_idx) > args.bg_sample:
        neg_idx = rng.choice(neg_idx, args.bg_sample, replace=False)
    keep = np.concatenate([pos_idx, neg_idx])
    X = tokens[keep]
    y = labels[keep]

    print(f"t-SNE on {len(X)} tokens ({len(pos_idx)} filament, {len(neg_idx)} background)")
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"{meta['file_name']}\n{len(anns)} filaments")
    axes[0].axis("off")

    bg = y == 0
    fg = y == 1
    axes[1].scatter(emb[bg, 0], emb[bg, 1], s=3, c="#9aa0a6", alpha=0.35, label="background")
    axes[1].scatter(emb[fg, 0], emb[fg, 1], s=14, c="#d62728", alpha=0.9, label="filament")
    axes[1].set_title(f"DINOv2 ViT-S/14 token t-SNE\n{len(X)} tokens")
    axes[1].legend(loc="best", markerscale=2)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Token-grid label map, to show how coarse 14 px tokens are.
    axes[2].imshow(labels.reshape(grid, grid), cmap="Reds", interpolation="nearest")
    axes[2].set_title(f"token labels ({grid}x{grid}, 14 px/token)")
    axes[2].axis("off")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
