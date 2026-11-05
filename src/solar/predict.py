"""Generate a submission CSV for the test set.

    python -m solar.predict --ckpt runs/resnet34_123/best.pt --out submissions/sub.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .data import normalise_image, parse_preprocess, preprocess_image
from .models import build_model
from .postprocess import labelmap_to_rle, prob_to_labelmap


def _forward(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    # Deep supervision returns {"logits", "sides"}; score the fused map.
    if isinstance(out, dict):
        out = out["logits"]
    elif isinstance(out, tuple):
        out = out[0]
    return torch.sigmoid(out)


@torch.no_grad()
def predict_prob(
    model,
    image: torch.Tensor,
    tta: bool = False,
    tile: int | None = None,
) -> np.ndarray:
    """Probability map for one (1, H, W) image.

    ``tta`` averages the 4 flip variants. Measured +0.008 PQ locally, and it is
    free at inference.

    ``tile`` runs the model over ``tile``-sized crops and stitches the result.
    Required for ViT encoders, which are strongly scale-sensitive: a model
    trained on 518 px crops predicts 7.7x less area when fed a full 2048 px
    image. Harmless for CNNs.
    """
    if tile:
        _, h, w = image.shape
        prob = np.zeros((h, w), np.float32)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                crop = image[:, y0:y1, x0:x1].unsqueeze(0)
                p = _forward(model, crop)
                if tta:
                    p = (
                        p
                        + torch.flip(_forward(model, torch.flip(crop, [3])), [3])
                        + torch.flip(_forward(model, torch.flip(crop, [2])), [2])
                        + torch.flip(_forward(model, torch.flip(crop, [2, 3])), [2, 3])
                    ) / 4
                prob[y0:y1, x0:x1] = p[0, 0].float().cpu().numpy()
        return prob

    x = image.unsqueeze(0)
    p = _forward(model, x)
    if tta:
        p = (
            p
            + torch.flip(_forward(model, torch.flip(x, [3])), [3])
            + torch.flip(_forward(model, torch.flip(x, [2])), [2])
            + torch.flip(_forward(model, torch.flip(x, [2, 3])), [2, 3])
        ) / 4
    return p[0, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-dir", default="MAGFiLO_1.0_Kaggle_2026/test/test_images")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument(
        "--close-k",
        type=int,
        default=0,
        help="morphological closing kernel before labelling (0 = off). "
        "Must match the value chosen by solar.tune, or the submission does "
        "not reproduce the tuned local PQ.",
    )
    ap.add_argument("--tta", action="store_true", help="4x flip test-time augmentation")
    ap.add_argument("--tile", type=int, default=None, help="tiled inference size")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    threshold = args.threshold if args.threshold is not None else cfg["eval"]["threshold"]
    min_area = args.min_area if args.min_area is not None else cfg["eval"]["min_area"]
    close_k = args.close_k
    tile = args.tile if args.tile is not None else cfg["eval"].get("tile")
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    pp_steps = parse_preprocess(cfg["data"].get("preprocess"))
    mask_offdisk = cfg["data"].get("mask_offdisk", False)
    offdisk_value = cfg["data"].get("offdisk_value", -3.0)

    test_dir = Path(args.test_dir)
    files = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    print(
        f"{len(files)} test images | threshold={threshold} min_area={min_area} "
        f"close_k={close_k} tta={args.tta} tile={tile}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    empty: list[str] = []
    counts: list[int] = []
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filament_id", "segmentation_rle"])

        for path in tqdm(files, desc="predict"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            # Preprocessing and normalisation must match training exactly, or
            # the model sees a distribution it was never fitted on.
            img, disk = preprocess_image(img, pp_steps)
            x = normalise_image(
                img,
                mean,
                std,
                disk=disk,
                mask_offdisk=mask_offdisk,
                offdisk_value=offdisk_value,
            )
            tensor = torch.from_numpy(x).unsqueeze(0).to(device)

            prob = predict_prob(model, tensor, tta=args.tta, tile=tile)
            labelmap = prob_to_labelmap(
                prob, threshold=threshold, min_area=min_area, close_k=close_k
            )
            rles = labelmap_to_rle(labelmap)

            stem = path.stem
            for inst, counts_rle in rles.items():
                writer.writerow([f"{stem}_{inst}", counts_rle])
                rows += 1
            counts.append(len(rles))
            if not rles:
                empty.append(stem)

    print(f"wrote {rows} filaments -> {out_path}")
    if counts:
        print(
            f"filaments/image: mean {np.mean(counts):.2f} "
            f"min {min(counts)} max {max(counts)}"
        )
    if empty:
        print(f"WARNING: {len(empty)} images got zero predictions: {empty[:8]}")


if __name__ == "__main__":
    main()
