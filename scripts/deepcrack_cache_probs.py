"""Cache val probabilities for the finetuned DeepCrack checkpoint.

``solar.tune`` can sweep post-processing, but it rebuilds the model via
``build_model`` (which would re-download/reload pretrained weights) and does not
dedup val files. This caches the exact 142-file dedup'd val set that all the
DeepCrack numbers are reported on, so a post-processing sweep is directly
comparable to the 0.3531 baseline.

    python scripts/deepcrack_cache_probs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import anns_to_labelmap, build_ann_index, load_coco  # noqa: E402
from solar.deepcrack import DeepCrack  # noqa: E402

cv2.setNumThreads(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/deepcrack_ft_256_long/best.pt")
    ap.add_argument("--out", default=None, help="defaults to <weights dir>/val_probs")
    args = ap.parse_args()

    weights = args.weights
    out_dir = (
        ROOT / args.out
        if args.out
        else (ROOT / weights).parent / "val_probs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / "splits/date_seed42.json").read_text())

    seen: set[str] = set()
    ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[:142]

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    model.load_state_dict(
        torch.load(ROOT / weights, map_location="cpu", weights_only=False)["model"]
    )
    model.eval()

    for i, iid in enumerate(tqdm(ids, desc="cache")):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
        np.save(out_dir / f"{i:04d}_prob.npy", prob.astype(np.float16))
        np.save(
            out_dir / f"{i:04d}_label.npy",
            anns_to_labelmap(ann_index[iid], im["height"], im["width"]).astype(np.int32),
        )

    print(f"cached {len(ids)} -> {out_dir}")


if __name__ == "__main__":
    main()
