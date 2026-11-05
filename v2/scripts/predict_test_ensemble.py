"""5-fold ensemble inference on the test set -> submission CSV.

Why this exists
---------------
The OOF ensemble number (0.4346) is contaminated: each training file is held out
by exactly one fold, so 4 of the 5 ensemble members trained on it. The test set
is the only data **no fold has ever seen**, so an ensemble there is honest. It
has no GT, so the only readout is the public leaderboard — 2-decimal, ~90 images,
±0.02 noise — but it is a real measurement rather than a leaky one.

How the predictions are combined
--------------------------------
**Probability averaging, before thresholding.** Each fold produces a sigmoid map
in [0, 1]; the five are averaged and the mean map is thresholded once. This is
the meaningful combination for a segmentation ensemble:

* Averaging *after* binarisation (mask voting) throws away the confidence
  information that makes an ensemble better than its members, and turns a
  disagreement about a boundary pixel into a coin flip.
* Averaging in probability space lets a pixel that 3 of 5 folds are confident
  about survive even if 2 are not, which is exactly the variance reduction we
  want. It also keeps the map calibrated, which Phase 2's ``q̂`` regressor needs.

Post-processing is the setting tuned on the **honest** OOF sweep
(thr 0.50 / min_area 256 / close 15 -> PQ 0.4220), not the leaky ensemble sweep.
The leaky sweep preferred 0.45, but that preference was fitted on contaminated
predictions and is not trustworthy.

    python v2/scripts/predict_test_ensemble.py
    python v2/scripts/predict_test_ensemble.py --tta --threshold 0.45
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import normalise_image, parse_preprocess, preprocess_image  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import labelmap_to_rle, prob_to_labelmap  # noqa: E402
from solar.predict import predict_prob  # noqa: E402

RUNS = ROOT / "v2/runs"
TEST_DIR = ROOT / "MAGFiLO_1.0_Kaggle_2026/test/test_images"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-area", type=int, default=256)
    ap.add_argument("--close-k", type=int, default=15)
    ap.add_argument("--tta", action="store_true", help="4x flip TTA per fold")
    ap.add_argument("--out", default=str(ROOT / "submissions/seg_v2_ensemble.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # All folds share one data config; take it from fold 0.
    ckpt0 = torch.load(RUNS / f"seg_v2_fold{args.folds[0]}/best.pt",
                       map_location="cpu", weights_only=False)
    cfg = ckpt0["cfg"]
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    pp_steps = parse_preprocess(cfg["data"].get("preprocess"))
    mask_offdisk = cfg["data"].get("mask_offdisk", False)
    offdisk_value = cfg["data"].get("offdisk_value", -3.0)

    models = []
    for f in args.folds:
        ck = torch.load(RUNS / f"seg_v2_fold{f}/best.pt", map_location="cpu",
                        weights_only=False)
        m = build_model(ck["cfg"]).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
        print(f"fold {f}: epoch {ck.get('epoch')} val pq {ck.get('pq', 0):.4f}")

    files = sorted(p for p in TEST_DIR.iterdir()
                   if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    print(f"\n{len(files)} test images | {len(models)} folds | "
          f"thr {args.threshold} area {args.min_area} close {args.close_k} "
          f"tta {args.tta}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    counts: list[int] = []
    empty: list[str] = []

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filament_id", "segmentation_rle"])

        for path in tqdm(files, desc="predict"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(path)
            # Preprocessing and normalisation must match training exactly.
            img, disk = preprocess_image(img, pp_steps)
            x = normalise_image(img, mean, std, disk=disk,
                                mask_offdisk=mask_offdisk,
                                offdisk_value=offdisk_value)
            tensor = torch.from_numpy(x).unsqueeze(0).to(device)

            # Average in probability space, then threshold once.
            acc = None
            for m in models:
                p = predict_prob(m, tensor, tta=args.tta)
                acc = p if acc is None else acc + p
            prob = acc / len(models)

            labelmap = prob_to_labelmap(
                prob, threshold=args.threshold, min_area=args.min_area,
                close_k=args.close_k,
            )
            rles = labelmap_to_rle(labelmap)
            for inst, counts_rle in rles.items():
                writer.writerow([f"{path.stem}_{inst}", counts_rle])
                rows += 1
            counts.append(len(rles))
            if not rles:
                empty.append(path.stem)

    print(f"\nwrote {rows} filaments -> {out_path}")
    if counts:
        print(f"filaments/image: mean {np.mean(counts):.2f} "
              f"min {min(counts)} max {max(counts)}")
    if empty:
        print(f"WARNING: {len(empty)} images got zero predictions: {empty[:8]}")


if __name__ == "__main__":
    main()
