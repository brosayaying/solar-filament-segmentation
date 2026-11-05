"""Score a DeepCrack checkpoint on the held-out val split (full-image inference).

Used to check whether a change helped before spending GPU time on finetuning.
Full-image inference is the right protocol here: the CNN family is scale-robust
(``docs/UNDERFITTING.md``: full image 0.3757 vs tiled 1024 0.3737), so there is
no need to tile, and full-image is what the submission does.

    python scripts/deepcrack_eval.py --weights runs/deepcrack_bn_recalib.pt
    python scripts/deepcrack_eval.py --weights runs/deepcrack_ft_256/best.pt --disk

Reports PQ/SQ/RQ over a threshold x min_area sweep. ``--disk`` masks predictions
to the solar disk, which is worth checking separately because the zero-shot
failure was concentrated on the limb.
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
from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


def disk_mask(img: np.ndarray, thr: int = 25) -> np.ndarray:
    m = (img > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones_like(m)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    disk = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(disk)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return filled


@torch.no_grad()
def predict(model, img: np.ndarray, device, mean: float, std: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    return torch.sigmoid(model(t))[0, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint; omit for raw pretrained")
    ap.add_argument("--n", type=int, default=60, help="val images")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--std", type=float, default=1.0)
    ap.add_argument("--disk", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--min-areas", type=int, nargs="+", default=[64, 512, 2048])
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    # One annotator variant per file: averaging over variants measures agreement
    # with a particular human, not correctness (docs/METRIC_CEILING.md).
    seen: set[str] = set()
    val_ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        val_ids.append(iid)
    val_ids = val_ids[: args.n]

    model = DeepCrack(in_chans=1, ngf=64, pretrained=args.weights is None).to(device)
    if args.weights:
        ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck))
        print(f"loaded {args.weights}")
    model.eval()

    probs, gts = [], []
    for iid in tqdm(val_ids, desc="predict"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        p = predict(model, img, device, args.mean, args.std)
        if args.disk:
            p = p * disk_mask(img)
        probs.append(p)
        gts.append(anns_to_labelmap(ann_index[iid], im["height"], im["width"]))

    results = []
    for thr in args.thresholds:
        for ma in args.min_areas:
            acc = PQAccumulator()
            for p, g in zip(probs, gts):
                acc.update(prob_to_labelmap(p, threshold=thr, min_area=ma), g)
            s = acc.summary()
            s.update(threshold=thr, min_area=ma)
            results.append(s)
            print(
                f"  thr={thr:.2f} min_area={ma:5d}  PQ={s['pq']:.4f} "
                f"SQ={s['sq']:.4f} RQ={s['rq']:.4f} "
                f"TP={s['tp']} FP={s['fp']} FN={s['fn']}"
            )

    best = max(results, key=lambda r: r["pq"])
    print(
        f"\nbest PQ={best['pq']:.4f} at thr={best['threshold']} "
        f"min_area={best['min_area']} (n={len(val_ids)})"
    )
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "weights": args.weights,
                    "n_images": len(val_ids),
                    "disk": args.disk,
                    "mean": args.mean,
                    "std": args.std,
                    "best": best,
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
