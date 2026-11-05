"""Where does PQ go, given a near-perfect probability map?

The fusion audit shows the fused map has pixel AUC 0.9968 -- it ranks filament
pixels above background almost perfectly. Yet PQ is 0.35. Those two facts are
only compatible if the errors are concentrated in a thin band that is a tiny
fraction of all pixels but a large fraction of each *instance's* area.

This script measures exactly that:

1. Pixel-level IoU of the binarised map against GT, at the best global threshold.
2. What fraction of the error pixels lie within k px of the GT boundary. If most
   of the error is a 1-3 px band, the model has found every filament and is only
   off on its edge.
3. Per-instance best IoU, and how much PQ an oracle per-image threshold would buy.

    python scripts/deepcrack_error_anatomy.py --weights runs/deepcrack_ft_256_long/best.pt
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
from solar.metrics import PQAccumulator, match_instances  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


@torch.no_grad()
def predict(model, img, device, mean, std):
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    return torch.sigmoid(model(t))[0, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=142)
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--std", type=float, default=1.0)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.45, 0.6, 0.75])
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    seen: set[str] = set()
    val_ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        val_ids.append(iid)
    val_ids = val_ids[: args.n]

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck))
    model.eval()
    print(f"loaded {args.weights}  ({len(val_ids)} val files)")

    # ---- Pass 1: global threshold sweep, plus pixel-IoU and error-band stats.
    accs = {t: PQAccumulator() for t in args.thresholds}
    pix_inter = {t: 0 for t in args.thresholds}
    pix_union = {t: 0 for t in args.thresholds}
    band_hits = {t: np.zeros(6, np.int64) for t in args.thresholds}
    band_tot = {t: 0 for t in args.thresholds}

    for iid in tqdm(val_ids, desc="pass1"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        p = predict(model, img, device, args.mean, args.std)
        g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        gbin = (g > 0).astype(np.uint8)

        # Distance (in px) from each pixel to the nearest GT boundary.
        edge = cv2.morphologyEx(gbin, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        dist = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5)

        for t in args.thresholds:
            pred = (p > t).astype(np.uint8)
            accs[t].update(prob_to_labelmap(p, threshold=t, min_area=args.min_area), g)
            pix_inter[t] += int((pred & gbin).sum())
            pix_union[t] += int((pred | gbin).sum())

            err = pred != gbin
            n_err = int(err.sum())
            band_tot[t] += n_err
            if n_err:
                d = dist[err]
                for k in range(6):
                    band_hits[t][k] += int((d <= k).sum())

    print("\n=== global threshold sweep ===")
    print(f"{'thr':>5} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'pixIoU':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    best_t, best_pq = None, -1
    for t in args.thresholds:
        s = accs[t].summary()
        piou = pix_inter[t] / max(pix_union[t], 1)
        print(
            f"{t:5.2f} {s['pq']:7.4f} {s['sq']:7.4f} {s['rq']:7.4f} {piou:7.4f} "
            f"{s['tp']:5d} {s['fp']:5d} {s['fn']:5d}"
        )
        if s["pq"] > best_pq:
            best_t, best_pq = t, s["pq"]

    print(f"\n=== where the error pixels are (thr={best_t}) ===")
    print("distance from GT boundary -> share of all misclassified pixels")
    tot = band_tot[best_t]
    for k in range(6):
        share = band_hits[best_t][k] / max(tot, 1)
        print(f"  <= {k} px : {share*100:6.2f}%")

    # ---- Pass 2: oracle per-image threshold (upper bound on post-processing).
    print("\n=== oracle per-image threshold (upper bound, uses GT) ===")
    oracle_pq = 0.0
    oracle_t = []
    for iid in tqdm(val_ids, desc="pass2"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        p = predict(model, img, device, args.mean, args.std)
        g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        best = (-1.0, None)
        for t in np.arange(0.2, 0.91, 0.05):
            acc = PQAccumulator()
            acc.update(prob_to_labelmap(p, threshold=float(t), min_area=args.min_area), g)
            s = acc.summary()
            if s["pq"] > best[0]:
                best = (s["pq"], float(t))
        oracle_pq += best[0]
        oracle_t.append(best[1])
    oracle_pq /= len(val_ids)
    print(f"  mean per-image best PQ = {oracle_pq:.4f}")
    print(f"  median best threshold  = {np.median(oracle_t):.2f}")
    print(f"  global-threshold PQ    = {best_pq:.4f}")
    print(f"  headroom from per-image thresholding = {oracle_pq - best_pq:+.4f}")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "weights": args.weights,
                    "n_images": len(val_ids),
                    "sweep": {
                        str(t): {
                            **accs[t].summary(),
                            "pixel_iou": pix_inter[t] / max(pix_union[t], 1),
                        }
                        for t in args.thresholds
                    },
                    "best_threshold": best_t,
                    "error_band_share": {
                        f"le_{k}px": band_hits[best_t][k] / max(tot, 1) for k in range(6)
                    },
                    "oracle_per_image_pq": oracle_pq,
                    "oracle_median_threshold": float(np.median(oracle_t)),
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
