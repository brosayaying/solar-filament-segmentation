"""Finish the decoder analysis once training has completed.

Runs the three measurements §21 needs and prints a markdown table ready to paste
into ``docs/DEEPCRACK_FINETUNE.md`` §21.3:

1. Best val PQ from the training log.
2. Full-val PQ/SQ/RQ at a threshold sweep (``deepcrack_eval`` protocol).
3. Per-head ranking vs localisation (``deepcrack_head_precision`` protocol), so
   the decoder can be compared against the no-decoder numbers: fused AUC 0.9968,
   pixel IoU 0.4987.

    python scripts/deepcrack_decoder_report.py
    python scripts/deepcrack_decoder_report.py --run runs/deepcrack_decoder_512
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

# Reference numbers from the no-decoder model (§20), for the comparison table.
NODEC_FUSED_AUC = 0.9968
NODEC_FUSED_IOU = 0.4987
NODEC_PQ = 0.3521
UNET_PQ = 0.4361


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC. ``labels`` must be bool (uint8 would index, not mask)."""
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def predict(model, img, device, mean, std):
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    return torch.sigmoid(model(t))[0, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/deepcrack_decoder_512")
    ap.add_argument("--n", type=int, default=142)
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--std", type=float, default=1.0)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.45, 0.55, 0.7])
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--auc-pixels", type=int, default=200000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = ROOT / args.run
    log = run / "log.jsonl"
    if not log.exists():
        raise SystemExit(f"no log at {log}")

    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    best = max(rows, key=lambda r: r["val_pq"])
    print(f"=== {args.run} ===")
    print(f"epochs trained : {len(rows)}")
    print(f"best val PQ    : {best['val_pq']:.4f} @ epoch {best['epoch']}")
    print(f"final val PQ   : {rows[-1]['val_pq']:.4f}")
    if len(rows) < 160:
        print(f"WARNING: run has only {len(rows)} epochs; results are preliminary.")

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

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False, decoder=True).to(device)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck))
    model.eval()

    accs = {t: PQAccumulator() for t in args.thresholds}
    inter = {t: 0 for t in args.thresholds}
    union = {t: 0 for t in args.thresholds}
    auc_sum = 0.0
    rng = np.random.default_rng(0)

    for iid in tqdm(val_ids, desc="eval"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        p = predict(model, img, device, args.mean, args.std)
        g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        gbin = (g > 0).astype(bool)

        flat = gbin.ravel()
        sub = rng.choice(len(flat), size=min(args.auc_pixels, len(flat)), replace=False)
        auc_sum += auc(p.ravel()[sub], flat[sub])

        for t in args.thresholds:
            pred = p > t
            inter[t] += int((pred & gbin).sum())
            union[t] += int((pred | gbin).sum())
            accs[t].update(prob_to_labelmap(p, threshold=t, min_area=args.min_area), g)

    n = len(val_ids)
    print(f"\n=== full val (n={n}, min_area={args.min_area}) ===")
    print(f"{'thr':>5} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'pixIoU':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    best_t, best_s = None, None
    for t in args.thresholds:
        s = accs[t].summary()
        iou = inter[t] / max(union[t], 1)
        print(
            f"{t:5.2f} {s['pq']:7.4f} {s['sq']:7.4f} {s['rq']:7.4f} {iou:7.4f} "
            f"{s['tp']:5d} {s['fp']:5d} {s['fn']:5d}"
        )
        if best_s is None or s["pq"] > best_s["pq"]:
            best_t, best_s = t, s

    fused_auc = auc_sum / n
    fused_iou = inter[best_t] / max(union[best_t], 1)

    print(f"\n=== decoder vs no-decoder (§20) ===")
    print(f"{'metric':<22} {'no decoder':>12} {'decoder':>10} {'delta':>9}")
    print(
        f"{'fused pixel AUC':<22} {NODEC_FUSED_AUC:>12.4f} {fused_auc:>10.4f} "
        f"{fused_auc - NODEC_FUSED_AUC:>+9.4f}"
    )
    print(
        f"{'fused pixel IoU':<22} {NODEC_FUSED_IOU:>12.4f} {fused_iou:>10.4f} "
        f"{fused_iou - NODEC_FUSED_IOU:>+9.4f}"
    )
    print(
        f"{'best PQ':<22} {NODEC_PQ:>12.4f} {best_s['pq']:>10.4f} "
        f"{best_s['pq'] - NODEC_PQ:>+9.4f}"
    )
    print(f"{'resnet34 UNet PQ':<22} {UNET_PQ:>12.4f}")

    print("\n=== markdown for §21.3 ===")
    print("| model | params | fused AUC | fused IoU | best PQ |")
    print("|---|---|---|---|---|")
    print(f"| DeepCrack, no decoder | 14.72 M | {NODEC_FUSED_AUC:.4f} | "
          f"{NODEC_FUSED_IOU:.4f} | {NODEC_PQ:.4f} |")
    print(f"| **DeepCrack + decoder** | 24.90 M | {fused_auc:.4f} | "
          f"{fused_iou:.4f} | **{best_s['pq']:.4f}** |")
    print(f"| resnet34 UNet (reference) | 24.4 M | — | — | {UNET_PQ:.4f} |")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "run": args.run,
                    "epochs": len(rows),
                    "best_log_pq": best["val_pq"],
                    "best_log_epoch": best["epoch"],
                    "n_images": n,
                    "fused_auc": fused_auc,
                    "fused_iou": fused_iou,
                    "best_threshold": best_t,
                    "best": best_s,
                    "sweep": {str(t): accs[t].summary() for t in args.thresholds},
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
