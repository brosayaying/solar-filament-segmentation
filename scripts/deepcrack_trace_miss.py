"""Trace a missed filament through DeepCrack's five side outputs.

DeepCrack is a VGG-style encoder with a 1x1 side head at each of five depths
(stride 1, 2, 4, 8, 16), all upsampled to input resolution and fused by a 1x1
conv. That structure makes "where does the signal fall off" a directly
answerable question: if the shallow side heads respond and the deep ones do not,
the failure is in the deep features; if none respond, the input itself is not
being read as a filament.

For a chosen GT instance this reports, per side output:

* mean/max probability inside the GT mask
* mean probability in a surrounding ring (the local background)
* the resulting contrast, and the AUC of that side output for this instance

    python scripts/deepcrack_trace_miss.py --file 20150627133754Bh --gt-id 5
    python scripts/deepcrack_trace_miss.py --file 20150627133754Bh --gt-id 5 --save
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

SIDE_NAMES = ["side1 (stride 1)", "side2 (stride 2)", "side3 (stride 4)",
              "side4 (stride 8)", "side5 (stride 16)", "fused"]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC; avoids a sklearn dependency for one number."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with tie handling via average ranks.
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/deepcrack_ft_256_long/best.pt")
    ap.add_argument("--file", default="20150627133754Bh")
    ap.add_argument("--gt-id", type=int, default=5)
    ap.add_argument("--pad", type=int, default=80)
    ap.add_argument("--ring", type=int, default=41)
    ap.add_argument("--save", action="store_true", help="write a figure")
    ap.add_argument("--out", default="docs/DEEPCRACK_FINETUNE/trace")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    by_file = {im["file_name"]: im for im in coco["images"]}

    im = by_file[args.file + ".jpeg"]
    iid = im["id"]
    h, w = im["height"], im["width"]
    img = cv2.imread(
        str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
    )
    gt = anns_to_labelmap(ann_index[iid], h, w)
    if args.gt_id > int(gt.max()):
        raise SystemExit(f"gt_id {args.gt_id} > n_gt {int(gt.max())}")
    gm = gt == args.gt_id

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    model.load_state_dict(
        torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)["model"]
    )
    model.eval()

    x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = model.net(x)
    probs = [torch.sigmoid(o)[0, 0].float().cpu().numpy() for o in outs]

    # Local ring around the GT instance = the background it must be separated from.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.ring, args.ring))
    ring = cv2.dilate(gm.astype(np.uint8), k).astype(bool) & ~gm
    local = gm | ring

    print(f"{args.file}  gt_id={args.gt_id}  area={int(gm.sum())}  "
          f"n_gt={int(gt.max())}")
    print(f"local window: {int(local.sum())} px "
          f"({int(gm.sum())} filament + {int(ring.sum())} ring)\n")
    print(f"{'side output':>18} {'mean_in':>8} {'max_in':>8} {'mean_ring':>10} "
          f"{'contrast':>9} {'AUC':>7} {'frac>0.45':>10}")
    print("-" * 78)

    rows = []
    for name, p in zip(SIDE_NAMES, probs):
        inside = p[gm]
        outside = p[ring]
        a = auc(p[local], gm[local])
        row = {
            "side": name,
            "mean_in": float(inside.mean()),
            "max_in": float(inside.max()),
            "mean_ring": float(outside.mean()),
            "contrast": float(inside.mean() - outside.mean()),
            "auc": a,
            "frac_above_thr": float((inside > 0.45).mean()),
        }
        rows.append(row)
        print(f"{name:>18} {row['mean_in']:>8.4f} {row['max_in']:>8.4f} "
              f"{row['mean_ring']:>10.4f} {row['contrast']:>+9.4f} "
              f"{a:>7.3f} {row['frac_above_thr']:>10.1%}")

    # Where does it fall off? First side output whose AUC drops below 0.5.
    print()
    aucs = [r["auc"] for r in rows[:5]]
    best = int(np.nanargmax(aucs))
    print(f"best side output: {SIDE_NAMES[best]} (AUC {aucs[best]:.3f})")
    if aucs[best] < 0.6:
        print("=> NO side output separates this filament from its surroundings.")
        print("   The failure is upstream of the side heads: the encoder never")
        print("   represents this structure as distinct from background.")
    else:
        print("=> The encoder DOES represent it; the loss is in fusion/thresholding.")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.file}_gt{args.gt_id}.json").write_text(
        json.dumps({"file": args.file, "gt_id": args.gt_id,
                    "gt_area": int(gm.sum()), "sides": rows}, indent=2)
    )

    if args.save:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ys, xs = np.nonzero(gm)
        y0, y1 = max(0, ys.min() - args.pad), min(h, ys.max() + args.pad)
        x0, x1 = max(0, xs.min() - args.pad), min(w, xs.max() + args.pad)

        fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
        axes = axes.ravel()
        axes[0].imshow(img[y0:y1, x0:x1], cmap="gray")
        axes[0].set_title("input", fontsize=11)
        ov = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        ov[gm[y0:y1, x0:x1]] = (0.4 * ov[gm[y0:y1, x0:x1]] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        axes[1].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        axes[1].set_title("GT (green)", fontsize=11)
        for i, (name, p) in enumerate(zip(SIDE_NAMES, probs)):
            ax = axes[i + 2]
            im2 = ax.imshow(p[y0:y1, x0:x1], cmap="inferno", vmin=0, vmax=1)
            ax.set_title(f"{name}\nAUC={rows[i]['auc']:.3f}", fontsize=10)
            plt.colorbar(im2, ax=ax, fraction=0.046)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{args.file} gt_id={args.gt_id} — missed FN traced through the network",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"{args.file}_gt{args.gt_id}.png", dpi=110)
        plt.close(fig)
        print(f"wrote {out / f'{args.file}_gt{args.gt_id}.png'}")


if __name__ == "__main__":
    main()
