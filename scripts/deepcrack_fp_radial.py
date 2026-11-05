"""Where do the 'pure' false positives actually sit?

The TP/FP figures showed that several FPs with no GT overlap are arcs along the
solar limb. That contradicts the earlier claim that limb FPs were eliminated,
which was measured as a *mean over the whole limb ring* -- a metric that dilutes
a localized arc into near-zero.

This measures the signed radial distance of each pure FP from the disk edge, so
"at the limb" is a per-instance property rather than an average.

    python scripts/deepcrack_fp_radial.py
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
from solar.metrics import pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


def disk_geometry(img: np.ndarray, thr: int = 25):
    m = (img > thr).astype(np.uint8)
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx = stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] / 2.0
    cy = stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT] / 2.0
    r = float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi))
    return float(cx), float(cy), r


def main() -> None:
    weights = "runs/deepcrack_ft_256_long/best.pt"
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

    rows = []
    for iid in ids:
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        h, w = img.shape
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

        pred = prob_to_labelmap(prob, 0.5, 512)
        gt = anns_to_labelmap(ann_index[iid], h, w)
        n_pred, n_gt = int(pred.max()), int(gt.max())
        if n_pred == 0 or n_gt == 0:
            continue

        iou = pairwise_iou(pred, gt, n_pred, n_gt)
        # Separate match flags per side. Using one shared set is a bug: pred and
        # gt ids are both 0-based, so a matched GT id would falsely mark the
        # prediction with the same index as matched.
        matched_pred = np.zeros(n_pred, dtype=bool)
        matched_gt = np.zeros(n_gt, dtype=bool)
        for flat in np.argsort(iou, axis=None)[::-1]:
            i, j = divmod(int(flat), n_gt)
            if iou[i, j] <= 0.5:
                break
            if matched_pred[i] or matched_gt[j]:
                continue
            matched_pred[i] = True
            matched_gt[j] = True

        cx, cy, r = disk_geometry(img)
        yy, xx = np.mgrid[0:h, 0:w]
        rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        for pid in range(n_pred):
            if matched_pred[pid]:
                continue
            best = float(iou[pid].max())
            if best > 0.1:
                continue  # split/merge, analysed separately
            d = rad[pred == pid + 1]
            if len(d) == 0:
                continue
            rows.append(
                {
                    "file": Path(im["file_name"]).stem,
                    "area": int((pred == pid + 1).sum()),
                    "signed_dist": float(np.median(d) - r),
                    "best_iou": best,
                }
            )

    dist = np.array([r["signed_dist"] for r in rows])
    area = np.array([r["area"] for r in rows])
    near = np.abs(dist) < 40

    print(f"pure FPs analysed: {len(rows)}")
    print(f"  |dist from disk edge| < 40 px (LIMB): {near.sum():4d}  ({near.mean():.1%})")
    print(f"  elsewhere:                            {(~near).sum():4d}  ({(~near).mean():.1%})")
    print(f"  median signed distance: {np.median(dist):+.1f} px")
    print(f"  limb-FP median area: {np.median(area[near]) if near.any() else 0:.0f} px")
    print(f"  other-FP median area: {np.median(area[~near]) if (~near).any() else 0:.0f} px")

    out = ROOT / "docs/DEEPCRACK_FINETUNE/tp_fp/fp_radial.json"
    out.write_text(
        json.dumps(
            {
                "n_pure_fp": len(rows),
                "n_limb": int(near.sum()),
                "n_other": int((~near).sum()),
                "frac_limb": float(near.mean()),
                "median_signed_dist": float(np.median(dist)),
                "rows": sorted(rows, key=lambda r: -r["area"])[:200],
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
