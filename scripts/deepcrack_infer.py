"""Run the pretrained DeepCrack network (yhlleo/DeepSegmentor) on MAGFiLO images.

No retraining. This is a zero-shot transfer probe: DeepCrack was trained on
asphalt crack images, where the target is a *thin dark curve on a textured
background*. Filaments are also thin dark curves on a textured background, so
the question is whether the learned "thin dark structure" prior transfers.

    python scripts/deepcrack_infer.py --n 6 --out docs/DEEPCRACK

The network is vendored in ``src/solar/deepcrack_net.py``. Only the weights
(``pretrained_net_G.pth``, Google Drive id ``1MAwIlWw27feU8EEkP8pgbia7kcuxBwEG``)
are external, extracted to ``/media/yasir/HDD/deepcrack_weights``.

Preprocessing follows ``test_example.ipynb`` exactly: RGB, ``ToTensor`` then
``Normalize((0.5,)*3, (0.5,)*3)``, i.e. ``x/255*2 - 1``. The network is fully
convolutional (5 stride-2 pools, 1x1 side heads, bilinear upsample back to the
input size), so 2048x2048 runs in one pass at ~3.3 GB.
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
from solar.deepcrack_net import define_deepcrack  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

WEIGHTS = Path("/media/yasir/HDD/deepcrack_weights/extracted/deepcrack/pretrained_net_G.pth")
DATA = ROOT / "MAGFiLO_1.0_Kaggle_2026"

cv2.setNumThreads(4)


def build_model(device: torch.device):
    model = define_deepcrack(in_nc=3, num_classes=1, ngf=64).eval()
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device)


def preprocess(img_gray: np.ndarray) -> torch.Tensor:
    """Grayscale uint8 HxW -> 1x3xHxW tensor in [-1, 1] (upstream convention)."""
    rgb = np.repeat(img_gray[..., None], 3, axis=2).astype(np.float32) / 255.0
    x = rgb * 2.0 - 1.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict(model, img_gray: np.ndarray, device: torch.device) -> np.ndarray:
    """Return the fused sigmoid probability map at the input resolution."""
    x = preprocess(img_gray).to(device)
    fused = model(x)[-1]
    return torch.sigmoid(fused)[0, 0].float().cpu().numpy()


def disk_mask(img: np.ndarray, thr: int = 25, margin: int = 0) -> np.ndarray:
    """Largest bright blob = solar disk, holes filled, optionally eroded."""
    m = (img > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones_like(m)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    disk = (lab == biggest).astype(np.uint8)
    disk = cv2.morphologyEx(disk, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(disk)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
        filled = cv2.erode(filled, k)
    return filled


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_val(model, device, n_images: int, thresholds, min_areas, use_disk: bool):
    """Sweep threshold / min_area on the held-out val split and report PQ."""
    coco = load_coco(DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / "splits/date_seed42.json").read_text())

    # One annotator variant per file, so the val set is not triple-counted.
    seen: set[str] = set()
    val_ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        val_ids.append(iid)
    val_ids = val_ids[:n_images]

    probs, gts = [], []
    for iid in val_ids:
        im = images[iid]
        img = cv2.imread(str(DATA / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        prob = predict(model, img, device)
        if use_disk:
            prob = prob * disk_mask(img)
        probs.append(prob)
        gts.append(anns_to_labelmap(ann_index[iid], im["height"], im["width"]))

    results = []
    for thr in thresholds:
        for ma in min_areas:
            acc = PQAccumulator()
            for prob, gt in zip(probs, gts):
                acc.update(prob_to_labelmap(prob, threshold=thr, min_area=ma), gt)
            s = acc.summary()
            s.update(threshold=thr, min_area=ma)
            results.append(s)
            print(
                f"  thr={thr:.2f} min_area={ma:5d}  PQ={s['pq']:.4f} "
                f"SQ={s['sq']:.4f} RQ={s['rq']:.4f} "
                f"TP={s['tp']} FP={s['fp']} FN={s['fn']}"
            )
    return results, len(val_ids)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


def save_panel(img, prob, gt, out_path: Path, title: str, thr: float, min_area: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = prob_to_labelmap(prob, threshold=thr, min_area=min_area)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if gt is not None:
        overlay[gt > 0] = (0.5 * overlay[gt > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    pred_vis = overlay.copy()
    pred_vis[lab > 0] = (0.35 * pred_vis[lab > 0] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"{title}\ninput")
    im = axes[1].imshow(prob, cmap="inferno", vmin=0, vmax=max(0.5, float(prob.max())))
    axes[1].set_title(f"DeepCrack prob (max {prob.max():.2f})")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(pred_vis)
    axes[2].set_title(f"pred @ thr={thr} min_area={min_area}\n{int(lab.max())} instances")
    if gt is not None:
        axes[3].imshow(overlay)
        axes[3].set_title(f"GT (green)\n{int(gt.max())} instances")
    else:
        axes[3].axis("off")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="number of test images to visualise")
    ap.add_argument("--out", default="docs/DEEPCRACK")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--score", action="store_true", help="also sweep PQ on the val split")
    ap.add_argument("--score-n", type=int, default=40)
    ap.add_argument("--disk", action="store_true", help="mask predictions to the solar disk")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    test_files = sorted((DATA / "test/test_images").glob("*.jpeg"))[: args.n]
    print(f"{len(test_files)} test images -> {out}")

    stats = []
    for path in test_files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        prob = predict(model, img, device)
        if args.disk:
            prob = prob * disk_mask(img)
        lab = prob_to_labelmap(prob, threshold=args.threshold, min_area=args.min_area)
        np.save(out / f"{path.stem}_prob.npy", prob.astype(np.float16))
        save_panel(img, prob, None, out / f"{path.stem}.png", path.stem, args.threshold, args.min_area)
        row = {
            "file": path.name,
            "prob_max": float(prob.max()),
            "prob_mean": float(prob.mean()),
            "frac_gt_0.5": float((prob > 0.5).mean()),
            "frac_gt_0.1": float((prob > 0.1).mean()),
            "n_instances": int(lab.max()),
            "pred_area_frac": float((lab > 0).mean()),
        }
        stats.append(row)
        print(
            f"  {path.name}  max={row['prob_max']:.3f} mean={row['prob_mean']:.4f} "
            f"area@0.1={row['frac_gt_0.1']:.4f} inst={row['n_instances']}"
        )

    (out / "test_stats.json").write_text(json.dumps(stats, indent=2))

    if args.score:
        print("\nPQ sweep on val split (one annotator variant per file):")
        results, n = score_val(
            model,
            device,
            args.score_n,
            thresholds=[0.05, 0.1, 0.2, 0.3, 0.4],
            min_areas=[64, 512, 2048],
            use_disk=args.disk,
        )
        (out / "val_pq.json").write_text(
            json.dumps({"n_images": n, "results": results}, indent=2)
        )


if __name__ == "__main__":
    main()
