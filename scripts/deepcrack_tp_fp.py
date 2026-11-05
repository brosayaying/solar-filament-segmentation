"""Extract and visualise true-positive and false-positive instances.

Runs the finetuned DeepCrack checkpoint over the val split, matches predictions
to GT with the same greedy IoU>0.5 rule the PQ metric uses, and crops each
matched (TP) and unmatched (FP) instance for inspection.

    python scripts/deepcrack_tp_fp.py --n 142

Writes to ``docs/DEEPCRACK_FINETUNE/tp_fp/``:

* ``tp_best.png``      highest-IoU true positives
* ``tp_marginal.png``  TPs closest to the 0.5 boundary
* ``fp_split.png``     FPs that overlap a GT (splits / merges)
* ``fp_pure.png``      FPs with no GT overlap at all
* ``summary.json``     counts and per-instance metadata

The split/pure distinction matters: ``docs/PQ_DIAGNOSIS.md`` found most FPs are
splits of an already-matched GT, which is a different failure from predicting
filament where there is none.
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
from solar.metrics import pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


def match_pairs(iou: np.ndarray, threshold: float = 0.5):
    """Greedy one-to-one matching, mirroring ``metrics.match_instances``.

    Returns ``(pairs, fp_ids, fn_ids)`` where pairs are ``(pred_id, gt_id, iou)``
    using 1-based instance ids as produced by ``prob_to_labelmap``.
    """
    n_pred, n_gt = iou.shape
    matched_pred = np.zeros(n_pred, dtype=bool)
    matched_gt = np.zeros(n_gt, dtype=bool)
    pairs = []
    for flat in np.argsort(iou, axis=None)[::-1]:
        i, j = divmod(int(flat), n_gt)
        if iou[i, j] <= threshold:
            break
        if matched_pred[i] or matched_gt[j]:
            continue
        matched_pred[i] = True
        matched_gt[j] = True
        pairs.append((i + 1, j + 1, float(iou[i, j])))
    fp = [i + 1 for i in range(n_pred) if not matched_pred[i]]
    fn = [j + 1 for j in range(n_gt) if not matched_gt[j]]
    return pairs, fp, fn


def crop_box(mask: np.ndarray, pad: int, h: int, w: int):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    )


@torch.no_grad()
def predict(model, img, device, mean, std):
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    return torch.sigmoid(model(t))[0, 0].float().cpu().numpy()


def render(examples, title, out_path, ncols=4):
    """One row per example: input, GT overlay, pred overlay, probability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not examples:
        return
    n = len(examples)
    # Keep the figure physically small so the saved PNG is not downscaled when
    # viewed; at 4.2in/column the 9pt titles became illegible.
    fig, axes = plt.subplots(n, ncols, figsize=(3.6 * ncols, 3.7 * n))
    if n == 1:
        axes = axes[None, :]

    for r, ex in enumerate(examples):
        img, gt, pred, prob = ex["img"], ex["gt"], ex["pred"], ex["prob"]

        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(ex["file"], fontsize=11)

        ov_gt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        ov_gt[gt > 0] = (0.45 * ov_gt[gt > 0] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        axes[r, 1].imshow(ov_gt)
        axes[r, 1].set_title("GT", fontsize=11)

        ov_pr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        ov_pr[pred > 0] = (0.35 * ov_pr[pred > 0] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
        axes[r, 2].imshow(ov_pr)
        axes[r, 2].set_title(ex["detail"], fontsize=11)

        im = axes[r, 3].imshow(prob, cmap="inferno", vmin=0, vmax=1)
        axes[r, 3].set_title(f"prob max={prob.max():.2f}", fontsize=11)
        plt.colorbar(im, ax=axes[r, 3], fraction=0.046)

        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path.name} ({n} examples)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/deepcrack_ft_256_long/best.pt")
    ap.add_argument("--n", type=int, default=142, help="val images to scan")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--mean", type=float, default=0.0)
    ap.add_argument("--std", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--pad", type=int, default=48, help="context px around instance")
    ap.add_argument("--per-figure", type=int, default=4)
    ap.add_argument("--out", default="docs/DEEPCRACK_FINETUNE/tp_fp")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    seen: set[str] = set()
    ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[: args.n]

    model = DeepCrack(in_chans=1, ngf=64, pretrained=False).to(device)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.weights} (epoch {ck.get('epoch')})")

    tps, fps = [], []
    n_tp = n_fp = n_fn = 0

    for iid in tqdm(ids, desc="scan"):
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        h, w = img.shape
        prob = predict(model, img, device, args.mean, args.std)
        pred = prob_to_labelmap(prob, threshold=args.threshold, min_area=args.min_area)
        gt = anns_to_labelmap(ann_index[iid], h, w)

        n_pred, n_gt = int(pred.max()), int(gt.max())
        if n_pred == 0 and n_gt == 0:
            continue
        if n_pred == 0:
            n_fn += n_gt
            continue
        if n_gt == 0:
            n_fp += n_pred
            continue

        iou = pairwise_iou(pred, gt, n_pred, n_gt)
        pairs, fp_ids, fn_ids = match_pairs(iou, 0.5)
        n_tp += len(pairs)
        n_fp += len(fp_ids)
        n_fn += len(fn_ids)

        stem = Path(im["file_name"]).stem

        for pid, gid, iou_val in pairs:
            box = crop_box((pred == pid) | (gt == gid), args.pad, h, w)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            tps.append(
                dict(
                    file=stem,
                    kind="TP",
                    label=f"iou={iou_val:.3f}",
                    detail=f"iou={iou_val:.3f}",
                    iou=iou_val,
                    area=int((pred == pid).sum()),
                    img=img[y0:y1, x0:x1].copy(),
                    gt=(gt[y0:y1, x0:x1] == gid).astype(np.uint8),
                    pred=(pred[y0:y1, x0:x1] == pid).astype(np.uint8),
                    prob=prob[y0:y1, x0:x1].copy(),
                )
            )

        for pid in fp_ids:
            m = pred == pid
            # Does this FP overlap any GT at all? Splits/merges do; noise does not.
            best = float(iou[pid - 1].max()) if n_gt else 0.0
            kind = "split" if best > 0.1 else "pure"
            box = crop_box(m, args.pad, h, w)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            fps.append(
                dict(
                    file=stem,
                    kind=kind,
                    label=f"area={int(m.sum())}",
                    detail=f"bestIoU={best:.2f} area={int(m.sum())}",
                    best_iou=best,
                    area=int(m.sum()),
                    img=img[y0:y1, x0:x1].copy(),
                    gt=(gt[y0:y1, x0:x1] > 0).astype(np.uint8),
                    pred=(pred[y0:y1, x0:x1] == pid).astype(np.uint8),
                    prob=prob[y0:y1, x0:x1].copy(),
                )
            )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nTP={n_tp}  FP={n_fp}  FN={n_fn}")
    fp_split = [e for e in fps if e["kind"] == "split"]
    fp_pure = [e for e in fps if e["kind"] == "pure"]
    print(f"  FP split/merge (overlaps GT): {len(fp_split)}")
    print(f"  FP pure (no GT overlap):      {len(fp_pure)}")

    k = args.per_figure
    tp_best = sorted(tps, key=lambda e: -e["iou"])[:k]
    tp_marg = sorted(tps, key=lambda e: abs(e["iou"] - 0.5))[:k]
    fp_split_top = sorted(fp_split, key=lambda e: -e["area"])[:k]
    fp_pure_top = sorted(fp_pure, key=lambda e: -e["area"])[:k]

    render(tp_best, "True positives — highest IoU", out / "tp_best.png")
    render(tp_marg, "True positives — closest to the 0.5 boundary", out / "tp_marginal.png")
    render(fp_split_top, "False positives — overlap a GT (splits / merges)", out / "fp_split.png")
    render(fp_pure_top, "False positives — no GT overlap", out / "fp_pure.png")

    summary = {
        "weights": args.weights,
        "n_images": len(ids),
        "threshold": args.threshold,
        "min_area": args.min_area,
        "tp": n_tp,
        "fp": n_fp,
        "fn": n_fn,
        "fp_split": len(fp_split),
        "fp_pure": len(fp_pure),
        "tp_iou_median": float(np.median([e["iou"] for e in tps])) if tps else None,
        "fp_area_median": float(np.median([e["area"] for e in fps])) if fps else None,
        "examples": {
            "tp_best": [{k2: e[k2] for k2 in ("file", "iou", "area")} for e in tp_best],
            "tp_marginal": [{k2: e[k2] for k2 in ("file", "iou", "area")} for e in tp_marg],
            "fp_split": [{k2: e[k2] for k2 in ("file", "best_iou", "area")} for e in fp_split_top],
            "fp_pure": [{k2: e[k2] for k2 in ("file", "best_iou", "area")} for e in fp_pure_top],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
