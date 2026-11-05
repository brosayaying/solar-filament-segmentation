"""Sweep post-processing hyperparameters against cached validation probabilities.

Inference is the expensive part, so probabilities are computed once and cached.
The sweep then only re-runs thresholding, morphology, and connected components,
which makes a 40-config grid cost minutes instead of hours.

    python -m solar.tune --ckpt runs/baseline_resnet34/best.pt
    python -m solar.tune --ckpt runs/baseline_resnet34/best.pt --recompute
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FilamentDataset, collate_val, load_coco
from .metrics import PQAccumulator
from .models import build_model
from .postprocess import prob_to_labelmap_watershed


def build_cache(ckpt: dict, cfg: dict, split: dict, cache_dir: Path, device, num_workers: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data"]["root"])
    coco = load_coco(root / cfg["data"]["ann_file"])
    ds = FilamentDataset(
        root,
        coco,
        split["val"],
        mode="val",
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        preprocess=cfg["data"].get("preprocess"),
        mask_offdisk=cfg["data"].get("mask_offdisk", False),
        offdisk_value=cfg["data"].get("offdisk_value", -3.0),
        preprocess_cache=cfg["data"].get("preprocess_cache"),
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for i, batch in enumerate(tqdm(loader, desc="cache probs")):
        with torch.no_grad():
            logits = model(batch["image"].to(device, non_blocking=True))
        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        np.save(cache_dir / f"{i:04d}_prob.npy", prob.astype(np.float16))
        np.save(cache_dir / f"{i:04d}_label.npy", batch["label"][0].numpy().astype(np.int32))


def load_cache(cache_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for p in sorted(cache_dir.glob("*_prob.npy")):
        probs.append(np.load(p).astype(np.float32))
        labels.append(np.load(p.with_name(p.name.replace("_prob", "_label"))))
    if not probs:
        raise FileNotFoundError(f"no cached probabilities in {cache_dir}")
    return probs, labels


def labelmap_from_prob(
    prob: np.ndarray,
    threshold: float,
    min_area: int,
    close_k: int,
    method: str = "cc",
    h_maxima: int = 8,
    min_distance: int = 12,
) -> np.ndarray:
    """Threshold -> optional closing -> instance labelling."""
    if method == "watershed":
        p = prob
        if close_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            p = cv2.morphologyEx((prob > threshold).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            p = p.astype(np.float32)
        return prob_to_labelmap_watershed(
            p, threshold, min_area, h_maxima=h_maxima, min_distance=min_distance
        )

    binary = (prob > threshold).astype(np.uint8)
    if close_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if binary.sum() == 0:
        return np.zeros(prob.shape, np.int32)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros(prob.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def evaluate(
    probs: list[np.ndarray],
    labels: list[np.ndarray],
    threshold: float,
    min_area: int,
    close_k: int,
    method: str = "cc",
    h_maxima: int = 8,
    min_distance: int = 12,
    bootstrap: bool = False,
    n_boot: int = 1000,
) -> dict:
    """Score one post-processing setting.

    ``bootstrap=True`` adds a per-image bootstrap CI for PQ/SQ/RQ. Resampling
    *images* and recomputing the global aggregate is the correct uncertainty for
    a dataset-level PQ; the per-image p10-p90 spread is not (it is the spread of
    individual images, not the standard error of the mean).
    """
    acc = PQAccumulator(threshold=0.5)
    for prob, label in zip(probs, labels):
        acc.update(
            labelmap_from_prob(
                prob, threshold, min_area, close_k, method, h_maxima, min_distance
            ),
            label,
            record=bootstrap,
        )
    out = acc.summary()
    if bootstrap:
        ci = acc.bootstrap_ci(n_boot=n_boot)
        for key, (lo, hi) in ci.items():
            out[f"{key}_lo"] = lo
            out[f"{key}_hi"] = hi
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--cache", default=None, help="defaults to <ckpt dir>/val_probs")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--thresholds", default="0.35,0.45,0.5,0.55,0.65")
    ap.add_argument("--min-areas", default="0,64,128,256")
    ap.add_argument("--close-ks", default="0,5")
    ap.add_argument("--method", default="cc", choices=["cc", "watershed"])
    ap.add_argument("--h-maximas", default="4,8,16")
    ap.add_argument("--min-distances", default="12")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="add a per-image bootstrap CI to every grid point (slower)",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(args.cache) if args.cache else ckpt_path.parent / "val_probs"
    if args.recompute or not cache_dir.exists() or not any(cache_dir.glob("*_prob.npy")):
        with open(args.split) as fh:
            split = json.load(fh)
        build_cache(ckpt, cfg, split, cache_dir, device, args.num_workers)

    probs, labels = load_cache(cache_dir)
    print(f"{len(probs)} cached validation images from {cache_dir}")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    min_areas = [int(x) for x in args.min_areas.split(",")]
    close_ks = [int(x) for x in args.close_ks.split(",")]
    h_maximas = [int(x) for x in args.h_maximas.split(",")]
    min_distances = [int(x) for x in args.min_distances.split(",")]

    if args.method == "watershed":
        grid = list(itertools.product(thresholds, min_areas, close_ks, h_maximas, min_distances))
    else:
        grid = [(t, a, c, 0, 0) for t, a, c in itertools.product(thresholds, min_areas, close_ks)]

    results = []
    for thr, area, ck, hm, md in tqdm(grid, desc="sweep"):
        m = evaluate(
            probs,
            labels,
            thr,
            area,
            ck,
            args.method,
            hm,
            md,
            bootstrap=args.bootstrap,
            n_boot=args.n_boot,
        )
        results.append(
            {
                "method": args.method,
                "threshold": thr,
                "min_area": area,
                "close_k": ck,
                "h_maxima": hm,
                "min_distance": md,
                **m,
            }
        )

    results.sort(key=lambda r: r["pq"], reverse=True)

    print(
        f"\n{'thr':>5} {'min_area':>9} {'close':>6} {'hmax':>5} {'mdis':>5} "
        f"{'PQ':>7} {'SQ':>7} {'RQ':>7} {'tp':>6} {'fp':>6} {'fn':>6}"
    )
    print("-" * 80)
    for r in results[:15]:
        print(
            f"{r['threshold']:>5.2f} {r['min_area']:>9} {r['close_k']:>6} "
            f"{r['h_maxima']:>5} {r['min_distance']:>5} "
            f"{r['pq']:>7.4f} {r['sq']:>7.4f} {r['rq']:>7.4f} "
            f"{r['tp']:>6} {r['fp']:>6} {r['fn']:>6}"
        )

    if args.bootstrap and results:
        b = results[0]
        print(
            f"\nbest PQ {b['pq']:.4f} "
            f"95% CI [{b['pq_lo']:.4f}, {b['pq_hi']:.4f}]  "
            f"(SQ [{b['sq_lo']:.4f}, {b['sq_hi']:.4f}], "
            f"RQ [{b['rq_lo']:.4f}, {b['rq_hi']:.4f}])"
        )

    out = ckpt_path.parent / "tune_results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nbest PQ={results[0]['pq']:.4f} -> {out}")


if __name__ == "__main__":
    main()
