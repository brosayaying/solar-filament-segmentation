"""SUPERSEDED -- kept only to reproduce the measurement in docs/PREPROCESSING.md §3.

This script applies preprocessing at *eval* time against a checkpoint trained on
raw input. That cannot answer whether preprocessing helps: the model has already
adapted to raw statistics, so feeding it flattened or CLAHE input measures a
distribution shift, not a benefit. Run against `runs/long_160ep_512` it reports
CLAHE as +0.10 PQ when the normalisation is wrong and -0.095 when it is right --
the sign flips with an unrelated bug, which is exactly why the approach is
untrustworthy.

The valid test is to *retrain* with preprocessing and compare the two trained
models, each served with its own preprocessing:

    python scripts/compare_runs.py runs/long_160ep_512 runs/norm_disk_flatten

Kept because the numbers it produced (with the correct normalisation) are quoted
in the docs as the reason eval-time probing was abandoned.

Original docstring follows.

---

A/B the preprocessing pipeline against existing checkpoints.

Variants:
    raw        no preprocessing (current pipeline)
    flatten    radial flattening only
    clahe      CLAHE only
    both       flatten -> blur -> CLAHE (EdgeAttNet's order)
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
from solar.metrics import PQAccumulator  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402
from solar.preprocess import preprocess  # noqa: E402

cv2.setNumThreads(4)

VARIANTS = {
    "raw": dict(flatten=False, clahe=False, blur=False),
    "flatten": dict(flatten=True, clahe=False, blur=False),
    "clahe": dict(flatten=False, clahe=True, blur=True),
    "both": dict(flatten=True, clahe=True, blur=True),
}


def load_model(weights: Path, device):
    """Build the architecture from the run's own config.json, then load weights.

    The ablation must work for any checkpoint (resnet34 UNet, DeepCrack, decoder),
    so the architecture is read from the run directory rather than assumed.
    """
    cfg_path = weights.parent / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json next to {weights}; cannot infer architecture")
    cfg = json.loads(cfg_path.read_text())
    m = build_model(cfg)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model", ck))
    arch = cfg["model"].get("arch", "unet")
    print(f"architecture: {arch} (from {cfg_path.name})")
    return m.to(device).eval()


@torch.no_grad()
def predict(model, img_f32, device, mean, std):
    x = (img_f32 / 255.0 - mean) / std
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    return torch.sigmoid(model(t))[0, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=142)
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--mean", type=float, default=None, help="default: from run config")
    ap.add_argument("--std", type=float, default=None, help="default: from run config")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.45, 0.55, 0.7])
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
    ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[: args.n]

    model = load_model(ROOT / args.weights, device)

    # Normalisation MUST come from the run's config. Defaulting to mean=0/std=1
    # silently breaks any checkpoint trained with mean=0.5/std=0.25 (the resnet34
    # UNet), which drops its PQ from 0.4152 to 0.056 and makes every variant look
    # like a huge win.
    run_cfg = json.loads((ROOT / args.weights).parent.joinpath("config.json").read_text())
    mean = args.mean if args.mean is not None else run_cfg["data"].get("mean", 0.0)
    std = args.std if args.std is not None else run_cfg["data"].get("std", 1.0)
    print(f"loaded {args.weights}  ({len(ids)} val files, device={device})")
    print(f"normalisation: mean={mean} std={std} (from run config)")

    # Per-variant, per-image PQ so the paired test is possible.
    # Two passes: pass 1 finds each variant's best threshold, pass 2 records
    # per-image PQ at that threshold. Doing it in one pass would require knowing
    # the best threshold before seeing all the images.
    per_image: dict[str, np.ndarray] = {}
    summaries: dict[str, dict] = {}
    best_thr: dict[str, float] = {}

    for name, kw in VARIANTS.items():
        accs = {t: PQAccumulator() for t in args.thresholds}
        for iid in tqdm(ids, desc=f"{name} (sweep)", leave=False):
            im = images[iid]
            raw = cv2.imread(
                str(data_root / "train/train_images" / im["file_name"]),
                cv2.IMREAD_GRAYSCALE,
            )
            img = preprocess(raw, **kw) if any(kw.values()) else raw.astype(np.float32)
            p = predict(model, img, device, mean, std)
            g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
            for t in args.thresholds:
                accs[t].update(
                    prob_to_labelmap(p, threshold=t, min_area=args.min_area), g
                )
        best_t, best_s = max(
            ((t, accs[t].summary()) for t in args.thresholds),
            key=lambda r: r[1]["pq"],
        )
        best_thr[name] = best_t
        summaries[name] = {"threshold": best_t, **best_s}

    for name, kw in VARIANTS.items():
        pqs = []
        for iid in tqdm(ids, desc=f"{name} (paired)", leave=False):
            im = images[iid]
            raw = cv2.imread(
                str(data_root / "train/train_images" / im["file_name"]),
                cv2.IMREAD_GRAYSCALE,
            )
            img = preprocess(raw, **kw) if any(kw.values()) else raw.astype(np.float32)
            p = predict(model, img, device, mean, std)
            g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
            a1 = PQAccumulator()
            a1.update(
                prob_to_labelmap(p, threshold=best_thr[name], min_area=args.min_area), g
            )
            pqs.append(a1.summary()["pq"])
        per_image[name] = np.array(pqs)

    print(f"\n=== PQ by variant (n={len(ids)}, min_area={args.min_area}) ===")
    print(f"{'variant':>9} {'thr':>5} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    for name in VARIANTS:
        s = summaries[name]
        print(
            f"{name:>9} {s['threshold']:5.2f} {s['pq']:7.4f} {s['sq']:7.4f} "
            f"{s['rq']:7.4f} {s['tp']:5d} {s['fp']:5d} {s['fn']:5d}"
        )

    print("\n=== paired per-image test vs raw (thr=0.55) ===")
    base = per_image["raw"]
    for name in VARIANTS:
        if name == "raw":
            continue
        d = per_image[name] - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        verdict = "REAL" if lo > 0 else ("worse" if hi < 0 else "n.s.")
        print(
            f"  {name:>9}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"t={d.mean()/se:+.2f}  better on {(d>0).sum()}/{len(d)}  -> {verdict}"
        )

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "weights": args.weights,
                    "n_images": len(ids),
                    "min_area": args.min_area,
                    "summaries": summaries,
                    "paired_vs_raw": {
                        name: {
                            "mean_delta": float((per_image[name] - base).mean()),
                            "ci_low": float(
                                (per_image[name] - base).mean()
                                - 1.96
                                * (per_image[name] - base).std(ddof=1)
                                / np.sqrt(len(base))
                            ),
                            "ci_high": float(
                                (per_image[name] - base).mean()
                                + 1.96
                                * (per_image[name] - base).std(ddof=1)
                                / np.sqrt(len(base))
                            ),
                        }
                        for name in VARIANTS
                        if name != "raw"
                    },
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
