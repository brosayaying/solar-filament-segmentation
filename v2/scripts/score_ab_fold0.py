"""Score the fold-0 A/B: radial flattening vs raw.

Both arms are trained on the same fold-0 split with identical everything except
``data.preprocess`` and the normalisation constants that go with it. The
comparison is therefore a **paired per-image test** on the difference, not two
independent aggregates: both models see the same 142 validation files, so the
per-image deltas are paired and the t-interval on the mean delta is the right
uncertainty.

This is the experiment `docs/PREPROCESSING.md` never actually ran. That doc
measured flattening by feeding flattened input to a *raw-trained* checkpoint,
which measures distribution shift, not the benefit of preprocessing. The retrain
config existed (`configs/norm_disk_flatten.json`) but was never executed.

    python v2/scripts/score_ab_fold0.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import FilamentDataset, collate_val, load_coco  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402
from solar.predict import predict_prob  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

SPLIT = ROOT / "v2/splits/week5_seed42/fold0.json"
ARMS = ["raw", "flatten"]
# Post-processing chosen on the full-val sweep of the baseline (Phase 0.4).
THR, MIN_AREA, CLOSE_K = 0.65, 256, 15


def per_image_pq(ckpt_path: Path, device: str) -> tuple[dict[str, float], list[dict]]:
    """Run one arm over fold-0 val and return per-image PQ stats."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    coco = load_coco(ROOT / cfg["data"]["root"] / cfg["data"]["ann_file"])
    split = json.load(open(SPLIT))

    ds = FilamentDataset(
        ROOT / cfg["data"]["root"],
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
        ds, batch_size=1, shuffle=False, num_workers=6,
        collate_fn=collate_val, pin_memory=True,
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    acc = PQAccumulator(threshold=0.5)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"][0].to(device)
            prob = predict_prob(model, image, tta=False)
            pred = prob_to_labelmap(
                prob, threshold=THR, min_area=MIN_AREA, close_k=CLOSE_K
            )
            acc.update(pred, batch["label"][0].numpy(), record=True)

    return acc.summary(), acc.per_image


def paired_ci(delta: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, t statistic, and 95 % t-interval for a paired per-image delta."""
    n = len(delta)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else 0.0
    crit = float(stats.t.ppf(0.975, n - 1))
    return mean, t, mean - crit * se, mean + crit * se


def image_pq(stats: dict) -> float:
    denom = stats["tp"] + 0.5 * stats["fp"] + 0.5 * stats["fn"]
    return stats["iou_sum"] / denom if denom > 0 else 0.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict[str, dict] = {}

    for arm in ARMS:
        ckpt = ROOT / f"v2/runs/ab_fold0_{arm}/best.pt"
        if not ckpt.exists():
            print(f"MISSING {ckpt} -- train this arm first")
            raise SystemExit(1)
        print(f"scoring {arm} ...")
        summary, per_image = per_image_pq(ckpt, device)
        results[arm] = {"summary": summary, "per_image": per_image}
        print(
            f"  {arm:8s} PQ {summary['pq']:.4f}  SQ {summary['sq']:.4f}  "
            f"RQ {summary['rq']:.4f}  tp {summary['tp']} fp {summary['fp']} "
            f"fn {summary['fn']}"
        )

    a = np.array([image_pq(s) for s in results["raw"]["per_image"]])
    b = np.array([image_pq(s) for s in results["flatten"]["per_image"]])
    delta = b - a
    mean, t, lo, hi = paired_ci(delta)

    print("\n=== paired per-image test: flatten - raw ===")
    print(f"  n = {len(delta)} images")
    print(f"  mean delta PQ = {mean:+.4f}   t = {t:+.2f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = (
        "flatten WINS (CI excludes 0)"
        if lo > 0
        else "raw WINS (CI excludes 0)"
        if hi < 0
        else "no significant difference (CI includes 0)"
    )
    print(f"  verdict: {verdict}")

    out = ROOT / "v2/runs/ab_fold0_results.json"
    with open(out, "w") as fh:
        json.dump(
            {
                "postprocess": {"threshold": THR, "min_area": MIN_AREA, "close_k": CLOSE_K},
                "arms": {k: v["summary"] for k, v in results.items()},
                "paired": {
                    "n": len(delta),
                    "mean_delta": mean,
                    "t": t,
                    "ci_low": lo,
                    "ci_high": hi,
                    "verdict": verdict,
                },
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
