"""Paired comparison: Phase 1 (seg_v2, 5-fold OOF) vs the resnet34 baseline.

The two models were evaluated on different file sets, so their headline numbers
are not comparable:

    norm_disk_raw   142 files (the old date-split val), 232 annotator entries
    seg_v2 OOF      707 files (all of them), 1,154 annotator entries

Pooled PQ over 707 files and pooled PQ over 142 files are different quantities.
The only honest comparison is on the **intersection**: the 142 files that are
out-of-fold for seg_v2 *and* were the baseline's val set. Both models are then
scored against the same GT, on the same images, and the per-image deltas are
paired.

    python v2/scripts/compare_phase1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"
BASE_PROBS = ROOT / "runs/norm_disk_raw/val_probs"
BASE_SPLIT = ROOT / "splits/date_seed42.json"
ANN = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

# Best setting for each model, from its own sweep.
SEG = dict(threshold=0.50, min_area=256, close_k=15)
BASE = dict(threshold=0.65, min_area=256, close_k=15)


def image_pq(stats: dict) -> float:
    denom = stats["tp"] + 0.5 * stats["fp"] + 0.5 * stats["fn"]
    return stats["iou_sum"] / denom if denom > 0 else 0.0


def paired_ci(delta: np.ndarray) -> tuple[float, float, float, float]:
    n = len(delta)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else 0.0
    crit = float(stats.t.ppf(0.975, n - 1))
    return mean, t, mean - crit * se, mean + crit * se


def main() -> None:
    coco = json.load(open(ANN))
    images = {i["id"]: i for i in coco["images"]}
    ann_index: dict[str, list] = {}
    for a in coco["annotations"]:
        ann_index.setdefault(a["image_id"], []).append(a)

    val_ids = sorted(json.load(open(BASE_SPLIT))["val"])
    # val_probs are written in sorted val-id order, one entry per image_id.
    base_files = sorted(BASE_PROBS.glob("*_prob.npy"))
    assert len(base_files) == len(val_ids), (len(base_files), len(val_ids))

    # Group the baseline's annotator entries by file stem.
    by_stem: dict[str, list[str]] = {}
    for iid in val_ids:
        by_stem.setdefault(Path(images[iid]["file_name"]).stem, []).append(iid)

    seg_acc = PQAccumulator(threshold=0.5)
    base_acc = PQAccumulator(threshold=0.5)
    seg_per, base_per = [], []

    for path, iid in zip(base_files, val_ids):
        stem = Path(images[iid]["file_name"]).stem
        oof_path = next(OOF.glob(f"fold*/{stem}.npz"), None)
        if oof_path is None:
            continue

        # --- baseline: one prob per image_id, scored against that entry's GT
        bprob = np.load(path).astype(np.float32)
        bpred = prob_to_labelmap(bprob, **BASE)
        bstats = base_acc.update(bpred, _labelmap(ann_index[iid]), record=True)
        base_per.append(bstats)

        # --- seg_v2: one prob per file, scored against every copy of that file
        with np.load(oof_path, allow_pickle=False) as z:
            sprob = z["prob"].astype(np.float32)
            slabels = z["labels"]
        spred = prob_to_labelmap(sprob, **SEG)
        # Aggregate this file's copies into one per-image record so the pairing
        # is per file, matching the baseline's per-entry records.
        agg = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
        for lab in slabels:
            s = seg_acc.update(spred, lab, record=False)
            for k in agg:
                agg[k] += s[k]
        seg_per.append(agg)

    n = min(len(seg_per), len(base_per))
    a = np.array([image_pq(s) for s in base_per[:n]])
    b = np.array([image_pq(s) for s in seg_per[:n]])
    delta = b - a
    mean, t, lo, hi = paired_ci(delta)

    print(f"paired on {n} files (baseline val ∩ seg_v2 OOF)")
    print(f"  baseline  PQ {base_acc.pq:.4f}  SQ {base_acc.sq:.4f}  RQ {base_acc.rq:.4f}")
    print(f"  seg_v2    PQ {seg_acc.pq:.4f}  SQ {seg_acc.sq:.4f}  RQ {seg_acc.rq:.4f}")
    print(f"\n  mean per-image delta = {mean:+.4f}   t = {t:+.2f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = (
        "seg_v2 WINS" if lo > 0 else "baseline WINS" if hi < 0
        else "no significant difference"
    )
    print(f"  verdict: {verdict}")

    out = ROOT / "v2/runs/seg_v2/compare_phase1.json"
    with open(out, "w") as fh:
        json.dump(
            {
                "n_files": n,
                "baseline": {"pq": base_acc.pq, "sq": base_acc.sq, "rq": base_acc.rq, **BASE},
                "seg_v2": {"pq": seg_acc.pq, "sq": seg_acc.sq, "rq": seg_acc.rq, **SEG},
                "paired": {"mean_delta": mean, "t": t, "ci_low": lo, "ci_high": hi,
                           "verdict": verdict},
            },
            fh, indent=2,
        )
    print(f"\nwrote {out}")


def _labelmap(anns: list) -> np.ndarray:
    from solar.data import anns_to_labelmap

    return anns_to_labelmap(anns, 2048, 2048)


if __name__ == "__main__":
    main()
