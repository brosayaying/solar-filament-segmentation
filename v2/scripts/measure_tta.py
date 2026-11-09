"""Measure the TTA effect on the shipped pipeline, on pooled OOF.

The paper claims 4x flip TTA is "worth roughly +0.008 PQ locally". That number
came from an early resnet34 experiment at a fixed threshold (docs/PQ_DIAGNOSIS.md),
not from the ConvNeXt-Small ensemble that was actually submitted. This script
measures it on the shipped system so the claim is either supported or corrected.

Two OOF dumps are compared, identical except for the ``tta`` flag:

    v2/runs/seg_v2/oof         tta=True   (what the local numbers were measured on)
    v2/runs/seg_v2/oof_notta   tta=False  (what the submitted CSV was generated with)

Both are scored twice: once with the Phase 1 post-processing, once with the
Phase 2 GBDT re-scorer at the shipped tau. The GBDT is refitted per dump, because
its candidate features come from the probability map and therefore differ.

    python v2/scripts/measure_tta.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.instances import (  # noqa: E402
    candidate_features,
    generate_candidates,
    label_candidates,
)
from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

RUNS = ROOT / "v2/runs"
P2 = RUNS / "phase2"


def per_image_pq(acc: PQAccumulator) -> np.ndarray:
    out = []
    for s in acc.per_image:
        d = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
        out.append(s["iou_sum"] / d if d > 0 else 0.0)
    return np.asarray(out, np.float64)


def paired_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)))


def build_candidates(oof_dir: Path, thr: float, min_area: int):
    """Candidate features for every OOF file, in a stable file order.

    ``label_candidates`` returns ``q``, the fraction of annotator copies in
    which the candidate matches some GT instance at IoU > 0.5. That is the
    target the shipped GBDT was fitted on, so it is the target used here.
    """
    paths = sorted(oof_dir.glob("fold*/*.npz"))
    X, y, groups = [], [], []
    keys = None
    for i, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
        cand = generate_candidates(prob, threshold=thr, min_area=min_area)
        n = int(cand.max())
        if n == 0:
            continue
        q, _, _ = label_candidates(cand, labels)
        feats = [candidate_features(prob, cand, cid) for cid in range(1, n + 1)]
        if keys is None:
            keys = list(feats[0].keys())
        X.extend([[f[k] for k in keys] for f in feats])
        y.extend(q.tolist())
        groups.extend([i] * n)
    return paths, np.asarray(X, np.float32), np.asarray(y, np.float32), \
        np.asarray(groups, np.int32)


def score(oof_dir: Path, X, y, groups, args):
    qhat = np.zeros(len(y), np.float32)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0,
        )
        m.fit(X[tr], y[tr])
        qhat[te] = m.predict(X[te])

    paths = sorted(oof_dir.glob("fold*/*.npz"))
    acc1 = PQAccumulator(threshold=0.5)
    acc2 = PQAccumulator(threshold=0.5)
    for i, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
        pred1 = prob_to_labelmap(
            prob, threshold=args.p1_threshold, min_area=args.p1_min_area,
            close_k=args.p1_close_k,
        )
        for lab in labels:
            acc1.update(pred1, lab, record=True)

        cand = generate_candidates(
            prob, threshold=args.p2_threshold, min_area=args.p2_min_area
        )
        n = int(cand.max())
        v = qhat[groups == i]
        if n == 0 or len(v) != n:
            pred2 = cand
        else:
            keep = np.concatenate([[False], v > args.tau])
            lut = np.zeros(n + 1, np.int32)
            nxt = 1
            for cid in range(1, n + 1):
                if keep[cid]:
                    lut[cid] = nxt
                    nxt += 1
            pred2 = lut[cand]
        for lab in labels:
            acc2.update(pred2, lab, record=True)
    return acc1, acc2, qhat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--p2-threshold", type=float, default=0.30)
    ap.add_argument("--p2-min-area", type=int, default=128)
    ap.add_argument("--p1-threshold", type=float, default=0.50)
    ap.add_argument("--p1-min-area", type=int, default=256)
    ap.add_argument("--p1-close-k", type=int, default=15)
    ap.add_argument("--out", default=str(P2 / "tta_effect.json"))
    args = ap.parse_args()

    res = {}
    for tag, d in (("tta", RUNS / "seg_v2/oof"),
                   ("no_tta", RUNS / "seg_v2/oof_notta")):
        n = len(list(d.glob("fold*/*.npz")))
        print(f"\n=== {tag} ({n} files) ===", flush=True)
        paths, X, y, groups = build_candidates(d, args.p2_threshold,
                                               args.p2_min_area)
        print(f"  candidates: {len(y)}  mean q: {y.mean():.4f}", flush=True)
        acc1, acc2, qhat = score(d, X, y, groups, args)
        s1, s2 = acc1.summary(), acc2.summary()
        print(f"  phase1: PQ {s1['pq']:.4f}  SQ {s1['sq']:.4f}  RQ {s1['rq']:.4f}")
        print(f"  phase2: PQ {s2['pq']:.4f}  SQ {s2['sq']:.4f}  RQ {s2['rq']:.4f}")
        res[tag] = {"phase1": s1, "phase2": s2,
                    "n_candidates": int(len(y)),
                    "corr": float(np.corrcoef(qhat, y)[0, 1]),
                    "_acc1": acc1, "_acc2": acc2}

    print("\n=== TTA effect (tta - no_tta) ===")
    for stage in ("phase1", "phase2"):
        a = res["no_tta"][stage]["pq"]
        b = res["tta"][stage]["pq"]
        print(f"  {stage}: {a:.4f} -> {b:.4f}  ({b - a:+.4f})")

    for stage, key in (("phase1", "_acc1"), ("phase2", "_acc2")):
        m, lo, hi = paired_ci(per_image_pq(res["no_tta"][key]),
                              per_image_pq(res["tta"][key]))
        print(f"  {stage} paired: {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
        res[f"{stage}_paired"] = {"mean": m, "lo": lo, "hi": hi}

    for tag in res:
        res[tag].pop("_acc1", None)
        res[tag].pop("_acc2", None)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
