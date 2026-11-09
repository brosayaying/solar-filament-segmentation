"""Paired bootstrap CI for the *shipped* Phase 2 configuration.

Why this exists
---------------
``eval_phase2_final.py`` reports the paired CI for whichever row has the best
pooled OOF PQ. That row is the rank-average blend (``blend0.7`` at a quantile
threshold), which is **not** what was submitted. The shipped scorer is the GBDT
alone at an absolute ``tau = 0.35``, chosen because a quantile threshold depends
on the candidate score distribution and that distribution shifts on test.

So the CI in the paper must be the CI for the shipped configuration, otherwise
the reported interval describes a system that was never run on the test set.
This script computes exactly that, and nothing else, so it finishes in a couple
of minutes instead of sweeping ~30 accumulators.

    python v2/scripts/paired_shipped.py
    python v2/scripts/paired_shipped.py --tau 0.30
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

from solar.instances import generate_candidates  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

OOF = ROOT / "v2/runs/seg_v2/oof"
P2 = ROOT / "v2/runs/phase2"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--p2-threshold", type=float, default=0.30)
    ap.add_argument("--p2-min-area", type=int, default=128)
    ap.add_argument("--p1-threshold", type=float, default=0.50)
    ap.add_argument("--p1-min-area", type=int, default=256)
    ap.add_argument("--p1-close-k", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(P2 / "paired_shipped.json"))
    args = ap.parse_args()

    z = np.load(P2 / "candidates.npz", allow_pickle=True)
    X, y, groups = z["cand_X"], z["cand_y"], z["cand_group"]

    # Same OOF q-hat the shipped scorer was fitted with: GroupKFold on the
    # candidate's source file, so no candidate is scored by a model that saw it.
    qhat = np.zeros(len(y), np.float32)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0,
        )
        m.fit(X[tr], y[tr])
        qhat[te] = m.predict(X[te])
    print(f"GBDT corr {np.corrcoef(qhat, y)[0, 1]:.4f}", flush=True)

    paths = sorted(OOF.glob("fold*/*.npz"))
    acc1 = PQAccumulator(threshold=0.5)
    acc2 = PQAccumulator(threshold=0.5)

    for i, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as zz:
            prob = zz["prob"].astype(np.float32)
            labels = zz["labels"]

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

        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    s1, s2 = acc1.summary(), acc2.summary()
    mean_d, lo, hi = paired_ci(per_image_pq(acc1), per_image_pq(acc2), args.n_boot)

    print()
    print(f"phase1 (thr {args.p1_threshold} area {args.p1_min_area} "
          f"close {args.p1_close_k}): PQ {s1['pq']:.4f}  SQ {s1['sq']:.4f}  "
          f"RQ {s1['rq']:.4f}  tp {s1['tp']} fp {s1['fp']} fn {s1['fn']}")
    print(f"shipped (gbdt tau {args.tau}): PQ {s2['pq']:.4f}  SQ {s2['sq']:.4f}  "
          f"RQ {s2['rq']:.4f}  tp {s2['tp']} fp {s2['fp']} fn {s2['fn']}")
    print(f"pooled delta: {s2['pq'] - s1['pq']:+.4f}")
    print(f"paired per-image: {mean_d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")

    json.dump({"phase1": s1, "shipped": s2, "tau": args.tau,
               "pooled_delta": s2["pq"] - s1["pq"],
               "paired": {"mean": mean_d, "lo": lo, "hi": hi},
               "n_files": len(paths)},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
