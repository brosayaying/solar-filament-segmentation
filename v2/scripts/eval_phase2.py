"""Phase 2 acceptance test: paired comparison against Phase 1.

The sweep in ``score_phase2.py`` compares against its own low-threshold baseline,
which is not the number that matters. This script compares the Phase 2 pipeline
against the actual Phase 1 configuration (thr 0.50 / area 256 / close 15) on the
same files, with a paired bootstrap CI over per-image PQ.

Paired, not pooled: the two pipelines see identical images, so the per-image
difference removes image difficulty and is far more sensitive than comparing two
pooled numbers with overlapping CIs.

    python v2/scripts/eval_phase2.py
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
    """Per-image PQ from recorded stats, in the order they were added."""
    out = []
    for s in acc.per_image:
        denom = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
        out.append(s["iou_sum"] / denom if denom > 0 else 0.0)
    return np.asarray(out, np.float64)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Bootstrap CI on mean(b - a), resampling images."""
    rng = np.random.default_rng(seed)
    d = b - a
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--p2-threshold", type=float, default=0.30)
    ap.add_argument("--p2-min-area", type=int, default=128)
    ap.add_argument("--p2-close-k", type=int, default=0)
    ap.add_argument("--p1-threshold", type=float, default=0.50)
    ap.add_argument("--p1-min-area", type=int, default=256)
    ap.add_argument("--p1-close-k", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(P2 / "acceptance.json"))
    args = ap.parse_args()

    # --- fit q_hat out-of-fold, grouped by file ----------------------------
    z = np.load(P2 / "candidates.npz", allow_pickle=True)
    X, y, groups = z["cand_X"], z["cand_y"], z["cand_group"]
    q_hat = np.zeros(len(y), np.float32)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0,
        )
        m.fit(X[tr], y[tr])
        q_hat[te] = m.predict(X[te])
    print(f"q_hat corr {np.corrcoef(q_hat, y)[0, 1]:.4f}")

    paths = sorted(OOF.glob("fold*/*.npz"))
    q_by_file = {i: q_hat[groups == i] for i in range(len(paths)) if (groups == i).any()}

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
        qh = q_by_file.get(i)
        if n == 0 or qh is None or len(qh) != n:
            pred2 = cand
        else:
            keep = np.concatenate([[False], qh > args.tau])
            lut = np.zeros(n + 1, np.int32)
            nxt = 1
            for cid in range(1, n + 1):
                if keep[cid]:
                    lut[cid] = nxt
                    nxt += 1
            pred2 = lut[cand]
        for lab in labels:
            acc2.update(pred2, lab, record=True)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    s1, s2 = acc1.summary(), acc2.summary()
    print()
    print(f"{'':22s} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'tp':>6} {'fp':>6} {'fn':>6}")
    print("-" * 62)
    for name, s in (("Phase 1", s1), ("Phase 2", s2)):
        print(f"{name:22s} {s['pq']:>7.4f} {s['sq']:>7.4f} {s['rq']:>7.4f} "
              f"{s['tp']:>6d} {s['fp']:>6d} {s['fn']:>6d}")

    a, b = per_image_pq(acc1), per_image_pq(acc2)
    mean_d, lo, hi = paired_bootstrap(a, b, args.n_boot)
    print()
    print(f"pooled delta      : {s2['pq'] - s1['pq']:+.4f}")
    print(f"paired per-image  : {mean_d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")
    print(f"FP  {s1['fp']} -> {s2['fp']}  "
          f"({(s2['fp'] - s1['fp']) / s1['fp']:+.1%})  target <= -25%")
    print(f"FN  {s1['fn']} -> {s2['fn']}  "
          f"({(s2['fn'] - s1['fn']) / s1['fn']:+.1%})  target <= +5%")

    checks = {
        "pq_gain_ge_0.03": (s2["pq"] - s1["pq"]) >= 0.03,
        "paired_ci_excludes_0": lo > 0 or hi < 0,
        "fp_down_ge_25pct": (s2["fp"] - s1["fp"]) / s1["fp"] <= -0.25,
        "fn_up_le_5pct": (s2["fn"] - s1["fn"]) / s1["fn"] <= 0.05,
    }
    print()
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    json.dump(
        {"phase1": s1, "phase2": s2, "delta": s2["pq"] - s1["pq"],
         "paired": {"mean": mean_d, "lo": lo, "hi": hi},
         "checks": checks, "params": vars(args)},
        open(args.out, "w"), indent=2,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
