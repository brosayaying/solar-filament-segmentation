"""Phase 2b: expected-value decision rule + closing, vs Phase 1.

Two problems with the first Phase 2 attempt (``eval_phase2.py``):

**1. The decision rule was a fixed threshold on ``q_hat``.** That is not what the
metric rewards. Derive it properly. Keeping a candidate changes the PQ numerator
by ``s`` with probability ``q`` (it is a TP) and by 0 otherwise; it changes the
denominator by ``1`` with probability ``q`` and by ``0.5`` otherwise. So

    E[gain] = q*s - PQ*(0.5 + 0.5*q)

and the candidate should be kept iff ``q*s > PQ*(0.5 + 0.5*q)``. A fixed
``q_hat > tau`` rule ignores ``s`` entirely, so it throws away well-drawn
candidates that happen to have moderate ``q`` -- which is exactly the FN increase
that failed the acceptance test.

**2. ``close_k`` was a confound.** Phase 1 closes with a 15 px ellipse before
labelling; the first Phase 2 run did not close at all. Part of the measured FP
reduction and all of the FN increase may be attributable to dropping closing
rather than to the scorer. This script sweeps ``close_k`` so the two effects are
separated.

    python v2/scripts/eval_phase2b.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
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


def fit_oof(X, y, groups, n_splits=5, **kw):
    oof = np.zeros(len(y), np.float32)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0, **kw
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def relabel(mask: np.ndarray, close_k: int, min_area: int) -> np.ndarray:
    """Close, then label connected components, dropping small ones."""
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    if mask.sum() == 0:
        return np.zeros(mask.shape, np.int32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = np.zeros(mask.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


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
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-threshold", type=float, default=0.30)
    ap.add_argument("--p2-min-area", type=int, default=128)
    ap.add_argument("--p1-threshold", type=float, default=0.50)
    ap.add_argument("--p1-min-area", type=int, default=256)
    ap.add_argument("--p1-close-k", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(P2 / "phase2b_results.json"))
    args = ap.parse_args()

    z = np.load(P2 / "candidates.npz", allow_pickle=True)
    X, y, groups = z["cand_X"], z["cand_y"], z["cand_group"]
    iou = z["cand_iou"]

    q_hat = fit_oof(X, y, groups)
    s_hat = fit_oof(X, iou, groups)
    print(f"q_hat corr {np.corrcoef(q_hat, y)[0, 1]:.4f}   "
          f"s_hat corr {np.corrcoef(s_hat, iou)[0, 1]:.4f}", flush=True)

    paths = sorted(OOF.glob("fold*/*.npz"))
    q_by = {i: q_hat[groups == i] for i in range(len(paths)) if (groups == i).any()}
    s_by = {i: s_hat[groups == i] for i in range(len(paths)) if (groups == i).any()}

    # Grid: EV margin c (keep iff q*s > c*PQ*(0.5+0.5q)) x close_k.
    cs = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    close_ks = [0, 15]
    PQ_REF = 0.44  # the PQ the rule is evaluated at; swept implicitly by c

    accs = {(c, k): PQAccumulator(threshold=0.5) for c in cs for k in close_ks}
    acc1 = PQAccumulator(threshold=0.5)

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
        qh, sh = q_by.get(i), s_by.get(i)
        if n == 0 or qh is None or len(qh) != n:
            for key in accs:
                for lab in labels:
                    accs[key].update(cand, lab, record=True)
            continue

        qh = np.clip(qh, 0.0, 1.0)
        sh = np.clip(sh, 0.0, 1.0)
        denom = 0.5 + 0.5 * qh
        score = np.where(denom > 0, qh * sh / denom, 0.0)

        for c in cs:
            keep = np.concatenate([[False], score > c * PQ_REF])
            mask = keep[cand]
            for k in close_ks:
                pred = relabel(mask, k, args.p2_min_area)
                for lab in labels:
                    accs[(c, k)].update(pred, lab, record=True)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    s1 = acc1.summary()
    print()
    print(f"Phase 1 (thr {args.p1_threshold} area {args.p1_min_area} "
          f"close {args.p1_close_k}): PQ {s1['pq']:.4f}  "
          f"tp {s1['tp']} fp {s1['fp']} fn {s1['fn']}")
    print()
    print(f"{'c':>5} {'close':>6} {'PQ':>7} {'SQ':>7} {'RQ':>7} "
          f"{'tp':>6} {'fp':>6} {'fn':>6} {'dPQ':>8} {'dFP':>7} {'dFN':>7}")
    print("-" * 82)
    rows = []
    for c in cs:
        for k in close_ks:
            s2 = accs[(c, k)].summary()
            d = s2["pq"] - s1["pq"]
            dfp = (s2["fp"] - s1["fp"]) / s1["fp"]
            dfn = (s2["fn"] - s1["fn"]) / s1["fn"]
            rows.append({"c": c, "close_k": k, **s2, "dPQ": d,
                         "dFP": dfp, "dFN": dfn})
            print(f"{c:>5.1f} {k:>6d} {s2['pq']:>7.4f} {s2['sq']:>7.4f} "
                  f"{s2['rq']:>7.4f} {s2['tp']:>6d} {s2['fp']:>6d} {s2['fn']:>6d} "
                  f"{d:>+8.4f} {dfp:>+7.1%} {dfn:>+7.1%}")

    best = max(rows, key=lambda r: r["pq"])
    a = per_image_pq(acc1)
    b = per_image_pq(accs[(best["c"], best["close_k"])])
    mean_d, lo, hi = paired_ci(a, b, args.n_boot)
    print()
    print(f"best: c {best['c']} close {best['close_k']} -> PQ {best['pq']:.4f} "
          f"({best['dPQ']:+.4f})")
    print(f"paired per-image: {mean_d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")
    print(f"FP {best['dFP']:+.1%} (target <= -25%)   "
          f"FN {best['dFN']:+.1%} (target <= +5%)")

    json.dump({"phase1": s1, "rows": rows, "best": best,
               "paired": {"mean": mean_d, "lo": lo, "hi": hi}},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
