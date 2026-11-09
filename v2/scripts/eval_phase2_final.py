"""Phase 2 final evaluation: GBDT vs CNN vs blend, in the PQ pipeline.

The CNN alone reaches ``corr(q_hat, q) = 0.507``, below the GBDT's 0.575. That is
a real result, not a bug: 6,166 training crops from 695 images is small for a
ResNet-18, and the scalar features already encode most of what the crop shows
(area, elongation, mean probability, contrast). The CNN's advantage would be
context the scalars cannot express -- whether the surrounding field looks like
the rest of the image -- and 15 epochs on 5k crops is not enough to learn it.

So the question is whether the two are *complementary*. They see different
information, so a blend may beat either. This script measures:

* GBDT alone, swept over tau
* CNN alone, swept over tau
* rank-average blend, swept over blend weight and tau

and reports the paired per-image CI against Phase 1 for the best of each.

    python v2/scripts/eval_phase2_final.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
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
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-threshold", type=float, default=0.30)
    ap.add_argument("--p2-min-area", type=int, default=128)
    ap.add_argument("--p1-threshold", type=float, default=0.50)
    ap.add_argument("--p1-min-area", type=int, default=256)
    ap.add_argument("--p1-close-k", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--paired-variant", default=None,
                    help="variant to report the paired CI for (default: best OOF)")
    ap.add_argument("--paired-tau", type=float, default=None,
                    help="tau to report the paired CI for (default: best OOF)")
    ap.add_argument("--out", default=str(P2 / "final.json"))
    args = ap.parse_args()

    z = np.load(P2 / "candidates.npz", allow_pickle=True)
    X, y, groups = z["cand_X"], z["cand_y"], z["cand_group"]

    gbdt = np.zeros(len(y), np.float32)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0,
        )
        m.fit(X[tr], y[tr])
        gbdt[te] = m.predict(X[te])

    cnn = np.load(P2 / "cnn_qhat.npz")["q_hat"]
    assert len(cnn) == len(gbdt), (len(cnn), len(gbdt))
    print(f"GBDT corr {np.corrcoef(gbdt, y)[0, 1]:.4f}   "
          f"CNN corr {np.corrcoef(cnn, y)[0, 1]:.4f}   "
          f"blend corr {np.corrcoef(0.5 * rankdata(gbdt) + 0.5 * rankdata(cnn), y)[0, 1]:.4f}",
          flush=True)

    # Rank-average so the two scorers' arbitrary output scales do not matter.
    rg = rankdata(gbdt) / len(gbdt)
    rc = rankdata(cnn) / len(cnn)

    variants: dict[str, np.ndarray] = {"gbdt": gbdt, "cnn": cnn}
    for w in (0.3, 0.5, 0.7):
        variants[f"blend{w}"] = w * rg + (1 - w) * rc

    taus = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    # Rank-based variants need quantile thresholds instead of absolute ones.
    rank_taus = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    paths = sorted(OOF.glob("fold*/*.npz"))
    by_file = {i: {} for i in range(len(paths))}
    for name, v in variants.items():
        for i in range(len(paths)):
            m = groups == i
            if m.any():
                by_file[i][name] = v[m]

    acc1 = PQAccumulator(threshold=0.5)
    accs = {(n, t): PQAccumulator(threshold=0.5)
            for n in variants for t in (rank_taus if n.startswith("blend") else taus)}

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
        vals = by_file.get(i, {})
        if n == 0 or not vals or len(vals.get("gbdt", [])) != n:
            for key in accs:
                for lab in labels:
                    accs[key].update(cand, lab, record=True)
            continue

        for (name, t), acc in accs.items():
            v = vals[name]
            keep = np.concatenate([[False], v > t])
            lut = np.zeros(n + 1, np.int32)
            nxt = 1
            for cid in range(1, n + 1):
                if keep[cid]:
                    lut[cid] = nxt
                    nxt += 1
            pred = lut[cand]
            for lab in labels:
                acc.update(pred, lab, record=True)

        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    s1 = acc1.summary()
    print()
    print(f"Phase 1: PQ {s1['pq']:.4f}  tp {s1['tp']} fp {s1['fp']} fn {s1['fn']}")
    print()
    print(f"{'variant':10s} {'tau':>5} {'PQ':>7} {'SQ':>7} {'RQ':>7} "
          f"{'tp':>6} {'fp':>6} {'fn':>6} {'dPQ':>8}")
    print("-" * 68)
    rows = []
    for (name, t), acc in accs.items():
        s = acc.summary()
        rows.append({"variant": name, "tau": t, **s, "dPQ": s["pq"] - s1["pq"]})
        print(f"{name:10s} {t:>5.2f} {s['pq']:>7.4f} {s['sq']:>7.4f} "
              f"{s['rq']:>7.4f} {s['tp']:>6d} {s['fp']:>6d} {s['fn']:>6d} "
              f"{s['pq'] - s1['pq']:>+8.4f}")

    best = max(rows, key=lambda r: r["pq"])

    # The best OOF row and the shipped configuration are not the same thing.
    # The best OOF row is the rank-average blend, whose threshold is a quantile
    # of the candidate score distribution; that distribution shifts on test
    # (different instrument, 11.7% of test files are 2021-22 against 5.7% of
    # train), so a quantile threshold does not transfer. The shipped scorer is
    # the GBDT at an absolute tau. Report the CI for whichever is asked for.
    if args.paired_variant is None:
        sel = (best["variant"], best["tau"])
    else:
        sel = (args.paired_variant, args.paired_tau)
        if sel not in accs:
            raise SystemExit(f"no accumulator for {sel}; have {sorted(accs)}")
    sel_row = next(r for r in rows
                   if r["variant"] == sel[0] and r["tau"] == sel[1])

    a = per_image_pq(acc1)
    b = per_image_pq(accs[sel])
    mean_d, lo, hi = paired_ci(a, b, args.n_boot)
    print()
    print(f"best OOF  : {best['variant']} tau {best['tau']} -> PQ "
          f"{best['pq']:.4f} ({best['dPQ']:+.4f})")
    print(f"paired for: {sel[0]} tau {sel[1]} -> PQ {sel_row['pq']:.4f} "
          f"({sel_row['dPQ']:+.4f})")
    print(f"paired per-image: {mean_d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")
    print(f"FP {(sel_row['fp'] - s1['fp']) / s1['fp']:+.1%} (target <= -25%)   "
          f"FN {(sel_row['fn'] - s1['fn']) / s1['fn']:+.1%} (target <= +5%)")

    json.dump({"phase1": s1, "rows": rows, "best": best,
               "paired_for": {"variant": sel[0], "tau": sel[1], **sel_row},
               "paired": {"mean": mean_d, "lo": lo, "hi": hi}},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
