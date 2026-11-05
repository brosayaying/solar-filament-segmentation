"""Phase 2.3: per-instance candidate scoring, evaluated honestly on OOF.

Trains a gradient-boosted model on the scalar candidate features to predict
``q`` -- the fraction of annotator copies in which a candidate has IoU > 0.5 --
then rebuilds label maps keeping only candidates with ``q_hat > tau``.

**Cross-validation is by file, not by row.** Rows from the same image are highly
correlated (same illumination, same filaments, same model errors); splitting them
randomly would leak and produce a number that looks valid and is not. This
project has already been burned four times by exactly that failure mode.

The baseline for comparison is the *same* candidate generation with no filtering,
so the measured delta is attributable to the scorer alone and not to the change
in threshold/min-area.

    python v2/scripts/score_phase2.py --sweep
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

OOF = ROOT / "v2/runs/seg_v2/oof"
P2 = ROOT / "v2/runs/phase2"


def load_dataset(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "cand_X": z["cand_X"],
        "cand_y": z["cand_y"],
        "cand_group": z["cand_group"],
        "cand_keys": [str(k) for k in z["cand_keys"]],
        "threshold": float(z["threshold"]),
        "min_area": int(z["min_area"]),
    }


def fit_predict_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """Out-of-fold predictions with file-grouped CV."""
    oof = np.zeros(len(y), np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=0,
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def score_sweep(
    paths: list[Path],
    q_hat_by_file: dict[int, np.ndarray],
    taus: list[float],
    thr: float,
    min_area: int,
) -> dict[float, PQAccumulator]:
    """Score every ``tau`` in a single pass over the OOF dumps.

    Candidate generation is the expensive part (~0.8 s/file), so it is done once
    and the label maps are rebuilt per ``tau`` from the cached candidate map.
    """
    accs = {t: PQAccumulator(threshold=0.5) for t in taus}
    for i, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as z:
            prob = z["prob"].astype(np.float32)
            labels = z["labels"]
        cand = generate_candidates(prob, threshold=thr, min_area=min_area)
        n = int(cand.max())

        qh = q_hat_by_file.get(i)
        if n == 0 or qh is None or len(qh) != n:
            for t in taus:
                for lab in labels:
                    accs[t].update(cand, lab)
            continue

        for t in taus:
            keep = np.concatenate([[False], qh > t])
            lut = np.zeros(n + 1, np.int32)
            nxt = 1
            for cid in range(1, n + 1):
                if keep[cid]:
                    lut[cid] = nxt
                    nxt += 1
            pred = lut[cand]
            for lab in labels:
                accs[t].update(pred, lab)
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(P2 / "candidates.npz"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--out", default=str(P2 / "phase2_results.json"))
    args = ap.parse_args()

    d = load_dataset(Path(args.dataset))
    X, y, groups = d["cand_X"], d["cand_y"], d["cand_group"]
    thr, min_area = d["threshold"], d["min_area"]
    print(f"{len(y)} candidates from {len(np.unique(groups))} files")
    print(f"candidate generation: thr {thr} area {min_area}")

    q_hat = fit_predict_oof(X, y, groups, args.n_splits)
    print(f"q_hat: min {q_hat.min():.3f} med {np.median(q_hat):.3f} "
          f"max {q_hat.max():.3f}")
    print(f"corr(q_hat, q) = {np.corrcoef(q_hat, y)[0, 1]:.4f}")

    paths = sorted(OOF.glob("fold*/*.npz"))
    q_by_file = {}
    for i in range(len(paths)):
        m = groups == i
        if m.any():
            q_by_file[i] = q_hat[m]

    taus = [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60] \
        if args.sweep else [args.tau]

    print()
    print(f"{'tau':>5} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'tp':>6} {'fp':>6} {'fn':>6}")
    print("-" * 48, flush=True)
    accs = score_sweep(paths, q_by_file, taus, thr, min_area)
    results = []
    for tau in taus:
        s = accs[tau].summary()
        results.append({"tau": tau, **s})
        print(f"{tau:>5.2f} {s['pq']:>7.4f} {s['sq']:>7.4f} {s['rq']:>7.4f} "
              f"{s['tp']:>6d} {s['fp']:>6d} {s['fn']:>6d}", flush=True)

    best = max(results, key=lambda r: r["pq"])
    base = results[0]
    print()
    print(f"baseline (tau 0, no filtering): PQ {base['pq']:.4f}  "
          f"fp {base['fp']}  fn {base['fn']}")
    print(f"best     (tau {best['tau']:.2f})        : PQ {best['pq']:.4f}  "
          f"fp {best['fp']}  fn {best['fn']}")
    print(f"delta: {best['pq'] - base['pq']:+.4f}   "
          f"fp {best['fp'] - base['fp']:+d} "
          f"({(best['fp'] - base['fp']) / max(1, base['fp']):+.1%})   "
          f"fn {best['fn'] - base['fn']:+d} "
          f"({(best['fn'] - base['fn']) / max(1, base['fn']):+.1%})")

    json.dump(
        {"results": results, "best": best, "baseline": base,
         "corr": float(np.corrcoef(q_hat, y)[0, 1]),
         "params": {"threshold": thr, "min_area": min_area,
                    "n_splits": args.n_splits}},
        open(args.out, "w"), indent=2,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
