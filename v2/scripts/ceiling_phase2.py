"""Phase 2 ceiling analysis: is the limit the scorer or the candidate set?

The GBDT scorer reaches ``corr(q_hat, q) = 0.575`` and buys +0.0166, but loses
244 true positives doing it. Before spending GPU on the ResNet-18 in the plan,
measure what a *perfect* scorer would achieve on the same candidate set. That
separates two very different problems:

* if perfect ``q`` gives a large gain, the candidate set is fine and the scorer
  is the bottleneck -> build the CNN
* if perfect ``q`` gives a small gain, the candidate set itself is wrong (too
  few candidates, or the wrong ones) -> fix 2.1, not 2.3

Three oracles are computed:

``q>0``      keep any candidate that matches GT in at least one copy
``q>=0.5``   keep any candidate that matches in the majority of copies
``q>=1``     keep only candidates every annotator drew

and, for reference, the same rule applied to the *predicted* ``q_hat``.

    python v2/scripts/ceiling_phase2.py
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-threshold", type=float, default=0.30)
    ap.add_argument("--p2-min-area", type=int, default=128)
    ap.add_argument("--p1-threshold", type=float, default=0.50)
    ap.add_argument("--p1-min-area", type=int, default=256)
    ap.add_argument("--p1-close-k", type=int, default=15)
    ap.add_argument("--out", default=str(P2 / "ceiling.json"))
    args = ap.parse_args()

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

    paths = sorted(OOF.glob("fold*/*.npz"))
    q_by = {i: y[groups == i] for i in range(len(paths)) if (groups == i).any()}
    qh_by = {i: q_hat[groups == i] for i in range(len(paths)) if (groups == i).any()}

    rules = {
        "phase1": None,
        "oracle_q>0": ("q", 0.0),
        "oracle_q>=0.5": ("q", 0.5),
        "oracle_q>=1.0": ("q", 1.0),
        "pred_q>0.35": ("qh", 0.35),
        "pred_q>0.30": ("qh", 0.30),
    }
    accs = {k: PQAccumulator(threshold=0.5) for k in rules}

    for i, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as zz:
            prob = zz["prob"].astype(np.float32)
            labels = zz["labels"]

        pred1 = prob_to_labelmap(
            prob, threshold=args.p1_threshold, min_area=args.p1_min_area,
            close_k=args.p1_close_k,
        )
        for lab in labels:
            accs["phase1"].update(pred1, lab)

        cand = generate_candidates(
            prob, threshold=args.p2_threshold, min_area=args.p2_min_area
        )
        n = int(cand.max())
        q = q_by.get(i)
        qh = qh_by.get(i)
        if n == 0 or q is None or len(q) != n:
            for k in rules:
                if k == "phase1":
                    continue
                for lab in labels:
                    accs[k].update(cand, lab)
            continue

        for k, spec in rules.items():
            if spec is None:
                continue
            src, thr = spec
            v = q if src == "q" else qh
            keep = np.concatenate([[False], v > thr])
            lut = np.zeros(n + 1, np.int32)
            nxt = 1
            for cid in range(1, n + 1):
                if keep[cid]:
                    lut[cid] = nxt
                    nxt += 1
            pred = lut[cand]
            for lab in labels:
                accs[k].update(pred, lab)

        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    base = accs["phase1"].summary()
    print()
    print(f"{'rule':16s} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'tp':>6} {'fp':>6} "
          f"{'fn':>6} {'dPQ':>8}")
    print("-" * 70)
    rows = {}
    for k in rules:
        s = accs[k].summary()
        rows[k] = s
        print(f"{k:16s} {s['pq']:>7.4f} {s['sq']:>7.4f} {s['rq']:>7.4f} "
              f"{s['tp']:>6d} {s['fp']:>6d} {s['fn']:>6d} "
              f"{s['pq'] - base['pq']:>+8.4f}")

    print()
    print("Interpretation:")
    print(f"  perfect q, keep q>0   : {rows['oracle_q>0']['pq'] - base['pq']:+.4f} "
          f"<- ceiling for THIS candidate set")
    print(f"  predicted q, tau 0.35 : {rows['pred_q>0.35']['pq'] - base['pq']:+.4f} "
          f"<- what the GBDT achieves")
    gap = (rows["oracle_q>0"]["pq"] - rows["pred_q>0.35"]["pq"])
    print(f"  scorer headroom       : {gap:+.4f}")

    json.dump({"phase1": base, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
