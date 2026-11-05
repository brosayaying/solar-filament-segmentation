"""Build the Phase 2 candidate dataset from the Phase 1 OOF dumps.

Streams one OOF file at a time (loading all 707 needs ~12 GB and was OOM-killed
before). For each file:

1. generate low-threshold candidates (2.1)
2. compute per-candidate features and ``q`` labels
3. compute per-pair merge features and merge labels
4. append to a growing list of arrays

Output is a single ``.npz`` with ``cand_X``, ``cand_y``, ``pair_X``, ``pair_y``
and the group id (file stem) for each row, so that any downstream model can be
cross-validated by *file* rather than by row. Rows from the same image are
highly correlated; splitting them randomly would leak.

    python v2/scripts/build_phase2_dataset.py --limit 5      # smoke test
    python v2/scripts/build_phase2_dataset.py                # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.instances import (  # noqa: E402
    candidate_features,
    generate_candidates,
    label_candidates,
    label_pairs,
    pair_features,
)

OOF = ROOT / "v2/runs/seg_v2/oof"
OUT = ROOT / "v2/runs/phase2"

# Pairs further apart than this are not merge candidates. 40 px is the plan's
# value; the EDT crop margin must be at least as large.
MAX_GAP = 40.0
MARGIN = 60


def build_file(path: Path, thr: float, min_area: int) -> dict | None:
    with np.load(path, allow_pickle=False) as z:
        prob = z["prob"].astype(np.float32)
        labels = z["labels"]
        meta = json.loads(str(z["meta"]))

    cand = generate_candidates(prob, threshold=thr, min_area=min_area)
    n_cand = int(cand.max())
    if n_cand == 0:
        return None

    q, best_iou, gid = label_candidates(cand, labels)

    # --- per-candidate rows -------------------------------------------------
    feats = [candidate_features(prob, cand, cid) for cid in range(1, n_cand + 1)]
    keys = list(feats[0].keys())
    cand_X = np.asarray([[f[k] for k in keys] for f in feats], np.float32)

    # --- per-pair rows ------------------------------------------------------
    # Only pairs whose bounding boxes are within MAX_GAP + a margin can be
    # close enough; testing every pair is O(n^2) and n can be ~40.
    boxes = np.asarray(
        [
            [
                np.nonzero(cand == cid)[0].min(),
                np.nonzero(cand == cid)[0].max(),
                np.nonzero(cand == cid)[1].min(),
                np.nonzero(cand == cid)[1].max(),
            ]
            for cid in range(1, n_cand + 1)
        ],
        np.int32,
    )

    cache: dict[int, tuple] = {}
    pair_rows: list[dict] = []
    pair_ids: list[tuple[int, int]] = []
    for a in range(1, n_cand + 1):
        ya0, ya1, xa0, xa1 = boxes[a - 1]
        for b in range(a + 1, n_cand + 1):
            yb0, yb1, xb0, xb1 = boxes[b - 1]
            # Cheap bbox rejection before the expensive EDT.
            dy = max(0, max(ya0, yb0) - min(ya1, yb1))
            dx = max(0, max(xa0, xb0) - min(xa1, xb1))
            if np.hypot(dy, dx) > MAX_GAP + 10:
                continue
            f = pair_features(prob, cand, a, b, cache)
            if f is None or f["gap"] > MAX_GAP:
                continue
            pair_rows.append(f)
            pair_ids.append((a, b))

    if pair_rows:
        pkeys = list(pair_rows[0].keys())
        pair_X = np.asarray([[f[k] for k in pkeys] for f in pair_rows], np.float32)
        pairs = np.asarray(pair_ids, np.int32)
        pair_y = label_pairs(gid, pairs)
    else:
        pkeys = []
        pair_X = np.zeros((0, 0), np.float32)
        pairs = np.zeros((0, 2), np.int32)
        pair_y = np.zeros(0, np.float32)

    return {
        "stem": meta["stem"],
        "fold": meta["fold"],
        "n_copies": meta["n_copies"],
        "cand_X": cand_X,
        "cand_y": q,
        "cand_iou": best_iou.max(axis=1),
        "cand_keys": keys,
        "pair_X": pair_X,
        "pair_y": pair_y,
        "pair_keys": pkeys,
        "pairs": pairs,
        "n_cand": n_cand,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--min-area", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "candidates.npz"))
    args = ap.parse_args()

    paths = sorted(OOF.glob("fold*/*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} OOF files | thr {args.threshold} area {args.min_area}")

    cand_X, cand_y, cand_iou, cand_group = [], [], [], []
    pair_X, pair_y, pair_group = [], [], []
    cand_keys: list[str] = []
    pair_keys: list[str] = []

    t0 = time.time()
    for i, p in enumerate(paths):
        r = build_file(p, args.threshold, args.min_area)
        if r is None:
            continue
        cand_keys = r["cand_keys"]
        pair_keys = r["pair_keys"]
        cand_X.append(r["cand_X"])
        cand_y.append(r["cand_y"])
        cand_iou.append(r["cand_iou"])
        cand_group.append(np.full(len(r["cand_y"]), i, np.int32))
        if len(r["pair_y"]):
            pair_X.append(r["pair_X"])
            pair_y.append(r["pair_y"])
            pair_group.append(np.full(len(r["pair_y"]), i, np.int32))
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            el = time.time() - t0
            print(
                f"  {i + 1}/{len(paths)}  {el:6.1f}s  "
                f"cand {sum(len(c) for c in cand_y):6d}  "
                f"pair {sum(len(c) for c in pair_y):6d}  "
                f"eta {el / (i + 1) * (len(paths) - i - 1):5.0f}s",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        cand_X=np.concatenate(cand_X) if cand_X else np.zeros((0, 0), np.float32),
        cand_y=np.concatenate(cand_y) if cand_y else np.zeros(0, np.float32),
        cand_iou=np.concatenate(cand_iou) if cand_iou else np.zeros(0, np.float32),
        cand_group=np.concatenate(cand_group) if cand_group else np.zeros(0, np.int32),
        pair_X=np.concatenate(pair_X) if pair_X else np.zeros((0, 0), np.float32),
        pair_y=np.concatenate(pair_y) if pair_y else np.zeros(0, np.float32),
        pair_group=np.concatenate(pair_group) if pair_group else np.zeros(0, np.int32),
        cand_keys=np.asarray(cand_keys),
        pair_keys=np.asarray(pair_keys),
        threshold=np.float32(args.threshold),
        min_area=np.int32(args.min_area),
    )
    print(f"wrote {out}")
    print(f"  candidates {sum(len(c) for c in cand_y)}")
    print(f"  pairs      {sum(len(c) for c in pair_y)}")


if __name__ == "__main__":
    main()
