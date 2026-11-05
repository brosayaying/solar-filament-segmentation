"""Fit the Phase 2 candidate scorer on all 707 files and persist it.

``eval_phase2_final.py`` fits the GBDT inside a ``GroupKFold`` loop and discards
each fold model -- it only ever needed OOF predictions. Test inference needs a
model trained on everything, which is what this produces.

The CNN is not refit: ``cnn_fold0..4.pt`` already exist and averaging all five is
the correct test-time behaviour for a k-fold ensemble.

    python v2/scripts/fit_phase2_scorer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

P2 = ROOT / "v2/runs/phase2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(P2 / "gbdt_qhat.joblib"))
    args = ap.parse_args()

    z = np.load(P2 / "candidates.npz", allow_pickle=True)
    X, y = z["cand_X"], z["cand_y"]
    keys = [str(k) for k in z["cand_keys"]]
    print(f"{X.shape[0]} candidates, {X.shape[1]} features, "
          f"{len(np.unique(z['cand_group']))} files")

    # Identical hyper-parameters to the OOF evaluation, so the model applied to
    # test is the same function that produced the measured +0.0186.
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0,
    )
    m.fit(X, y)

    # In-sample correlation is optimistic by construction; reported only as a
    # sanity check that the fit converged, never as a result.
    print(f"in-sample corr {np.corrcoef(m.predict(X), y)[0, 1]:.4f} "
          f"(OOF was 0.5749 -- this number is optimistic, do not quote it)")

    joblib.dump({"model": m, "features": keys}, args.out)
    print(f"wrote {args.out}")

    meta = {
        "features": keys,
        "n_train": int(X.shape[0]),
        "n_files": int(len(np.unique(z["cand_group"]))),
        "threshold": float(z["threshold"]),
        "min_area": int(z["min_area"]),
        "oof_corr": 0.5749,
    }
    (P2 / "gbdt_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {P2 / 'gbdt_meta.json'}")


if __name__ == "__main__":
    main()
