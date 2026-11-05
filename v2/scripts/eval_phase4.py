"""Does PVTv2-B2 add anything to the ConvNeXt ensemble? Honest OOF measurement.

The question
------------
Phase 4 trained five PVTv2-B2 folds to sit alongside the five ConvNeXt-Small
folds from Phase 1. PVTv2 scores *worse* solo (0.3906 vs 0.4177), which is
expected and not the point. The point is whether it fails *differently* enough
that averaging the two helps.

Why the obvious approach is wrong
---------------------------------
`score_ensemble.py` runs all five checkpoints over all 707 train files and
averages. That number (0.4346) is **leaky**: each file is held out by exactly
one fold, so four of the five members trained on it. The gain it reports is
mostly memorisation. It must not be quoted, and it cannot be used to judge
Phase 4 either -- adding more members that memorised the same files just adds
more memorisation.

The honest design
-----------------
Both `seg_v2_fold{f}` and `phase4_pvt_fold{f}` were trained on the *same* split
(`v2/splits/week5_seed42/fold{f}.json`). So for fold *f*, both checkpoints held
out exactly the same files. Averaging those two predictions on fold *f*'s val
set is therefore leak-free, and pooling the five such 2-member ensembles over
all 707 files gives an honest pooled OOF PQ.

This measures a 2-member ensemble, not a 10-member one. That is a deliberate
trade: it answers "does a PVTv2 member add value to a ConvNeXt member that saw
the same data?" without any contamination. A 10-member number would be larger
and meaningless.

What is reported
----------------
* pooled PQ for ConvNeXt alone, PVTv2 alone, and the average of the two
* the paired delta (ensemble - ConvNeXt) with a bootstrap CI over files
* prediction correlation between the two members, which is the mechanism: if
  the correlation is as high as ConvNeXt-to-ConvNeXt, there is no diversity to
  exploit and the ensemble gain is just variance reduction

    python v2/scripts/eval_phase4.py
    python v2/scripts/eval_phase4.py --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

CONV = ROOT / "v2/runs/seg_v2/oof"
PVT = ROOT / "v2/runs/phase4/oof"
OUT = ROOT / "v2/runs/phase4/eval_phase4.json"

# Phase 1 pooled OOF, for orientation.
PHASE1_POOLED = 0.4220


def fold_files(oof_dir: Path, fold: int) -> dict[str, Path]:
    d = oof_dir / f"fold{fold}"
    if not d.exists():
        return {}
    return {p.stem: p for p in sorted(d.glob("*.npz"))}


def load_pair(conv_p: Path, pvt_p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (conv_prob, pvt_prob, labels) for one file."""
    with np.load(conv_p, allow_pickle=False) as z:
        conv = z["prob"].astype(np.float32)
        labels = z["labels"]
    with np.load(pvt_p, allow_pickle=False) as z:
        pvt = z["prob"].astype(np.float32)
    if conv.shape != pvt.shape:
        raise SystemExit(f"shape mismatch {conv_p.name}: {conv.shape} vs {pvt.shape}")
    return conv, pvt, labels


class Variant:
    """Accumulates pooled PQ and per-file PQ for one prediction variant.

    Streaming rather than batch: holding all 707 files x 3 float32 2048^2 maps
    is ~35 GB, which is an OOM on this 46 GB box. This project has already been
    killed by that once (`load_oof()` on the full set), so every variant is
    scored one file at a time and only the scalars are retained.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.pooled = PQAccumulator(threshold=0.5)
        self.per_file: dict[str, float] = {}

    def add(self, stem: str, prob: np.ndarray, labels: np.ndarray,
            thr: float, area: int, close_k: int) -> None:
        pred = prob_to_labelmap(
            prob, threshold=thr, min_area=area, close_k=close_k
        )
        # Pooled: every annotator copy contributes, which is what the official
        # scorer does.
        for lab in labels:
            self.pooled.update(pred, lab)
        # Per-file: pooled over that file's copies. The bootstrap resamples
        # *files*, because three annotator copies of one image are not three
        # independent observations.
        acc = PQAccumulator(threshold=0.5)
        for lab in labels:
            acc.update(pred, lab)
        self.per_file[stem] = acc.summary()["pq"]

    def summary(self) -> dict:
        return self.pooled.summary()


def paired_bootstrap(
    a: dict[str, float], b: dict[str, float], n: int = 2000, seed: int = 42
) -> dict:
    """Bootstrap the mean of (b - a) over files."""
    stems = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in stems], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "n_files": len(stems),
        "delta": float(d.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "p_positive": float((means > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-area", type=int, default=256)
    ap.add_argument("--close-k", type=int, default=15)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    # Collect the files both members held out, fold by fold.
    pairs: list[tuple[str, Path, Path]] = []
    for f in range(5):
        conv = fold_files(CONV, f)
        pvt = fold_files(PVT, f)
        shared = sorted(set(conv) & set(pvt))
        if not shared:
            print(f"fold {f}: no shared files (conv {len(conv)}, pvt {len(pvt)})")
            continue
        if len(shared) != len(conv) or len(shared) != len(pvt):
            print(
                f"fold {f}: WARNING conv {len(conv)} / pvt {len(pvt)} / "
                f"shared {len(shared)} -- using the intersection"
            )
        for s in shared:
            pairs.append((s, conv[s], pvt[s]))

    if not pairs:
        raise SystemExit("no paired OOF files found; run dump_phase4_oof.sh first")

    print(f"paired files: {len(pairs)}")

    def run(thr: float, area: int, close_k: int) -> dict[str, Variant]:
        """Score all three variants in one streaming pass over the files."""
        variants = {
            "conv": Variant("ConvNeXt (Phase 1)"),
            "pvt": Variant("PVTv2 (Phase 4)"),
            "ens": Variant("ensemble (avg)"),
        }
        for stem, cp, pp in pairs:
            c, p, lab = load_pair(cp, pp)
            variants["conv"].add(stem, c, lab, thr, area, close_k)
            variants["pvt"].add(stem, p, lab, thr, area, close_k)
            variants["ens"].add(stem, 0.5 * (c + p), lab, thr, area, close_k)
        return variants

    # Prediction correlation, computed once. Pearson on a stride-4 subsample:
    # the full 4.2M pixels are highly autocorrelated, so the subsample estimates
    # the same quantity for 1/16 the cost.
    corrs: list[float] = []
    for stem, cp, pp in pairs:
        c, p, _ = load_pair(cp, pp)
        cs = c[::4, ::4].ravel()
        ps = p[::4, ::4].ravel()
        if cs.std() > 1e-6 and ps.std() > 1e-6:
            corrs.append(float(np.corrcoef(cs, ps)[0, 1]))
    mean_corr = float(np.mean(corrs))
    print(f"mean conv/pvt prediction correlation: {mean_corr:.4f}")

    if args.sweep:
        print(f"\n{'thr':>5}{'area':>7}{'close':>7}{'conv':>9}{'pvt':>9}{'ens':>9}{'delta':>9}")
        print("-" * 55)
        best = None
        for thr in (0.40, 0.45, 0.50, 0.55, 0.60):
            for area in (128, 256, 512):
                for ck in (0, 15):
                    v = run(thr, area, ck)
                    sc, sp, se = (v[k].summary() for k in ("conv", "pvt", "ens"))
                    d = se["pq"] - sc["pq"]
                    print(
                        f"{thr:>5.2f}{area:>7}{ck:>7}"
                        f"{sc['pq']:>9.4f}{sp['pq']:>9.4f}{se['pq']:>9.4f}{d:>+9.4f}"
                    )
                    if best is None or se["pq"] > best[0]:
                        best = (se["pq"], thr, area, ck)
        print(f"\nbest ensemble setting: PQ {best[0]:.4f} at thr {best[1]} "
              f"area {best[2]} close {best[3]}")
        return

    thr, area, ck = args.threshold, args.min_area, args.close_k
    v = run(thr, area, ck)
    sc, sp, se = (v[k].summary() for k in ("conv", "pvt", "ens"))

    print(f"\npost-processing: thr {thr} / min_area {area} / close {ck}")
    print(f"{'':<22}{'PQ':>8}{'SQ':>8}{'RQ':>8}")
    print("-" * 46)
    for name, s in (("ConvNeXt (Phase 1)", sc), ("PVTv2 (Phase 4)", sp),
                    ("ensemble (avg)", se)):
        print(f"{name:<22}{s['pq']:>8.4f}{s['sq']:>8.4f}{s['rq']:>8.4f}")

    boot_ens = paired_bootstrap(v["conv"].per_file, v["ens"].per_file,
                                n=args.bootstrap)
    boot_pvt = paired_bootstrap(v["conv"].per_file, v["pvt"].per_file,
                                n=args.bootstrap)

    print(f"\npaired vs ConvNeXt (n={boot_ens['n_files']} files, "
          f"{args.bootstrap} bootstraps):")
    print(f"  ensemble  {boot_ens['delta']:+.4f}  "
          f"CI [{boot_ens['ci_lo']:+.4f}, {boot_ens['ci_hi']:+.4f}]  "
          f"P(>0)={boot_ens['p_positive']:.3f}")
    print(f"  PVTv2     {boot_pvt['delta']:+.4f}  "
          f"CI [{boot_pvt['ci_lo']:+.4f}, {boot_pvt['ci_hi']:+.4f}]  "
          f"P(>0)={boot_pvt['p_positive']:.3f}")

    # The pre-registered criterion.
    gain = se["pq"] - sc["pq"]
    passed = gain >= 0.01 and boot_ens["ci_lo"] > 0
    print(f"\nacceptance (>= +0.0100 and CI excludes 0): "
          f"{'PASS' if passed else 'FAIL'}  (gain {gain:+.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(
            {
                "n_files": len(pairs),
                "postprocess": {"threshold": thr, "min_area": area, "close_k": ck},
                "convnext": sc,
                "pvtv2": sp,
                "ensemble": se,
                "mean_corr_conv_pvt": mean_corr,
                "bootstrap_ensemble_vs_conv": boot_ens,
                "bootstrap_pvt_vs_conv": boot_pvt,
                "phase1_pooled_reference": PHASE1_POOLED,
                "acceptance_passed": bool(passed),
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
