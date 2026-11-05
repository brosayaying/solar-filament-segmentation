"""Paired per-image comparison of two trained checkpoints.

    python scripts/compare_runs.py runs/long_160ep_512 runs/norm_disk_raw
    python scripts/compare_runs.py runs/long_160ep_512 runs/norm_disk_flatten --n 142

Each checkpoint is served exactly as it was trained: architecture, mean/std,
preprocessing steps and post-processing threshold all come from that run's own
``config.json``. This is the whole point -- a model trained on flattened input
must be *evaluated* on flattened input, and a model trained on raw input on raw
input. Feeding either one the other's preprocessing measures a distribution
shift, not a difference in method.

Contrast with ``scripts/preprocess_ablation.py``, which applied preprocessing at
eval time against a raw-trained model. That cannot answer whether preprocessing
helps (see ``docs/PREPROCESSING.md`` §3), so it is superseded by this script.

Reports
-------
* PQ / SQ / RQ for each checkpoint over the same val files
* a paired per-image test on the *difference*, because a point estimate without
  a confidence interval is not a result (``docs/IDEAS_TO_EXPLORE.md``)

The paired test is the right one here: both models see the same 142 images, so
the per-image differences remove image difficulty and only the method effect
remains. The reported CI is a 95% t-interval on the mean per-image delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import (  # noqa: E402
    anns_to_labelmap,
    build_ann_index,
    load_coco,
    normalise_image,
    parse_preprocess,
    preprocess_image,
)
from solar.metrics import PQAccumulator  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


class Runner:
    """One trained checkpoint, served exactly as configured.

    ``threshold_override`` forces one binarisation threshold on every run. This
    matters when comparing against checkpoints trained before the switch to the
    competition's 0.5: those runs stored a *tuned* 0.55, and honouring it would
    confound the preprocessing difference with a post-processing difference.
    """

    def __init__(
        self,
        run_dir: Path,
        device: torch.device,
        threshold_override: float | None = None,
    ) -> None:
        self.run_dir = run_dir
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            raise SystemExit(f"no config.json in {run_dir}")
        self.cfg = json.loads(cfg_path.read_text())
        d = self.cfg["data"]
        self.name = run_dir.name
        self.mean = float(d.get("mean", 0.5))
        self.std = float(d.get("std", 0.25))
        self.steps = parse_preprocess(d.get("preprocess"))
        self.mask_offdisk = bool(d.get("mask_offdisk", False))
        self.offdisk_value = float(d.get("offdisk_value", -3.0))
        # 0.5 is the competition's binarisation threshold. Older runs stored a
        # tuned value (0.55). Default is to force 0.5 on everything so the
        # comparison measures preprocessing, not post-processing.
        self.cfg_threshold = float(self.cfg["eval"].get("threshold", 0.5))
        self.threshold = (
            float(threshold_override) if threshold_override is not None else 0.5
        )
        self.min_area = int(self.cfg["eval"].get("min_area", 512))

        weights = run_dir / "best.pt"
        if not weights.exists():
            raise SystemExit(f"no best.pt in {run_dir}")
        self.model = build_model(self.cfg).to(device).eval()
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck.get("model", ck))
        self.device = device

    def describe(self) -> str:
        pp = "+".join(sorted(self.steps)) or "none"
        note = ""
        if self.threshold != self.cfg_threshold:
            note = f" (override; cfg had {self.cfg_threshold:.2f})"
        return (
            f"{self.name:24s} pp={pp:20s} mean={self.mean:.4f} std={self.std:.4f} "
            f"thr={self.threshold:.2f}{note}"
        )

    @torch.no_grad()
    def predict(self, raw: np.ndarray) -> np.ndarray:
        img, disk = preprocess_image(raw, self.steps)
        x = normalise_image(
            img,
            self.mean,
            self.std,
            disk=disk,
            mask_offdisk=self.mask_offdisk,
            offdisk_value=self.offdisk_value,
        )
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)
        return torch.sigmoid(self.model(t))[0, 0].float().cpu().numpy()


def paired_ci(delta: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, t-statistic and 95% t-interval for a paired per-image delta."""
    from scipy import stats

    n = len(delta)
    mean = float(delta.mean())
    if n < 2:
        return mean, 0.0, float("nan"), float("nan")
    se = float(delta.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        return mean, float("inf"), mean, mean
    t = mean / se
    crit = float(stats.t.ppf(0.975, n - 1))
    return mean, t, mean - crit * se, mean + crit * se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="baseline run directory (e.g. runs/long_160ep_512)")
    ap.add_argument("run_b", help="candidate run directory")
    ap.add_argument("--n", type=int, default=142, help="number of val files (deduplicated)")
    ap.add_argument("--split", default="splits/date_seed42.json")
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--root", default="MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="binarisation threshold forced on every run (default 0.5, the "
        "competition's). Pass a number to test a tuned operating point instead.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = ROOT / args.root
    coco = load_coco(root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / args.split).read_text())

    # Deduplicate by file: multi-annotator variants of one image would weight it
    # more than once and turn val PQ into a measure of annotator agreement.
    seen: set[str] = set()
    ids: list[str] = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[: args.n]
    print(f"{len(ids)} val files (deduplicated)\n")

    runs = [
        Runner(ROOT / args.run_a, device, args.threshold),
        Runner(ROOT / args.run_b, device, args.threshold),
    ]
    for r in runs:
        print(r.describe())
    print()

    img_dir = root / "train/train_images"
    per_image: dict[str, list[float]] = {r.name: [] for r in runs}
    accs = {r.name: PQAccumulator() for r in runs}

    for iid in tqdm(ids, desc="compare"):
        im = images[iid]
        raw = cv2.imread(str(img_dir / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        gt = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        for r in runs:
            prob = r.predict(raw)
            pred = prob_to_labelmap(prob, threshold=r.threshold, min_area=r.min_area)
            accs[r.name].update(pred, gt)
            # Per-image PQ for the paired test. A single-image accumulator gives
            # that image's own PQ, which is 0 when it has no TP.
            a = PQAccumulator()
            a.update(pred, gt)
            per_image[r.name].append(a.summary()["pq"])

    print(f"\n=== aggregate PQ (n={len(ids)}, threshold={args.threshold:.2f}) ===")
    print(f"{'run':>24} {'thr':>5} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    for r in runs:
        s = accs[r.name].summary()
        print(
            f"{r.name:>24} {r.threshold:5.2f} {s['pq']:7.4f} {s['sq']:7.4f} "
            f"{s['rq']:7.4f} {s['tp']:5d} {s['fp']:5d} {s['fn']:5d}"
        )

    a_vals = np.array(per_image[runs[0].name])
    b_vals = np.array(per_image[runs[1].name])
    delta = b_vals - a_vals
    mean, t, lo, hi = paired_ci(delta)

    print(f"\n=== paired per-image test: {runs[1].name} - {runs[0].name} ===")
    verdict = "REAL" if (lo > 0 or hi < 0) else "n.s."
    if lo <= 0 <= hi:
        verdict = "n.s. (CI spans 0)"
    elif mean > 0:
        verdict = "REAL improvement"
    else:
        verdict = "REAL regression"
    print(f"  mean delta : {mean:+.4f}")
    print(f"  95% CI     : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  t          : {t:+.2f}  (df={len(delta) - 1})")
    print(f"  better on  : {int((delta > 0).sum())}/{len(delta)} images")
    print(f"  identical  : {int((delta == 0).sum())}/{len(delta)} images")
    print(f"  -> {verdict}")

    print("\n  Context: the decoder experiment moved PQ by +0.040 with CI")
    print("  [+0.025, +0.056] and was still not enough for the leaderboard.")
    print("  Anything under ~0.02 here should be treated as noise.")

    if args.out:
        dest = ROOT / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(
                {
                    "n_images": int(len(delta)),
                    "runs": [
                        {
                            "name": r.name,
                            "config": r.cfg["data"],
                            "threshold": r.threshold,
                            "metrics": accs[r.name].summary(),
                        }
                        for r in runs
                    ],
                    "paired_delta": {
                        "a": runs[0].name,
                        "b": runs[1].name,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "t": t,
                        "df": int(len(delta) - 1),
                        "n_better": int((delta > 0).sum()),
                        "n_identical": int((delta == 0).sum()),
                        "verdict": verdict,
                    },
                    "per_image": {k: [float(x) for x in v] for k, v in per_image.items()},
                },
                indent=2,
            )
        )
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
