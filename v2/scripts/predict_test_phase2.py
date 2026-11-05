"""Phase 2 test inference: 5-fold ensemble -> candidates -> q̂ -> submission.

Two stages, because the expensive part (5 model forwards per image) should not be
repeated while tuning the cheap part (candidate scoring).

``--stage probs``
    Run the 5-fold ensemble on the 180 test images and cache the averaged
    probability maps to ``v2/runs/phase2/test_probs/``. ~6 min.

``--stage submit``
    Load the cached maps, generate candidates, score them with the persisted
    GBDT, threshold, and write the submission CSV. ~2 min.

The scorer is the **GBDT alone at tau 0.35**, not the rank-average blend that
scored marginally higher on OOF (0.4406 vs 0.4386). The blend's +0.0020 is inside
the paired CI width, and its threshold is a *quantile*, which depends on the
candidate distribution -- a distribution that differs on test (different
instrument, 11.7 % of test files are 2021-22 against 5.7 % of train). An absolute
threshold on a calibrated score transfers; a rank threshold does not.

    python v2/scripts/predict_test_phase2.py --stage probs
    python v2/scripts/predict_test_phase2.py --stage submit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import normalise_image, parse_preprocess, preprocess_image  # noqa: E402
from solar.instances import candidate_features, generate_candidates  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import labelmap_to_rle  # noqa: E402
from solar.predict import predict_prob  # noqa: E402

RUNS = ROOT / "v2/runs"
P2 = RUNS / "phase2"
TEST_DIR = ROOT / "MAGFiLO_1.0_Kaggle_2026/test/test_images"
PROBS = P2 / "test_probs"


def stage_probs(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt0 = torch.load(RUNS / f"seg_v2_fold{args.folds[0]}/best.pt",
                       map_location="cpu", weights_only=False)
    cfg = ckpt0["cfg"]
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    pp_steps = parse_preprocess(cfg["data"].get("preprocess"))
    mask_offdisk = cfg["data"].get("mask_offdisk", False)
    offdisk_value = cfg["data"].get("offdisk_value", -3.0)

    models = []
    for f in args.folds:
        ck = torch.load(RUNS / f"seg_v2_fold{f}/best.pt", map_location="cpu",
                        weights_only=False)
        m = build_model(ck["cfg"]).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
        print(f"fold {f}: epoch {ck.get('epoch')} val pq {ck.get('pq', 0):.4f}")

    files = sorted(p for p in TEST_DIR.iterdir()
                   if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    PROBS.mkdir(parents=True, exist_ok=True)
    print(f"\n{len(files)} test images | {len(models)} folds | tta {args.tta}")

    for path in tqdm(files, desc="predict"):
        out = PROBS / f"{path.stem}.npy"
        if out.exists() and not args.overwrite:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img, disk = preprocess_image(img, pp_steps)
        x = normalise_image(img, mean, std, disk=disk,
                            mask_offdisk=mask_offdisk,
                            offdisk_value=offdisk_value)
        tensor = torch.from_numpy(x).unsqueeze(0).to(device)

        acc = None
        for m in models:
            p = predict_prob(m, tensor, tta=args.tta)
            acc = p if acc is None else acc + p
        prob = (acc / len(models)).astype(np.float16)
        np.save(out, prob)

    print(f"wrote {len(files)} probability maps -> {PROBS}")


def stage_submit(args) -> None:
    bundle = joblib.load(P2 / "gbdt_qhat.joblib")
    model, keys = bundle["model"], bundle["features"]
    print(f"scorer: {len(keys)} features, tau {args.tau}")

    files = sorted(PROBS.glob("*.npy"))
    if not files:
        raise SystemExit(f"no cached probabilities in {PROBS}; run --stage probs")
    print(f"{len(files)} cached probability maps")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    counts: list[int] = []
    empty: list[str] = []
    n_cand_total = 0
    n_kept_total = 0

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filament_id", "segmentation_rle"])

        for path in tqdm(files, desc="score"):
            prob = np.load(path).astype(np.float32)
            cand = generate_candidates(
                prob, threshold=args.cand_threshold, min_area=args.cand_min_area
            )
            n = int(cand.max())
            n_cand_total += n

            if n == 0:
                counts.append(0)
                empty.append(path.stem)
                continue

            feats = [candidate_features(prob, cand, cid) for cid in range(1, n + 1)]
            X = np.asarray([[f[k] for k in keys] for f in feats], np.float32)
            q_hat = model.predict(X)

            keep = np.concatenate([[False], q_hat > args.tau])
            n_kept_total += int(keep.sum())

            lut = np.zeros(n + 1, np.int32)
            nxt = 1
            for cid in range(1, n + 1):
                if keep[cid]:
                    lut[cid] = nxt
                    nxt += 1
            labelmap = lut[cand]

            rles = labelmap_to_rle(labelmap)
            for inst, counts_rle in rles.items():
                writer.writerow([f"{path.stem}_{inst}", counts_rle])
                rows += 1
            counts.append(len(rles))
            if not rles:
                empty.append(path.stem)

    print(f"\nwrote {rows} filaments -> {out_path}")
    print(f"candidates {n_cand_total} -> kept {n_kept_total} "
          f"({n_kept_total / max(1, n_cand_total):.1%})")
    if counts:
        print(f"filaments/image: mean {np.mean(counts):.2f} "
              f"min {min(counts)} max {max(counts)}")
    if empty:
        print(f"WARNING: {len(empty)} images got zero predictions: {empty[:8]}")

    (P2 / "test_submit_meta.json").write_text(json.dumps({
        "tau": args.tau,
        "cand_threshold": args.cand_threshold,
        "cand_min_area": args.cand_min_area,
        "n_images": len(files),
        "n_candidates": n_cand_total,
        "n_kept": n_kept_total,
        "n_filaments": rows,
        "empty_images": empty,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probs", "submit"], required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--cand-threshold", type=float, default=0.30)
    ap.add_argument("--cand-min-area", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "submissions/seg_v2_phase2.csv"))
    args = ap.parse_args()

    if args.stage == "probs":
        stage_probs(args)
    else:
        stage_submit(args)


if __name__ == "__main__":
    main()
