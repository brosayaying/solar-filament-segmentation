"""Measure the 5-fold ensemble on the full train set.

The OOF dumps give each file a prediction from *one* fold — the fold that held it
out. That is the right thing for an honest single-model number, but it cannot
measure the ensemble, because the ensemble needs all five checkpoints to predict
the same image.

Two stages, deliberately separated:

1. **infer** — run every fold checkpoint over every train file, average the
   probabilities, write one float16 ``.npy`` per file. ~15 min.
2. **score** — stream those files back and sweep post-processing. Cheap, and
   re-runnable without repeating inference.

The first version kept all 707 probability maps in RAM (13 GB) and re-rasterised
the GT for every grid point, which was both slow and an OOM risk on a box that
already has other work resident. Splitting the stages fixes both.

    python v2/scripts/score_ensemble.py infer
    python v2/scripts/score_ensemble.py score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import FilamentDataset, collate_val, load_coco  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402
from solar.models import build_model  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402
from solar.predict import predict_prob  # noqa: E402

ANN = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
RUNS = ROOT / "v2/runs"
PROB_DIR = RUNS / "seg_v2/ensemble_probs"


def infer(folds: list[int], tta: bool, num_workers: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coco = load_coco(ANN)

    first_id: dict[str, str] = {}
    for img in coco["images"]:
        first_id.setdefault(Path(img["file_name"]).stem, img["id"])
    stems = sorted(first_id)

    ckpt0 = torch.load(RUNS / f"seg_v2_fold{folds[0]}/best.pt",
                       map_location="cpu", weights_only=False)
    cfg = ckpt0["cfg"]
    ds = FilamentDataset(
        ROOT / cfg["data"]["root"], coco, [first_id[s] for s in stems],
        mode="val",
        mean=cfg["data"]["mean"], std=cfg["data"]["std"],
        preprocess=cfg["data"].get("preprocess"),
        mask_offdisk=cfg["data"].get("mask_offdisk", False),
        offdisk_value=cfg["data"].get("offdisk_value", -3.0),
        preprocess_cache=cfg["data"].get("preprocess_cache"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_val,
                        pin_memory=True)

    models = []
    for f in folds:
        ck = torch.load(RUNS / f"seg_v2_fold{f}/best.pt", map_location="cpu",
                        weights_only=False)
        m = build_model(ck["cfg"]).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
        print(f"fold {f}: epoch {ck.get('epoch')} pq {ck.get('pq', 0):.4f}")

    PROB_DIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc="ensemble infer"):
            stem = Path(batch["file_name"][0]).stem
            out = PROB_DIR / f"{stem}.npy"
            if out.exists():
                continue
            image = batch["image"][0].to(device)
            acc = None
            for m in models:
                p = predict_prob(m, image, tta=tta)
                acc = p if acc is None else acc + p
            np.save(out, (acc / len(models)).astype(np.float16))
    print(f"wrote {len(list(PROB_DIR.glob('*.npy')))} probs -> {PROB_DIR}")


def score() -> None:
    from solar.data import anns_to_labelmap

    coco = load_coco(ANN)
    images = {i["id"]: i for i in coco["images"]}

    anns_by_id: dict[str, list] = {}
    for a in coco["annotations"]:
        anns_by_id.setdefault(a["image_id"], []).append(a)
    ids_by_stem: dict[str, list[str]] = {}
    for iid in anns_by_id:
        ids_by_stem.setdefault(Path(images[iid]["file_name"]).stem, []).append(iid)

    # Rasterise GT once, not once per grid point.
    print("rasterising GT ...")
    gt: dict[str, list[np.ndarray]] = {}
    for stem, ids in tqdm(ids_by_stem.items(), desc="gt"):
        gt[stem] = [anns_to_labelmap(anns_by_id[i], 2048, 2048) for i in ids]

    grid = [(t, a, c) for t in (0.45, 0.5, 0.55) for a in (256, 512) for c in (0, 15)]
    results = []
    print(f"\n{'thr':>5} {'area':>6} {'close':>6} {'PQ':>7} {'SQ':>7} {'RQ':>7}")
    print("-" * 46)
    for thr, area, ck in grid:
        acc = PQAccumulator(threshold=0.5)
        for path in sorted(PROB_DIR.glob("*.npy")):
            stem = path.stem
            if stem not in gt:
                continue
            prob = np.load(path).astype(np.float32)
            pred = prob_to_labelmap(prob, threshold=thr, min_area=area, close_k=ck)
            for lab in gt[stem]:
                acc.update(pred, lab)
        s = acc.summary()
        results.append({"threshold": thr, "min_area": area, "close_k": ck, **s})
        print(f"{thr:>5.2f} {area:>6} {ck:>6} {s['pq']:>7.4f} {s['sq']:>7.4f} "
              f"{s['rq']:>7.4f}")

    results.sort(key=lambda r: r["pq"], reverse=True)
    best = results[0]
    print(f"\nbest ensemble: thr {best['threshold']} area {best['min_area']} "
          f"close {best['close_k']} -> PQ {best['pq']:.4f}")

    out = RUNS / "seg_v2/ensemble_results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["infer", "score"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    if args.stage == "infer":
        infer(args.folds, args.tta, args.num_workers)
    else:
        score()


if __name__ == "__main__":
    main()
