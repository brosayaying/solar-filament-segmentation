"""Are the 'full' false positives actually unannotated filaments?

A "full FP" is a prediction with no overlap against the *one* annotator variant
used as GT. But 296 files were annotated by 2-3 people independently, and
instance counts disagree by up to 3x. So a region unlabelled by annotator A may
be labelled by annotator B -- in which case it is a real filament, and training
it as a hard negative would teach the model to suppress real filaments.

This checks each full FP against the *other* annotators of the same file.

    python scripts/deepcrack_fp_other_annotators.py
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import anns_to_labelmap, build_ann_index, load_coco  # noqa: E402
from solar.metrics import pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


def labelmap(prob, thr, min_area, close_k):
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx((prob > thr).astype(np.uint8), cv2.MORPH_CLOSE, k)
        return prob_to_labelmap(binary.astype(np.float32), thr, min_area)
    return prob_to_labelmap(prob, thr, min_area)


def main() -> None:
    cache = ROOT / "runs/deepcrack_ft_256_long/val_probs"
    files = sorted(glob.glob(str(cache / "*_prob.npy")))

    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / "splits/date_seed42.json").read_text())

    # All image_ids per file_name, so we can reach the other annotators.
    by_file: dict[str, list[str]] = defaultdict(list)
    for im in coco["images"]:
        by_file[im["file_name"]].append(im["id"])

    seen: set[str] = set()
    ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[: len(files)]

    n_full = 0
    n_confirmed = 0          # overlaps another annotator -> real filament
    n_multi_annot_file = 0   # full FPs on files that HAVE other annotators
    per_case = []

    for pf, iid in zip(files, ids):
        prob = np.load(pf).astype(np.float32)
        gt = np.load(pf.replace("_prob.npy", "_label.npy"))
        im = images[iid]
        fn = im["file_name"]
        h, w = im["height"], im["width"]

        pred = labelmap(prob, 0.45, 512, 15)
        n_pred, n_gt = int(pred.max()), int(gt.max())
        if n_pred == 0:
            continue

        if n_gt == 0:
            iou = np.zeros((n_pred, 0))
            mp = np.zeros(n_pred, bool)
        else:
            iou = pairwise_iou(pred, gt, n_pred, n_gt)
            mp = np.zeros(n_pred, bool)
            mg = np.zeros(n_gt, bool)
            for flat in np.argsort(iou, axis=None)[::-1]:
                i, j = divmod(int(flat), n_gt)
                if iou[i, j] <= 0.5:
                    break
                if mp[i] or mg[j]:
                    continue
                mp[i] = True
                mg[j] = True

        others = [o for o in by_file[fn] if o != iid]
        if not others:
            continue

        # Union of the other annotators' masks for this file.
        other_mask = np.zeros((h, w), np.uint8)
        for o in others:
            other_mask |= anns_to_labelmap(ann_index.get(o, []), h, w) > 0

        for pid in range(n_pred):
            if mp[pid]:
                continue
            best = float(iou[pid].max()) if n_gt else 0.0
            if best > 0.1:
                continue  # partial, not our concern here
            n_full += 1
            n_multi_annot_file += 1
            m = pred == pid + 1
            # Fraction of this FP covered by another annotator's labels.
            cover = float((m & (other_mask > 0)).sum()) / max(1, int(m.sum()))
            if cover > 0.5:
                n_confirmed += 1
                per_case.append(
                    {
                        "file": Path(fn).stem,
                        "area": int(m.sum()),
                        "other_annotator_coverage": round(cover, 3),
                        "n_other_annotators": len(others),
                    }
                )

    print(f"full FPs on multi-annotator files: {n_full}")
    print(f"  of which >50% covered by ANOTHER annotator: {n_confirmed} "
          f"({n_confirmed / max(1, n_full):.1%})")
    print()
    if n_confirmed:
        print("These are almost certainly real filaments that the GT annotator missed.")
        print("Training them as hard negatives would suppress true detections.")
        print()
        print("largest confirmed cases:")
        for c in sorted(per_case, key=lambda x: -x["area"])[:10]:
            print(f"  {c['file']:22s} area={c['area']:5d} "
                  f"other-annotator coverage={c['other_annotator_coverage']:.0%} "
                  f"({c['n_other_annotators']} other annotator(s))")

    out = ROOT / "docs/DEEPCRACK_FINETUNE/fp_other_annotators.json"
    out.write_text(
        json.dumps(
            {
                "full_fp_multi_annot": n_full,
                "confirmed_by_other_annotator": n_confirmed,
                "frac_confirmed": n_confirmed / max(1, n_full),
                "cases": sorted(per_case, key=lambda x: -x["area"])[:100],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
