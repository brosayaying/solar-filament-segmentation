"""Is watershed the right tool for the split false positives?

The TP/FP analysis found 346 FPs that overlap a GT (splits/merges) and 340 with
no overlap. The natural instinct is "use watershed to split merged blobs". But
that is only correct if the failure is *merged* instances.

If instead the model produces several *disconnected* fragments of one filament,
connected components already separates them and watershed -- which splits
connected blobs -- cannot merge them back. The fix would be the opposite:
morphological closing to join fragments before labelling.

This measures which failure actually dominates, so the sweep is aimed correctly.

    python scripts/deepcrack_split_diagnosis.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.metrics import pairwise_iou  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)


def main() -> None:
    cache = ROOT / "runs/deepcrack_ft_256_long/val_probs"
    files = sorted(glob.glob(str(cache / "*_prob.npy")))

    n_split = n_pure = 0
    # For each split FP, how many GT instances does it overlap, and how many
    # separate predicted components cover the GT it belongs to?
    frag_counts = []
    merge_counts = []

    for pf in files:
        prob = np.load(pf).astype(np.float32)
        label = np.load(pf.replace("_prob.npy", "_label.npy"))
        pred = prob_to_labelmap(prob, 0.5, 512)
        n_pred, n_gt = int(pred.max()), int(label.max())
        if n_pred == 0 or n_gt == 0:
            continue

        iou = pairwise_iou(pred, label, n_pred, n_gt)
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

        for pid in range(n_pred):
            if mp[pid]:
                continue
            if iou[pid].max() > 0.1:
                n_split += 1
            else:
                n_pure += 1

        # For each GT, how many predicted components overlap it meaningfully?
        # >1 means the model fragmented one filament into pieces (needs merging).
        for gid in range(n_gt):
            overlapping = int((iou[:, gid] > 0.1).sum())
            if overlapping > 1:
                frag_counts.append(overlapping)

        # For each prediction, how many GTs does it overlap? >1 means merged.
        for pid in range(n_pred):
            overlapping = int((iou[pid] > 0.1).sum())
            if overlapping > 1:
                merge_counts.append(overlapping)

    print(f"images: {len(files)}")
    print(f"split/merge FPs: {n_split}   pure FPs: {n_pure}")
    print()
    print("GT instances covered by >1 predicted component (FRAGMENTATION):")
    print(f"  count: {len(frag_counts)}")
    if frag_counts:
        a = np.array(frag_counts)
        print(f"  mean components per such GT: {a.mean():.2f}  max: {a.max()}")
    print()
    print("Predictions overlapping >1 GT (MERGING):")
    print(f"  count: {len(merge_counts)}")
    if merge_counts:
        a = np.array(merge_counts)
        print(f"  mean GTs per such prediction: {a.mean():.2f}  max: {a.max()}")
    print()
    if len(frag_counts) > len(merge_counts):
        print("=> FRAGMENTATION dominates. Watershed (splits blobs) is the WRONG tool;")
        print("   closing / merging is the right direction.")
    else:
        print("=> MERGING dominates. Watershed is the right tool.")

    out = ROOT / "docs/DEEPCRACK_FINETUNE/split_diagnosis.json"
    out.write_text(
        json.dumps(
            {
                "n_images": len(files),
                "split_fp": n_split,
                "pure_fp": n_pure,
                "fragmented_gt": len(frag_counts),
                "frag_mean_components": float(np.mean(frag_counts)) if frag_counts else 0.0,
                "merged_pred": len(merge_counts),
                "merge_mean_gts": float(np.mean(merge_counts)) if merge_counts else 0.0,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
