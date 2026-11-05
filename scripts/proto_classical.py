"""Quick prototype: local-contrast thresholding baseline, evaluated with PQ."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar.data import anns_to_labelmap, build_ann_index, load_coco  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402

ROOT = Path("MAGFiLO_1.0_Kaggle_2026")
cv2.setNumThreads(4)


def disk_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of the solar disk (largest bright blob)."""
    m = (img > 25).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones_like(img, np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lab == k).astype(np.uint8)
    # fill holes
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(out)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def residual(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    return f - bg


def label_from_resid(res: np.ndarray, disk: np.ndarray, thr: float, min_area: int) -> np.ndarray:
    binary = ((res < -thr) & (disk > 0)).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros(res.shape, np.int32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros(res.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out


def main() -> None:
    coco = load_coco(ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    split = json.load(open("splits/date_seed42.json"))
    ann_index = build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}
    val = split["val"][: int(sys.argv[1]) if len(sys.argv) > 1 else 40]

    cache = {}
    for iid in val:
        meta = images[iid]
        img = cv2.imread(str(ROOT / "train/train_images" / meta["file_name"]), cv2.IMREAD_GRAYSCALE)
        cache[iid] = (img, disk_mask(img), residual(img), anns_to_labelmap(ann_index[iid], 2048, 2048))

    for thr in [4, 6, 8, 10, 12, 15]:
        for min_area in [128, 512]:
            acc = PQAccumulator()
            for iid in val:
                img, disk, res, gt = cache[iid]
                pred = label_from_resid(res, disk, thr, min_area)
                acc.update(pred, gt)
            s = acc.summary()
            print(f"thr={thr:>4} min_area={min_area:>4}  PQ={s['pq']:.4f} SQ={s['sq']:.3f} RQ={s['rq']:.3f} "
                  f"tp={s['tp']} fp={s['fp']} fn={s['fn']}")


if __name__ == "__main__":
    main()
