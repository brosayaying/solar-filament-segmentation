"""Shared classical-CV utilities: disk masking, local contrast, caching, eval."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("MAGFiLO_1.0_Kaggle_2026")
cv2.setNumThreads(4)


def disk_mask(img: np.ndarray, thr: int = 25) -> np.ndarray:
    """Binary mask of the solar disk (largest bright blob, holes filled)."""
    m = (img > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones_like(img, np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out[:] = 0
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def residual(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Local background subtraction: negative values are darker than surroundings."""
    f = img.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), sigma)


def load_split(path: str = "splits/date_seed42.json") -> dict:
    return json.load(open(path))


def load_coco_index():
    from solar.data import build_ann_index, load_coco

    coco = load_coco(ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    return coco, build_ann_index(coco), {i["id"]: i for i in coco["images"]}


def read_image(file_name: str) -> np.ndarray:
    img = cv2.imread(str(ROOT / "train/train_images" / file_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(file_name)
    return img


def components(binary: np.ndarray, min_area: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Label connected components, dropping those below ``min_area``."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = np.zeros(binary.shape, np.int32)
    nxt = 1
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_area:
            continue
        out[lab == c] = nxt
        nxt += 1
    return out, stats
