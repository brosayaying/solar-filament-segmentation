"""Stage-1 dataset: crops with CenterNet targets, plus box decoding at inference."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import build_ann_index, load_coco
from .detector_loss import STRIDE, encode_targets

cv2.setNumThreads(0)


class DetectionDataset(Dataset):
    """Returns image crops plus CenterNet targets built from COCO bboxes.

    ``foreground_bias`` oversamples crops that contain at least one filament.
    Filaments cover only ~0.36% of an image, so uniform random crops are ~50%
    empty at 512^2 and the detector would spend most of its gradient on
    background.
    """

    def __init__(
        self,
        root: str | Path,
        coco: dict[str, Any],
        image_ids: Iterable[str],
        patch_size: int = 512,
        mean: float = 0.5,
        std: float = 0.25,
        augment: bool = True,
        foreground_bias: float = 0.75,
        max_tries: int = 12,
        with_size: bool = True,
    ) -> None:
        self.root = Path(root)
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.foreground_bias = foreground_bias
        self.max_tries = max_tries
        self.with_size = with_size

        self.ann_index = build_ann_index(coco)
        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(image_ids)
        self.img_dir = self.root / "train/train_images"

    def __len__(self) -> int:
        return len(self.image_ids)

    def _boxes(self, image_id: str) -> list[np.ndarray]:
        return [np.asarray(a["bbox"], np.float32) for a in self.ann_index.get(image_id, [])]

    def _centers(self, image_id: str) -> list[np.ndarray]:
        """Spine midpoint per annotation, used as the heatmap centre.

        The bbox centre lands on the filament only ~51% of the time because
        filaments are thin and curved; the spine midpoint hits 99.9%.
        """
        out: list[np.ndarray] = []
        for a in self.ann_index.get(image_id, []):
            spine = a.get("spine")
            if spine:
                pts = np.asarray(spine, np.float32).reshape(-1, 2)
                out.append(pts[len(pts) // 2])
            else:
                x, y, w, h = a["bbox"]
                out.append(np.array([x + w / 2, y + h / 2], np.float32))
        return out

    def _sample_window(self, boxes: list[np.ndarray], h: int, w: int) -> tuple[int, int]:
        ps = self.patch_size
        if h <= ps or w <= ps:
            return 0, 0

        if boxes and random.random() < self.foreground_bias:
            for _ in range(self.max_tries):
                b = random.choice(boxes)
                x, y, bw, bh = b
                # Centre the crop near the box, jittered by up to half a patch.
                cx = x + bw / 2 + random.uniform(-ps / 2, ps / 2)
                cy = y + bh / 2 + random.uniform(-ps / 2, ps / 2)
                x0 = int(np.clip(cx - ps / 2, 0, w - ps))
                y0 = int(np.clip(cy - ps / 2, 0, h - ps))
                if self._has_box(boxes, x0, y0, ps):
                    return x0, y0
        return random.randint(0, w - ps), random.randint(0, h - ps)

    @staticmethod
    def _has_box(boxes: list[np.ndarray], x0: int, y0: int, ps: int) -> bool:
        for x, y, bw, bh in boxes:
            if x < x0 + ps and x + bw > x0 and y < y0 + ps and y + bh > y0:
                return True
        return False

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        meta = self.images[image_id]
        img = cv2.imread(str(self.img_dir / meta["file_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.img_dir / meta["file_name"])

        h, w = img.shape
        ps = self.patch_size
        boxes = self._boxes(image_id)
        centers = self._centers(image_id)
        x0, y0 = self._sample_window(boxes, h, w)

        crop = img[y0 : y0 + ps, x0 : x0 + ps]
        if crop.shape[0] < ps or crop.shape[1] < ps:
            crop = cv2.copyMakeBorder(
                crop, 0, ps - crop.shape[0], 0, ps - crop.shape[1], cv2.BORDER_REFLECT
            )

        # Shift boxes into crop space and clip to the crop.
        local: list[np.ndarray] = []
        local_centers: list[np.ndarray] = []
        for box, ctr in zip(boxes, centers):
            x, y, bw, bh = box
            nx0, ny0 = max(x, x0), max(y, y0)
            nx1, ny1 = min(x + bw, x0 + ps), min(y + bh, y0 + ps)
            if nx1 - nx0 < 2 or ny1 - ny0 < 2:
                continue
            local.append(np.array([nx0 - x0, ny0 - y0, nx1 - nx0, ny1 - ny0], np.float32))
            local_centers.append(np.array([ctr[0] - x0, ctr[1] - y0], np.float32))

        if self.augment:
            crop, local, local_centers = self._augment(crop, local, local_centers)

        x = (crop.astype(np.float32) / 255.0 - self.mean) / self.std
        target = encode_targets(local, ps, ps, STRIDE, centers=local_centers, with_size=self.with_size)

        out = {
            "image": torch.from_numpy(x).unsqueeze(0),
            "heatmap": target["heatmap"],
            "offset": target["offset"],
            "mask": target["mask"],
            "n_boxes": len(local),
            "image_id": image_id,
        }
        if "size" in target:
            out["size"] = target["size"]
        return out

    @staticmethod
    def _augment(
        img: np.ndarray, boxes: list[np.ndarray], centers: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        h, w = img.shape
        if random.random() < 0.5:
            img = img[:, ::-1]
            boxes = [np.array([w - (b[0] + b[2]), b[1], b[2], b[3]], np.float32) for b in boxes]
            centers = [np.array([w - c[0], c[1]], np.float32) for c in centers]
        if random.random() < 0.5:
            img = img[::-1]
            boxes = [np.array([b[0], h - (b[1] + b[3]), b[2], b[3]], np.float32) for b in boxes]
            centers = [np.array([c[0], h - c[1]], np.float32) for c in centers]
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k)
            for _ in range(k):
                boxes = [np.array([b[1], w - (b[0] + b[2]), b[3], b[2]], np.float32) for b in boxes]
                centers = [np.array([c[1], w - c[0]], np.float32) for c in centers]
                h, w = w, h
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), boxes, centers


def collate_detection(batch: list[dict]) -> dict[str, Any]:
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "offset": torch.stack([b["offset"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "n_boxes": [b["n_boxes"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }
    if "size" in batch[0]:
        out["size"] = torch.stack([b["size"] for b in batch])
    return out


def nms(boxes: np.ndarray, iou_thr: float = 0.3) -> np.ndarray:
    """Greedy non-maximum suppression on ``[x, y, w, h]`` boxes."""
    if len(boxes) == 0:
        return boxes
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = np.argsort(-areas)

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou < iou_thr]
    return boxes[keep]


def decode_points(
    pred: dict[str, torch.Tensor],
    topk: int = 200,
    threshold: float = 0.25,
    stride: int = STRIDE,
    nms_dist: float = 40.0,
) -> list[np.ndarray]:
    """Decode spine-mode output into ``(N, 2)`` centre points in input pixels.

    Suppression is by Euclidean distance rather than box IoU, since there are no
    boxes. ``nms_dist`` is in input pixels and is the single most important
    decode knob: at 12 px the detector emits many peaks per filament and F1 is
    0.31; at 40 px it is 0.44. Filaments are long, so peaks along one spine are
    far apart and need a generous suppression radius.
    """
    hm = torch.sigmoid(pred["heatmap"])
    hm = F_maxpool3x3(hm)
    b, _, hh, hw = hm.shape
    out: list[np.ndarray] = []

    for i in range(b):
        scores = hm[i, 0].flatten()
        k = min(topk, scores.numel())
        top_scores, top_idx = scores.topk(k)
        keep = top_scores > threshold
        top_scores, top_idx = top_scores[keep], top_idx[keep]

        ys = (top_idx // hw).float()
        xs = (top_idx % hw).float()
        off = pred["offset"][i].reshape(2, -1)[:, top_idx]
        cx = ((xs + off[0]) * stride).cpu().numpy()
        cy = ((ys + off[1]) * stride).cpu().numpy()
        pts = np.stack([cx, cy], axis=1)

        if nms_dist > 0 and len(pts) > 1:
            order = np.argsort(-top_scores.cpu().numpy())
            pts = pts[order]
            kept: list[np.ndarray] = []
            for p in pts:
                if all(np.hypot(*(p - q)) >= nms_dist for q in kept):
                    kept.append(p)
            pts = np.array(kept, np.float32) if kept else np.zeros((0, 2), np.float32)
        out.append(pts)
    return out


def decode_boxes(
    pred: dict[str, torch.Tensor],
    topk: int = 200,
    threshold: float = 0.25,
    stride: int = STRIDE,
    nms_thr: float = 0.3,
) -> list[np.ndarray]:
    """Turn CenterNet outputs into ``[x, y, w, h]`` boxes in input pixels.

    NMS is required. The 3x3 max-pool only removes duplicates within one cell;
    when the heatmap is diffuse (as it is for thin, low-contrast filaments) a
    single filament produces many peaks across neighbouring cells. Measured on
    validation, adding NMS took precision from 0.034 to 0.362 and F1 from 0.059
    to 0.215 -- a 4x improvement from this one change.
    """
    hm = torch.sigmoid(pred["heatmap"])
    hm = F_maxpool3x3(hm)
    b, _, hh, hw = hm.shape
    out: list[np.ndarray] = []

    for i in range(b):
        scores = hm[i, 0].flatten()
        k = min(topk, scores.numel())
        top_scores, top_idx = scores.topk(k)
        keep = top_scores > threshold
        top_scores, top_idx = top_scores[keep], top_idx[keep]

        ys = (top_idx // hw).float()
        xs = (top_idx % hw).float()
        off = pred["offset"][i].reshape(2, -1)[:, top_idx]
        size = pred["size"][i].reshape(2, -1)[:, top_idx]

        cx = (xs + off[0]) * stride
        cy = (ys + off[1]) * stride
        bw = torch.exp(size[0])
        bh = torch.exp(size[1])

        boxes = torch.stack([cx - bw / 2, cy - bh / 2, bw, bh], dim=1).cpu().numpy()
        if nms_thr > 0:
            boxes = nms(boxes, nms_thr)
        out.append(boxes)
    return out


def F_maxpool3x3(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
