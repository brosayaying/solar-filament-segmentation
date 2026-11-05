"""Positive-biased, limb-aware crop dataset for DeepCrack finetuning.

The zero-shot probe failed in a specific way: the network predicted the solar
limb and essentially nothing else (FP 921, TP 0). Two properties of uniform
random cropping make that failure hard to fix by finetuning:

* **Filaments are 2.6% of pixels.** A uniform 256 crop is usually empty
  background, so almost every gradient step teaches "nothing here".
* **The limb is never a labelled example.** If crops are drawn from the disk
  interior, the limb is neither a positive nor a negative. The model is free to
  keep firing on it, and the FP failure simply moves to inference time.

This dataset fixes both by mixing three sampling modes:

``positive``
    Centre on a random point along a random annotation polygon edge. Filaments
    are thin, so an edge point is on the filament. Guarantees the crop contains
    a target.
``limb``
    Centre on the disk boundary, at a random angle and a radius jittered by
    ``limb_width``. The limb is *not* annotated, so it lands in the mask as
    background -- exactly the negative example the probe was missing.
``uniform``
    Plain random crop, so the model still sees the overall background
    distribution and does not overfit to filament neighbourhoods.

Disk geometry is cached per image as ``(cx, cy, r)`` rather than as a full
2048x2048 mask: 707 masks would be ~2.9 GB, three floats are 24 bytes.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import FilamentDataset, anns_to_mask, build_ann_index

cv2.setNumThreads(0)


def disk_geometry(img: np.ndarray, thr: int = 25) -> tuple[float, float, float]:
    """Centre and equivalent radius of the solar disk (largest bright blob)."""
    m = (img > thr).astype(np.uint8)
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        h, w = img.shape
        return w / 2.0, h / 2.0, min(h, w) / 2.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[k]
    r = math.sqrt(max(1.0, float(stats[k, cv2.CC_STAT_AREA])) / math.pi)
    return float(cx), float(cy), float(r)


class PositiveCropDataset(Dataset):
    """``patch_size`` crops with positive-biased and limb-aware sampling.

    Parameters
    ----------
    p_positive, p_limb:
        Probabilities of the positive and limb modes. The remainder is uniform.
        They must sum to <= 1.
    limb_width:
        Half-width in px of the annulus around the disk edge that limb crops are
        drawn from. The limb ring is a few px wide; 24 gives the crop enough
        context to see the disk boundary as well as the ring itself.
    """

    def __init__(
        self,
        root: str | Path,
        coco: dict[str, Any],
        image_ids: Iterable[str],
        patch_size: int = 256,
        mean: float = 0.0,
        std: float = 1.0,
        samples_per_image: int = 8,
        p_positive: float = 0.5,
        p_limb: float = 0.2,
        limb_width: int = 24,
        augment: bool = True,
        img_dir: str | Path | None = None,
        soft_labels: bool = False,
    ) -> None:
        if p_positive + p_limb > 1.0 + 1e-9:
            raise ValueError(
                f"p_positive + p_limb must be <= 1, got {p_positive} + {p_limb}"
            )
        self.root = Path(root)
        self.patch_size = int(patch_size)
        self.mean = float(mean)
        self.std = float(std)
        self.samples_per_image = int(samples_per_image)
        self.p_positive = float(p_positive)
        self.p_limb = float(p_limb)
        self.limb_width = int(limb_width)
        self.augment = bool(augment)
        self.soft_labels = bool(soft_labels)

        self.ann_index = build_ann_index(coco)
        self.images = {img["id"]: img for img in coco["images"]}
        self.img_dir = (
            Path(img_dir) if img_dir is not None else self.root / "train/train_images"
        )
        if not self.img_dir.exists():
            raise FileNotFoundError(f"image directory not found: {self.img_dir}")

        # Group every annotator variant of a file together. 296 of the 707 files
        # were annotated independently by 2-3 people, and their instance counts
        # disagree by up to 3x, so which variant you train on materially changes
        # the target.
        self._file_anns: dict[str, list[str]] = {}
        for iid in image_ids:
            fn = self.images[iid]["file_name"]
            self._file_anns.setdefault(fn, []).append(iid)

        if self.soft_labels:
            # One sample stream per *file*, not per annotator variant: the target
            # is the mean over annotators, so the variants are no longer separate
            # examples. Without this dedup a 3-annotator file would be seen 3x.
            self.image_ids = [ids[0] for ids in self._file_anns.values()]
        else:
            self.image_ids = list(image_ids)

        self._geom: dict[str, tuple[float, float, float]] = {}

    def __len__(self) -> int:
        return len(self.image_ids) * self.samples_per_image

    # -- helpers ----------------------------------------------------------- #

    def _load(self, file_name: str) -> np.ndarray:
        img = cv2.imread(str(self.img_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.img_dir / file_name)
        return img

    def _geometry(self, image_id: str, img: np.ndarray) -> tuple[float, float, float]:
        g = self._geom.get(image_id)
        if g is None:
            g = disk_geometry(img)
            self._geom[image_id] = g
        return g

    def _positive_center(self, image_id: str) -> tuple[float, float] | None:
        """Random point on a random polygon edge of a random annotation.

        With soft labels the pool is every annotator's polygons for this file, so
        crops are centred on the union of what anyone marked -- otherwise the
        sampler would only ever visit the regions one particular annotator drew.
        """
        if self.soft_labels:
            pool: list[dict] = []
            for iid in self._file_anns[self.images[image_id]["file_name"]]:
                pool.extend(self.ann_index.get(iid) or [])
        else:
            pool = self.ann_index.get(image_id) or []
        if not pool:
            return None
        polys = random.choice(pool).get("segmentation") or []
        if not polys:
            return None
        pts = np.asarray(random.choice(polys), np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return None
        if len(pts) == 1:
            return float(pts[0, 0]), float(pts[0, 1])
        i = random.randrange(len(pts))
        a, b = pts[i], pts[(i + 1) % len(pts)]
        p = a + random.random() * (b - a)
        return float(p[0]), float(p[1])

    def _soft_mask(self, file_name: str, ps: int, x0: int, y0: int) -> np.ndarray:
        """Mean of every annotator's binary mask for this file, in [0, 1].

        A pixel labelled by 2 of 3 annotators gets 0.667. This is the target the
        official scorer effectively rewards: it evaluates against every annotator
        variant, so a prediction in the consensus region is a TP for all of them,
        while a prediction only one annotator drew is a TP for that one and an FP
        for the others.

        Averaging also removes the "unlabelled filament" problem that makes hard
        negatives unsafe: a region only one annotator marked becomes a soft 0.33
        rather than a hard 0, so the model is not taught to suppress it.
        """
        ids = self._file_anns[file_name]
        acc = np.zeros((ps, ps), np.float32)
        for iid in ids:
            acc += anns_to_mask(self.ann_index.get(iid, []), ps, ps, offset=(x0, y0))
        return acc / len(ids)

    def _limb_center(self, image_id: str, img: np.ndarray) -> tuple[float, float]:
        cx, cy, r = self._geometry(image_id, img)
        ang = random.uniform(0.0, 2.0 * math.pi)
        rr = r + random.uniform(-self.limb_width / 2.0, self.limb_width / 2.0)
        return cx + rr * math.cos(ang), cy + rr * math.sin(ang)

    # -- item -------------------------------------------------------------- #

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx % len(self.image_ids)]
        meta = self.images[image_id]
        img = self._load(meta["file_name"])
        h, w = img.shape
        ps = self.patch_size

        u = random.random()
        center: tuple[float, float] | None = None
        if u < self.p_positive:
            center = self._positive_center(image_id)
        elif u < self.p_positive + self.p_limb:
            center = self._limb_center(image_id, img)
        if center is None:
            center = (random.uniform(0.0, w), random.uniform(0.0, h))

        # Jitter so the target is not always dead-centre; without this the model
        # can learn a positional prior instead of the appearance of a filament.
        cx = center[0] + random.uniform(-ps / 4.0, ps / 4.0)
        cy = center[1] + random.uniform(-ps / 4.0, ps / 4.0)
        x0 = max(0, min(int(round(cx - ps / 2.0)), max(0, w - ps)))
        y0 = max(0, min(int(round(cy - ps / 2.0)), max(0, h - ps)))

        crop = img[y0 : y0 + ps, x0 : x0 + ps]
        if self.soft_labels:
            mask = self._soft_mask(meta["file_name"], ps, x0, y0)
        else:
            mask = anns_to_mask(self.ann_index.get(image_id, []), ps, ps, offset=(x0, y0))

        # Only reachable if an image is smaller than the patch; all MAGFiLO
        # images are 2048x2048, so this is a safety net rather than a hot path.
        if crop.shape != (ps, ps):
            dy, dx = ps - crop.shape[0], ps - crop.shape[1]
            crop = cv2.copyMakeBorder(crop, 0, dy, 0, dx, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, dy, 0, dx, cv2.BORDER_REFLECT)

        if self.augment:
            crop, mask = FilamentDataset._augment(crop, mask)

        x = crop.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return {
            "image": torch.from_numpy(x).unsqueeze(0),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "image_id": image_id,
        }
