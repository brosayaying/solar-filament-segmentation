"""Dataset, COCO loading, and leakage-free splitting.

Key design points
-----------------
* Images are 2048x2048 8-bit grayscale (PIL mode ``L``).
* The annotation JSON has 1154 image entries but only 707 unique files: the same
  observation was annotated independently by several annotators, disambiguated by
  an ``<annotator>-<filename>`` id. Splitting on ``image_id`` therefore leaks the
  same underlying image into train and val.
* Filaments persist for days, so images from the same date (and station) are
  near-duplicates. The default split groups by **date** to keep validation honest.
* Masks are rasterised on the fly with ``cv2.fillPoly``. Polygons are small, so
  this is far cheaper than caching 1154 x 2048^2 masks (~4.8 GB).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import apply_clahe, disk_geometry, radial_flatten

cv2.setNumThreads(0)

# --------------------------------------------------------------------------- #
# COCO helpers
# --------------------------------------------------------------------------- #


def load_coco(path: str | Path) -> dict[str, Any]:
    """Load the MAGFiLO COCO annotation file."""
    with open(path) as fh:
        return json.load(fh)


def build_ann_index(coco: dict[str, Any]) -> dict[str, list[dict]]:
    """Map ``image_id`` -> list of annotations."""
    index: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        index[ann["image_id"]].append(ann)
    return dict(index)


def group_key(file_name: str, group_by: str) -> str:
    """Derive the split group for an image filename.

    Filenames look like ``YYYYMMDDHHMMSS<station>.jpeg``.
    """
    stem = Path(file_name).stem
    if group_by == "date":
        return stem[:8]
    if group_by == "month":
        return stem[:6]
    if group_by == "year":
        return stem[:4]
    if group_by == "station":
        return stem[14:16]
    if group_by == "file_name":
        return stem
    raise ValueError(f"unknown group_by={group_by!r}")


def make_split(
    coco: dict[str, Any],
    group_by: str = "date",
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Group-aware train/val split over ``image_id``.

    All annotator variants of one file land in the same group, and (by default)
    all images from one calendar date land in the same group.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for img in coco["images"]:
        groups[group_key(img["file_name"], group_by)].append(img["id"])

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    total = sum(len(v) for v in groups.values())
    target = int(round(total * val_frac))

    val_keys: list[str] = []
    n_val = 0
    for k in keys:
        if n_val >= target:
            break
        val_keys.append(k)
        n_val += len(groups[k])

    val_set = set(val_keys)
    train_ids = [i for k in keys if k not in val_set for i in groups[k]]
    val_ids = [i for k in val_keys for i in groups[k]]

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "meta": {
            "group_by": group_by,
            "val_frac": val_frac,
            "seed": seed,
            "n_groups": len(keys),
            "n_val_groups": len(val_keys),
            "n_train_images": len(train_ids),
            "n_val_images": len(val_ids),
        },
    }


# --------------------------------------------------------------------------- #
# Rasterisation
# --------------------------------------------------------------------------- #


def _poly_to_rle(polygons: Sequence, height: int, width: int):
    """Encode polygons with pycocotools, the official rasterisation.

    ``cv2.fillPoly`` on rounded integer vertices produces masks that are ~11.5%
    larger than the official ones (median IoU 0.88 against pycocotools). The
    annotation ``area`` field matches pycocotools exactly, and the hidden test GT
    is decoded from RLE, so anything else silently mis-measures PQ by ~0.02.
    """
    from pycocotools import mask as mask_utils

    rles = mask_utils.frPyObjects(list(polygons), height, width)
    return mask_utils.merge(rles) if isinstance(rles, list) else rles


def rasterise(
    anns: Sequence[dict],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
    labelmap: bool = False,
) -> np.ndarray:
    """Rasterise annotations using the official pycocotools polygon decoder.

    ``offset`` crops a window out of the full-size mask, which is how patch
    training gets its targets. Cropping after decoding (rather than translating
    polygon coordinates) keeps the rasterisation identical to the official one.

    Returns a uint8 0/1 mask, or an int32 label map when ``labelmap=True``.
    """
    from pycocotools import mask as mask_utils

    ox, oy = offset
    full_h = height + int(round(oy))
    full_w = width + int(round(ox))
    out = np.zeros((height, width), np.int32 if labelmap else np.uint8)

    for i, ann in enumerate(anns, start=1):
        seg = ann["segmentation"]
        if not seg:
            continue
        rle = _poly_to_rle(seg, full_h, full_w)
        m = mask_utils.decode(rle)
        if m.ndim == 3:  # merged multi-polygon RLE decodes to (H, W, 1)
            m = m[:, :, 0]
        crop = m[int(round(oy)) : int(round(oy)) + height, int(round(ox)) : int(round(ox)) + width]
        if crop.shape != (height, width):
            continue
        out[crop.astype(bool)] = i if labelmap else 1
    return out


def anns_to_mask(
    anns: Sequence[dict],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Rasterise all annotations into a single binary mask (uint8 0/1)."""
    return rasterise(anns, height, width, offset, labelmap=False)


def anns_to_labelmap(
    anns: Sequence[dict],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Rasterise each annotation into its own instance id (int32, 0 = background)."""
    return rasterise(anns, height, width, offset, labelmap=True)


def anns_to_spine_mask(
    anns: Sequence[dict],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
    thickness: int = 3,
) -> np.ndarray:
    """Rasterise filament spines (centerlines) into a binary mask.

    The spine is the filament's medial axis. It is not used by the PQ metric, but
    predicting it forces the network to learn *which* pixels belong to the same
    filament, which is exactly the instance-identity signal the semantic head
    lacks. It also gives a clean seed source for instance separation.
    """
    mask = np.zeros((height, width), np.uint8)
    ox, oy = offset
    for ann in anns:
        spine = ann.get("spine")
        if not spine:
            continue
        pts = np.asarray(spine, np.float32).reshape(-1, 2).copy()
        pts[:, 0] -= ox
        pts[:, 1] -= oy
        pts = np.round(pts).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(mask, tuple(pts[0]), max(1, thickness // 2), 1, -1)
        else:
            cv2.polylines(mask, [pts], False, 1, thickness=max(1, thickness))
    return mask


# --------------------------------------------------------------------------- #
# Preprocessing + normalisation (shared by train, val and inference)
# --------------------------------------------------------------------------- #


def parse_preprocess(spec: str | None) -> set[str]:
    """Parse a ``"flatten+blur+clahe"`` spec into a set of step names."""
    if not spec:
        return set()
    steps = {s.strip() for s in spec.replace("+", ",").split(",") if s.strip()}
    unknown = steps - {"flatten", "blur", "clahe"}
    if unknown:
        raise ValueError(f"unknown preprocess steps: {sorted(unknown)}")
    return steps


def disk_mask(img: np.ndarray, geom: tuple[float, float, float] | None = None) -> np.ndarray:
    """Boolean disk mask for a raw image."""
    cx, cy, r = geom if geom is not None else disk_geometry(img)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0)) <= 1.0


def preprocess_image(
    img: np.ndarray, steps: set[str]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply full-resolution preprocessing. Returns ``(image, disk_mask)``.

    Must run at full resolution: radial flattening estimates its background
    profile from concentric annuli spanning the whole disk, so a 512px crop
    cannot supply it.

    ``disk_mask`` is ``None`` when no step needs it, so callers can skip the
    (relatively expensive) connected-component pass.
    """
    if not steps:
        return img, None
    # One geometry pass, shared by the mask and the flattening.
    geom = disk_geometry(img)
    disk = disk_mask(img, geom)
    out = img.astype(np.float32)
    if "flatten" in steps:
        out = radial_flatten(out, *geom)
    if "blur" in steps:
        out = cv2.GaussianBlur(out, (3, 3), 0.7)
    if "clahe" in steps:
        out = apply_clahe(out, mask=disk)
    return out, disk


def normalise_image(
    img: np.ndarray,
    mean: float,
    std: float,
    disk: np.ndarray | None = None,
    mask_offdisk: bool = False,
    offdisk_value: float = -3.0,
) -> np.ndarray:
    """Scale to [0,1] and standardise, optionally replacing off-disk pixels.

    ``mean``/``std`` should be measured over the *disk*, not the whole frame:
    28% of each image is off-disk black, which drags the mean down and inflates
    the variance. See ``scripts/disk_stats.py``.
    """
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    if mask_offdisk and disk is not None:
        x = np.where(disk, x, offdisk_value)
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class FilamentDataset(Dataset):
    """MAGFiLO dataset.

    ``mode="train"`` returns random ``patch_size`` crops with light augmentation.
    ``mode="val"`` returns the full-resolution image plus an instance label map.
    """

    def __init__(
        self,
        root: str | Path,
        coco: dict[str, Any],
        image_ids: Iterable[str],
        mode: str = "train",
        patch_size: int = 512,
        mean: float = 0.5,
        std: float = 0.25,
        augment: bool = True,
        samples_per_image: int = 1,
        img_dir: str | Path | None = None,
        tiled: bool = False,
        spine_thickness: int = 3,
        dedup_files: bool = False,
        preprocess: str | None = None,
        mask_offdisk: bool = False,
        offdisk_value: float = -3.0,
        preprocess_cache: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.mode = mode
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.augment = augment and mode == "train"
        self.samples_per_image = samples_per_image
        self.tiled = tiled and mode == "train"
        self.spine_thickness = spine_thickness

        # Full-resolution preprocessing, applied *before* cropping because
        # radial flattening needs the whole disk to estimate its profile.
        # Comma/plus separated subset of {flatten, blur, clahe}; None disables.
        #
        # This must be applied identically at train, val and test time. Applying
        # it only at eval time (as scripts/preprocess_ablation.py did) is a
        # distribution shift, not a test of preprocessing: a model trained on
        # raw input has already adapted to raw statistics, so feeding it
        # flattened/CLAHE input measures the shift, not the benefit.
        self.preprocess = preprocess
        self._pp_steps = parse_preprocess(preprocess)

        # mask_offdisk=True replaces off-disk pixels with a fixed normalised
        # value. With disk-derived statistics the sky already lands ~3.3 sigma
        # below the disk mean, so this is belt-and-braces; it matters when the
        # normalisation is global, where the sky sits only ~2 sigma out.
        self.mask_offdisk = mask_offdisk
        self.offdisk_value = offdisk_value
        # Cache *geometry*, not the mask. A boolean 2048^2 mask is 4 MB, and
        # caching one per file across 8 dataloader workers reaches ~18 GB of
        # resident memory, which pushed epoch time from 38 s to 298 s through
        # allocation pressure. Three floats per file costs nothing.
        self._geom_cache: dict[str, tuple[float, float, float]] = {}

        # Optional on-disk cache of preprocessed images (see
        # scripts/build_preprocess_cache.py). Flattening costs ~0.5 s per
        # 2048^2 image, which is ~12 h of redundant work over a 160-epoch run.
        self.preprocess_cache = Path(preprocess_cache) if preprocess_cache else None
        if self.preprocess_cache is not None and not self.preprocess_cache.exists():
            raise FileNotFoundError(
                f"preprocess cache not found: {self.preprocess_cache}. Build it with "
                f"scripts/build_preprocess_cache.py --preprocess {preprocess}"
            )

        self.ann_index = build_ann_index(coco)
        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(image_ids)

        # 296 image files were annotated independently by 2-3 annotators, so the
        # COCO file has 1154 image entries for 707 files and 65% of val entries
        # are duplicates of another entry. Averaging over annotator variants
        # makes val PQ a measure of agreement with a particular human rather than
        # correctness, and it is capped by inter-annotator disagreement
        # (~0.36 PQ, see docs/METRIC_CEILING.md). Keeping one entry per file
        # makes the number interpretable and comparable to a single-annotation
        # test set.
        if dedup_files:
            seen: set[str] = set()
            kept: list[str] = []
            for iid in self.image_ids:
                fname = self.images[iid]["file_name"]
                if fname in seen:
                    continue
                seen.add(fname)
                kept.append(iid)
            self.image_ids = kept

        # Deterministic tiling: split each 2048x2048 observation into a fixed
        # grid of patch_size tiles. For 2048/1024 this is exactly 4 quadrants,
        # so every pixel is seen once per epoch and no region is skipped.
        # Unlike random cropping this loses no detail and never truncates a
        # filament at a random offset, but it does remove translation diversity,
        # so augmentation matters more here.
        self.tiles: list[tuple[str, int, int]] = []
        if self.tiled:
            ps = self.patch_size
            for iid in self.image_ids:
                meta = self.images[iid]
                h, w = int(meta["height"]), int(meta["width"])
                ys = list(range(0, max(1, h - ps + 1), ps)) or [0]
                xs = list(range(0, max(1, w - ps + 1), ps)) or [0]
                for y0 in ys:
                    for x0 in xs:
                        # Clamp so the last row/column still yields a full tile.
                        self.tiles.append((iid, min(x0, max(0, w - ps)), min(y0, max(0, h - ps))))

        # Both train and val images live in train_images; only the test set has
        # its own folder, and it carries no annotations.
        if img_dir is not None:
            self.img_dir = Path(img_dir)
        elif mode == "test":
            self.img_dir = self.root / "test/test_images"
        else:
            self.img_dir = self.root / "train/train_images"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"image directory not found: {self.img_dir}")

    def __len__(self) -> int:
        if self.tiled:
            return len(self.tiles)
        return len(self.image_ids) * (self.samples_per_image if self.mode == "train" else 1)

    def _load_image(self, file_name: str) -> np.ndarray:
        path = self.img_dir / file_name
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _geom(self, file_name: str, img: np.ndarray) -> tuple[float, float, float]:
        """Disk geometry for ``file_name``, cached as three floats."""
        g = self._geom_cache.get(file_name)
        if g is None:
            g = disk_geometry(img)
            self._geom_cache[file_name] = g
        return g

    def _cached(self, file_name: str) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Read a preprocessed image from disk, or None if not cached."""
        if self.preprocess_cache is None or not self._pp_steps:
            return None
        stem = Path(file_name).stem
        path = self.preprocess_cache / f"{stem}.npy"
        if not path.exists():
            # Fall through to computing it rather than failing: a partially
            # built cache should not stop a run.
            return None
        out = np.load(path).astype(np.float32)
        disk = None
        if self.mask_offdisk:
            disk_path = self.preprocess_cache / f"{stem}.disk.npy"
            disk = np.load(disk_path) if disk_path.exists() else disk_mask(out)
        return out, disk

    def _prepare(self, img: np.ndarray, file_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply full-resolution preprocessing. Returns (image, disk mask)."""
        if not self._pp_steps and not self.mask_offdisk:
            return img, None

        hit = self._cached(file_name)
        if hit is not None:
            return hit

        if not self._pp_steps:
            # Only masking is requested, so the disk is all we need.
            return img, disk_mask(img, self._geom(file_name, img))
        out, disk = preprocess_image(img, self._pp_steps)
        return out, disk

    def _normalise(self, img: np.ndarray, disk: np.ndarray | None = None) -> torch.Tensor:
        x = normalise_image(
            img,
            self.mean,
            self.std,
            disk=disk,
            mask_offdisk=self.mask_offdisk,
            offdisk_value=self.offdisk_value,
        )
        return torch.from_numpy(x).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.tiled:
            image_id, x0, y0 = self.tiles[idx]
        else:
            image_id, x0, y0 = self.image_ids[idx % len(self.image_ids)], None, None

        meta = self.images[image_id]

        # Try the preprocessed cache first: loading a 2048^2 float16 .npy is
        # ~10 ms against ~60 ms to decode the JPEG plus ~500 ms to flatten it.
        hit = self._cached(meta["file_name"])
        if hit is not None:
            img, disk = hit
        else:
            img, disk = self._prepare(self._load_image(meta["file_name"]), meta["file_name"])
        anns = self.ann_index.get(image_id, [])

        if self.mode == "train":
            return self._get_train(img, anns, image_id, x0, y0, disk)
        return self._get_val(img, anns, image_id, disk)

    # -- train ------------------------------------------------------------- #

    def _get_train(
        self,
        img: np.ndarray,
        anns: list[dict],
        image_id: str,
        fixed_x0: int | None = None,
        fixed_y0: int | None = None,
        disk: np.ndarray | None = None,
    ) -> dict[str, Any]:
        h, w = img.shape
        ps = self.patch_size

        if fixed_x0 is not None and fixed_y0 is not None:
            # Tiled mode: deterministic, non-overlapping window.
            x0, y0 = fixed_x0, fixed_y0
        elif h > ps and w > ps:
            y0 = random.randint(0, h - ps)
            x0 = random.randint(0, w - ps)
        else:
            y0 = x0 = 0
            ps_h, ps_w = min(ps, h), min(ps, w)
            img = cv2.copyMakeBorder(img, 0, ps - ps_h, 0, ps - ps_w, cv2.BORDER_REFLECT)
            h, w = img.shape

        crop = img[y0 : y0 + ps, x0 : x0 + ps]
        crop_disk = disk[y0 : y0 + ps, x0 : x0 + ps] if disk is not None else None
        mask = anns_to_mask(anns, ps, ps, offset=(x0, y0))
        spine = anns_to_spine_mask(anns, ps, ps, offset=(x0, y0), thickness=self.spine_thickness)

        if self.augment:
            crop, mask, spine, crop_disk = self._augment(crop, mask, spine, crop_disk)

        return {
            "image": self._normalise(crop, crop_disk),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "spine": torch.from_numpy(spine).float().unsqueeze(0),
            "image_id": image_id,
        }

    @staticmethod
    def _augment(
        img: np.ndarray,
        mask: np.ndarray,
        spine: np.ndarray | None = None,
        disk: np.ndarray | None = None,
    ) -> tuple:
        # Slicing with a negative step yields negative strides, which torch
        # cannot wrap; every geometric op is followed by ascontiguousarray.
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
            if spine is not None:
                spine = spine[:, ::-1]
            if disk is not None:
                disk = disk[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
            if spine is not None:
                spine = spine[::-1]
            if disk is not None:
                disk = disk[::-1]
        k = random.randint(0, 3)
        if k:
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if spine is not None:
                spine = np.rot90(spine, k)
            if disk is not None:
                disk = np.rot90(disk, k)
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12, 12)
            # Preserve float32 for preprocessed images: flattening produces
            # sub-integer values and quantising them back to uint8 would throw
            # away the precision the preprocessing just recovered.
            dtype = np.float32 if img.dtype == np.float32 else np.uint8
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(dtype)
        # Always return four values so the caller can unpack unconditionally.
        # A variable-length tuple here silently broke when disk was None.
        return (
            np.ascontiguousarray(img),
            np.ascontiguousarray(mask),
            np.ascontiguousarray(spine) if spine is not None else None,
            np.ascontiguousarray(disk) if disk is not None else None,
        )

    # -- val --------------------------------------------------------------- #

    def _get_val(
        self,
        img: np.ndarray,
        anns: list[dict],
        image_id: str,
        disk: np.ndarray | None = None,
    ) -> dict[str, Any]:
        h, w = img.shape
        label = anns_to_labelmap(anns, h, w)
        return {
            "image": self._normalise(img, disk),
            "label": torch.from_numpy(label),
            "image_id": image_id,
            "file_name": self.images[image_id]["file_name"],
        }


def collate_train(batch: list[dict]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    # Not every sampler emits a spine target (PositiveCropDataset does not), so
    # fall back to zeros. The spine loss is only applied when the model has an
    # aux_spine head, and spine_weight defaults to 0.
    if "spine" in batch[0]:
        spines = torch.stack([b["spine"] for b in batch])
    else:
        spines = torch.zeros_like(masks)
    return {
        "image": images,
        "mask": masks,
        "spine": spines,
        "image_id": [b["image_id"] for b in batch],
    }


def collate_val(batch: list[dict]) -> dict[str, Any]:
    # Full-resolution images differ in size only in principle; all are 2048^2.
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": [b["label"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "file_name": [b["file_name"] for b in batch],
    }
