"""Frozen-backbone token embeddings for representation analysis.

Two backbones are supported, and they differ in ways that matter for this task:

* ``vit_small_patch14_dinov2`` -- 14 px tokens, 384 dims, CLS prefix token,
  NCHW, accepts ``dynamic_img_size`` so the whole 2048x2048 image goes through
  in one shot. 2048 is not a multiple of 14, so the image is reflect-padded to
  2058 (147 x 14).
* ``samvit_base_patch16`` -- 16 px tokens, 768 dims, no prefix token, NHWC
  internally, and it does *not* accept ``dynamic_img_size``. Global attention
  over 16,384 tokens needs ~50 GB, so the image is processed as four
  1024x1024 tiles (3.1 GB each). 2048 is an exact multiple of 16, so tiling
  loses nothing at the seams.

Both adapters return the same thing: ``(tokens, coords)`` where ``tokens`` is
``(N, C)`` float32 and ``coords`` is ``(N, 2)`` float32 giving each token's
centre in *original image pixel* coordinates. Keeping the coordinate mapping in
the adapter is what lets the metrics code stay backbone-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Embedding:
    """Tokens for one image plus the geometry needed to map them back."""

    tokens: np.ndarray  # (N, C) float32
    coords: np.ndarray  # (N, 2) float32, (y, x) in original pixels
    patch: int  # token stride in pixels
    grid: tuple[int, int]  # (rows, cols) of the token lattice

    def __len__(self) -> int:
        return len(self.tokens)


def load_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _to_rgb_norm(gray: np.ndarray) -> np.ndarray:
    """HxW uint8 -> HxWx3 float32, ImageNet-normalised."""
    rgb = np.repeat(gray[:, :, None].astype(np.float32) / 255.0, 3, axis=2)
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


class DinoAdapter:
    """Whole-image DINOv2. Pads to a multiple of the patch size."""

    name = "dinov2_vits14"

    def __init__(self, model_name: str = "vit_small_patch14_dinov2", device=None):
        import timm

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            timm.create_model(
                model_name, pretrained=True, num_classes=0, dynamic_img_size=True
            )
            .to(self.device)
            .eval()
        )
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.dim = int(self.model.embed_dim)
        self.n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))

    def _pad(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Symmetric padding so both sides are multiples of the patch size."""
        ph = (-h) % self.patch
        pw = (-w) % self.patch
        return ph // 2, ph - ph // 2, pw // 2, pw - pw // 2

    @torch.no_grad()
    def embed(self, gray: np.ndarray) -> Embedding:
        h, w = gray.shape
        top, bottom, left, right = self._pad(h, w)
        padded = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_REFLECT)
        x = torch.from_numpy(_to_rgb_norm(padded).transpose(2, 0, 1)).unsqueeze(0)

        feats = self.model.forward_features(x.to(self.device))
        tokens = feats[0, self.n_prefix :].float().cpu().numpy()

        rows = padded.shape[0] // self.patch
        cols = padded.shape[1] // self.patch
        assert tokens.shape[0] == rows * cols, (tokens.shape, rows, cols)

        # Token (i, j) centre in padded coords, shifted back by the pad.
        ys = (np.arange(rows) * self.patch + self.patch // 2 - top).astype(np.float32)
        xs = (np.arange(cols) * self.patch + self.patch // 2 - left).astype(np.float32)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        coords = np.stack([gy.ravel(), gx.ravel()], axis=1)
        return Embedding(tokens, coords, self.patch, (rows, cols))


class SamAdapter:
    """Tiled SAM ViT encoder. ``forward_features`` returns a neck map, so the
    transformer blocks are driven directly to get raw tokens."""

    name = "sam_vitb16"

    def __init__(
        self,
        model_name: str = "samvit_base_patch16",
        tile: int = 1024,
        device=None,
    ):
        import timm

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            timm.create_model(model_name, pretrained=True, num_classes=0)
            .to(self.device)
            .eval()
        )
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.dim = int(self.model.embed_dim)
        self.tile = tile

    @torch.no_grad()
    def _tile_tokens(self, tile_rgb: np.ndarray) -> np.ndarray:
        """(tile, tile, 3) float32 -> (n, n, C) token grid."""
        import timm

        m = self.model
        x = torch.from_numpy(tile_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t = m.patch_embed(x)  # NHWC
        if t.shape[1:3] == m.pos_embed.shape[1:3]:
            t = t + m.pos_embed
        else:
            t = t + timm.layers.resample_abs_pos_embed_nhwc(m.pos_embed, t.shape[1:3])
        t = m.norm_pre(m.patch_drop(m.pos_drop(t)))
        t = m.blocks(t)
        return t[0].float().cpu().numpy()

    def embed(self, gray: np.ndarray) -> Embedding:
        h, w = gray.shape
        rgb = _to_rgb_norm(gray)
        n = self.tile // self.patch

        all_tokens, all_coords = [], []
        for y0 in range(0, h, self.tile):
            for x0 in range(0, w, self.tile):
                y1, x1 = min(y0 + self.tile, h), min(x0 + self.tile, w)
                crop = rgb[y0:y1, x0:x1]
                # Pad partial edge tiles up to a full tile.
                if crop.shape[0] != self.tile or crop.shape[1] != self.tile:
                    crop = cv2.copyMakeBorder(
                        crop,
                        0,
                        self.tile - crop.shape[0],
                        0,
                        self.tile - crop.shape[1],
                        cv2.BORDER_REFLECT,
                    )
                grid = self._tile_tokens(crop)  # (n, n, C)
                all_tokens.append(grid.reshape(-1, grid.shape[-1]))

                ys = (y0 + np.arange(n) * self.patch + self.patch // 2).astype(np.float32)
                xs = (x0 + np.arange(n) * self.patch + self.patch // 2).astype(np.float32)
                gy, gx = np.meshgrid(ys, xs, indexing="ij")
                all_coords.append(np.stack([gy.ravel(), gx.ravel()], axis=1))

        tokens = np.concatenate(all_tokens, axis=0)
        coords = np.concatenate(all_coords, axis=0)
        rows = len(range(0, h, self.tile)) * n
        cols = len(range(0, w, self.tile)) * n
        return Embedding(tokens, coords, self.patch, (rows, cols))


ADAPTERS = {"dinov2": DinoAdapter, "sam": SamAdapter}


def build_adapter(kind: str, **kwargs):
    if kind not in ADAPTERS:
        raise ValueError(f"unknown backbone {kind!r}; choose from {sorted(ADAPTERS)}")
    return ADAPTERS[kind](**kwargs)


# --------------------------------------------------------------------------- #
# Labels and metrics
# --------------------------------------------------------------------------- #


def rasterize(anns: list[dict], height: int, width: int) -> np.ndarray:
    """Union of all filament polygons as a uint8 mask."""
    mask = np.zeros((height, width), np.uint8)
    for ann in anns:
        for poly in ann["segmentation"]:
            pts = np.round(np.asarray(poly, np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def token_labels(emb: Embedding, mask: np.ndarray) -> np.ndarray:
    """1 if the token's centre pixel falls inside the filament mask."""
    h, w = mask.shape
    ys = np.clip(np.round(emb.coords[:, 0]).astype(int), 0, h - 1)
    xs = np.clip(np.round(emb.coords[:, 1]).astype(int), 0, w - 1)
    return mask[ys, xs].astype(np.int8)


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def linear_auc(tokens: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0):
    """Cross-validated AUC of a linear probe. Returns (mean, std)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_pos = int(labels.sum())
    if n_pos < folds or n_pos == len(labels):
        return float("nan"), float("nan")

    X = _l2norm(tokens.astype(np.float32))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, labels):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), labels[tr])
        p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        scores.append(roc_auc_score(labels[te], p))
    return float(np.mean(scores)), float(np.std(scores))


def knn_filament_agreement(tokens: np.ndarray, labels: np.ndarray, k: int = 1):
    """For each filament token, does its nearest neighbour also lie on a filament?

    Reported separately for all tokens (dominated by background) and for
    filament tokens only (the number that actually matters). Chance level for
    the filament-only figure is the filament prevalence.
    """
    from sklearn.neighbors import NearestNeighbors

    X = _l2norm(tokens.astype(np.float32))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    _, idx = nn.kneighbors(X)
    neighbour_labels = labels[idx[:, 1:]]  # drop self

    overall = float((neighbour_labels == labels[:, None]).mean())
    fg = labels == 1
    if fg.sum() == 0:
        return overall, float("nan")
    fg_agree = float((neighbour_labels[fg] == 1).mean())
    return overall, fg_agree


def token_coverage(emb: Embedding, labels: np.ndarray, mask: np.ndarray):
    """How much of the filament area is covered by filament-labelled tokens.

    Each token is treated as a ``patch x patch`` square. Recall is the fraction
    of true filament pixels inside a positive token; precision is the fraction
    of positive-token area that is truly filament. This is the ceiling on what
    any downstream head can recover at this token resolution.
    """
    h, w = mask.shape
    pred = np.zeros((h, w), np.uint8)
    half = emb.patch // 2
    for (cy, cx), lab in zip(emb.coords, labels):
        if lab:
            y0, y1 = int(cy) - half, int(cy) + half
            x0, x1 = int(cx) - half, int(cx) + half
            pred[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 1

    inter = int((pred & mask).sum())
    recall = inter / max(1, int(mask.sum()))
    precision = inter / max(1, int(pred.sum()))
    return float(recall), float(precision)
