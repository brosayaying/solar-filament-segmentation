"""Solar-disk preprocessing: radial flattening and CLAHE.

Motivated by `docs/LITERATURE_REVIEW.md` §2. EdgeAttNet (arXiv 2509.02964
§III-A), trained on MAGFiLO — the same dataset as this competition — preprocesses
every image with:

1. normalise to [0, 1]
2. locate the disk (Hough circle) and mask off-disk to zero
3. **radial flattening** — annular background estimate, divided out
4. Gaussian blur, 3x3, sigma 0.7
5. **CLAHE** inside the disk mask

Steps 3 and 5 are implemented here. They address a defect this repo measured but
never fixed (`docs/DISK_MASKING.md`):

    r/R 0.4-0.6   recall 0.730
    r/R 0.8-1.0   recall 0.281   <- 2.6x collapse, 11% of all filament area

The cause is limb darkening. Filaments are dark-on-bright everywhere except the
outer ring, where the background drops to 78.4 and the contrast *inverts*
(+12 -> -24). The model learns "dark on bright" and the outer ring is
"bright on dark".

Radial flattening removes the cause: estimate the mean intensity in concentric
annuli, smooth that 1-D profile, interpolate it back to a 2-D background, and
divide. After that the filament/background contrast has the same sign everywhere.

Why this is worth doing rather than assuming: it is a *preprocessing* change, so
it can be measured against existing checkpoints without retraining the model
class. See `scripts/preprocess_ablation.py`.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "disk_geometry",
    "radial_background",
    "radial_flatten",
    "apply_clahe",
    "preprocess",
]


def radial_coord(
    h: int, w: int, cx: float, cy: float, r: float
) -> np.ndarray:
    """Normalised radius ``r/R`` for every pixel, in [0, ~1.4].

    Split out because both the background estimate and the flattening need it,
    and building it costs a full-image ``mgrid`` + ``sqrt`` (~40 ms at 2048^2).
    """
    yy, xx = np.mgrid[0:h, 0:w]
    return (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0)).astype(np.float32)


def disk_geometry(img: np.ndarray, thr: int = 25) -> tuple[float, float, float]:
    """Centre and equivalent radius of the solar disk.

    Uses threshold + largest connected component rather than a Hough transform.
    EdgeAttNet uses Hough; on these GONG images the disk is the only large bright
    blob, so the cheaper method is equivalent and has no tuning parameters. The
    equivalent radius (from area) is used rather than the bounding-box half-width
    because the disk is occasionally clipped by the frame edge.

    Returns
    -------
    (cx, cy, r) in pixels.
    """
    m = (img > thr).astype(np.uint8)
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        h, w = img.shape
        return w / 2.0, h / 2.0, min(h, w) / 2.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[k]
    r = float(np.sqrt(max(1.0, float(stats[k, cv2.CC_STAT_AREA])) / np.pi))
    return float(cx), float(cy), r


def radial_background(
    img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    n_bins: int = 64,
    smooth: int = 9,
) -> np.ndarray:
    """Estimate the radial intensity profile as a full-size 2-D background.

    Parameters
    ----------
    n_bins:
        Number of concentric annuli. 64 over a ~1000 px radius gives ~16 px
        annuli, coarse enough to average out filaments (which are 8-22 px wide)
        and fine enough to follow limb darkening.
    smooth:
        Width of the 1-D smoothing kernel over the profile, in bins. Must be odd.
        Without this the profile inherits filament structure from whichever
        annulus happens to contain more of them.

    Returns
    -------
    Float32 array, same shape as ``img``, the estimated background. Off-disk
    pixels are filled with the outermost valid bin so the division is safe.
    """
    h, w = img.shape
    rr = radial_coord(h, w, cx, cy, r)

    # Bin index per pixel, clipped to the disk.
    idx = np.clip((rr * n_bins).astype(np.int32), 0, n_bins - 1)
    inside = rr <= 1.0

    # Median per annulus: robust to filaments occupying part of a bin.
    #
    # The profile is a smooth 1-D curve with 64 bins, so it does not need every
    # pixel: a stride-4 subsample still leaves ~4k samples per annulus, and the
    # median of 4k samples is indistinguishable from the median of 65k. This
    # matters because the naive version does 64 full-image boolean masks per
    # image, which is ~0.5 s at 2048^2 and dominates the data loader.
    stride = max(1, int(np.sqrt(h * w / 262144)))  # target ~512^2 samples
    sl = (slice(None, None, stride), slice(None, None, stride))
    idx_s = idx[sl].ravel()
    inside_s = inside[sl].ravel()
    vals_s = img[sl].ravel()

    prof = np.full(n_bins, np.nan, np.float32)
    keep = inside_s
    if keep.any():
        # One sort instead of 64 masks: order by bin, then take per-bin medians.
        order = np.argsort(idx_s[keep], kind="stable")
        b_sorted = idx_s[keep][order]
        v_sorted = vals_s[keep][order]
        edges = np.searchsorted(b_sorted, np.arange(n_bins + 1))
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if hi > lo:
                prof[b] = float(np.median(v_sorted[lo:hi]))

    # Fill empty bins (possible near the very centre or beyond the disk edge).
    valid = ~np.isnan(prof)
    if not valid.any():
        return np.full_like(img, float(np.median(img)), dtype=np.float32)
    bins = np.arange(n_bins)
    prof = np.interp(bins, bins[valid], prof[valid]).astype(np.float32)

    # Smooth the profile so filament structure does not leak into the background.
    if smooth > 1:
        k = smooth if smooth % 2 == 1 else smooth + 1
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, k), 0).ravel()

    # Interpolate back to 2-D. Off-disk pixels get the outermost bin value.
    bg = prof[np.clip(idx, 0, n_bins - 1)]
    return bg.astype(np.float32)


def radial_flatten(
    img: np.ndarray,
    cx: float | None = None,
    cy: float | None = None,
    r: float | None = None,
    n_bins: int = 64,
    smooth: int = 9,
    eps: float = 1.0,
    clip: float | None = 2.0,
) -> np.ndarray:
    """Divide out the radial intensity profile so contrast is flat across the disk.

    Returns a float32 image with the same scale as the input (0-255). Off-disk
    pixels are set to 0, matching EdgeAttNet's convention.

    ``eps`` guards against division by near-zero in the darkest annuli. It is in
    the same units as the input, so 1.0 means "never divide by less than 1 DN".

    ``clip`` caps the output at ``clip`` x the median background. These images
    carry a thin *bright* chromospheric ring at r/R ~ 0.90-0.94 (p99 = 218 DN
    against a local background of 55). Dividing by the background amplifies it
    ~4x, to +7 sigma after standardisation, where it would dominate the gradient
    and swamp the faint filaments in the same annulus. Filaments are dark, so
    clipping only the upper tail leaves them untouched. ``None`` disables.
    """
    if cx is None or cy is None or r is None:
        cx, cy, r = disk_geometry(img)

    f = img.astype(np.float32)
    bg = radial_background(f, cx, cy, r, n_bins=n_bins, smooth=smooth)
    med = float(np.median(bg))
    out = f / np.maximum(bg, eps) * med
    if clip is not None:
        out = np.minimum(out, clip * med)

    # Off-disk -> 0, as EdgeAttNet does.
    h, w = img.shape
    out[radial_coord(h, w, cx, cy, r) > 1.0] = 0.0
    return out.astype(np.float32)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile: int = 8,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """CLAHE, applied only inside ``mask`` (the disk).

    ``clip_limit=2.0`` and ``tile=8`` are OpenCV's defaults and are what
    EdgeAttNet-style pipelines use. Higher clip limits amplify noise; the point
    here is to lift faint filaments in low-intensity regions without blowing out
    the bright disk centre.

    Pixels outside ``mask`` are left untouched, so this composes with
    :func:`radial_flatten` (which already zeroes them).
    """
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    eq = clahe.apply(u8)
    if mask is None:
        return eq.astype(np.float32)
    out = u8.copy()
    out[mask] = eq[mask]
    return out.astype(np.float32)


def preprocess(
    img: np.ndarray,
    flatten: bool = True,
    clahe: bool = True,
    blur: bool = True,
    n_bins: int = 64,
    smooth: int = 9,
    clip_limit: float = 2.0,
    tile: int = 8,
    blur_sigma: float = 0.7,
) -> np.ndarray:
    """Full pipeline: flatten -> blur -> CLAHE, all inside the disk mask.

    Order matters. Flattening must come first, because CLAHE's local histogram
    equalisation would otherwise be dominated by the limb-darkening gradient and
    would amplify it. Blurring before CLAHE suppresses the high-frequency noise
    that CLAHE would otherwise boost.

    Returns float32 in the input's scale (0-255), off-disk zeroed.
    """
    cx, cy, r = disk_geometry(img)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0)) <= 1.0

    out = img.astype(np.float32)
    if flatten:
        out = radial_flatten(out, cx, cy, r, n_bins=n_bins, smooth=smooth)
    if blur and blur_sigma > 0:
        out = cv2.GaussianBlur(out, (3, 3), blur_sigma)
    if clahe:
        out = apply_clahe(out, clip_limit=clip_limit, tile=tile, mask=disk)
    return out.astype(np.float32)
