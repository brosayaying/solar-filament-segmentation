# Solar disk masking

Question: should we exclude the sun disk and only consider the inside region, and
are the current experiments doing that?

Short answer: **no, the main pipeline does no disk masking — but it does not need
to for the off-disk case.** Both models already predict 0% off-disk, and the
ground truth is 0% off-disk. The real problem is not the black surround, it is
the **limb**: recall collapses from 0.73 to 0.28 in the outer 20% of the disk
radius, because limb darkening inverts the filament/background contrast there.

## What `scripts/patch_clf.py` does

It masks the disk and erodes by a margin before using patches:

```python
def disk_mask(img, thr=25):
    """Binary mask of the solar disk (largest bright blob, holes filled)."""
    m = (img > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    ...
    inner = cv2.erode(disk, k)          # k = ellipse(2*margin+1), margin=40
    valid = inner[::PATCH, ::PATCH][:GRID, :GRID] > 0
```

So `patch_clf` excludes both the off-disk surround *and* a 40 px rim band.

## What the main pipeline does

Nothing. `src/solar/data.py`, `models.py`, `train.py`, `detector*.py` contain no
disk logic at all — the only `cv2.circle` call is for rasterising spine
thickness. The full 2048² image, including the black surround, goes into the
model.

## Does that cost us anything? Measured

### Off-disk: no

| quantity | value |
|---|---|
| disk covers | 73.1% of image (67.7–79.9%) |
| off-disk surround | 26.9% of pixels |
| **GT filament pixels off-disk** | **0.00%** |
| **segmentation model predicted area off-disk** | **0.00%** |
| **detector points off-disk** | **0.0%** (0/49) |

The annotations are already strictly inside the disk, and both models learned to
ignore the black surround on their own. Applying `disk_mask` to the baseline's
probability map changed PQ by **exactly zero** (0.3445 → 0.3445). So masking the
surround is a no-op for scoring — it is not where the loss is.

### The limb: yes, this is the real problem

Recall of GT filament pixels, binned by normalised radius `r/R` from the disk
centre (baseline resnet34, 24 val images):

| r/R | recall | GT pixels |
|---|---|---|
| 0.0–0.2 | 0.483 | 9,963 |
| 0.2–0.4 | 0.597 | 69,345 |
| 0.4–0.6 | **0.730** | 114,323 |
| 0.6–0.8 | 0.642 | 186,962 |
| 0.8–1.0 | **0.281** | 58,662 |

Recall peaks mid-disk and **collapses to 0.28 in the outer ring** — a 2.6× drop.
The outer ring holds 58,662 GT pixels, ~11% of all filament area, so this is not
a rounding error.

### Why: limb darkening inverts the contrast

Mean intensity of filament vs on-disk background, by radius:

| r/R | filament | background | contrast |
|---|---|---|---|
| 0.0–0.2 | 127.0 | 139.3 | +12.3 |
| 0.2–0.4 | 126.9 | 138.9 | +12.1 |
| 0.4–0.6 | 119.6 | 135.9 | +16.3 |
| 0.6–0.8 | 114.0 | 125.4 | +11.4 |
| 0.8–1.0 | 102.5 | 78.4 | **−24.0** |

Filaments are *dark on bright* everywhere except the outer ring, where the
background drops to 78.4 (limb darkening) and the sign flips. The model is
looking for dark-on-bright and the outer ring is bright-on-dark.

Part of that outer-ring number is mask contamination — `disk_mask` with
`thr=25` includes some of the darkening limb itself as "background". Eroding the
mask fixes the sign:

| erode | contrast at r/R 0.8–1.0 |
|---|---|
| 0 px | −22.1 |
| 40 px | −10.1 |
| 80 px | −2.4 |
| 120 px | **+5.9** |

So the true limb band is roughly the outer 80–120 px of the disk, and inside it
the signal genuinely weakens and eventually inverts.

## What this means

1. **Do not bother masking the off-disk surround.** It is free — both models
   already ignore it, and masking changes PQ by 0.0. Adding it would be code
   with no measured benefit.

2. **Do handle the limb.** This is where 11% of filament area sits at 0.28
   recall. Options, cheapest first:
   - **Radial normalisation**: divide by a smoothed radial intensity profile
     (or subtract it) so contrast is flat across the disk. This directly
     removes the cause.
   - **Limb-aware loss weighting**: up-weight the outer ring so the model does
     not learn to ignore it.
   - **Exclude the outer ~100 px** if the competition's GT never puts filaments
     there — but it does (58k pixels), so this would trade recall for nothing.

3. **`patch_clf`'s 40 px margin is too small** for this effect. The contrast is
   still −10 at 40 px erosion and only reaches positive at ~120 px. If that
   script is used for anything downstream, the margin should be larger or the
   radial normalisation applied instead.

## Reproducing

```bash
python scripts/disk_analysis.py
```

Uses `disk_mask` from `scripts/img_stats.py` (the same implementation as
`patch_clf.py`).
