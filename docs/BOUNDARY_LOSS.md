# Boundary-aware loss: tried, boundary got better, PQ did not

The PQ diagnosis said the errors are boundary precision (median best IoU 0.604,
0% above 0.9). So a loss that targets the contour should help. It does — the
contour metrics improve exactly as designed. PQ goes nowhere, because the gain is
spent on extra false positives.

## What was built

`src/solar/losses.py` gained two options, selected by `train.loss`:

* **`boundary`** — `(1-w)·(BCE+Dice) + w·Dice_band`, where `Dice_band` is Dice
  restricted to a band around the GT contour.
* **`boundary_weighted`** — BCE+Dice with per-pixel weight `1 + α·band`, keeping
  the global pressure while emphasising the contour.

Both use `max_pool2d` morphology (dilate − erode = contour, then dilate by
`radius`), so they are cheap, differentiable, and AMP-safe.

Band coverage at `radius=3`, by filament width:

| width | band covers (% GT) | band covers (% image) |
|---|---|---|
| 6 px | 100% | 3.8% |
| 16 px | 52% | 4.5% |
| 30 px | 30% | 4.9% |
| 60 px | 17% | 5.6% |

For a ~30 px filament the loss concentrates on 30% of the positive pixels — which
is the point: contour disagreement is no longer diluted by interior pixels whose
label is not in doubt.

## Result: the mechanism works

Matched config (`configs/boundary.json` vs the baseline: resnet34, 512² crops,
25 epochs, identical split and postprocessing). Both cached and evaluated at
matched epochs.

| | baseline | boundary | Δ |
|---|---|---|---|
| best PQ (thr 0.55) | 0.3619 | 0.3642 | +0.002 |
| **median best IoU** | 0.604 | **0.621** | **+0.017** |
| **GT instances with IoU > 0.5** | 63.6% | **69.4%** | **+5.8 pts** |
| GT instances with IoU > 0.7 | 29.6% | **31.8%** | **+2.2 pts** |
| SQ (thr 0.75) | 0.6846 | 0.6801 | −0.005 |
| **predicted / GT instance count** | 464/417 = **1.11** | 574/417 = **1.38** | **+24%** |

The boundary loss did precisely what it was asked to do: **5.8 percentage points
more GT instances clear the IoU 0.5 matching bar**, and the median IoU moved up.
That is the metric it optimises, and it moved.

But PQ is flat, because the model now emits **574 instances where there are 417**.
At the tuned threshold it finds more true positives than the baseline (288 vs 264)
*and* more false positives (286 vs 200). The extra detections cost as much as the
better boundaries gain.

## Why

Optimising a band around the contour makes the model more willing to commit to a
weak, thin, locally-contrasty region — it improves the delineation of anything it
has already decided is a filament, but it also lowers the bar for deciding that a
few faint pixels are a filament. More instances, better shaped, more of them
wrong.

The plain Dice term was acting as an implicit *count* regulariser: it rewards
matching total positive area, which suppresses speculative detections. Shifting
weight onto a band weakens that pressure.

## What this rules out

The boundary hypothesis was the best-supported one from the diagnosis and it is now
tested. Its specific claim — that better contours convert near-miss instances into
matches — **is confirmed** (69.4% vs 63.6% above IoU 0.5). What it did not do is
improve PQ, because the same change trades recall for precision at roughly 1:1.

So the remaining gap is **not** purely a boundary problem. Fixing contours is
necessary but not sufficient; any such fix has to come with instance-count control,
or it cannibalises itself.

## Options this suggests

1. **Pair it with a count/precision term.** The boundary gain is real but leaks
   into FPs. A loss that penalises predicted instance count against an estimate
   (or a stronger `min_area`, or the proximity merge that already gave +0.011)
   could harvest the +5.8 pts instead of paying for it. This is the natural next
   attempt and is cheap to test.
2. **Lower `boundary_weight`.** 0.5 was a guess. 0.2–0.3 may keep the contour
   benefit with less of the precision loss. One run to find out.
3. **Keep the `boundary_weighted` variant** (weighted BCE+Dice rather than a
   separate band Dice) — it preserves the area-matching pressure that the pure
   band term sacrifices, so it may not inflate the count.

None of these are likely to be large. The larger levers remain resolution
(1024² tiles, currently only 12 epochs) and the limb normalisation.

## Reproducing

```bash
python -m solar.train --config configs/boundary.json
```

Config: `configs/boundary.json` (`loss: boundary`, `boundary_weight: 0.5`,
`boundary_radius: 3`). Checkpoint: `runs/boundary_r34_512/best.pt`.

Note: `PQAccumulator` uses the matching IoU threshold from `eval.threshold`
(0.55 in `validate()`), while `scripts/pq_decompose.py` defaults to 0.5. Numbers
from the two are not directly comparable — the tables above hold the threshold
fixed within each comparison.
