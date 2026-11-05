# Where the PQ is actually going

PQ has been stuck at ~0.40 local / 0.33 public. This is a decomposition of the
error, not an attempt at a fix. It identifies which lever has room.

## Calibration

The public leaderboard top is **0.69**; we are at **0.33**. So there is a lot of
headroom, and the wall is not fundamental.

## The ceiling is not the problem

Using the **ground-truth mask as a binary mask** and running connected components
to split instances:

| min_area | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|
| 128 | **1.0000** | 1.0000 | 1.0000 | 417 | 0 | 0 |
| 512 | 0.9456 | 1.0000 | 0.9456 | 374 | 0 | 43 |

**Connected components separate these instances perfectly.** Every GT filament is
its own connected blob. So instance separation — watershed, NMS, merge heuristics,
the whole two-stage detect-then-separate programme — has *nothing left to win*.
The entire gap is mask quality.

That also retroactively explains the watershed negative result: there was nothing
to split.

## Where the errors are (baseline resnet34, 64 val images)

PQ 0.3958, SQ 0.6839, RQ 0.5787 (tp=261, fp=224, fn=156).

### False negatives: 156

| bucket | count | % FN | % GT |
|---|---|---|---|
| no overlap at all | 63 | 40.4% | 15.1% |
| merged into a multi-GT prediction | 4 | 2.6% | 1.0% |
| **overlapped but IoU < 0.5** | **89** | **57.1%** | **21.3%** |

### False positives: 224

| bucket | count | % FP | % pred |
|---|---|---|---|
| no GT overlap | 93 | 41.5% | 19.2% |
| **split of an already-matched GT** | **131** | **58.5%** | **27.0%** |

### The IoU distribution

Best IoU achieved per GT instance:

| IoU | count | share |
|---|---|---|
| 0.0–0.1 | 65 | 15.7% |
| 0.1–0.3 | 27 | 6.5% |
| 0.3–0.5 | 59 | 14.2% |
| 0.5–0.7 | 141 | 34.0% |
| 0.7–0.9 | 123 | 29.6% |
| 0.9–1.0 | **0** | **0.0%** |

Median best IoU is **0.604**. **Two thirds of instances are found** (IoU > 0.5)
and 30% are between 0.7 and 0.9. Almost none are near-perfect.

This is the signature of **imprecise boundaries**, not missed objects. A better
model has 34% + 30% = 64% of instances to convert from "good enough" or "barely
missing" into a match, plus a large chunk of SQ on top.

## What has been exhausted

### Post-processing (max +0.007)

Threshold × min_area grid, 64 images:

| thr | min_area | PQ |
|---|---|---|
| 0.55 | 512 | 0.3958 |
| 0.65 | 512 | 0.4031 |
| 0.75 | 512 | 0.4103 |
| 0.80 | 512 | 0.4104 |

Plateau at **0.4104** from 0.4031. Also tried:

| method | best PQ | delta |
|---|---|---|
| morphological close (k 3–13) | 0.4076 | +0.005 |
| proximity merge (4 px) | 0.4137 | +0.011 |
| hysteresis (2 thresholds) | 0.3928 | −0.011 |
| mask dilate / erode ±1–3 px | 0.385 / 0.377 | negative |

Mask thickness is already correct (dilate and erode both hurt), so it is not an
area-bias problem.

### Per-image threshold adaptation (unreachable but informative)

Picking the best threshold *per image* with the GT in hand gives PQ **0.4412**, and
the median best threshold is **0.85** — way above the 0.55 being used. Matching
predicted area to GT area per image gives only 0.4061, so it is not simply an
area-calibration issue.

Practical version: raise the global threshold to ~0.75–0.80 (+0.015) and consider
a cheap per-image statistic (e.g. predicted area percentile) to select it without
GT.

### Free win: 4× flip TTA (+0.008)

| thr | plain | 4× flip TTA |
|---|---|---|
| 0.65 | 0.3793 | 0.3932 |
| 0.75 | 0.3870 | 0.3945 |

Small but free and it composes with anything else.

### Already ruled out

* Watershed — nothing to split (confirmed above: GT mask + CC = 1.0)
* Stride-2 detector — documented negative
* Box detector — box is 73% background
* Frozen DINOv2 — matches SQ, loses RQ
* Token-level classification — 28% of filament pixels have no positive token

## The one lever with real room

Everything above is detail. The evidence says:

1. Instance separation is solved (oracle = 1.0).
2. The errors are boundary precision: median best IoU 0.604, 0% above 0.9.
3. The model was trained on **512² random crops** for 25 epochs at resnet34.

The strongest hypothesis is that **the model is under-resolved and under-trained**,
not that the method is wrong:

* **Resolution**: thin structures need pixels. 512² crops of a 2048² image throw
  away 94% of each image per sample. The tiled-1024 run scored 0.4300 locally —
  the best number so far — but it was only 12 epochs.
* **Boundary supervision**: BCE + soft Dice at full-image scale barely penalises a
  boundary that is 2 px off on a 30 px wide filament. A **boundary-aware loss**
  (distance-transform weighting, or an explicit edge term) targets exactly the
  0.6→0.8 IoU band where 34% of instances sit.
* **Schedule**: 25 epochs on ~900 samples, cosine-decayed, with val PQ still
  improving at the last epoch in several runs.

## Recommended order

1. **Boundary-aware loss + longer schedule at 1024² tiles.** This targets the
   measured error (IoU 0.6–0.9) directly. Highest expected value.
2. **Threshold 0.75–0.80 + 4× TTA** as a free +0.02 baseline improvement while (1)
   trains.
3. **Limb radial normalisation** ([docs/DISK_MASKING.md](DISK_MASKING.md)) — the
   outer ring is at 0.28 recall and holds 11% of filament area.
4. Only then revisit architecture.

Do **not** spend more time on instance separation, watershed, NMS, or
post-processing heuristics. The oracle says they are worthless here.

## Reproducing

```bash
python scripts/pq_decompose.py --n 64
```

Caches probabilities to `/tmp/pq_cache.npz` for the post-processing sweeps.
