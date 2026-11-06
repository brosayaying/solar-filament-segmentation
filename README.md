# Solar Filament Segmentation — MAGFiLO 2026

Harness for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
Competition details, metric definition, and submission rules: [`docs/COMPETITION.md`](docs/COMPETITION.md).

## Results

| Stage | Pooled OOF PQ | 95% CI | Public LB |
|---|---|---|---|
| Phase 1 — segmentation only | 0.4220 | [0.4144, 0.4299] | 0.36 |
| Phase 2 — + candidate re-scoring | **0.4406** | [0.4109, 0.4703] | **0.38** |

Paired gain of Phase 2 over Phase 1: **+0.0154**, CI [+0.0109, +0.0199] over
2,000 bootstraps on 707 files. Pooled OOF scores every file with the fold that
did not train on it, against every annotator copy — the same protocol as the
official scorer.

Two experiments failed and are documented rather than hidden:

| Stage | Change | Result |
|---|---|---|
| Phase 3 | GONG label transfer, 3.1× training data | **−0.0150** PQ |
| Phase 4 | PVTv2-B2 ensemble diversity | **−0.0113** PQ, corr 0.913 |

## Reproducing the submission

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. weights (~994 MB, from the GitHub release)
python scripts/fetch_weights.py

# 3. competition data
#    Download from Kaggle into ./MAGFiLO_1.0_Kaggle_2026/
#    (train/train_images, train/*.json, test/test_images)

# 4. predictions -> submission CSV
python v2/scripts/predict_test_phase2.py --stage probs
python v2/scripts/predict_test_phase2.py --stage submit
```

`notebooks/pipeline.ipynb` walks through the same steps with explanations and
visualisations, and is the fastest way to understand the method. It runs the
whole pipeline end to end — weights, preprocessing, ensemble, candidate
scoring, submission CSV — and is the notebook the competition asks for.

`notebooks/report_draft.ipynb` is a separate tool: it renders the technical
report from `v2/results/summary.json`. It does not run inference.

Inference runs fully offline: `build_model` is called with `pretrained=False`
because the checkpoint supplies every weight, so no Hugging Face download is
needed. Verified with `load_state_dict(..., strict=True)` reporting all keys
matched.

## Quick start (development)

```bash
source .venv/bin/activate          # torch 2.6 + cu124, timm, pycocotools

python -m solar.split              # build the leakage-free split (once)
python -m solar.train --config configs/quick.json      # ~30 s smoke run
python -m solar.train --config configs/baseline.json   # real baseline
./scripts/compare.sh               # rank all runs by val PQ
```

## Layout

```
src/solar/
  data.py         COCO loading, group-aware split, patch dataset, rasterisation
  models.py       TimmUNet — any timm backbone as encoder; DinoUNet for ViTs
  metrics.py      Panoptic Quality (vectorised pairwise IoU)
  losses.py       BCE + soft Dice
  postprocess.py  prob map -> instances -> COCO RLE counts
  train.py        training loop with AMP, cosine LR, val PQ
  predict.py      test-set inference -> submission CSV
  split.py        split creation + leakage report
  detector.py     stage-1 CenterNet (box or spine-point mode)
  detector_data.py  detection dataset, target encoding, decode + NMS
  detector_loss.py  focal loss, Gaussian targets, size/offset L1
  train_detector.py stage-1 training + point/box matching
  embed.py        frozen-backbone token adapters (DINOv2, SAM) + probe metrics
configs/          base / quick / baseline / tiled1024 / detector / spine_centernet
scripts/          sweep.sh, compare.sh, submit.sh, compare_embeddings.py
```

## Swapping backbones

The encoder is any `timm` model with `features_only=True`, so a sweep is a
one-flag change:

```bash
python -m solar.train --set model.backbone=convnext_tiny
./scripts/sweep.sh resnet18 resnet34 convnext_tiny efficientnet_b0 mit_b0
```

`in_chans=1` is passed to `timm`, which adapts the first conv layer for
grayscale input automatically.

**Fixed-size backbones (swin).** Windowed transformers hard-assert that the input
matches the size the model was built for, and their `features_only` path ignores
`dynamic_img_size`. Pass `model.img_size` to build them at the right size:

```bash
python -m solar.train --set model.backbone=swin_tiny_patch4_window7_224 \
                         model.img_size=384 data.patch_size=384
```

This only works if validation also runs at that size. Since validation runs at
full 2048², swin is **not usable with the current eval loop** — it fails with
`Input height (2048) doesn't match model (384)`. Supporting it would require
tiled inference at 384² with stitching, which is a separate piece of work.
Conv backbones (resnet, convnext, efficientnet) are size-agnostic and unaffected.

## The split

`python -m solar.split` writes `splits/date_seed42.json`.

Two leakage sources are handled:

1. **Annotator duplicates.** The JSON has 1,154 image entries but only 707 unique
   files — the same observation annotated by several people. Splitting on
   `image_id` would put the same image in train and val.
2. **Temporal near-duplicates.** Filaments persist for days, so images from the
   same date are highly correlated. The default groups by **date**.

Verified: 0 file overlap, 0 date overlap between train and val.

| | entries | unique files | dates | annotations |
|---|---|---|---|---|
| train | 922 | 565 | 522 | 6,558 |
| val | 232 | 142 | 134 | 1,641 |

## Metric

Panoptic Quality, computed globally over the dataset (not averaged per image):

```
PQ = Σ IoU(TP) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Reported alongside SQ (mean IoU of matches) and RQ (recognition quality), since
PQ = SQ × RQ and the split tells you whether to fix masks or instance separation.

## Baseline approach

Binary semantic segmentation + connected components:

1. UNet over a pretrained timm encoder, 512² random crops, flips/rot90/brightness.
2. BCE + Dice loss, AdamW, cosine schedule, AMP.
3. Threshold, drop small components, label the rest as instances.

This is deliberately simple. Its known weakness is exactly what PQ punishes:
connected components cannot separate touching filaments, so it fragments or
merges them. The next step is an instance-aware head (or watershed on the
distance transform) to improve RQ.

## Results

**Deduplicated val** (142 entries, one annotation per unique file) is the honest
metric — see the metric-ceiling section below. Numbers in the first table are from
that set; the second table keeps the older 64-image training-loop numbers for
continuity.

| run | PQ (dedup val) | SQ | RQ | public LB | notes |
|---|---|---|---|---|---|
| resnet34, 512², 25 ep | 0.3768 | 0.6935 | 0.5434 | 0.33 | baseline |
| resnet34, 512², **160 ep** | 0.4152 | 0.6996 | 0.5934 | **0.34** | +0.038 local, +0.01 public |
| resnet34, 1024² tiled, 12 ep | 0.3942 | 0.6935 | 0.5684 | 0.33 | |
| resnet34, 1024² tiled, **warm + 60 ep** | **0.4195** | 0.6968 | 0.6021 | — | t=−0.18 vs 512²-160ep |
| **human vs human** (reference) | **0.3719** | 0.6505 | 0.5718 | — | 61 imgs, 2nd annotator |

**Public LB: 0.34, rank 578/1152 (44th percentile).** Top is 0.69, p90 is 0.40.

> **These local numbers use the old cv2 rasterisation and are ~0.02 optimistic.**
> On the official pycocotools GT the 160-epoch model scores **0.3986** on full val
> (95% CI [0.379, 0.416]), not 0.4186. See [docs/CORRECTIONS.md](docs/CORRECTIONS.md).
> The table is kept for continuity; re-run with the fixed rasteriser before
> comparing new experiments.

Tiled 1024² is **statistically indistinguishable** from 512² (paired per-image
t = −0.18, tiled better on 67/142 images) despite 6× the epoch time. Long training
is the only real gain, and it saturates.

Training-loop numbers (64-image subset, used for checkpoint selection, so
optimistic — not comparable to the table above):

| run | local val PQ | public LB | notes |
|---|---|---|---|
| resnet34, 512² random crops, 25 ep | 0.3619 | **0.30** | untuned postprocess |
| resnet34, 512² random crops, 25 ep | 0.4101 | **0.33** | tuned postprocess only |
| resnet34, 1024² tiled quadrants, 12 ep | 0.4300 | **0.33** | +0.02 local, 0 public |
| resnet34, 512², **160 ep** | 0.4361 | — | saturated; SQ flat since ep 25 |
| resnet34, 1024² tiled, warm + 60 ep | 0.4343 | — | no gain over 512²-160ep |
| resnet34, 512², **boundary loss**, 25 ep | 0.3642 | — | contours better, PQ flat |
| DINOv2 ViT-S/14 frozen + conv decoder, 518², 40 ep | 0.3409 | — | see below |

All local numbers use the same split so they are directly comparable, but note
**two different evaluation subsets are in play** and they are not comparable to
each other:

* training-loop numbers (`val_images: 64`) — used for checkpoint selection, so
  they are optimistic (the checkpoint was picked on these images);
* the tiled-vs-512 table below uses the **full 232-entry val set** (1641 GT
  instances) and is the honest comparison.

Do not compare a 64-image number against a full-val number. The 64-image subset
happens to be harder, so full-val scores run ~0.02–0.04 higher.

### ⚠️ Corrections: two earlier conclusions were wrong

**Read [docs/CORRECTIONS.md](docs/CORRECTIONS.md).** A review (`FEEDBACK.md`)
found real errors; I verified each against the data.

**1. Masks were rasterised wrong.** The repo used `cv2.fillPoly`; the official GT
is pycocotools. cv2 masks are **11.5% larger** (median IoU 0.882 against the
official rasterisation), and the annotation `area` field matches pycocotools
exactly. Re-scoring our best model on the official GT:

| GT | PQ | SQ | RQ |
|---|---|---|---|
| cv2 (what we used) | 0.4186 | 0.7011 | 0.5971 |
| **pycocotools (official)** | **0.3986** | 0.6914 | 0.5764 |

**−0.020 PQ from evaluation alone**, and the model was also *trained* on the fat
masks. Fixed: `data.rasterise()` now uses pycocotools throughout.

**2. "Humans disagree as much as we err" was the wrong conclusion.** I claimed
inter-annotator agreement (~0.36 PQ) caps model performance and that we should
stop optimising past ~0.45. **That is false.** A consensus predictor beats each
individual annotator, because annotator errors are independent:

| predictor | PQ vs individual annotators |
|---|---|
| one annotator vs another | 0.3593 |
| **majority-vote consensus of 3** | **0.6087** |

Human-vs-human is a **reference line, not a ceiling**. The public leader at 0.69
is consistent with this, and there is a real gap from our ~0.40 to 0.61+.

Also corrected: the per-image p10–p90 spread was quoted as a "noise band" when
the standard error of a dataset-level PQ is ~0.01 (`PQAccumulator.bootstrap_ci`
now reports 95% CIs; ours is [0.379, 0.416] on full val). And the "26% transfer
rate" was derived from 2-decimal leaderboard numbers — retracted.

What survives: 296 files really are annotated 2–3 times. That is an
**opportunity** (train on consensus targets, which is what the official scorer
rewards) rather than a cap.

### Where the PQ is going (read this before trying anything)

Public LB top is **0.69**; we are at **0.34**. The wall is not fundamental.

**Resolution is not the limit, quantitatively.** Quantising the *ground-truth*
mask to a coarse grid and running connected components bounds what any predictor
on that grid can achieve:

| grid | PQ |
|---|---|
| stride 1 | 0.9456 |
| stride 2 | 0.8698 |
| **stride 4** | **0.7691** |
| stride 8 | 0.5613 |

Even at stride 4 a perfect prediction scores **0.769**; we are at 0.40. And the
architecture already uses a stride-2 stem (`act1`) plus stride-4 `layer1` — it is
not subsampling-only. Filament width has only weak effect: the 8–22 px bulk sits
at median IoU ~0.6, and only ~10 of 270 instances are thinner than 8 px.

**Instance separation is solved.** GT mask + connected components gives
**PQ = 1.0000**. Every filament is its own blob, so watershed / NMS / merge
heuristics have nothing to win. This also explains why watershed made things worse.

**Post-processing is exhausted** (max +0.007). Free wins: threshold 0.75–0.80
(+0.015) and 4× flip TTA (+0.008).

**Training budget is not the gap either — and the 160-ep run tells us *why*.**
A 160-epoch run reaches best PQ **0.4361** (ep 126) vs 0.3619 for 25 epochs. But
the gain decomposes cleanly:

| component | ep 25 | ep 126 | gain over 135 more epochs |
|---|---|---|---|
| **SQ** (mask quality) | 0.6861 | 0.7012 | **+0.014** |
| **RQ** (instance detection) | 0.5188 | 0.6219 | **+0.095** |
| PQ | 0.3560 | 0.4361 | +0.074 |

**SQ saturated by epoch 25.** 135 additional epochs bought 0.014 of boundary
quality. Almost all the progress was RQ, i.e. learning to stop over-predicting:
the predicted/GT instance ratio fell from 1.36 (ep 10) → 1.16 (ep 25) → 0.93
(ep 110) → 1.01 (ep 160), converging on the correct count.

This is consistent with the boundary-loss result: that experiment raised
IoU>0.5 coverage 63.6% → 69.4% but *also* raised the count to 1.38, and PQ went
down. Count control, not contour precision, is what the model was missing.

SQ is now 0.701. Against a *self-consistent* mask quantised to a stride-4 grid the
ceiling is 0.822, which looks like 0.12 of headroom — but against a *different
human annotator* SQ is only 0.612–0.653 (see the metric-ceiling section above).
**We are already above the human-to-human SQ ceiling**, so that "headroom" is
annotation ambiguity, not model capacity.

> **Correction:** an earlier version of this section claimed the model was
> "underfitting" because train PQ (0.337) was below val PQ (0.401). That was a
> sampling artefact — the first 60 train images are much denser than val (8.85 vs
> 6.48 instances/image) and PQ is anti-correlated with instance count. On a random
> train sample the ordering is correct (train 0.417 vs val 0.400).

**So the remaining lever is SQ / representation capacity**, not schedule and not
post-processing. Full analysis in [docs/UNDERFITTING.md](docs/UNDERFITTING.md) and
[docs/PQ_DIAGNOSIS.md](docs/PQ_DIAGNOSIS.md), reproduce with
`python scripts/train_val_gap.py`.

## Resolution strategy

Nothing is ever downsampled — the model always sees 1:1 pixels. Two different
strategies apply:

| phase | resolution | strategy |
|---|---|---|
| train (random crops) | 512² | random window, `tiled=false` |
| train (tiled) | 1024² | fixed 2×2 quadrants, `tiled=true` |
| validation | 2048² | whole image |
| inference | 2048² | whole image |

Full-resolution inference costs only 0.10 s/image and 4.3 GB, so there is no
need for tiled or sliding-window **inference** — only training is windowed.

### Why 512² random crops were wasteful

Filament mask area is just **0.36% of an image**, and the median filament bbox
diagonal is 112 px. Measured on 400 sampled training crops:

| | 384² | 512² | 768² | 1024² |
|---|---|---|---|---|
| crops containing no filament | 59.7% | **50.5%** | 19.0% | 13.0% |
| filaments touching crop but cut off | 66.7% | **50.9%** | 32.8% | 22.6% |

So half of all 512² crops were pure background, and half of the filaments a crop
did touch were truncated by the crop edge. Truncation teaches the model straight
blobby boundaries where filaments actually taper.

`data.tiled=true` with `patch_size=1024` replaces random cropping with exactly 4
non-overlapping quadrants per image (3,688 tiles for 922 train images). Every
pixel is seen once per epoch, nothing is truncated at a random offset, and empty
tiles fall to 22%.

Trade-off: tiling removes translation diversity, so it relies more on the
flip/rot90 augmentation. Random cropping at 512² supplies far more variety per
epoch (184,400 possible positions vs 4 fixed tiles).

### Result: tiled wins locally, not on the leaderboard

```
model                        PQ      SQ      RQ     tp     fp     fn
512² random crops          0.4101  0.6820  0.6013   1073    855    568
1024² tiled quadrants      0.4300  0.6802  0.6321   1074    683    567
```

Identical TP (1073 vs 1074) and FN (568 vs 567), but **172 fewer false
positives** — RQ went 0.6013 → 0.6321 while SQ was unchanged. That is exactly the
predicted effect: less truncation means fewer spurious blobs, not better masks.

Public LB stayed at **0.33** (reported to 2 decimals), so the +0.02 local gain did
not translate. This is the second time local improvements failed to move the
leaderboard, which reinforces that local PQ tracks *ranking* well but not
absolute score.

Training cost: ~2,900 s for 12 epochs at 1024² (batch 8, 14 GB VRAM) versus
~900 s for 25 epochs at 512². Roughly 6× the wall-clock per unit of progress, so
the 1024² setting needs a shorter schedule and is the more expensive default.

Public LB context: top 0.69, median 0.34, 1,152 teams. Our 0.34 ranks 578/1152
(44th percentile); p90 is 0.40.

### Post-processing matters more than the backbone

A 40-config sweep over `(threshold, min_area, close_k)` on cached validation
probabilities moved PQ from 0.3619 → 0.4101 with **no retraining**. The driver is
`min_area`: every spurious component costs 0.5 FP in the PQ denominator, so
raising the floor from 64 px to 512 px removed ~260 false positives.

```
thr  min_area  PQ      SQ      RQ     fp    fn
0.55      512  0.4101  0.6820  0.6013   855   568
0.45      512  0.4097  0.6805  0.6021   869   560
0.55      384  0.4073  0.6817  0.5975  1009   512
0.45      256  0.4010  0.6788  0.5906  1115   486
0.55      768  0.3919  0.6857  0.5716   628   733
0.45     1536  0.2987  0.6903  0.4327   257  1117
```

Note the shape: SQ keeps *rising* as `min_area` grows (0.682 → 0.690) while RQ
falls. The model's masks are good; the problem is that it emits too many
instances. That is the RQ bottleneck in one picture.

### Backbone sweep (6 epochs, 384², not converged)

| backbone | params | PQ |
|---|---|---|
| resnet34 (25 ep, 512²) | 23.9M | **0.3619** |
| convnext_tiny | 28.7M | 0.2670 |
| efficientnet_b0 | 4.3M | 0.2032 |
| resnet18 | 12.0M | 0.1945 |
| convnext_small | 50.4M | 0.2026 |
| resnet50 | 25.4M | 0.1364 |

These are short runs, so they rank *learning speed*, not final quality. resnet34
at 25 epochs is the only converged entry. Bigger is not automatically better here
— resnet50 and convnext_small underperform resnet34 at equal budget, which is
consistent with the task being limited by instance separation rather than
capacity.

## Two-stage pipeline (in progress)

The single-stage model has no notion of instance identity, so PQ's recognition
quality (RQ) is the ceiling. The two-stage plan is: (1) locate each filament,
(2) segment inside each located region.

### Stage 1: spine CenterNet

`python -m solar.train_detector --config configs/spine_centernet.json`

CenterNet with **heatmap + offset only** — no size head. Target is the **spine
midpoint**, one point per filament.

Why no box and no size head:

| measurement | value |
|---|---|
| bbox centre lands on filament | 50.8% |
| mask centroid lands on filament | 72.7% |
| **spine midpoint lands on filament** | **99.9%** |
| filament area / bbox area (mean) | 0.274 |

The first number is the important one. CenterNet's focal loss supervises only the
single centre cell as positive, so using the bbox centre meant **half the training
targets sat on background pixels**. Switching to the spine midpoint fixed it.

The second number explains why the box variant plateaus: a *perfect* box is still
73% background, so IoU-0.5 matching is hopeless by construction. The size head
also never learned, systematically undershooting (pred median w 54 vs GT 74).

#### Results

| stage-1 variant | point F1 (24px) | mask-hit recall |
|---|---|---|
| box CenterNet, no NMS | 0.059 | — |
| box CenterNet, + NMS | 0.242 | — |
| spine CenterNet, 20 ep | 0.433 | 0.589 |
| **spine CenterNet, 30 ep** | **0.461** | **0.650** |

#### The metric was measuring the wrong thing

The first version scored stage 1 with point F1 at a 24 px radius around the spine
midpoint. That under-reports badly. Measuring the distance from each ground-truth
filament to its nearest prediction:

| distance | share |
|---|---|
| 0-12 px | 26.5% |
| 12-24 px | 19.5% |
| 24-48 px | 17.0% |
| 48-96 px | 9.8% |
| >96 px | 27.2% |
| **median** | **27.7 px** |

The median miss sits just outside the 24 px cutoff, so F1 was partly measuring an
arbitrary radius rather than whether the filament was found.

What stage 2 actually needs is a seed *anywhere on the filament*, so the metric is
now **mask-hit recall**: does at least one predicted point land inside the
filament's own mask? It is scale-free and needs no radius. Both are reported;
mask-hit is the one to optimise.

#### Stride 2: tried, and it is worse

Hypothesis: the decoder already walks up to the stride-2 encoder level, then
upsamples to stride 4 — so the finest features are being thrown away. Predicting
at stride 2 should halve localisation error.

Result: **mask-hit recall 0.10-0.37 vs 0.59** at stride 4. Clear regression.

Cause: stride 2 quadruples the heatmap cell count, pushing the
negative/positive ratio from **7,282 to 29,127**. The focal loss collapses — the
heatmap never exceeds 0.22 confidence and the model trades recall for precision
(P 0.55, R 0.12). The extra resolution is not worth the imbalance for a
single-point target.

Stride 4 is the default and is documented as such in `detector.py`.

#### Longer schedule helps

30 epochs vs 20: mask-hit 0.589 → 0.650, point F1 0.433 → 0.461. Still improving
at the end, so more epochs are likely worth it.

#### Decode matters as much as the model

Two decode settings dominate. Both were wrong on the first attempt.

**NMS is mandatory.** I wrote "CenterNet needs no NMS" in the first version. True
on COCO, false here: the 3x3 max-pool only dedupes within a cell, and a diffuse
heatmap over a thin low-contrast filament produces many peaks per object. Adding
NMS took the box variant from F1 0.059 to 0.215 — a 4x jump from one change.

**Suppression radius must be large.** For points, suppression is by Euclidean
distance, not IoU. Sweeping it:

```
nms_dist   thr    P      R      F1
   12     0.20  0.211  0.555  0.306
   20     0.20  0.321  0.522  0.397
   30     0.20  0.384  0.481  0.427
   40     0.20  0.416  0.463  0.438   <- best
   60     0.20  0.419  0.445  0.431
```

12 px is about one filament *width*, so it seemed reasonable. But filaments are
long, so peaks spread along one spine are much further apart. 40 px is the
operating point.

#### Verifying the detector actually localizes

Worth measuring separately from F1, because F1 conflates localisation, sizing,
and duplicate suppression. Checking whether the top-40 heatmap peaks per image
land on a filament pixel:

| | |
|---|---|
| peaks on filament | **84.7%** (847/1000) |
| random baseline | ~0.4% |

So localisation was never the problem — the box representation and missing NMS
were.

### Boundary-aware loss: the contours improved, PQ did not

The diagnosis said the errors are boundary precision (median best IoU 0.604, 0%
above 0.9), so a contour-focused loss should help. Built as `loss: boundary` in
`src/solar/losses.py` — `(1-w)·(BCE+Dice) + w·Dice_band`, where the band is
`dilate(dilate(gt)−erode(gt), radius)`, computed with `max_pool2d`.

**It did exactly what it was asked to do:**

| | baseline | boundary |
|---|---|---|
| median best IoU | 0.604 | **0.621** |
| GT instances with IoU > 0.5 | 63.6% | **69.4%** |
| GT instances with IoU > 0.7 | 29.6% | **31.8%** |
| best PQ | 0.3619 | 0.3642 |
| **predicted / GT instances** | 464/417 = 1.11 | **574/417 = 1.38** |

**+5.8 percentage points more instances clear the matching bar** — the mechanism
is confirmed. But PQ is flat, because the model now emits 574 instances where
there are 417. At the tuned threshold it finds more TPs (288 vs 264) *and* more FPs
(286 vs 200). The gain is spent.

Cause: optimising a contour band makes the model more willing to commit to faint
thin regions. Plain Dice was acting as an implicit count regulariser (it rewards
matching total positive area) and shifting weight to a band weakens that.

So the gap is **not purely** a boundary problem. Fixing contours is necessary but
has to come with instance-count control. Full write-up in
[docs/BOUNDARY_LOSS.md](docs/BOUNDARY_LOSS.md).

### Solar disk masking: not needed off-disk, but the limb is a real problem
`scripts/patch_clf.py` masks the solar disk and erodes by 40 px. The main
pipeline does **no** disk masking at all. Measured, that turns out to be fine for
the surround and bad for the limb.

**Off-disk is a non-issue.** The disk covers 73% of the image, so 27% is black
surround — but GT filaments are **0.00%** off-disk, the segmentation model
predicts **0.00%** off-disk, and the detector emits **0/49** points off-disk.
Applying `disk_mask` to the baseline's probabilities changed PQ by exactly zero
(0.3445 → 0.3445). Both models already learned to ignore the surround.

**The limb is where the loss is.** Recall of GT filament pixels by normalised
radius `r/R`:

| r/R | recall | GT px |
|---|---|---|
| 0.0–0.2 | 0.483 | 9,963 |
| 0.2–0.4 | 0.597 | 69,345 |
| 0.4–0.6 | **0.730** | 114,323 |
| 0.6–0.8 | 0.642 | 186,962 |
| 0.8–1.0 | **0.281** | 58,662 |

A 2.6× drop in the outer ring, which holds ~11% of all filament area. Cause is
limb darkening: filaments are dark-on-bright everywhere (contrast +12 to +16)
except the outer ring, where background falls to 78.4 and the sign **inverts**
(−24.0). The model looks for dark-on-bright and the limb is bright-on-dark.

`patch_clf`'s 40 px margin is too small for this — contrast is still −10 at 40 px
erosion and only turns positive at ~120 px. Radial normalisation (subtract a
smoothed radial intensity profile) is the direct fix. Full write-up in
[docs/DISK_MASKING.md](docs/DISK_MASKING.md), reproduce with
`python scripts/disk_analysis.py`.

### End-to-end DINOv2: works, but does not beat from-scratch

A plain `TimmUNet` over DINOv2 is broken by construction: `features_only`
returns three levels **all at stride 14**, so the decoder is a 14× bilinear
upsample with no fine-scale skips. `DinoUNet` fixes this by tapping different
block depths for a pseudo-pyramid and building the fine scales from a conv stem
on the raw image.

Result (40 ep, 518² crops, frozen encoder, 1.19 M trainable params):

| run | PQ | SQ | RQ |
|---|---|---|---|
| resnet34, 512², from scratch | **0.3619** | 0.6791 | **0.5328** |
| DINOv2 frozen + conv decoder | 0.3409 | **0.6817** | 0.5000 |

**SQ is identical** — boundary quality is just as good. The whole gap is **RQ**
(instance detection): it finds 231 instances vs 284 and misses 186 vs 133. That
matches the embedding analysis: at 14 px tokens ~28% of filament pixels have no
positive token, so the model under-predicts area.

This measures *frozen features*, not fine-tuned DINOv2 — the encoder was frozen
by design as a cheap ablation. Fine-tuning is the obvious untested variable.
Full write-up in [docs/DINOV2_SEGMENTATION.md](docs/DINOV2_SEGMENTATION.md).

#### A validation bug this exposed

The first run scored PQ 0.03 and looked collapsed. It was not: the model was
trained on 518² crops but **validated on full 2048² images**. ViTs are strongly
scale-sensitive — the same content predicts 1.46× the GT area at 1022² but only
0.19× at 2048². Tiled inference at the training scale recovered it
(PQ 0.045 → 0.218). `eval.tile` now controls this, and it matters for **any**
ViT encoder, not just this experiment.

### Frozen embeddings: DINOv2 vs SAM
Before investing in a token-level pipeline, I checked whether pretrained
self-supervised features already separate filaments. They do — linear-probe AUC
is 0.99 for both DINOv2 ViT-S/14 and SAM ViT-B/16 — but that is a weak test on a
0.1–1.1% positive rate. The binding constraint is token resolution: at 14–16 px
tokens, **28–32% of filament pixels have no positive token at all**, because the
median filament is only ~30 px wide (~2 tokens across).

DINOv2 wins on every metric (recall 0.719 vs 0.681, precision 0.723 vs 0.692,
AUC 0.993 vs 0.977) and is 2× faster and 4× smaller. Most of that win is the
finer 14 px stride, not better features. Conclusion: keep the pixel-level
decoder; these features are useful as an initialisation or auxiliary loss, not
as a replacement output space. Full write-up in
[docs/EMBEDDINGS.md](docs/EMBEDDINGS.md).

### Watershed: tried, and it makes things worse

Hypothesis: connected components merge touching filaments, so a distance-transform
watershed should split them and lift RQ.

Result: **PQ 0.3348 vs 0.4101 for connected components** — a clear regression.

```
method      thr  min_area  hmax  mdist     PQ      SQ      RQ     tp     fp     fn
cc         0.55       512     -      -  0.4101  0.6820  0.6013   1073    855    568
watershed  0.55       512     4     16  0.3348  0.6721  0.4981    919   1130    722
watershed  0.55       512     8     16  0.3141  0.6780  0.4632    743    824    898
```

**Why it fails.** The premise was wrong. Measuring instance counts against ground
truth over 60 validation images:

| | mean instances | median |
|---|---|---|
| ground truth | 6.48 | 6.0 |
| connected components | 7.48 | 7.0 |
| watershed | 7.75 | 6.5 |

The model already **over**-segments — it emits ~1 more instance per image than
there are filaments. Splitting further pushes TP down (1073 → 919) while FP rises
(855 → 1130), because each split piece drops below the IoU 0.5 match threshold and
becomes a simultaneous FP and FN. Watershed only helps when the baseline
*under*-segments.

This is a useful negative result: the RQ ceiling is not a post-processing problem.
The semantic head has no notion of instance identity, so no amount of geometric
splitting recovers it. Fixing RQ needs an instance-aware model (query-based head,
embedding + clustering, or a learned boundary/spine target).

Implementation notes, if this is revisited:
- `skimage.morphology.h_maxima` costs ~1.3 s per 2048² image and dominates the
  pipeline. Replaced with an erode/dilate reconstruction, ~50x faster.
- Filaments are thin, so the distance transform maxes out around 17 px with a mean
  of 4.7. `h_maxima` must be small (1–8), not the usual 10–20.
- `peak_local_max` rejects boolean input; the seed mask must be uint8.

### Local vs public gap

The local→public ratio is stable at **0.78–0.82** across four submissions:

| local (full val, cv2 GT) | public | ratio |
|---|---|---|
| 0.3768 | 0.33 | 0.88 |
| 0.4152 | 0.34 | 0.82 |

Train and test share the same year and station distributions with zero file
overlap, so this is not domain shift. Most of the gap is now explained by the
rasterisation bug (item 1 above): our local numbers were computed against masks
11.5% larger than the official ones, which inflates local PQ by ~0.02. The
remaining difference is plausibly the annotator-variant averaging.

**Do not infer a transfer rate from these numbers.** The public score is reported
to 2 decimals on a ~90-image split, so a 0.33 → 0.34 step is consistent with
anything from +0.005 to +0.015, and the public split has its own ±0.02 noise. An
earlier version of this section claimed a "26% transfer rate" and "local
compresses gains 4×" — both are retracted as unsupported by 2-decimal data. Pull
5-decimal scores from the Kaggle submissions API before drawing conclusions about
transfer.

## Notes

- Images are 2048² grayscale (`PIL` mode `L`). Do not treat them as RGB.
- Validation runs at full 2048² resolution; training uses crops.
- Submission RLE is always at 2048², counts only, no quotes.
