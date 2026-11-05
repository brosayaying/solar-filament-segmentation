# Why PQ stalls: two hypotheses tested, one surviving lead

> **Correction (important).** The first version of this document claimed the model
> was "underfitting" on the evidence that train PQ (0.337) was below val PQ
> (0.401). That comparison was **flawed**: it used the *first* 60 train images,
> which happen to be much denser than the val split (8.85 vs 6.48 instances per
> image), and PQ is strongly anti-correlated with instance count. On a random
> train sample the gap disappears: train 0.3979 vs val 0.4193, and
> density-matched it is 0.4182 vs 0.4356. **There is no meaningful train/val
> gap, so the model is neither badly overfitting nor dramatically underfitting.**
> The rest of this document is kept because the resolution measurements are
> correct and still rule out the subsampling hypothesis, but the "train longer"
> conclusion is much weaker than originally stated. See the corrected summary at
> the end.

Two hypotheses were tested:

1. **"We lose information because subsampling is 4."** Refuted. The architecture
   already has stride-2 features, and resolution is not quantitatively the binding
   constraint.
2. **"The model is underfitting."** Original measurement was wrong (density
   confound). The corrected reading is: train and val are close, so the model is
   near its capacity *for this architecture and schedule* — not starved of epochs.

## The stride-4 hypothesis does not hold

### The architecture is already fine at resolution

resnet34 gives five levels and `TimmUNet` uses all of them:

| level | module | stride | 512² input | 1024² input | channels |
|---|---|---|---|---|---|
| 0 | `act1` | **2** | 256² | 512² | 64 |
| 1 | `layer1` | 4 | 128² | 256² | 64 |
| 2 | `layer2` | 8 | 64² | 128² | 128 |
| 3 | `layer3` | 16 | 32² | 64² | 256 |
| 4 | `layer4` | 32 | 16² | 32² | 512 |

The decoder already receives a stride-2 feature (`act1`) and a stride-4 feature
(`layer1`), concatenates both, and upsamples to full resolution. It is not a
subsampled-only architecture.

### Resolution is not the limiting factor, quantitatively

Take the **ground-truth mask**, quantise it to a coarse grid, and run connected
components. This is the ceiling if the model's only error were quantisation:

| prediction grid | PQ | SQ | RQ |
|---|---|---|---|
| stride 1 (exact) | 0.9456 | 1.0000 | 0.9456 |
| stride 2 | 0.8698 | 0.9122 | 0.9536 |
| **stride 4** | **0.7691** | 0.8224 | 0.9352 |
| stride 8 | 0.5613 | 0.6917 | 0.8114 |
| stride 16 | 0.1822 | 0.5968 | 0.3052 |

Even at **stride 4** — the coarsest level that could plausibly matter — a
perfect prediction scores **0.769**. We are at 0.40. The gap is not quantisation.
(Stride 8 does hurt, 0.561, which is the real regime where thin structures break
down — but we are not there.)

### Filament width has only weak effect

Best IoU per GT instance, binned by a width proxy (area / max extent), 270
instances:

| width | n | median IoU | frac IoU>0.5 |
|---|---|---|---|
| 0–8 px | 10 | 0.157 | 30.0% |
| 8–14 px | 108 | 0.570 | 55.6% |
| 14–22 px | 114 | 0.600 | 64.9% |
| 22–35 px | 29 | 0.532 | 51.7% |
| 35+ px | 9 | 0.623 | 77.8% |

Only 10 of 270 instances are in the 0–8 px bin. The bulk sits at 8–22 px with
median IoU ~0.57–0.60. That is not a resolution cliff.

### The CNN does not care about inference scale

Unlike the ViT (where tiled inference at training scale gained 5×), the CNN is
scale-robust:

| model | full image | tiled 1024 | tiled 512 |
|---|---|---|---|
| baseline | 0.3757 | 0.3737 | 0.3484 |
| tiled-trained | 0.3841 | 0.3723 | 0.3585 |

No gain. The "train at 512, validate at 2048" mismatch that sabotaged DINOv2 is
not a factor for convolutions.

## The train/val comparison, corrected

The original claim used the first 60 entries of `split["train"]`. Those are
unrepresentative — train is denser than val and PQ falls with instance count
(corr = −0.19):

| sample | n | mean PQ | median PQ | inst/img |
|---|---|---|---|---|
| train, **first 60** | 60 | 0.337 | — | 8.85 |
| train, **random 60** | 60 | **0.3979** | 0.4038 | 7.20 |
| val | 60 | **0.4193** | 0.4129 | 6.48 |
| train, density-matched (<7 inst) | 29 | 0.4182 | — | 3.86 |
| val, density-matched (<7 inst) | 35 | 0.4356 | — | 3.69 |

On a random sample the gap is **0.021**, and density-matched it is **0.017**. That
is a normal, small generalisation gap. The dramatic "train is worse than val"
result was an artefact of taking the head of the split.

## What this means

* **Not under-resolved**: stride-4 GT still scores 0.769; we are at 0.40.
* **Not badly overfitting**: train and val are within 0.02.
* **Not a long-schedule problem per se**: the 160-epoch run is at epoch 36 with
  best PQ 0.3850 (epoch 23), versus 0.3619 for the 25-epoch baseline — real but
  modest improvement, and it is not tracking toward 0.69.

So the model is plateauing at ~0.40 because it is **near capacity for this
setup**. Longer training alone will not close a gap of 0.40 → 0.69.

## What the evidence points to instead

The error decomposition (see [PQ_DIAGNOSIS.md](PQ_DIAGNOSIS.md)) showed the loss
is concentrated in instances that are *found but poorly delineated*: median best
IoU 0.604, and FP are 59% "split of an already-matched GT". Combined with the
facts above, that suggests the limiting factor is **the model's ability to
represent these structures at all**, not the schedule, the post-processing, or the
stride per se.

Things this makes worth trying, in order:

1. **Higher-capacity decoder / different encoder family.** The 160-epoch run shows
   resnet34 + 128-dim decoder saturates near 0.38–0.39. A stronger encoder
   (convnext, or a deeper decoder with more decoder_dim) is the direct test of
   "capacity, not epochs".
2. **1024² tiles with the long schedule.** Tiled-1024 scored 0.4300 in only 12
   epochs — the best result so far. It has never been run long. Combining it with
   the 160-epoch budget is the most promising single experiment.
3. **Auxiliary supervision at the instance level.** The spine head is built and
   reaches mask-hit 0.650; using it to gate instances attacks the "split of a
   matched GT" FP bucket directly.

## Reproducing

```bash
python scripts/train_val_gap.py --n 60 --sample random
```

Reports train vs val PQ (random sample by default) and the stride-quantisation
ceiling. Use `--sample head` to reproduce the flawed first-60 comparison.
