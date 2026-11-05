# The metric ceiling: human annotators disagree as much as our model errors

> ## ⚠️ SUPERSEDED — the central claim in this document is wrong
>
> **Read [CORRECTIONS.md](CORRECTIONS.md) instead.**
>
> This document argues that inter-annotator agreement (~0.36 PQ) caps how well
> any model can score, and concludes "stop optimising local PQ past ~0.45".
> **That is false.** A consensus predictor beats each individual annotator by a
> wide margin, because annotator errors are independent and average out. Measured
> on 40 three-annotator files:
>
> | predictor | PQ vs individual annotators |
> |---|---|
> | one annotator vs another | 0.3593 |
> | **majority-vote consensus of 3** | **0.6087** |
>
> So human-vs-human is a *reference line*, not a ceiling. The public leader at
> 0.69 is consistent with this. There is a real gap from our ~0.40 to 0.61+.
>
> Two further errors here: the per-image p10–p90 spread was quoted as a "noise
> band" when the standard error of a dataset-level PQ is ~0.01 (see
> `PQAccumulator.bootstrap_ci`), and the local→public "transfer rate" was derived
> from 2-decimal leaderboard numbers.
>
> What survives: the dataset *does* have 296 files annotated 2–3 times, and the
> per-image variance is genuinely large. The duplicate structure is real and
> matters — but as an opportunity (train on consensus targets) rather than a cap.

This is the most consequential finding in the project so far. It reframes every
number in the README, and it changes what "improving PQ" can even mean.

## What the data actually is

The COCO file has **1154 image entries but only 707 unique image files**. 296
files appear 2–3 times under different `image_id`s, distinguished by a numeric
prefix:

```
050103-20111116063134Lh   <- annotator 050103
050102-20111116063134Lh   <- annotator 050102
050101-20111116063134Lh   <- annotator 050101
```

There are **37 distinct annotator prefixes** (`010101`, `010102`, `020101`,
`050103`, …). Each is a separate human annotation of the *same* underlying image.

| split | entries | unique files | entries in duplicated files |
|---|---|---|---|
| train | 922 | 565 | 592 (64%) |
| **val** | **232** | **142** | **151 (65%)** |

The validation set is **65% duplicate entries**. It scores the same 142 images
repeatedly, against 2–3 different human annotations each.

## Humans do not agree with each other

Scoring one annotator's polygons as a "prediction" against another annotator's
ground truth — same image, same task, both human — over all 296 multi-annotator
files (598 pairs):

| | PQ | SQ | RQ |
|---|---|---|---|
| mean | **0.361** | **0.612** | 0.560 |
| median | 0.364 | 0.635 | 0.571 |
| p10 | 0.152 | 0.562 | — |
| p90 | 0.575 | 0.707 | — |

Restricted to file-pairs present in the val split (61 pairs):

| | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|
| **human vs human** | **0.3716** | **0.6529** | 0.5691 | 243 | 174 | 194 |
| **our model (160 ep)** | **0.4361** | **0.7012** | 0.6219 | 250 | 137 | 167 |

**Our model scores higher than the average human annotator does against another
human annotator.** Not slightly — by 0.065 PQ and 0.048 SQ.

## What this means

### 1. The local metric is close to saturated, and 0.86 was never achievable

Earlier I computed that public 0.69 implies local ~0.86 and concluded we were far
from it. That inference assumed the local metric had a ceiling of 1.0. It does
not: **inter-annotator disagreement caps the meaningful local PQ around 0.36–0.50**
depending on how strictly you define agreement.

Our 0.4361 is already at or above the human baseline on this metric. Chasing
0.86 locally would mean fitting one annotator's idiosyncrasies better than other
humans do, which is overfitting to annotation noise, not learning the task.

### 2. SQ 0.701 is not "0.12 below the quantisation ceiling"

I previously compared SQ 0.701 to a stride-4 quantisation ceiling of 0.822 and
concluded there was 0.12 of headroom. That comparison was against a *self-consistent*
mask. Against a *different human*, SQ is 0.612–0.653. We are already **above** the
human-to-human SQ ceiling.

The flat SQ since epoch 25 is therefore not a capacity failure. It is what
saturation looks like.

### 3. Every experiment ranked by local PQ is partly ranking noise-fitting

The boundary loss raised IoU>0.5 coverage and lost PQ. The 160-epoch run gained
+0.074, mostly in RQ (count calibration). Both are real effects, but the *target*
they were being measured against is itself noisy at roughly ±0.15 PQ (p10 0.152 to
p90 0.575 between humans).

Differences of 0.02–0.05 between configurations are **inside the annotator noise
band**. Only larger effects should be believed, and the public leaderboard is the
tie-breaker.

### 4. The public LB ratio may be misleading

We observe local 0.41 → public 0.33, a ratio of 0.80, and attributed it to val
averaging multiple annotators while the test set uses one. That attribution looks
right, and it means **the public score is the honest signal** — it is the one not
diluted by annotator disagreement.

If the test set has a single annotation per image, then public PQ can exceed what
we can measure locally, because local PQ is capped by disagreement between
annotators. This would explain why local gains kept failing to transfer.

## Recommended change of strategy

**Stop optimising local PQ past ~0.45.** It is measuring agreement with a
particular annotator, not correctness.

Concretely:

1. **Use the public leaderboard as the primary signal** for any change expected to
   be worth less than ~0.05 local. Submit more, submit smaller increments.
2. **Deduplicate the validation set** — evaluate on one annotation per unique file
   (142 entries instead of 232). This removes the 65% duplication, makes local
   numbers interpretable, and makes them comparable to a single-annotation test
   set. This is a small change to `FilamentDataset` and should be done first,
   because it changes the meaning of every future comparison.
3. **Report human-vs-human as the reference line** on every results table, so it
   is obvious when a run is above or below the noise floor.
4. **Re-rank past experiments** under the deduplicated validation to see which
   conclusions survive. In particular the 160-epoch gain and the tiled-vs-512
   difference need re-checking.

## Reproducing

```bash
python scripts/annotator_agreement.py
```

Rasterises every annotator variant, scores each pair with `PQAccumulator`, and
prints the agreement distribution plus the val-subset human baseline.

## Caveat

This analysis assumes the numeric prefix identifies an annotator and that variants
are independent human attempts at the same image. The evidence is strong (three
variants per file, distinct prefixes, differing polygon counts and areas) but the
competition does not document it. If the variants are instead, say, different
sessions or instruments, the interpretation changes. Worth confirming from the
competition rules before acting on it — see `docs/COMPETITION.md`.
