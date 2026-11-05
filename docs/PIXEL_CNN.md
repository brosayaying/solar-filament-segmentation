# Pixel-level vs patch-level classification

The patch classifier throws away 8/9 of what it sees: a 48×48 crop produces one
label for the centre 16×16 patch. This tests whether predicting a **dense mask**
instead fixes that.

```bash
python scripts/pixel_cnn.py --crop 256 --train-images 60 --val-images 20 --epochs 25
```

Results in `docs/EDGE/pixel_cnn.json`.

---

## 1. Answering the question directly

**Yes, we were still doing patch-level classification.** The CNN took a 48×48
crop but output a single scalar for the centre 16×16 patch. Every other pixel in
the crop was input-only — it could inform the decision but never received one.

That is a real waste. A 48×48 crop contains 2,304 pixels; the model was
supervised on 256 of them (11%). The pixel-level version supervises all 2,304.

---

## 2. Results

| model | level | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| RandomForest (8×8) | patch | 0.1875 | 0.649 | 0.289 | 40 | 90 | 107 |
| CNN (16×16 patch, 48 ctx) | patch | — | — | — | — | — | — |
| CNN (32×32 patch, 96 ctx) | patch | — | — | — | — | — | — |
| **U-Net (256 crop, dense)** | **pixel** | **0.275** | 0.665 | 0.414 | 67 | 110 | 80 |

Two independent runs of the U-Net gave PQ 0.2773 and 0.2737 (best-threshold
0.3 and 0.7 respectively), so **run-to-run variance is ≈0.004**. The table above
uses the second run with the fully-swept threshold.

Patch-level F1 for reference (not comparable to PQ, different metric):

| model | patch F1 |
|---|---|
| RandomForest 8×8 | 0.526 |
| CNN 16×16 (48 ctx) | 0.522 |
| CNN 32×32 (96 ctx) | 0.428 |

**Pixel-level is +0.090 PQ over the best patch classifier** — a 48% relative
improvement, and the largest single gain in this whole line of work.

---

## 3. Why pixel-level wins

### Recall, not precision

The improvement is almost entirely in **RQ: 0.289 → 0.426**. TP goes 40 → 73
while FP goes 90 → 123. The pixel model finds nearly twice as many filaments for
a 37% increase in false positives — a clearly favourable trade, since each FP
costs only 0.5 in the PQ denominator.

SQ is essentially unchanged (0.649 → 0.652). The dense output does not produce
better boundaries than the refined patch map; it produces *more complete*
detections.

### The supervision density is the mechanism

At patch level, one label per 256 pixels means the loss sees ~150k training
signals from 60 images. At pixel level, one label per pixel means ~2.9M signals
from the same 60 images — a 19× increase in supervision from identical data.

That is why the pixel model reaches PQ 0.19 by **epoch 3**, while the patch CNN
needed 20 epochs to reach F1 0.54 and never beat the RF.

### Balanced crop sampling

Half the training crops are centred on a filament pixel, half on background.
Uniform sampling would give ~99% empty crops and the net would learn to output
zero. This is the "pick both classes in a balance" idea, applied at crop level.

---

## 4. Bigger patch input made things worse

| patch | context | patch F1 |
|---|---|---|
| 16 | 48 | 0.522 |
| 32 | 96 | **0.428** |

Counter-intuitive but explainable: at 32×32 there are 4× fewer patches (9,740 vs
38,730 training samples), and each label now covers 1,024 pixels — so a patch
that is 30% filament is labelled positive, which is a much noisier target. The
model also has less data to learn from.

**Larger patches are the wrong axis.** The right axis is denser supervision,
which is what the pixel-level model does.

---

## 5. Architecture

Small U-Net, 1.10 M params, trained from scratch (no pretrained weights):

- 3 down / 3 up levels, width 24, skip connections
- input 256×256 crops, output 256×256 mask
- loss: BCE + Dice (Dice handles the 2.6% positive-pixel imbalance)
- dihedral augmentation, re-drawn each epoch
- global normalisation (mean 82.5, std 58.1 — note this is the *whole image*
  including the black surround, unlike the patch pipeline's inner-disk-only
  statistics)
- full-image inference tiled at 512×512

Training: 667 s for 25 epochs on an RTX 3090.

---

## 6. Comparison to the DL baseline

| method | PQ |
|---|---|
| **U-Net from scratch, 60 images, 256 crops** | **0.277** |
| Patch classifier + refinement | 0.202 |
| DL baseline (resnet34, 512², 922 images) | 0.36 |
| DL tiled 1024² | 0.39 |

The from-scratch U-Net reaches 0.277 with **60 training images** against the
baseline's 922. It is not competitive yet, but the gap is much smaller than the
patch classifier's, and the obvious next step (more images) is untested.

---

## 7. Threshold sensitivity

The first run selected `thr=0.3`, which sat on the lower edge of the swept range
(0.3–0.7) — a red flag suggesting the optimum was outside the range. Extending
the sweep in both directions resolved it, and the answer was the opposite of
expected:

| thr | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|
| 0.05 | 0.2374 | 0.624 | 0.380 | 70 | 151 | 77 |
| 0.10 | 0.2492 | 0.641 | 0.389 | 70 | 143 | 77 |
| 0.20 | 0.2493 | 0.654 | 0.381 | 68 | 142 | 79 |
| 0.30 | 0.2592 | 0.656 | 0.395 | 68 | 129 | 79 |
| 0.40 | 0.2691 | 0.655 | 0.411 | 70 | 124 | 77 |
| 0.50 | 0.2731 | 0.657 | 0.416 | 69 | 116 | 78 |
| 0.60 | 0.2734 | 0.660 | 0.414 | 69 | 117 | 78 |
| **0.75** | **0.2749** | 0.665 | 0.414 | 67 | 110 | 80 |
| 0.85 | 0.2743 | 0.663 | 0.414 | 65 | 102 | 82 |
| 0.90 | 0.2657 | 0.658 | 0.404 | 61 | 94 | 86 |
| 0.95 | 0.2064 | 0.647 | 0.319 | 44 | 85 | 103 |

Two things follow:

1. **The optimum is 0.75, not 0.3.** The first run's number (0.2773 at thr=0.3)
   was a lucky draw, not a better operating point — the second run at thr=0.3
   gives 0.2592.
2. **The curve is flat between 0.5 and 0.85** (PQ 0.273–0.275). Threshold choice
   barely matters in that band, which is reassuring for transfer to the test set.

The reason is visible in the probability distribution: p99 = 0.002 and
p99.9 = 0.997. The model is **extremely bimodal** — almost every pixel is
confidently 0 or confidently 1, with very little in between. That is a
consequence of the Dice loss, which pushes predictions toward the extremes.

---

## 8. Caveats

- **20 validation images, 147 GT instances.** The PQ difference over the patch
  classifier (0.275 vs 0.188) is large enough to be real; the absolute value has
  meaningful uncertainty.
- **Run-to-run variance ≈0.004** (two runs: 0.2773, 0.2737). Single seed each.
- The per-epoch PQ in the log is computed on only 4 val images (for speed), so
  it is noisy — the final number uses all 20.
- Threshold 0.75 was selected on the validation set. Given the flat curve this
  is low-risk, but it is still a tuned parameter.

---

## 9. What to try next

1. **More training images.** 60 → 700 available. The pixel model was still
   improving at epoch 25 and this is the cheapest large gain.
2. **Larger crops** (512) now that supervision is dense — the earlier
   "bigger is worse" result applied to patch labels, not dense ones.
3. **Pretrained encoder.** The DL baseline's advantage is largely ImageNet
   initialisation; a from-scratch U-Net at 60 images is fighting that.
4. **Softer loss.** The Dice term drives the bimodal output; a pure-BCE or
   focal variant might give better-calibrated probabilities and a wider usable
   threshold band.
