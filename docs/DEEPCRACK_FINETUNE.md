# DeepCrack finetuning

Finetuning the pretrained DeepCrack network (`yhlleo/DeepSegmentor`) on MAGFiLO,
starting from the released `pretrained_net_G.pth`.

The zero-shot probe is in [DEEPCRACK/](DEEPCRACK/) and `scripts/deepcrack_infer.py`.
This document covers what happens when we stop treating it as a frozen probe and
start adapting it.

---

## 1. Why finetune at all

The zero-shot result was PQ 0.0 with **TP = 0, FP = 921, FN = 265**: the network
fires on the solar limb and essentially nowhere else. In-disk AUC was 0.47 under
the upstream `[-1, 1]` normalisation and 0.64 under `[0, 1]`.

That is a distribution mismatch, not a capacity limit. The weights encode a
"thin dark curve on textured background" prior that is genuinely relevant to
filaments; they are just being read out on data whose intensity distribution and
content they have never seen. So the weights are worth keeping and adapting.

---

## 2. Why 256x256 crops rather than full images

The decision is not about resolution. It is about matching the pretraining
distribution cheaply while keeping BatchNorm well-conditioned.

| consideration | 256 crops | full 2048 |
|---|---|---|
| matches pretraining scale | yes (trained on 256²) | no |
| BN batch statistics | batch 16, well-conditioned | batch 1-2, very noisy |
| gradient steps per unit compute | ~64x more | baseline |
| inference cost | full-image is free (see below) | same |

The last row is the key one. This CNN family is already scale-robust at
inference, so there is no inference-side reason to pay for full-image training.
From `docs/UNDERFITTING.md`:

| model | full image | tiled 1024 | tiled 512 |
|---|---|---|---|
| baseline | 0.3757 | 0.3737 | 0.3484 |
| tiled-trained | 0.3841 | 0.3723 | 0.3585 |

Train-small / infer-large costs essentially nothing. The DINOv2-style penalty
that made tiled inference mandatory for the ViT does not apply here.

Full-image training would also force a choice between freezing BN (throwing away
the adaptation we want) and SyncBN. At 256² with batch 16, BN adapts normally.

---

## 3. What actually matters: the sampler and the loss

Crop size is the least interesting part. Two other choices target the observed
failure directly.

### Positive-biased, limb-aware sampling (`src/solar/crop_data.py`)

Uniform cropping cannot fix the limb false positives. Filaments are 2.6% of
pixels, so a uniform 256 crop is usually empty background; and if crops are drawn
from the disk interior, the limb is never a labelled example at all — it is
neither positive nor negative, so the model is free to keep firing on it.

`PositiveCropDataset` mixes three modes:

| mode | p | purpose |
|---|---|---|
| `positive` | 0.5 | centre on a random point on a random polygon edge |
| `limb` | 0.2 | centre on the disk boundary, radius jittered by `limb_width` |
| `uniform` | 0.3 | plain random crop, keeps the background distribution honest |

The limb is unannotated, so limb crops land in the mask as **background** — the
negative example the probe was missing.

Measured on 40 sampled crops: **67.5% contain a target**, mean positive fraction
2.0%. Uniform sampling at 2.6% global density would give far fewer.

### Border-aware loss (`border_aware_loss` in `src/solar/losses.py`)

A 256 crop almost always truncates a filament. Without correction the model is
taught that filaments terminate abruptly at the crop edge, which shows up as an
SQ regression — and SQ is already where the error concentrates (median best IoU
0.604, `docs/PQ_DIAGNOSIS.md`).

Pixels within `border_margin` px of the edge are scaled linearly down to
`border_floor`. `floor=0` ignores the band entirely; a small floor keeps a weak
"not filament" signal.

---

## 4. Step 1: BatchNorm recalibration — a negative result

`scripts/deepcrack_bn_recalib.py` runs the network in `train()` mode under
`no_grad()` with `momentum=None` (cumulative moving average), so the running
statistics become the exact mean/var over 800 MAGFiLO crops rather than an
exponential blend with the original asphalt statistics.

The hypothesis was that the `[0,1]` vs `[-1,1]` sensitivity (AUC 0.64 vs 0.51)
was a BN statistics problem.

**It un-suppressed the network but degraded ranking.**

| weights | max prob | p99.9 | in-disk AUC | filament mean | background mean |
|---|---|---|---|---|---|
| raw pretrained | 0.123 | 0.035 | **0.639** | 0.0127 | 0.0113 |
| BN recalibrated | 0.887 | 0.530 | **0.425** | 0.0087 | 0.0099 |

Recalibration raised the output range by ~7x (max 0.12 → 0.89) but pushed in-disk
AUC *below chance*. The filament/background separation collapsed: 0.0127 vs
0.0113 raw (a real if tiny gap) became 0.0087 vs 0.0099 (inverted).

Interpretation: the original BN statistics were not simply "wrong". They were
suppressing the network's output, and that suppression happened to preserve a
weak but correct ranking. Replacing them with MAGFiLO statistics lets the network
express its (asphalt-shaped) features at full confidence, and those features rank
filaments *worse* than the suppressed version did.

**Conclusion: BN recalibration is not a useful standalone step here.** It is
still worth having as a script, because it is the right diagnostic — it isolates
"is this a statistics problem or a weights problem" and answers *weights*. The
finetune below therefore starts from the **raw pretrained weights**, not the
recalibrated ones.

---

## 5. Step 2: finetuning

```bash
python -m solar.train --config configs/deepcrack_ft.json
```

| setting | value | why |
|---|---|---|
| patch | 256 | matches pretraining |
| batch | 16 | BN well-conditioned |
| lr | 1e-4 | 3x lower than the from-scratch baseline; adapting, not relearning |
| sampler | `positive_crop` | p_pos 0.5 / p_limb 0.2 / uniform 0.3 |
| loss | `border_aware` | margin 16, floor 0.0 |
| normalisation | mean 0, std 1 | i.e. `x/255`, the `[0,1]` convention that scored AUC 0.64 |
| epochs | 12 | |
| init | raw `pretrained_net_G.pth` | per §4 |

Evaluation is full-image at 2048², which is free per §2.

---

## 6. Results

### Re-run on corrected (pycocotools) masks — the number to use

The first finetune (`runs/deepcrack_ft_256`) was **trained and evaluated on
`cv2.fillPoly` masks**, which are 11.5% too large (`docs/CORRECTIONS.md` §1). Its
0.338 was therefore measured against fat GT. After the rasterisation fix it
re-scores at **0.326** on full val — the −0.013 was evaluation alone.

`configs/deepcrack_ft_pycoco.json` warm-starts from that checkpoint
(`init_ckpt`, `missing=0 unexpected=0`) and trains 10 more epochs on the correct
targets. 15 minutes on an RTX 3090.

Full val, 142 unique files, one annotator variant each, `thr=0.5 min_area=512`:

| checkpoint | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| run 1, re-scored on pycoco GT | 0.3258 | 0.6612 | 0.4927 | 592 | 783 | 436 |
| **run 2 (warm start, pycoco targets)** | **0.3417** | 0.6710 | 0.5092 | 610 | 758 | 418 |

**Δ = +0.0158.** Paired bootstrap over images (2000 resamples): 95% CI
**[+0.0078, +0.0243]**, P(Δ>0) = 1.000. Per-model CIs are [0.306, 0.346] and
[0.322, 0.362], which overlap — but the *paired* test is the right one here
since both models see identical images, and it is unambiguous.

So retraining on correct targets recovered slightly more than the −0.013 that the
evaluation fix cost, netting a small real gain. The improvement is uniform: SQ,
RQ, TP up, FP and FN down.

### Threshold sweep (run 2)

| thr | min_area | PQ | SQ | RQ |
|---|---|---|---|---|
| 0.40 | 512 | 0.3362 | 0.6648 | 0.5057 |
| **0.50** | **512** | **0.3417** | 0.6710 | 0.5092 |
| 0.60 | 512 | 0.3226 | 0.6768 | 0.4766 |
| 0.70 | 512 | 0.3075 | 0.6762 | 0.4547 |

`min_area=512` wins at every threshold; 2048 costs ~0.18 PQ.

### Run 3: longer warm start — and an LR trap worth knowing about

`configs/deepcrack_ft_long.json` warm-starts from run 2 and trains 25 more
epochs. The first attempt **failed**, and the failure is instructive.

**The bug.** `train.py` restarts the cosine schedule at peak LR on every
invocation. Run 2's schedule had annealed to **lr = 0** by its final epoch, so
warm-starting from it at `lr=1e-4` applied a full-size step to a fully settled
model. Result:

| epoch | lr | PQ |
|---|---|---|
| 1 | 1.0e-4 | 0.2524 |
| 2 | 1.0e-4 | 0.0146 |
| 4 | 9.6e-5 | 0.3166 |
| **6** | 9.0e-5 | **0.0000** (tp=0, fp=1) |
| 7 | 8.5e-5 | 0.0088 |

The model collapsed into a degenerate zero-output state. Run 2 had shown the same
early oscillation (ep1 0.198 → ep2 0.070) but recovered by luck.

**The fix.** Probe both candidate LRs for 4 epochs before committing:

| lr | epoch 1 | 2 | 3 | 4 | behaviour |
|---|---|---|---|---|---|
| 3e-5 | 0.3629 | 0.3506 | 0.3745 | 0.3637 | oscillates |
| **1e-5** | 0.3534 | 0.3621 | 0.3643 | **0.3662** | monotone |

`lr=1e-5` with `warmup_epochs=0` is stable. `train.py` now prints a warning when
a warm start uses an LR >= the checkpoint's original LR, so this cannot recur
silently.

**Result** (full val, 142 files, `thr=0.5 min_area=512`):

| checkpoint | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| run 2 (10 ep) | 0.3417 | 0.6710 | 0.5092 | 610 | 758 | 418 |
| **run 3 (25 ep, warm)** | **0.3531** | 0.6767 | 0.5218 | 605 | 686 | 423 |

**Δ = +0.0114**, paired bootstrap 95% CI **[+0.0054, +0.0171]**, P(Δ>0) = 1.000.
The gain is almost entirely **FP: 758 → 686** (−72), with TP flat (610 → 605).
So the extra epochs are teaching the model to stop over-predicting, not to find
more filaments.

### Threshold sweep (run 3)

| thr | min_area | PQ | SQ | RQ |
|---|---|---|---|---|
| 0.40 | 512 | 0.3513 | 0.6678 | 0.5260 |
| **0.50** | **512** | **0.3531** | 0.6767 | 0.5218 |
| 0.60 | 512 | 0.3342 | 0.6796 | 0.4917 |

### The limb false positives are gone — but read the caveat

| weights | in-disk AUC | limb-ring mean prob | filament mean | background mean |
|---|---|---|---|---|
| raw pretrained | 0.650 | 0.0109 | 0.0128 | 0.0113 |
| run 1 finetuned | 0.992 | 0.0000 | 0.6179 | 0.0033 |

(n=12 images, corrected masks. Disk masking changes PQ by 0.0000.)

**Caveat, added after the TP/FP analysis in §10.** The `limb-ring mean prob`
column is a *mean over the whole ring*, which dilutes a localized arc to
near-zero. It is not evidence that no limb FP exists. The per-instance radial
analysis in §10 is the correct measurement, and it finds **0 of 340 pure FPs
within 40 px of the disk edge** — so the conclusion holds, but it holds because
of §10, not because of this table.

### Where it stands

| method | PQ |
|---|---|
| DeepCrack zero-shot | 0.000 |
| DeepCrack + BN recalibration | 0.000 |
| U-Net from scratch, 60 images, 256 crops | 0.277 |
| **DeepCrack finetuned, 256 crops, 47 epochs total** | **0.353** |
| DL baseline (resnet34, 512², 922 images) | 0.36 |
| DL tiled 1024² | 0.39 |

~50 minutes of total training now lands within 0.007 of a resnet34 baseline
trained on 922 images with ImageNet weights — comfortably inside the noise band,
i.e. **at parity** with the baseline.

### Convergence

Run 3 plateaued: PQ oscillates in 0.341–0.353 from epoch 4 onward with no trend,
and the best epoch (16) is not the last. Unlike run 2, this recipe is now
**converged** — more epochs at this LR will not help. Further gains need a
different lever (see §9).

---

## 7. Reproducing

```bash
# The network is vendored in src/solar/deepcrack_net.py -- no clone needed.
# Only the weights are external:
# pretrained_net_G.pth (Google Drive id 1MAwIlWw27feU8EEkP8pgbia7kcuxBwEG)
#   -> /media/yasir/HDD/deepcrack_weights/extracted/deepcrack/

python scripts/deepcrack_bn_recalib.py --n-images 200 --crops-per-image 4
python scripts/deepcrack_eval.py --n 60 --mean 0.0 --std 1.0
python -m solar.train --config configs/deepcrack_ft.json
python -m solar.train --config configs/deepcrack_ft_pycoco.json   # warm start
python -m solar.train --config configs/deepcrack_ft_long.json     # warm start, lr=1e-5
python scripts/deepcrack_eval.py --weights runs/deepcrack_ft_256_long/best.pt \
    --n 142 --mean 0.0 --std 1.0 --thresholds 0.4 0.5 0.6 --min-areas 512
python scripts/deepcrack_viz.py --n 4
```

`scripts/deepcrack_eval.py` scores any checkpoint on the val split with a
threshold x min_area sweep, and `--disk` masks predictions to the solar disk.

---

## 8. Caveats

* **142 val files, 1028 GT instances.** The run-to-run deltas are resolved
  (+0.0158 and +0.0114, both paired CIs exclude 0), but the difference against
  the resnet34 baseline (0.353 vs 0.36) is **not** resolved at this sample size
  and should not be read as "DeepCrack is worse".
* Run 1's PQ was originally reported as 0.338 using cv2 masks and the first 60
  sorted val ids. The subset bias turned out to be negligible here (first-60 vs
  full-142 agreed to 0.001), unlike the 160-epoch main run where it mattered.
* The BN recalibration negative result (§4) is measured on 12 images for the AUC
  and 60 for PQ. The direction is unambiguous (AUC below chance), but the exact
  magnitude is not.
* `p_positive`/`p_limb` were set from the failure mode, not tuned. A sweep is the
  obvious next experiment.
* Run 3 is converged, so 0.353 is this recipe's ceiling, not a floor.

---

## 9. What would move the number now

The error is concentrated in **RQ (0.52)**, not SQ (0.68). Run 3's gain came
entirely from cutting FP (758 → 686) with TP flat, which says the remaining
problem is instance bookkeeping, not pixel accuracy:

* **FP 686 vs TP 605** — more false instances than true ones. Many are splits of
  an already-matched GT (`docs/PQ_DIAGNOSIS.md`). `prob_to_labelmap_watershed`
  exists in `src/solar/postprocess.py` and has never been tried on this model.
* **FN 423** — instances missed entirely. Recall is weakest in the outer ring
  (`docs/CORRECTIONS.md` §7), which the limb sampler does not address because it
  only teaches the limb as *negative*.
* **TTA** — `predict.py --tta` gives +0.0024 on the main model, inside the CI.
  Cheap but not a lever.
* **Multi-annotator targets** — train on the consensus mask for the 296 files
  with 2–3 annotators. This is the highest-value item in `docs/CORRECTIONS.md`
  and applies here unchanged.

---

## 10. TP / FP analysis

```bash
python scripts/deepcrack_tp_fp.py --n 142 --per-figure 3
python scripts/deepcrack_fp_radial.py
python scripts/deepcrack_fp_inspect.py
```

Run 3 (`runs/deepcrack_ft_256_long/best.pt`, epoch 16) over the full val split.
Counts reproduce the eval exactly: **TP 605, FP 686, FN 423**.

Figures in `docs/DEEPCRACK_FINETUNE/tp_fp/`:

| file | contents |
|---|---|
| `tp_best.png` | highest-IoU TPs (0.856–0.859) |
| `tp_marginal.png` | TPs closest to the 0.5 boundary (0.5008–0.5039) |
| `fp_split.png` | FPs overlapping a GT (splits / merges) |
| `fp_pure.png` | FPs with no GT overlap |
| `fp_inspect.png` | pure FPs with the disk edge drawn |

### TP quality is bimodal

Median TP IoU is **0.681**. The best TPs are clean (0.86), but the marginal ones
sit at 0.5008–0.5039 — i.e. a cluster of instances that barely clear the 0.5
threshold. Those are worth ~0.5 TP each and are the most fragile part of the
score: a small SQ regression would flip them to FP+FN.

### FPs split roughly 50/50

| kind | count | median area |
|---|---|---|
| split/merge (best IoU > 0.1) | 346 | — |
| pure (no GT overlap) | 340 | 774 px |

The split/merge half is the known failure from `docs/PQ_DIAGNOSIS.md`: the model
finds a filament but fragments it, so one GT becomes two predictions. In
`fp_split.png` the GT (green) is a single elongated structure while the
prediction (red) is the same structure broken into pieces — and the probability
map shows a *continuous* ridge, so the split is happening in
`prob_to_labelmap`'s connected-components step, not in the network.

**That is the single most actionable finding here.** The network's probability
map is better than its instance output. `prob_to_labelmap_watershed` in
`src/solar/postprocess.py` exists precisely for this and has never been run on
this model.

### The pure FPs are interior, not limb

I initially read `fp_pure.png` as showing limb arcs. That was wrong, and the
per-instance measurement says so:

| measurement | value |
|---|---|
| pure FPs analysed | 340 |
| within 40 px of disk edge | **0 (0.0%)** |
| median signed distance from edge | **−268 px** (inside) |
| median area | 774 px |

`fp_inspect.png` draws the disk edge in yellow: the red FPs sit clearly inside
it. They are faint interior structures the annotators did not label — plausible
as real but sub-threshold filaments, or as noise the model over-fires on.

Two process notes, since both cost time:

* The `limb-ring mean prob` metric in §6 is a ring *average* and cannot detect a
  localized arc. It was the wrong statistic for the claim I made from it.
* My first radial script used one shared `matched` set for both prediction and GT
  ids. Since both are 0-based, a matched GT id falsely marked the prediction with
  the same index as matched, which silently dropped 112 of 340 FPs. Fixed; the
  corrected script reports all 340.

### What this implies for the next step

1. **Watershed post-processing** — the split FPs are a post-processing artefact,
   not a model failure. Cheapest available win.
2. **Raise `min_area` selectively** — pure FPs have median area 774 px, close to
   the 512 threshold. A higher threshold trades FN for FP; worth sweeping now
   that the FP area distribution is known.
3. **Do not chase the limb.** It is already clean; effort there is wasted.

---

## 11. Post-processing sweep

```bash
python scripts/deepcrack_cache_probs.py
python scripts/deepcrack_split_diagnosis.py
python scripts/deepcrack_postprocess_sweep.py --method cc \
    --thresholds 0.35,0.4,0.45,0.5 --min-areas 512,768 --close-ks 0,3,5,7,9,11,15
python scripts/deepcrack_postprocess_ci.py
```

### Watershed was the wrong tool — and the diagnosis says why

§10 suggested watershed, on the reasoning that split FPs are merged blobs. That
reasoning was **backwards**, and measuring it first saved a long sweep.

`deepcrack_split_diagnosis.py` separates the two possible failure modes:

| failure | definition | count |
|---|---|---|
| **fragmentation** | one GT covered by >1 predicted component | **100** (mean 2.21 components, max 4) |
| merging | one prediction overlapping >1 GT | 6 (mean 2.00) |

Fragmentation outnumbers merging **17:1**. The model is not merging filaments —
it is breaking them into pieces. Watershed *splits* connected blobs, so it can
only make this worse. The correct operation is the opposite: **morphological
closing**, to join fragments before labelling.

This also explains the `fp_split.png` figure: the probability ridge is
continuous, but the thresholded mask is not, so connected components emits
several instances for one filament.

A watershed sweep was started and killed after 5 minutes with no config
completed — `h_maxima=4` costs ~19 s/image at 2048², so the 72-config grid would
have taken hours. It was not worth finishing once the diagnosis showed the
direction was wrong.

### Closing works

Sweep over threshold × min_area × closing kernel (56 configs, 523 s):

| thr | min_area | close_k | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **0.45** | **512** | **15** | **0.3647** | 0.6654 | 0.5480 | 656 | 710 | 372 |
| 0.48 | 512 | 17 | 0.3646 | 0.6657 | 0.5477 | 646 | 685 | 382 |
| 0.45 | 512 | 11 | 0.3622 | 0.6675 | 0.5426 | 649 | 715 | 379 |
| 0.50 | 512 | 15 | 0.3591 | 0.6693 | 0.5365 | 628 | 685 | 400 |
| 0.50 | 512 | 0 | 0.3527 | 0.6763 | 0.5216 | 605 | 687 | 423 |

Paired bootstrap over images (2000 resamples), against the `close_k=0` baseline:

| config | Δ PQ | 95% CI | P(Δ>0) |
|---|---|---|---|
| thr=0.45, close=15 | **+0.0121** | [+0.0019, +0.0224] | 0.989 |
| thr=0.48, close=17 | **+0.0117** | [+0.0031, +0.0206] | 0.996 |

Both are real. The gain is **RQ 0.5216 → 0.5480** with SQ slightly down
(0.6763 → 0.6654): closing joins fragments, converting FP+FN pairs into TPs, at
the cost of slightly fatter boundaries. That is exactly the trade the diagnosis
predicted.

`min_area=512` wins at every threshold; 1024 and 2048 lose 0.05–0.19 PQ by
discarding real instances.

### Revised standing

| method | PQ |
|---|---|
| DeepCrack zero-shot | 0.000 |
| U-Net from scratch, 60 images | 0.277 |
| DeepCrack finetuned, no closing | 0.353 |
| **DeepCrack finetuned + closing (thr=0.45, k=15)** | **0.365** |
| DL baseline (resnet34, 512², 922 images) | 0.36 |
| DL tiled 1024² | 0.39 |

The post-processing change alone moves this **above** the resnet34 baseline,
with no retraining. Caveat: the baseline's 0.36 was measured without this sweep,
so the comparison is not like-for-like — the same closing may help it too.

### Caveat on the sweep

`close_k=15` and `thr=0.45` were selected on the same 142 images used to report
the gain, so the +0.012 is mildly optimistic (selection on the eval set). The
paired CI accounts for sampling noise but not for selection. A held-out split
would give the honest number; the effect is large enough (P>0.98) that it is
unlikely to vanish, but the true value is probably nearer +0.008 than +0.012.

---

## 12. Per-instance dumps

Every TP and FP is written as an individual 3-panel PNG (input | GT | prediction)
so failures can be inspected one at a time rather than through aggregate metrics.

```bash
python scripts/deepcrack_dump_tps.py
python scripts/deepcrack_dump_fps.py
```

Both default to the shipped config (`thr=0.45, min_area=512, close_k=15`), so the
two dumps are directly comparable. Counts reconcile exactly with the sweep:
**TP 656, FP 710, FN 372**.

### True positives — `docs/DEEPCRACK_FINETUNE/tp_dump/`

Bucketed by match quality, because the three groups fail differently:

| bucket | IoU range | count | what it means |
|---|---|---|---|
| `strong` | ≥ 0.75 | 128 | prediction essentially *is* the annotation |
| `good` | 0.60–0.75 | 351 | correct detection, boundary sloppiness |
| `marginal` | 0.50–0.60 | 177 | barely passed; most fragile to any SQ change |

`marginal` is 27% of all TPs. Those instances contribute ~0.5 TP each and would
flip to FP+FN under a small regression, so they are the ones to watch.

### False positives — `docs/DEEPCRACK_FINETUNE/fp_dump/`

| bucket | definition | count |
|---|---|---|
| `partial` | overlaps some GT (best IoU > 0.1) | 308 |
| `full` | no GT overlap at all | 402 |

Note the split moved relative to §10 (346/340 → 308/402) because §10 used
`thr=0.5, close_k=0` while this uses the shipped `thr=0.45, close_k=15`. Closing
joins fragments, so fewer FPs overlap a GT and more stand alone.

### Manifest and contact sheets

Each dump has a `manifest.csv` with per-instance metadata (`file`, `pred_id`,
`gt_id`, `iou`, `area`, `gt_area`, `n_pred`, `n_gt`, `image`) and paginated
contact sheets (`sheet_<bucket>_pNN.png`, 12 instances per page).

### Bug fixed: overlays were rendering blue

Both dump scripts built the overlay with `cv2.cvtColor(..., COLOR_GRAY2RGB)` and
then wrote it with `cv2.imwrite`, which expects **BGR**. Red and blue were
silently swapped, so the prediction overlay rendered blue instead of red. Fixed
to `COLOR_GRAY2BGR` with the red channel last. Verified by sampling saturated
pixels: mean BGR went from `[124, 134, 59]` (blue) to `[59, 134, 124]` (red).

This affected the earlier `fp_dump` too, so any visual read of those images
before this fix had the colours inverted.

---

## 13. Soft labels (mean over annotators) — no significant gain

Motivated by `docs/CORRECTIONS.md` §2 and by the finding in §12 that ~33% of
"full FPs" are filaments another annotator *did* label. Training on one
annotator's mask teaches the model to suppress structures a different annotator
considered real. Averaging over annotators removes that conflict.

```bash
python -m solar.train --config configs/deepcrack_ft_soft.json
```

### Implementation

`PositiveCropDataset(soft_labels=True)`:

* **Target** = mean of every annotator's binary mask for the file, so a pixel
  labelled by 2 of 3 annotators is 0.667. Verified values are exactly
  {0, ⅓, ½, ⅔, 1}.
* **Dedup**: one sample stream per *file*, not per annotator variant. Without
  this a 3-annotator file would be seen 3× per epoch. Train streams drop
  922 → 565.
* **Positive sampling** draws from the union of all annotators' polygons, so
  crops are centred on what *anyone* marked.
* Losses needed no change — BCE and soft Dice both accept fractional targets
  (verified: finite gradients).

### Result

Warm-started from run 3, 25 epochs, `lr=1e-5`. Best epoch 16.

| model | post-processing | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| hard labels (run 3) | thr=0.45, k=15 | 0.3647 | 0.6654 | 0.5480 | 656 | 710 | 372 |
| **soft labels** | thr=0.50, k=15 | **0.3651** | 0.6677 | 0.5468 | 634 | 657 | 394 |

**Δ = +0.0005**, paired bootstrap 95% CI **[−0.0073, +0.0087]**, P(Δ>0) = 0.552.

That is indistinguishable from zero. The soft model trades TP for FP
(656→634 TP, 710→657 FP) — it is more conservative, which is the expected
direction, but the two effects cancel in PQ.

### Why it probably did not help

1. **Only 42% of train files have >1 annotator** (113 with 2, 122 with 3, 330
   with 1). For the 330 single-annotator files the soft target is identical to
   the hard one, so the change only touches 235 of 565 files.
2. **The model was already warm-started from a converged checkpoint.** 25 epochs
   at `lr=1e-5` is a small perturbation; the target change may need training from
   the pretrained init to matter.
3. **PQ is thresholded at IoU 0.5.** Softening the target mostly moves boundary
   pixels, which affects SQ (0.6654 → 0.6677, +0.002) far more than the
   instance-level TP/FP/FN counts that dominate PQ.

### Verdict

Soft labels are **not** a win on this setup, but they are also not a loss, and
they are strictly more principled than picking one annotator arbitrarily. Two
untested variants that might convert it into a real gain:

* Train from the pretrained init rather than warm-starting, so the target change
  shapes the whole optimisation.
* Use the soft target only for the *loss*, but keep hard labels for the
  positive-sampling centre, or vice versa.

Given the effect size, neither is high priority compared to the RQ-side work in
§9.

### Bug fixed along the way

`deepcrack_cache_probs.py` had hardcoded `--weights`/`--out`, so passing
arguments silently did nothing and the script overwrote
`runs/deepcrack_ft_256_long/val_probs` with a different model's probabilities.
Both caches were rebuilt afterwards and the GT arrays verified identical, so no
reported number was affected — but the script now takes real arguments.

---

## 14. False negatives — the side we had not looked at

```bash
python scripts/deepcrack_dump_fns.py
```

372 FNs at the shipped config, bucketed by each FN's **best IoU against any
prediction**. The three buckets need completely different fixes.

| bucket | best IoU | count | share | meaning |
|---|---|---|---|---|
| `missed` | = 0 | 144 | 38.7% | nothing predicted anywhere near it |
| `partial` | 0–0.3 | 56 | 15.1% | something predicted, barely overlaps |
| **`near_miss`** | **0.3–0.5** | **172** | **46.2%** | found and mostly right, just under threshold |

Counts reconcile with the sweep: TP 656, FP 710, FN 372.

### The near-misses are truncation, not detection failure

This is the most actionable finding. For the 172 near-misses, comparing each GT
to its nearest prediction:

| measurement | median | interpretation |
|---|---|---|
| area ratio (pred / GT) | **0.67** | prediction is 33% too small |
| centroid offset | **26 px** | noticeably displaced |
| GT coverage by nearest pred | 0.51 | only half the GT is covered |

| pattern | share |
|---|---|
| pred smaller than GT (ratio < 0.8) | **60.5%** |
| pred larger than GT (ratio > 1.25) | 25.0% |
| centroid offset > 20 px | 61.0% |
| coverage > 0.7 (mostly found) | 23.8% |

So the dominant near-miss is **the model finding the filament but predicting a
shorter/thinner version of it**, displaced by ~26 px. The contact sheets show
this directly: green GT and red prediction trace the same structure, but red
covers only part of it or sits slightly off.

This is the *opposite* of the FP-side problem. There, closing (dilation) helped
because fragments needed joining. Here the prediction is already too small, so
closing helps these too — which is consistent with closing's +0.012 gain coming
from both sides at once.

### The "missed" bucket is not simply faint filaments

I expected the 144 zero-IoU FNs to be low-contrast structures below the model's
detection threshold. The contrast measurement does not support that cleanly:

| group | n | median contrast | p25 | p75 |
|---|---|---|---|---|
| TP (matched) | 656 | −17.2 | −20.7 | −14.0 |
| FN missed (IoU=0) | 144 | −14.3 | −20.1 | −11.0 |
| FN partial/near | 228 | −12.7 | −16.0 | −10.5 |

Missed FNs are *less* dark than TPs (median −14.3 vs −17.2), which is the right
direction, but the distributions overlap almost completely at p25 (−20.1 vs
−20.7). Contrast alone does not separate them, so "the model has a contrast
floor" is not a sufficient explanation.

The contact sheets suggest a different factor: several missed FNs sit in
**crowded, high-texture regions** (near sunspots, in the plage-like areas visible
in rows 4–5 of `sheet_missed_p01.png`). That is consistent with the model being
distracted by surrounding structure rather than unable to see the filament.

### What this implies

1. **Near-misses (172) are the cheapest target.** They are a segmentation-extent
   problem, not a detection problem. Anything that grows predictions slightly —
   more closing, a lower threshold, or a Dice-heavy loss — should convert some.
   Note the tension: growing predictions also grows FPs, so this needs a joint
   sweep rather than a one-sided change.
2. **`missed` (144) needs a different lever.** Not contrast. Likely more
   capacity or better hard-example mining in textured regions.
3. **`partial` (56) is small** and probably shares the near-miss mechanism.

### Caveat

The bucket boundaries (0.3, 0.5) are arbitrary. The `near_miss`/`partial` split
at 0.3 is a judgement call; the robust facts are the 46% share above 0.3 and the
0.67 median area ratio.

---

## 15. Tracing a miss through the network — the encoder sees it, the threshold loses it

```bash
python scripts/deepcrack_trace_miss.py --file 20150627133754Bh --gt-id 5 --save
```

DeepCrack has a 1x1 side head at each of five depths (stride 1, 2, 4, 8, 16), all
upsampled to input resolution and fused by a 1x1 conv. That makes "where does the
signal fall off" directly measurable: compute each side output's AUC for the
missed instance against its local background ring.

### Single trace: `20150627133754Bh` gt_id=5 (area 3380)

| side output | mean in GT | max in GT | mean ring | contrast | AUC | frac > 0.45 |
|---|---|---|---|---|---|---|
| side1 (stride 1) | 0.133 | 0.420 | 0.106 | +0.027 | 0.614 | 0.0% |
| side2 (stride 2) | 0.319 | 0.997 | 0.203 | +0.115 | 0.551 | 31.0% |
| side3 (stride 4) | 0.551 | 1.000 | 0.195 | +0.357 | 0.777 | 55.1% |
| **side4 (stride 8)** | 0.473 | 0.958 | 0.133 | +0.341 | **0.839** | 48.8% |
| side5 (stride 16) | 0.017 | 0.085 | 0.005 | +0.012 | 0.717 | 0.0% |
| **fused** | 0.270 | 0.994 | 0.018 | +0.252 | **0.875** | **26.7%** |

The fused output has **AUC 0.875** for this instance — it ranks the filament above
its background almost perfectly. Yet only **26.7%** of the filament's pixels
exceed the 0.45 threshold, so the instance never forms a component large enough
to survive `min_area=512`.

The figure (`docs/DEEPCRACK_FINETUNE/trace/20150627133754Bh_gt5.png`) shows why:
the fused map lights up the **top and bottom of the filament but not the middle**.
The GT is a long vertical structure; the prediction is two disconnected blobs at
its ends. That is exactly the fragmentation pattern from §11, seen at the level
of a single instance.

### This generalises — and it is not a detection failure

Same measurement over 40 randomly sampled missed FNs:

| side output | median AUC | frac AUC > 0.7 | frac AUC < 0.6 |
|---|---|---|---|
| side1 (stride 1) | 0.865 | 97.5% | 2.5% |
| side2 (stride 2) | 0.887 | 90.0% | 0.0% |
| **side3 (stride 4)** | **0.970** | **100.0%** | 0.0% |
| side4 (stride 8) | 0.848 | 95.0% | 2.5% |
| side5 (stride 16) | 0.704 | 50.0% | 35.0% |
| **fused** | **0.981** | **97.5%** | 0.0% |

**Median fused AUC is 0.981.** For 97.5% of missed FNs the fused output ranks the
filament above its surroundings with AUC > 0.7. These are not structures the
network fails to see.

### What this changes

The §14 conclusion — "`missed` (144) needs more capacity or hard-example mining"
— was **wrong**. The encoder represents these filaments with near-perfect
separability. The loss happens downstream, in the step from a probability map to
a thresholded instance:

1. **The probability is not confident enough.** Mean probability inside the GT is
   0.27 while the ring is 0.018 — a large *relative* gap but a small *absolute*
   one. A fixed 0.45 threshold sits above most of the filament.
2. **The response is spatially incomplete.** Even where the model is confident,
   it covers only the ends of a long filament, so the component is split and each
   piece may fall under `min_area`.

Both are post-processing problems, not model-capacity problems. The obvious
levers, in order of expected value:

* **Lower the threshold for these cases.** At thr=0.45 the fused mean is 0.27, so
  a threshold near 0.2 would capture much more. The global sweep already showed
  thr=0.45 beating 0.5, consistent with this.
* **Hysteresis thresholding** — seed at a high threshold, grow at a low one. This
  is the standard fix for exactly this failure (confident core, weak extent) and
  is not yet implemented. `scripts/eval_hyst.py` exists in the repo but has never
  been applied to DeepCrack.
* **`min_area` interacts** — a fragmented filament produces pieces below 512 px
  that get discarded. Lowering `min_area` recovers them but admits noise FPs.

### Caveat

The AUC is computed against a 41 px ring around the GT, which is a *local*
comparison. A high local AUC does not mean the model would rank this filament
above the whole image — only above its immediate surroundings. That is the right
comparison for "is the signal there", but it does not by itself prove a global
threshold change would work.

The 40-instance sample is also small; the median AUCs are stable enough to trust
in direction, but the exact fractions are not precise.

---

## 16. Deep supervision on the side heads — improved the sides, hurt the output

```bash
python -m solar.train --config configs/deepcrack_ft_ds.json
```

§15 motivated this: `fuse_conv` is a bias-free 1x1 conv, so fusion is the plain
linear sum `fused = Σ wᵢ·sideᵢ`, and the five side heads are **never supervised**
in upstream training — they are shaped only by gradient flowing back through
those five weights. A side head that disagrees with the others can therefore only
dilute the result.

### Implementation

* `DeepCrack(deep_supervision=True)` returns
  `{"logits": fused, "sides": [side1..side5]}` instead of a bare tensor. A dict,
  not a tuple, so it cannot be confused with the `(logits, spine_logits)`
  convention used by the timm models.
* `train.py`: `loss = seg_loss(fused) + side_weight * mean(seg_loss(side_i))`.
  **Averaged**, not summed, so `side_weight` does not scale with head count.
* `validate()` and `_predict_tiled()` unwrap the dict and score only the fused
  map, so checkpoint selection is unchanged.
* Verified: side-loss-only backward produces non-zero gradient on all five
  `side_conv*` heads, and all sides come out at input resolution.

### Result — it made things worse

25 epochs, warm-started from run 3, `lr=1e-5`, `side_weight=0.5`. Best epoch 14.

| model | post-processing | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| hard baseline (run 3) | thr=0.45, k=15 | **0.3647** | 0.6654 | 0.5480 | 656 | 710 | 372 |
| deep supervision | thr=0.45, k=15 | 0.3394 | 0.6665 | 0.5093 | 575 | 655 | 453 |

**Δ = −0.0253**, paired bootstrap 95% CI **[−0.0373, −0.0137]**, P(Δ>0) = 0.000.
A clear, unambiguous regression. The loss is in recall: TP 656→575, FN 372→453.

### But the side heads did improve — a lot

Median per-instance AUC over 20 images (local ring comparison):

| side output | baseline | deep supervision |
|---|---|---|
| side1 (stride 1) | 0.831 | 0.844 |
| side2 (stride 2) | 0.838 | **0.901** |
| side3 (stride 4) | 0.946 | 0.955 |
| side4 (stride 8) | 0.868 | **0.929** |
| side5 (stride 16) | 0.721 | **0.895** |
| **fused** | **0.970** | **0.964** |

Every side head improved, side5 dramatically (0.721 → 0.895 — it was the weakest
and got the most direct supervision). Yet the **fused** output got slightly worse.

### Why supervision of the parts hurt the whole

This is the interesting part, and it inverts the intuition that went into the
change.

Deep supervision makes each side head independently good *for its own depth*.
But the side heads sit at stride 1, 2, 4, 8, 16, so they see the same filament
with very different receptive fields and produce systematically different
extents. Before, gradient through `fuse_conv` was free to *cancel* that bias — it
could learn weights that suppressed whatever each head got wrong. Once each head
is pushed hard toward the same target, their errors become correlated (they all
over-predict the confident core and under-predict the weak extent, in the same
places), so the linear sum accumulates the same mistake five times instead of
averaging it away.

The evidence fits: RQ fell (0.548 → 0.509) while SQ was flat (0.665 → 0.667).
Supervision changed *which instances form*, not how well they are delineated.

Also note the fusion weights are frozen during this run in effect — `side_weight`
changes the side heads' behaviour, and `fuse_conv` must then adapt to five
*moving* targets. That is a harder optimisation than training the fused output
against a fixed-loss side stack.

### Verdict

**Deep supervision as implemented is a regression and should not be used.**
`configs/deepcrack_ft_ds.json` is kept for the record, not as a recommendation.

The mechanism suggests the opposite direction would be more promising: rather
than making the side heads individually accurate, *reduce* the number of fused
heads. If side5 (stride 16, weakest) is mostly adding correlated error, dropping
it from the fusion — or refitting only the five `fuse_conv` weights on a frozen
encoder — is a much smaller change with a clearer rationale. Neither is tested.

### Caveats

* One `side_weight` (0.5) at one LR. A weaker weight, or supervision from the
  pretrained init rather than warm-starting, might not regress. But the sign is
  consistent with the mechanism above, so I would not expect it to become a gain.
* The per-side AUC table is 20 images, ~140 instances. Direction is solid
  (side5 nearly +0.17), exact values are not.

---

## 17. Bug: PQAccumulator was matching at the wrong IoU threshold

Found while cross-checking two numbers that should have been identical: the
in-training eval reported PQ 0.3542 for the deep-supervision run, while the
post-processing sweep on cached probabilities from the *same* checkpoint gave
0.3322.

**Cause.** `PQAccumulator.threshold` is the **IoU matching threshold** (0.5 in
the official protocol), not a probability threshold. `train.py` passed
`cfg["eval"]["threshold"]` — the *probability* threshold, 0.45 — so every
in-training evaluation matched instances at **IoU > 0.45 instead of 0.5**.

| computation on the same cached probs | PQ |
|---|---|
| `PQAccumulator(threshold=0.5)` | 0.3322 |
| `PQAccumulator(threshold=0.45)` | 0.3543 |

The cache was verified fresh (live inference matches it to 2.4e-4), so this was
purely the metric.

**Scope.** Every `scripts/` file already used `threshold=0.5`; only `train.py`
was wrong. So all *reported* numbers in this document (which come from the
sweep) are correct, but **every `best.pt` in the repo was selected on the looser
metric**. Fixed in `train.py`, and the docstring on `PQAccumulator` now states
what the field means.

**Consequence.** Checkpoint selection was systematically biased toward models
that produce more marginal matches. The direction of each comparison is probably
unaffected (all runs shared the bias), but individual best epochs may not be the
true best under the correct metric.

---

## 18. From-scratch training — pretraining was not biasing us

All prior runs warm-started from `pretrained_net_G.pth` (asphalt crack weights).
To test whether that was helping or biasing, this run starts from **random
init**: `pretrained=false`, no `init_ckpt`, 60 epochs, `lr=3e-4`, 2 warmup
epochs. Verified random: `conv1` differs from the released weights, weight
std 0.0012.

```bash
python -m solar.train --config configs/deepcrack_scratch.json
```

### Learning curve

| epoch | loss | PQ | SQ | RQ |
|---|---|---|---|---|
| 1 | 5.30 | 0.054 | 0.596 | 0.091 |
| 5 | 0.47 | 0.262 | 0.642 | 0.409 |
| 10 | 0.41 | 0.300 | 0.650 | 0.462 |
| 20 | 0.37 | 0.325 | 0.657 | 0.494 |
| **34** | 0.35 | **0.351** | 0.670 | 0.524 |
| 45 | 0.33 | 0.347 | 0.669 | 0.519 |
| 60 | 0.33 | 0.345 | 0.668 | 0.517 |

It passes the from-scratch U-Net baseline (0.277) by epoch 9 and plateaus around
epoch 34. Loss 5.30 → 0.33; the first two epochs are spent escaping the random
initialisation (PQ 0.054 → 0.210 by epoch 3).

### Head-to-head, same post-processing (thr=0.45, close_k=15)

| init | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| pretrained (asphalt) | **0.3647** | 0.6654 | 0.5480 | 656 | 710 | 372 |
| **from scratch** | **0.3594** | 0.6631 | 0.5420 | 629 | 664 | 399 |

**Δ = −0.0052**, paired bootstrap 95% CI **[−0.0148, +0.0045]**, P(Δ>0) = 0.147.

**Not significant.** The pretrained weights are worth at most ~0.005 PQ here, and
the CI includes zero.

### What this means

1. **Pretraining was not biasing the results.** The asphalt-crack prior gives no
   meaningful advantage on filaments. This is consistent with the zero-shot
   probe (§1): the pretrained network scored PQ 0.0 out of the box, so whatever
   it contributed had to be relearned anyway.
2. **The architecture, not the init, is doing the work.** 14.7 M params trained
   from random on 565 files reaches the same place as the pretrained version.
3. **Practical implication:** the external dependency on
   `pretrained_net_G.pth` (a Google Drive download) can be dropped. That removes
   a fragile setup step. The network code is now vendored in
   `src/solar/deepcrack_net.py`, so the `/tmp/DeepSegmentor` clone is no longer
   needed either.
4. **Cost:** 60 epochs from scratch (~55 min) versus 47 epochs warm-started
   (~50 min) for the same result. Pretraining saves essentially nothing in wall
   time either, because the warm-started runs also needed many epochs to adapt.

### Caveats

* One seed, one LR schedule. From-scratch runs are more sensitive to
  initialisation and LR than warm-started ones, so a different seed could move
  this by more than the 0.005 gap.
* The from-scratch run used `lr=3e-4` with 2 warmup epochs; the warm-started runs
  used `lr=1e-5`. These are not matched, and the comparison is between *recipes*,
  not just initialisations.
* Both models were selected on the buggy IoU>0.45 metric (§17), so neither
  `best.pt` is guaranteed optimal under the correct metric.

---

## 19. Grayscale stem and vendored network

Two cleanups that follow from §18. Since pretraining buys nothing, the RGB input
it forced on us is pure overhead, and the external clone it required is pure
fragility.

### 19.1 Dropping the RGB repeat

The released weights have a 3-channel `conv1` because DeepCrack was trained on
RGB asphalt. The wrapper fed grayscale by repeating it to 3 channels, so the
first layer computed

$$\text{conv}_3([x, x, x]) = \sum_{c=1}^{3} W_{:,c} * x = \Big(\sum_{c=1}^{3} W_{:,c}\Big) * x$$

which is exactly a 1-channel convolution whose weights are the channel-sum of
the 3-channel ones. So the fold

```python
conv1_1ch.weight = conv1_3ch.weight.sum(dim=1, keepdim=True)   # (64,3,3,3) -> (64,1,3,3)
```

is not an approximation — it is the same function. Verified:

| Check | Result |
|---|---|
| Synthetic conv, repeat vs folded | max abs diff **8.3e-07** |
| Full network, pretrained weights | max abs diff **5.0e-06** |
| Old checkpoint through new path | max abs diff **6.1e-05** |
| Binarized pixels differing at thr=0.45, 8 real 2048² images | **0 / 33,554,432** |

The last row is the one that matters: the change cannot move the metric, because
not a single pixel flips across the threshold. Re-scoring
`runs/deepcrack_ft_256_long/best.pt` on the full val split gives **0.3521** vs the
recorded **0.3531** — a 0.001 gap from float32 reassociation in the fold, well
inside the ±0.005 run-to-run noise already documented.

What it buys: `conv1` drops from 1728 to 576 weights, and the first layer no
longer allocates a 3× copy of a 2048² image.

`DeepCrack.load_state_dict` applies the fold automatically whenever a 3-channel
`conv1` meets a 1-channel stem, so **every existing checkpoint in `runs/` still
loads unchanged** — no migration needed.

### 19.2 Vendoring the network

`src/solar/deepcrack_net.py` now holds `DeepCrackNet`, `define_deepcrack`, and the
two helpers it needed from upstream's `models/networks.py` (`get_norm_layer`,
`init_net`). The `/tmp/DeepSegmentor` clone is gone from the import path, so the
model survives a reboot.

Parity with upstream was checked directly: same 84 state-dict keys, and
bit-identical weights when both are built under the same seed. The saved
zero-shot probability maps in `docs/DEEPCRACK/*_prob.npy` reproduce to
`max|diff| < 1.2e-03` (JPEG decode + float32 reassociation), with max
probabilities matching to 4 decimals.

Layer names are load-bearing — `conv1.0.weight`, `side_conv1.weight`,
`fuse_conv.weight` — because the released weights and every checkpoint are keyed
by them. Do not rename or reorder.

### 19.3 Config

`configs/deepcrack_gray.json` is the from-scratch grayscale recipe: identical to
`deepcrack_scratch.json` except for the run name. Note that
`deepcrack_scratch.json` already declared `in_chans: 1` — before this change that
value only controlled the repeat, so it was silently running a 3-channel stem.
It now means what it says.

---

## 20. Why PQ stalls at ~0.35: the model localises, it does not segment

Four questions were asked: is the model saturated, does it need more/fewer
layers, is the fusion broken, and should the fusion have more parameters. All
four now have measured answers, and they point at one cause.

### 20.1 The fusion is not broken — it is the only thing working

`fuse_conv` is a bias-free 1x1 conv over the five side logits, so the fused map
is a fixed linear combination of them. Scoring each head separately
(`scripts/deepcrack_fusion_audit.py`, 142 val files):

| head | stride | AUC | best PQ |
|---|---|---|---|
| side1 | 1 | 0.6106 | 0.0000 |
| side2 | 2 | 0.8539 | 0.0671 |
| side3 | 4 | 0.9570 | 0.0802 |
| side4 | 8 | 0.9782 | 0.0262 |
| side5 | 16 | 0.6750 | 0.0000 |
| **fused** | 1 | **0.9968** | **0.3521** |

The fused map beats every individual head by a wide margin (0.3521 vs 0.0802).
Fusion is doing real work, not destroying information.

**More fusion parameters would not help.** Fitting an oracle fusion on the val
set — non-negative least squares on the side logits, the same functional form as
`fuse_conv` — gives weights `[0.000, 0.000, 0.021, 0.000, 0.000]`, i.e. it
collapses onto side3 alone and scores **PQ 0.0000**. A logistic regression with
per-head weights *and* a bias scores **0.0254**. Both are far worse than the
learned `fuse_conv` at 0.3521.

The reason is that the side heads are individually miscalibrated in different
directions, and only a *jointly trained* linear combination cancels that. The
learned weights `[0.178, 0.328, 0.425, 0.428, 0.260]` (sum 1.62) are not
recoverable by fitting on the outputs, because the fit sees each head's
miscalibration as signal. Adding parameters to the fusion would give it more
freedom to fit that noise.

### 20.2 The model is not saturated — the loss is

Training loss on the 160-epoch run is still falling while PQ is flat:

| epochs | mean train loss | best PQ |
|---|---|---|
| 21–40 | 0.1688 | 0.4038 |
| 141–160 | 0.1438 | 0.4301 |

Loss dropped 15% over 120 epochs for +0.026 PQ. The network is still finding
signal; the BCE+Dice objective has stopped being a good proxy for PQ. This is
objective saturation, not capacity saturation.

### 20.3 The real bottleneck: ranking is near-perfect, localisation is not

> **Superseded in part by §21.3.** The "shape headroom 0.4859" below is measured
> against a *self-consistent* mask. Against a *different human*, SQ 0.70 is
> already above the human-to-human line (0.6505), and the PQ arithmetic shows SQ
> would need to exceed 1.0 to reach 0.55. The real lever is RQ (instance
> bookkeeping), not shape. Read this section for the AUC/IoU measurement, not for
> the headroom conclusion.

The fused map has **pixel AUC 0.9968** — it ranks filament pixels above
background almost perfectly. Yet pixel IoU is **0.4987**. Those two facts are
only compatible if the probability map is a spatially *blurred* version of the
filament.

`scripts/deepcrack_head_precision.py` separates the two failure modes:

| | value |
|---|---|
| fused AUC (ranking) | 0.9968 |
| fused best IoU (localisation) | 0.4987 |
| IoU with an oracle per-image threshold | 0.5141 |
| **calibration headroom** | **+0.0154** |
| **shape headroom** | **0.4859** |

Threshold choice accounts for 3% of the gap. **97% is shape.** No amount of
threshold tuning, TTA, or post-processing can recover it — which is exactly what
the post-processing sweeps already found (max +0.012).

The error anatomy confirms the geometry: at thr=0.45, 25% of misclassified
pixels sit exactly on the GT boundary and 44% within 2 px. The model is drawing
the right filament a couple of pixels too fat or too thin.

### 20.4 Why: DeepCrack has no decoder

This is the architectural cause. DeepCrack takes a side head at each encoder
stage and bilinearly upsamples it straight to full resolution. There is no
decoder, no skip connections, no refinement at high resolution.

So the heads that rank best are the ones that localise worst:

| head | stride | AUC | IoU |
|---|---|---|---|
| side3 | 4 | 0.9570 | 0.2383 |
| side4 | 8 | **0.9782** | 0.2139 |
| side1 | 1 | 0.6106 | 0.0052 |

side4 has the best ranking of any single head but sits at stride 8 — its
predictions are bilinear interpolations of an 8x-downsampled map, so they cannot
have sharp edges. side1 is at full resolution but only two convs deep, so it has
almost no semantics (AUC 0.61).

**The architecture forces a choice between semantics and resolution, and the
fusion cannot recover what neither head has.** That is the whole story.

This also explains the cross-architecture comparison, which was previously
attributed to "DeepCrack is not worse, just unresolved":

| model | decoder | best PQ |
|---|---|---|
| DeepCrack (finetuned) | **no** | 0.3531 |
| DeepCrack (scratch) | **no** | 0.3507 |
| resnet34 UNet | yes | **0.4361** |
| resnet34 UNet, tiled | yes | 0.4343 |

The UNet is +0.083 ahead, and it is the only structural difference. The earlier
reading ("not resolved at this sample size") was too generous: the gap is
consistent across two independent UNet runs and matches the mechanism measured
here.

### 20.5 Answers to the four questions

| question | answer | evidence |
|---|---|---|
| Is the model saturated? | **No.** Loss still falling; AUC 0.9968. | §20.2 |
| More or fewer layers? | **Neither.** The encoder is fine; it lacks a decoder. | §20.4 |
| Is the fusion broken? | **No.** It is the best-performing component. | §20.1 |
| More parameters around the fusion? | **No.** Oracle fits score 0.00–0.03 vs 0.35. | §20.1 |

### 20.6 What would actually move PQ

Ranked by expected value, given the measurements above:

1. **Add a decoder with skip connections.** This is the one change that
   addresses the 97% shape headroom. Concretely: keep the five encoder stages,
   but instead of `side_convN -> upsample -> concat -> 1x1`, do
   `upsample + concat with the previous stage -> 3x3 conv` at each level, so
   high-resolution detail from `conv1`/`conv2` reaches the output. The UNet
   already in this repo is the existence proof (+0.083).
2. **Supervise at full resolution.** The current loss is computed on the fused
   map only, which is a linear combination of blurred inputs. A deep-supervision
   variant that weights side1/side2 more heavily would push gradient into the
   high-resolution path — but §16 showed naive deep supervision *hurt*
   (−0.025), so this needs the decoder first, not instead.
3. **Stop tuning post-processing.** 97% of the gap is shape; the sweeps have
   already found the +0.012 that is available.

What will *not* help, now measured rather than assumed: more fusion parameters,
more encoder layers, longer training, threshold tuning, TTA, and watershed.

### 20.7 Caveats

* The oracle fusion fits are trained and scored on the same val split, so they
  are optimistic — and they still lose badly. The conclusion is safe in the
  direction that matters.
* The overfit test (8 images, 120 epochs) reached only PQ 0.24, but at 960 total
  steps that is a step-count limit, not a capacity limit, so it is not evidence
  either way and is not used above.
* AUC is computed on a 200k-pixel subsample per image (standard error < 0.001);
  full 2048² sorts are CPU-bound and dominated the runtime.
* `scripts/deepcrack_fusion_audit.py` had a bug on first run: `scores[labels]`
  with a uint8 label array does *integer* indexing, not boolean masking, which
  produced AUC 0.43 for a model with PQ 0.35. Fixed by casting to bool; the
  numbers above are post-fix.

---

## 21. Adding a decoder

§20 concluded that 97% of the PQ gap is *shape*, and that the cause is
DeepCrack's lack of a decoder. This section implements one and reports what it
does.

### 21.1 What was added

`DeepCrackNet(decoder=True)` keeps the five encoder stages and the five side
heads, but replaces the fused output with a UNet-style decoder:

```
conv5 (stride 16) ──up──┐
                        concat ── up4 (3x3, 3x3) ──┐
conv4 (stride 8) ───────┘                          │
                        concat ── up3 (3x3, 3x3) ──┤
conv3 (stride 4) ──────────────────────────────────┘
                        concat ── up2 (3x3, 3x3) ──┐
conv2 (stride 2) ──────────────────────────────────┘
                        concat ── up1 (3x3, 3x3) ──┐
conv1 (stride 1) ──────────────────────────────────┘
                        head (1x1) -> logits
```

Each `upN` is two 3x3 convs with norm+ReLU, matching the resnet34 UNet decoder
already in this repo. The side heads are still built (1x1 convs, ~5 params each)
so `deep_supervision=True` keeps working, but `fused` now comes from the decoder.

| | params | state-dict keys |
|---|---|---|
| `decoder=False` (upstream) | 14.72 M | 84 |
| `decoder=True` | 24.90 M | 134 |

`decoder=False` is **bit-identical to upstream** (verified against the clone:
same 84 keys, same weights under the same seed), so every checkpoint in `runs/`
still loads. `pretrained=True` with `decoder=True` raises a `ValueError`, because
the released weights contain no decoder parameters.

### 21.2 A bug worth recording: the decoder was dead on arrival

The first run sat at **PQ 0.0000 with zero predictions for 14 epochs** while the
loss fell normally. The cause was not the architecture:

`init_net` applies `xavier_normal_(gain=0.02)` to *every* conv. That gain is the
pix2pix GAN-generator convention and is right for the encoder, but applied to a
four-block decoder it shrinks the output to nothing:

| decoder init | logit std | logit range | sigmoid range |
|---|---|---|---|
| gain 0.02 (inherited) | 0.0197 | [-0.08, 0.11] | [0.48, 0.53] |
| **gain 1.0 (fixed)** | **0.8098** | **[-2.61, 4.92]** | **[0.07, 0.99]** |

With gain 0.02 no pixel can ever cross a 0.5 threshold, so `prob_to_labelmap`
returns an empty mask and PQ is structurally 0 — the network is not slow, it is
mute. `define_deepcrack` now re-initialises the decoder blocks with `gain=1.0`
after `init_net` runs. The encoder is untouched.

This is a good example of why "PQ 0 with zero predictions" should be treated as a
bug signal rather than a slow start: the non-decoder scratch run reached PQ 0.21
by epoch 3, so 14 epochs of exactly zero was diagnostic.

### 21.3 Results

**The decoder helped: PQ 0.3521 → 0.3834 (+0.0313) on the full 142-file val set.**
The gain is entirely in RQ, not SQ — which is the opposite of what §20 predicted,
and it is the more useful outcome.

#### Final numbers (160 epochs, full val, n=142, thr=0.55, min_area=512)

| model | params | fused AUC | fused IoU | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| DeepCrack, no decoder | 14.72 M | 0.9968 | 0.4987 | 0.3521 | 0.6767 | 0.5218 | 605 | 686 | 423 |
| **DeepCrack + decoder** | 24.90 M | 0.9949 | **0.5248** | **0.3834** | 0.6733 | **0.5695** | 621 | 532 | 407 |
| resnet34 UNet (reference) | 24.4 M | — | — | 0.4152 | 0.6996 | 0.5934 | 586 | 361 | 442 |

Delta vs no-decoder: **PQ +0.0313, SQ −0.0034, RQ +0.0477, FP −154, FN −16.**

#### Where the gain came from: fewer false instances

The decoder did *not* improve shape. SQ is flat (0.6767 → 0.6733, inside noise)
and pixel IoU moved only +0.026. What changed is instance bookkeeping:

| | no decoder | decoder | delta |
|---|---|---|---|
| FP | 686 | 532 | **−154** |
| FN | 423 | 407 | −16 |
| FP/TP | 1.13 | **0.86** | −0.27 |
| RQ | 0.5218 | 0.5695 | **+0.0477** |

The no-decoder model produced more false instances than true ones (FP/TP 1.13).
The decoder cut that to 0.86. This is the §9 failure mode — splits of
already-matched GT — being partially fixed, presumably because the decoder's
higher-resolution output stops a single filament from fragmenting into several
blobs.

This also corrects my own mid-run read. At epoch 46 I reported FP/TP 1.25 and
concluded the decoder was making fragmentation *worse*. That was a 64-image
in-training subset at epoch 46; the final 142-image number is 0.86. The
preliminary conclusion was wrong and is retracted.

#### SQ did not move, and that is expected

| | PQ | SQ | RQ |
|---|---|---|---|
| human vs human (61 imgs, 2nd annotator) | 0.3719 | 0.6505 | 0.5718 |
| `long_160ep_512` (UNet, 142 imgs) | 0.4152 | 0.6996 | 0.5934 |
| `deepcrack_scratch` (no decoder) | 0.3507 | 0.6695 | 0.5239 |
| `deepcrack_decoder_512` | 0.3834 | 0.6733 | 0.5695 |

SQ 0.67–0.70 sits at or above the human-to-human SQ of 0.6505. The decoder's SQ
(0.6733) is indistinguishable from the no-decoder from-scratch run (0.6695).
That wall is annotation ambiguity, not capacity, so no architecture change will
move it. The decoder was worth building, but not for the reason §20 gave.

#### The arithmetic, updated

PQ = SQ × RQ. To reach PQ 0.55 from the decoder's numbers:

| | now | needed (other frozen) | delta |
|---|---|---|---|
| SQ (RQ frozen at 0.5695) | 0.6733 | 0.966 | +0.292 |
| RQ (SQ frozen at 0.6733) | 0.5695 | 0.817 | +0.247 |

SQ would need to reach 0.966 — essentially perfect, and far above the human
line. RQ needs +0.247, which is large but not impossible: the UNet already
reaches 0.5934, and the consensus-of-3 predictor reaches PQ 0.6087. **RQ remains
the only lever**, and the decoder's +0.048 RQ is a real step along it.

#### The comparison is still confounded

The decoder run uses `pretrained: false` (encoder from scratch), while the
no-decoder baselines use `pretrained: true`. So the +0.0313 mixes architecture
with initialisation. It is a *lower bound* on what the decoder can do, since the
decoder run had the weaker starting point — but it is not a clean measurement.

| run | pretrained encoder | decoder | patch | batch | lr | loss |
|---|---|---|---|---|---|---|
| `long_160ep_512` | **yes** | no | 512 | 12 | 3e-4 | bce_dice |
| `deepcrack_ft_256_long` | **yes** | no | 256 | 16 | 1e-5 | border_aware |
| `deepcrack_scratch` | no | no | 256 | 16 | 3e-4 | border_aware |
| `deepcrack_decoder_512` | no | **yes** | 512 | 12 | 3e-4 | bce_dice |

The clean test is `configs/deepcrack_decoder_pre.json` (pretrained encoder +
decoder). It is written and verified — 24.90 M params, all 84 encoder tensors
loaded, 42 decoder tensors at init, logits in [−2.26, 5.88] at init — but it is
**not yet trained**. Given the decoder's +0.031 from a random encoder, this is
the run most likely to beat the UNet's 0.4152.

#### Training curve

| epoch | loss | PQ | SQ | RQ |
|---|---|---|---|---|
| 1 | 0.6580 | 0.0000 | 0.0000 | 0.0000 |
| 21 | 0.2330 | 0.2305 | 0.6453 | 0.3571 |
| 41 | 0.1935 | 0.2863 | 0.6721 | 0.4259 |
| 61 | 0.1864 | 0.3209 | 0.6599 | 0.4863 |
| 81 | 0.1830 | 0.3504 | 0.6666 | 0.5257 |
| 101 | 0.1768 | 0.3695 | 0.6625 | 0.5577 |
| 131 | 0.1620 | 0.3862 | 0.6692 | 0.5770 |
| 160 | 0.1615 | 0.3744 | 0.6777 | 0.5525 |

Best in-training PQ 0.3896 at epoch 140 (64-image subset); final 0.3744. Loss was
still falling at epoch 160 (0.1620 → 0.1615), so this run is compute-limited, not
converged. A longer schedule would likely add a little more.

#### Cost

120 s/epoch vs 38.5 s for the UNet — 3.1× slower, 5.3 h for 160 epochs. The
decoder is 24.90 M params trained from scratch at 512², so most of that is the
missing ImageNet initialisation rather than the decoder blocks themselves.

#### Matched-epoch comparison (confounded)

| epoch | no-decoder PQ | SQ | RQ | decoder PQ | SQ | RQ |
|---|---|---|---|---|---|---|
| 10 | 0.3286 | 0.6808 | 0.4828 | 0.1926 | 0.6566 | 0.2934 |
| 20 | 0.3504 | 0.6905 | 0.5074 | 0.2119 | 0.6217 | 0.3409 |
| 30 | 0.3759 | 0.6929 | 0.5425 | 0.2147 | 0.6538 | 0.3283 |
| 38 | 0.3878 | 0.6938 | 0.5589 | 0.3059 | 0.6680 | 0.4579 |

The decoder trails at every matched epoch, but it starts from a random encoder
and the no-decoder run from pretrained weights, so this is expected and says
nothing about the architecture. The decoder closes the gap by epoch 160 and ends
ahead on the metric that matters (RQ).

### 21.4 Reproducing

```bash
python -m solar.train --config configs/deepcrack_decoder.json
python scripts/deepcrack_eval.py --weights runs/deepcrack_decoder_512/best.pt \
    --n 142 --mean 0.0 --std 1.0 --thresholds 0.3 0.45 0.55 0.7 --min-areas 512
python scripts/deepcrack_head_precision.py \
    --weights runs/deepcrack_decoder_512/best.pt --n 142
```





