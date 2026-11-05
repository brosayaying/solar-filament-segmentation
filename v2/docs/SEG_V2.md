# Phase 1 — 5-fold segmentation model v2

Status: **complete 2026-10-31 00:37**. Result: **null — no significant gain over
the resnet34 baseline.** See [Result](#result) below.

## Configuration

| | value |
|---|---|
| backbone | `convnext_small.fb_in22k_ft_in1k` (IN-22k) |
| `stem_ch` | 32 (stride-1 path) |
| `in_chans` | 1 |
| patch / batch | 512² / 12 |
| preprocess | `flatten` (cached, 707/707 train + 180/180 test) |
| `mean` / `std` | 0.4254 / 0.0809 (flattened disk-only) |
| AMP | **bf16** |
| epochs | 160, early stop patience 30, `min_delta` 0.002 |
| lr / wd | 3e-4 / 1e-4, cosine, 2 warm-up |
| loss | BCE + Dice |
| val | full fold, every epoch |
| split | `v2/splits/week5_seed42/fold{0..4}.json` |

### Why flattening, given the A/B was inconclusive

The fold-0 A/B ([`AB_FLATTEN.md`](AB_FLATTEN.md)) could not resolve the effect —
one fold is underpowered (224 images vs the 428 needed for 80 % power at 0.015).
But the direction and mechanism were consistent:

| | raw | flatten |
|---|---|---|
| PQ | 0.4041 | 0.4198 |
| FP | 808 | **732 (−9.4 %)** |
| TP | 1027 | 1029 (flat) |
| SQ | 0.6667 | 0.6762 |
| RQ | 0.6061 | 0.6208 |

Flattening removes spurious detections without losing real ones, and improves
both SQ and RQ. That is the signature of removing a nuisance the network would
otherwise spend capacity on — limb darkening inverts filament contrast in the
outer ring, which holds 11 % of filament area (`docs/DISK_MASKING.md`).

The 5-fold run **is** the properly-powered version of that A/B: ~1,120 val files
across the five folds, 5× the power, at no extra cost because the folds were
needed anyway. If flattening is worthless, the pooled OOF will show it.

### Fold 0 is reused, not retrained

`v2/runs/ab_fold0_flatten` was trained on exactly this config (`fold0.json`,
flatten, bf16, stem 32) and its `best.pt` holds the **true maximum** (epoch 110,
PQ 0.4177) — the early-stopping bug only affected the raw arm. Verified field by
field against `v2/configs/seg_v2.json` before reuse. Copied to
`v2/runs/seg_v2_fold0`, saving ~2.5 h.

## Cost

~116 s/epoch measured on fold 1 (909 train ids, 245 val entries). Folds 1–4 at
160 epochs ≈ 5.2 h each worst case, but early stopping fires around epoch 90–140
in practice, so expect **~15–18 h total** for the four remaining folds.

## Outputs

```
v2/runs/seg_v2_fold{0..4}/best.pt      per-fold checkpoints
v2/runs/seg_v2_fold{0..4}/log.jsonl    per-epoch metrics
v2/runs/seg_v2/oof/fold{0..4}/*.npz    OOF dumps (prob, labels[k], spines[k], meta)
```

The OOF dumps are the hard prerequisite for Phase 2: they carry the filename and
annotator-copy metadata that `tune.build_cache` never wrote, and `oof.py` refuses
to dump a checkpoint on files it was trained on.

## Acceptance (from `PLAN.md` §1.8)

| check | threshold | status |
|---|---|---|
| pooled OOF PQ | ≥ 0.46 | pending |
| paired CI vs `norm_disk_raw` | excludes 0 | pending |
| SQ | ≥ 0.68 (no regression) | pending |
| RQ | ≥ 0.66 | pending |
| fold spread | max − min ≤ 0.04 | pending |

If pooled OOF < 0.44 after all five folds, stop and diagnose with
`scripts/pq_decompose.py` before touching Phase 2.

## Resuming

The runner skips any fold whose `last.pt` exists, so it is safe to re-run after a
disconnect or crash:

```bash
tmux new -s seg -d 'bash v2/scripts/run_seg_v2.sh'
tmux attach -t seg
tail -f v2/runs/seg_v2.log
```

## Result — null

All five folds trained, 707/707 OOF dumps verified (no missing, extra,
duplicates, wrong-fold, or bad labels).

### Per fold

| fold | epochs | best ep | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 110 | 0.4177 | 0.6783 | 0.6158 | 920 | 514 | 634 |
| 1 | 118 | 109 | 0.4166 | 0.6768 | 0.6156 | 1052 | 512 | 802 |
| 2 | 146 | 116 | 0.4163 | 0.6830 | 0.6094 | 1008 | 603 | 689 |
| 3 | 136 | 106 | 0.4330 | 0.6769 | 0.6396 | 891 | 408 | 596 |
| 4 | 86 | 56 | 0.3911 | 0.6744 | 0.5798 | 928 | 666 | 679 |

Spread 0.0419 (threshold 0.04 — marginally over). Mean 0.4149.

Fold 4 is the outlier: it plateaued at 0.3911 and early-stopped at epoch 86, the
only fold to stop before 100. Its val set is the smallest (140 files) and its
loss curve was noisier from epoch 30 onward. Not a bug — the trajectory is
monotone-ish and the early stop fired correctly — just a harder fold.

### Pooled OOF (707 files, 1,154 annotator entries)

| thr | area | close | PQ | SQ | RQ |
|---|---|---|---|---|---|
| 0.50 | 256 | 15 | **0.4220** | 0.6765 | 0.6238 |
| 0.45 | 256 | 15 | 0.4217 | 0.6756 | 0.6241 |
| 0.55 | 256 | 15 | 0.4210 | 0.6774 | 0.6215 |
| 0.50 | 512 | 15 | 0.4199 | 0.6774 | 0.6198 |

Bootstrap 95 % CI on the best: **[0.4144, 0.4299]**.

### The honest comparison

Pooled OOF over 707 files and the baseline's 0.4225 over 142 files are **not
comparable** — different file sets, different annotator counts. The only honest
test is on the intersection: the 232 annotator entries (142 files) that are
out-of-fold for seg_v2 *and* were the baseline's val set.

`python v2/scripts/compare_phase1.py`:

| | PQ | SQ | RQ |
|---|---|---|---|
| baseline (`norm_disk_raw`, resnet34) | 0.4225 | 0.6778 | 0.6234 |
| **seg_v2** (ConvNeXt-S + stem + flatten) | **0.4264** | 0.6784 | 0.6286 |

Paired per-image test, n = 232:

```
mean per-image delta = +0.0019   t = +0.23   95% CI [-0.0139, +0.0176]
verdict: no significant difference
```

**Phase 1 bought nothing measurable.** +0.0039 pooled, +0.0019 per-image, CI
straddling zero. SQ moved +0.0006 and RQ +0.0052 — both inside noise.

### What this means

Phase 1 changed four things at once (backbone, stride-1 stem, flattening, 5-fold
CV) and the net effect is zero. That is informative even though it is negative:

* **The architecture is not the bottleneck.** ConvNeXt-Small with a stride-1
  path is not better than resnet34 at this data scale. The plan's §1.8 said stop
  and diagnose if pooled OOF < 0.44; it is 0.4220, so that applies.
* **Flattening is not the win the A/B hinted at.** The fold-0 A/B showed FP
  808 → 732; across five folds the effect does not survive. The A/B's
  bias-corrected CI already included zero, and this confirms it.
* **The 5-fold split is sound.** Spread 0.0419 is marginally over the 0.04
  threshold, driven entirely by fold 4. Folds 0–3 span 0.0167. Not worth
  switching to month grouping.
* **SQ is stuck at ~0.678** across every model tried in this project — resnet34,
  DeepCrack+decoder, ConvNeXt-S. That is a strong signal that the remaining
  headroom is not in the segmentation network.

### The 5-fold ensemble — and why its number is not usable

`python v2/scripts/score_ensemble.py infer && ... score`

| | PQ | SQ | RQ |
|---|---|---|---|
| single model (honest OOF) | 0.4220 | 0.6765 | 0.6238 |
| 5-fold ensemble | **0.4346** | 0.6780 | 0.6410 |
| apparent gain | +0.0126 | +0.0015 | +0.0172 |

**This number is contaminated by construction and must not be quoted as a
result.** Each file is held out by exactly one fold, so for any given file **4 of
the 5 ensemble members were trained on it**. The ensemble average therefore mixes
the genuine ensemble benefit with memorisation, inseparably.

The decomposition says the same thing: SQ moves +0.0015 (noise) while RQ moves
+0.0172. Boundary quality is what generalises; instance recall is what
memorisation improves. A real ensemble gain would move both.

To measure the ensemble honestly you need a held-out set that **no** fold saw —
which the 707 training files cannot provide. Options, in order of cost:

1. **Use the 180 test images.** No fold trained on them, so a 5-fold ensemble
   there is honest. But there is no GT, so it only gives a public LB number
   (2-decimal, ~90 images, ±0.02 noise).
2. **Hold out a 6th group.** Re-split into 6 groups, train 5, keep one untouched
   as a true ensemble-validation set. Costs one more fold of training (~4 h) and
   shrinks the training set by 1/6.
3. **Accept the single-model OOF number** (0.4220) as the honest Phase 1 result
   and treat the ensemble as an inference-time technique whose benefit is
   measured on the public LB instead.

Option 3 is what the plan should do. The ensemble is still worth shipping — it is
free at inference and cannot hurt — but its contribution is unmeasured, and
quoting 0.4346 would repeat exactly the mistake this project has already made
three times (cv2 GT, OOF contamination, fp16 divergence): a number that looks
valid and is not.

### Where the headroom actually is

RQ is 0.624 and SQ is 0.678, so PQ = SQ × RQ ≈ 0.42. The measured error anatomy
(`docs/PQ_DIAGNOSIS.md`) says 58.5 % of FPs are splits of an already-matched GT
and 57.1 % of FNs are found-but-IoU<0.5. Both are **instance-level** problems,
not pixel-level ones — which is exactly what Phase 2 (fragment merging +
per-instance `q̂`) targets, and why it should not be gated behind a segmentation
improvement that is not coming.

**Recommendation: go to Phase 2 now.** The OOF dumps it needs are verified and
in place. Re-tuning the segmentation model further is spending GPU on a
component that has been flat across three architectures.

---

## Test-set ensemble submission (2026-10-31)

The 5-fold ensemble was run on the 180 test images and submitted. This is the
**first honest ensemble measurement** in the project: no fold trained on any test
file, so the public score is not contaminated the way the OOF ensemble number
(0.4346) is.

| | local | public | gap |
|---|---|---|---|
| single model (`norm_disk_raw`) | 0.4225 | 0.35 | 0.0725 |
| 5-fold ensemble (test set) | — | **0.36** | — |

**The gap is stable across two different models: 0.0725 vs 0.0746.** That is the
useful finding. It means the local→public gap is a property of the *test set*
(different instrument, different years, harder filaments), not a property of the
model or of overfitting to the local split. The calibration constant from
`CALIBRATION.md` can therefore be applied to future local numbers with some
confidence.

### What was submitted

`v2/scripts/predict_test_ensemble.py`:

- 5 fold checkpoints (convnext_small + `stem_ch=32`, flatten preprocessing, bf16)
- **probabilities averaged across folds, then thresholded once** at
  thr 0.50 / min_area 256 / close_k 15
- 1,365 filaments, mean 7.58/image, max 19
- 3 images got zero predictions: `20170708164010Mh`, `20200310113230Ch`,
  `20211214174010Mh` — all 2017–2021, consistent with the known P14 issue

Averaging in probability space rather than voting on binarised masks is the
point: a pixel 3 of 5 folds are confident about survives even if 2 are not.
Mask voting discards exactly the confidence information that makes an ensemble
better than its members.

Validation before submit: all 1,365 RLEs decode at 2048², min area 256, no
nulls, no duplicate `filament_id`, 177/180 images represented.

### Reading the +0.01

The public LB is 2-decimal on ~90 images, so ±0.02 noise. **+0.01 is not
evidence that the ensemble helps.** It is consistent with "the ensemble is
neutral-to-slightly-positive", which is what theory predicts for 5 correlated
members of the same architecture trained on the same distribution. The ensemble
is still worth shipping — it is free at inference and cannot hurt — but its
contribution remains unmeasured at a resolution that matters.

This is the same conclusion the OOF analysis reached, now confirmed on clean
data: **the ensemble is not where the headroom is.**
