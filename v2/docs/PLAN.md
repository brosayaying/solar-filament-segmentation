# Phase 2 plan — from 0.42 local / 0.34 public to a legitimate 0.45+

Written 2026-10-29. Supersedes the *scheduling* in
[`../../docs/STRATEGY_V2.md`](../../docs/STRATEGY_V2.md) while keeping its
analysis. Everything below was re-measured on this machine today; where a number
contradicts STRATEGY_V2 it is called out in §2.

**Deadline: 2026-11-15 06:00 UTC — 17 days.** One RTX 3090 (24 GB), one person.

---

## 1. Where we actually are

### 1.1 The honest baseline is 0.4225, not 0.4152

Every full-val number in `README.md` and `docs/` was computed with the old
`cv2.fillPoly` rasteriser, which makes GT masks ~10.5 % too fat. `data.py` was
switched to pycocotools on 2026-10-27, so **all cached `val_probs/` directories
predate the fix and are stale** (checked: `runs/baseline_resnet34/val_probs` and
`runs/tiled1024_resnet34/val_probs` are dated 2026-10-23).

Re-scoring the best checkpoint that was *trained and evaluated* under the current
GT, on the full 232-entry val set:

| run | GT | norm | epochs | best val PQ |
|---|---|---|---|---|
| `long_160ep_512` | cv2 | 0.5 / 0.25 | 160 | 0.4361 (64-img) |
| `long_160ep_pycocotools` | cv2 | 0.5 / 0.25 | 160 | 0.3898 (64-img) |
| **`norm_disk_raw`** | **pycocotools** | **0.4463 / 0.1313** | 160 (stopped at 106) | **0.4124 (64-img)** |

Full-val sweep of `norm_disk_raw/best.pt` (232 entries, 1,641 GT instances,
pycocotools GT, IoU > 0.5):

| thr | min_area | close | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|---|---|
| 0.65 | 256 | 15 | **0.4225** | 0.6778 | 0.6234 | 1109 | 808 | 532 |
| 0.55 | 256 | 15 | 0.4219 | 0.6777 | 0.6225 | 1110 | 815 | 531 |
| 0.45 | 512 | 15 | 0.4197 | 0.6775 | 0.6195 | 1029 | 652 | 612 |
| 0.50 | 512 | 0 | 0.4177 | 0.6824 | 0.6121 | 1014 | 658 | 627 |

**Take this as the baseline: PQ 0.4225, SQ 0.678, RQ 0.623.** Public score of the
closest submitted artefact is 0.34.

Two things follow immediately:

* Disk-only normalisation is worth roughly **+0.02** (0.4124 vs 0.3898 on the
  same 64-image subset, and the pycocotools GT is the *harder* one, so the true
  gain is at least that). It is already in the codebase and already trained.
* **The local→public gap is measured: 0.0725** (0.4225 local → 0.35 public,
  submitted 2026-10-29). Not the 0.04 ± 0.02 STRATEGY_V2 assumes. See
  [`CALIBRATION.md`](CALIBRATION.md). Rank at 0.35 is ≈ 477 / 1,125.

### 1.2 What the data actually is (re-measured)

| fact | value |
|---|---|
| unique train files | 707 (1,154 image entries, 8,199 annotations) |
| annotator copies per file | 411 × 1, 145 × 2, 151 × 3 |
| annotations per file | median 9 |
| filament pixel fraction | median 0.31 %, mean 0.36 % |
| instance area p5/p25/p50/p75/p95 | 316 / 670 / 1228 / 2438 / 6946 px |
| instance-count disagreement between copies | median 2, mean 2.58 |
| `spine` present on | 8,199 / 8,199 annotations |
| test files | 180; 2021–22 share 11.7 % vs 5.7 % of train |
| distinct dates / ISO weeks / months | 656 / 343 / 118 |

Every annotation has a centreline, so the spine heatmap head is free supervision
with no extra labelling.

### 1.3 Compute, measured on this GPU

Batch throughput for `TimmUNet` (decoder_dim 128), forward+backward+step:

| backbone | size | bs | ms/step | samp/s | peak VRAM |
|---|---|---|---|---|---|
| resnet34 | 512 | 12 | 234 | 51.3 | 8.2 GB |
| resnet34 | 1024 | 4 | 321 | 12.4 | 10.8 GB |
| convnext_small | 512 | 8 | 321 | 24.9 | 9.9 GB |
| convnext_small | 768 | 4 | 348 | 11.5 | 11.2 GB |
| convnext_small | 1024 | 2 | 318 | 6.3 | 9.9 GB |
| pvt_v2_b2 | 512 | 8 | 207 | 38.7 | 9.1 GB |

At 565 files × 2 samples = 1,130 samples/epoch, a 160-epoch run costs:

| config | s/epoch | 160 epochs | 5 folds |
|---|---|---|---|
| convnext_small 512 bs8 | ~55 | 2.4 h | **12 h** |
| pvt_v2_b2 512 bs8 | ~40 | 1.8 h | **9 h** |
| convnext_small 768 bs4 | ~110 | 4.9 h | 25 h |

A 5-fold CV of the main model is therefore ~12 h, not a luxury. **The plan
assumes 5-fold CV everywhere**, which also yields five ensemble members for free.

### 1.4 The GONG archive is live and directly downloadable

Verified today, no credentials, no rate-limit hit:

```
https://gong2.nso.edu/HA/haf/YYYYMM/YYYYMMDD/YYYYMMDDHHMMSS<station>.fits.fz
```

* 2048 × 2048 int16, 2.8 MB per frame, **1-minute cadence**, all six stations.
* `astropy` 8.0.2 installed into `.venv` (was missing).
* Header carries `DATE-OBS`, `OBS-SITE`, `TELESCOP`.
* Disk has 186 GB free → ~60 k frames of headroom.

This makes STRATEGY_V2's phase C real rather than speculative. It is also the
single largest untapped asset: 707 labelled images against an effectively
unlimited supply of same-instrument, same-format frames.

### 1.5 What the leaderboard says

1,152 teams, 2026-10-24: median 0.34, p75 0.38, p90 0.40, p95 0.55, max 0.69.
Only **4 teams** sit between 0.41 and 0.55; 95 sit at ≥ 0.55. The 0.55+ cluster
is the leaked-MAGFiLO notebook and its variants (forum-confirmed). So the honest
frontier is ~0.41 and **anything ≥ 0.45 public is plausibly top-5 legitimate**.

---

## 2. Corrections to STRATEGY_V2

The analysis in §1–§3 of STRATEGY_V2 holds up. Four of its *prescriptions* do
not, and the schedule needs rewriting.

| STRATEGY_V2 claim | what the repo says | plan response |
|---|---|---|
| A2: radial flattening is "the cheapest known defect" and a win | `docs/PREPROCESSING.md` measured flatten at **−0.0048, CI [−0.0137, +0.0042]** — not significant — when bolted onto a raw-trained model. The retrain (`configs/norm_disk_flatten.json`) **was never run**. | **A/B run 2026-10-30, inconclusive** ([`AB_FLATTEN.md`](AB_FLATTEN.md)): delta +0.0154 as-run, but +0.0134 CI [−0.0012, +0.0279] after correcting an early-stopping bug. One fold is underpowered (needs 428 images for 80 % power at 0.015). Fold into Phase 1's 5-fold CV rather than re-running standalone. |
| A2: "normalise with on-disk statistics, not 0.5/0.25" | Already done and already trained: `norm_disk_raw` = 0.4124 vs 0.3898. | **Shipped.** Fold into the baseline, do not re-litigate. |
| A3: ConvNeXt-Small or EfficientNetV2-M | EfficientNetV2-M OOMs at 768² bs4 on this card (measured). ConvNeXt-Small fits at 512² bs8 and 768² bs4. | ConvNeXt-Small is the primary; EfficientNetV2-M only at 512². |
| A4: budget 100–150 epochs | `norm_disk_raw` was still improving at epoch 100 when it was killed; `long_160ep_512` peaked at 126. | 160 epochs, EMA, and **select on EMA not argmax** (FEEDBACK item 4). |
| §6: local→public drop is 0.04 ± 0.02 | Our own data says 0.08 (0.4225 → 0.34). | Measure it (§3.0.4) before setting a target. |
| D: SegFormer-B3 for diversity | `mit_*` / `segformer*` do not exist in timm 1.0.30 (checked). | Use **PVTv2-B2** (38.7 samp/s, 9.1 GB) as the transformer member. |

One thing STRATEGY_V2 gets right and that the repo under-weights: **the crop
sampling change is the cheapest large win available.** `tiled1024` removed 172
false positives with identical TP/FN purely by not truncating filaments, and
`long_160ep` bought +0.095 RQ. Both are sampling effects. Phase 1 makes sampling
a first-class design decision rather than a side effect of patch size.

---

## 3. What v2 is

**Decision: `v2/` is an artefact root, not a code fork.**

* Code stays in `src/solar/` and evolves in place. Rewriting 5.2 k lines of
  working, measured code with 17 days left is not a plan, it is a gamble.
* New modules are added alongside (`src/solar/folds.py`, `src/solar/rescore.py`,
  `src/solar/gong.py`, `src/solar/official.py`).
* **All new docs go to `v2/docs/`**, one file per experiment, figures in
  `v2/docs/<EXPERIMENT>/<figure>.png`.
* **All new runs go to `v2/runs/<experiment_name>/`** via
  `train.out_dir = "v2/runs"`. Old runs stay where they are, read-only, and are
  cited by path.
* Nothing in `v2/` is allowed to depend on a stale `val_probs/` cache.

Rationale: the failure mode that has cost this project the most time is
*unmeasured* changes (the 6-epoch backbone sweep, the eval-time preprocessing
probe with the wrong normalisation, the cv2 rasteriser). A clean artefact root
with a mandatory measurement gate is worth more than a clean code root.

---

## 4. Phase 0 — measurement infrastructure — **COMPLETE 2026-10-29**

Nothing in Phases 1–4 is trustworthy until this exists. It is ~1 day of work and
it is the highest-leverage day in the plan. Full write-up in
[`PHASE0.md`](PHASE0.md); all five acceptance checks pass.

One bug was caught that would have invalidated Phase 2: dumping the existing
`norm_disk_raw` checkpoint over the week-5 folds put **80 % of the "OOF" dumps
in-sample** (it was trained on 565 of 707 files). Those dumps looked valid —
right shape, right fold, right labels — and were deleted. `oof.py` now refuses
to dump a checkpoint on files it was trained on.

### 0.1 Five-fold, week-grouped split — **DONE**

`src/solar/folds.py` → `v2/splits/week5_seed42.json` plus
`v2/splits/week5_seed42/fold{0..4}.json` (the per-fold `{train, val}` format
`solar.train` reads).

* Group by **ISO week**, not date. Date grouping still puts adjacent days on
  opposite sides, and filaments persist for days. 343 week groups, 130 of them
  singletons, median 2 files/group, max 6.
* Greedy balance over shuffled week keys gives folds of **141 / 142 / 142 / 140 /
  142 files** (verified). Month grouping is the fallback if week folds turn out
  too noisy (118 groups, folds of 138–143).
* All annotator copies of a file stay in one fold. Assert 0 file overlap and 0
  week overlap between every train/val pair.
* Emit `folds: [[ids], ...]` plus `meta` with the leakage report.

**Acceptance:** `python -m solar.folds --inspect` prints 5 folds, 0 file overlap,
0 week overlap, and the per-fold file counts above.

### 0.2 Official-scorer harness — **DONE**

`src/solar/official.py`. The organisers' notebook is not in the repo, so this is
written from `docs/COMPETITION.md` §4 and validated against `PQAccumulator`
(agrees to 4 dp: 0.4225 / 0.6778 / 0.6234, identical TP/FP/FN):

* GT keyed `<annotator>-<image>`, predictions matched by bare `<image>`.
* pycocotools RLE decode, IoU **strictly > 0.5**, greedy 1:1.
* PQ pooled globally over all annotator copies, not averaged per image.
* Emit the same `filament_id,segmentation_rle` CSV shape as a submission, so the
  harness and the real submission path share one code path.

**Acceptance:** scoring `norm_disk_raw` through this harness reproduces
`PQAccumulator` to 4 dp on the same predictions. If it does not, the difference
is a bug in one of them and must be found before any further training.

### 0.3 OOF probability dumps with metadata — **DONE**

`src/solar/oof.py` writes, per val image:

```
v2/runs/<exp>/oof/<fold>/<file_stem>.npz
  prob      float16 (2048, 2048)   ensemble-averaged probability
  label     int32   (2048, 2048)   pycocotools instance labelmap
  spine     uint8   (2048, 2048)   GT centreline raster
  meta      json                   file_name, image_id, annotator ids, fold, r/R map stats
```

The current cache is `NNNN_prob.npy` + `NNNN_label.npy` with **no filename
mapping**, which makes it impossible to join OOF candidates back to annotations —
and Phase 2 is entirely built on that join. This is a hard prerequisite.

**Acceptance:** for every train file, exactly one OOF probability map exists
(from the fold where it was in val), and `meta.file_name` round-trips.

### 0.4 Calibrate local→public — **DONE 2026-10-29**

Submitted `norm_disk_raw/best.pt` at thr 0.65 / min_area 256 / close 15:
**local 0.4225 → public 0.35, gap 0.0725.** Full write-up in
[`CALIBRATION.md`](CALIBRATION.md).

Two things came out of it beyond the number:

* `predict.py` had no `close_k` support, so the tuned setting could not be
  reproduced from the CLI. Added `--close-k`; verified bit-identical to the
  tuning harness on all 232 val images (0 labelmap mismatches).
* The gap is 1.8× larger than STRATEGY_V2 assumed, which shifts every target in
  §13 down. **The real milestone is 0.40 public** (top 141), not 0.45.

**Re-calibrate after Phase 1.** If the gap shrinks when the model is trained on
soft/consensus targets, targets move up; if it grows, Phase 3 becomes the
highest-value phase rather than the highest-variance one.

### 0.5 Bootstrap CI on every reported number — **DONE**

`PQAccumulator.bootstrap_ci` already existed and is correct (resamples images,
recomputes the global aggregate). Wired into `tune.py` as `--bootstrap`:

```
best PQ 0.4225 95% CI [0.4052, 0.4392]  (SQ [0.6718, 0.6835], RQ [0.6014, 0.6473])
```

**No number is reported without a CI from here on**, and no change is
"shipped" on a point estimate.

---

## 5. Phase 1 — segmentation model v2 (days 2–5)

Target: **local 0.4225 → 0.46–0.48** (pooled OOF, 5-fold).

### 1.1 Input: 3 channels

`[flattened, raw-normalised, r/R]`, `in_chans=3`.

* `r/R` is one cheap channel that lets the network condition on limb geometry
  instead of inferring it from vignetting. Recall collapses 0.73 → 0.28 in the
  outer ring, which holds 11 % of filament area (`docs/DISK_MASKING.md`).
* **Flattening: unresolved, not settled.** The fold-0 A/B
  ([`AB_FLATTEN.md`](AB_FLATTEN.md)) gave +0.0154 as-run but +0.0134 CI
  [−0.0012, +0.0279] after correcting an early-stopping bug — inconclusive, and
  underpowered by design (224 images vs the 428 needed for 80 % power at 0.015).
  The FP count did drop 9.4 % with TP unchanged, so the direction is credible.
  **Resolve it inside Phase 1's 5-fold CV** (≈ 1,120 val files, 5× the power) as
  a 2-arm comparison, not as a separate experiment.
* The flatten cache already exists and covers 707/707 train and 180/180 test
  files (11 GB, `cache/preprocess/flatten`). No rebuild needed.

### 1.2 Architecture

`TimmUNet` + `convnext_small.fb_in22k_ft_in1k` (7×7 depthwise — the receptive
field is the point, since the cue is long-range coherence of a thin ridge against
same-width texture), `decoder_dim=128`, plus:

* **stride-1 path**: concatenate the stem/stride-2 features into the final
  decoder block so an 11 px object is not reconstructed from stride-4 alone.
  This is a small change to `TimmUNet.forward` and it is the difference between
  the DeepCrack failure (no decoder, PQ 0.35) and the UNet (0.44).
* **spine aux head** (`aux_spine=true`, already implemented, never trained with
  `spine_weight > 0`). Gaussian σ ≈ 3 px along the provided centreline. Free
  supervision that encodes connectivity, and Phase 2 consumes it.

### 1.3 Loss

`BCE(soft target) + Lovász-hinge + 0.3 · clDice`, spine head on focal/MSE.

* BCE keeps the pixel map **calibrated** — Phase 2's `q̂` regressor needs a
  probability, and Dice alone destroys calibration.
* Lovász-hinge optimises IoU directly, i.e. the 0.5 cliff. The measured IoU
  distribution has **0 instances above 0.9** and 34 % in 0.5–0.7, so the mass is
  sitting right on the cliff; this is where the points are.
* clDice (Shit et al., CVPR 2021) is built for tubular structures and penalises
  exactly the fragmentation that produces **58.5 % of our false positives**.
* Soft targets: for a file with k copies the pixel target is the mean of the k
  rasterised masks, and each **file** is sampled once per epoch, not once per
  copy (which over-weights multi-annotator files 3×).

None of clDice, Lovász, or EMA exists in the repo today (grepped). All three are
new code in `src/solar/losses.py` and `src/solar/train.py`.

### 1.4 Sampling — the cheapest win

* Crops **768²**, sampled **70 % centred on a random annotation / 30 % uniformly
  on-disk**. This fixes the measured problem that 512² random crops waste ~50 %
  on pure background and truncate ~51 % of touched filaments, without giving up
  translation diversity. `tiled1024` proved the mechanism (+172 fewer FP, same
  TP/FN) at 6× the cost; this gets the same effect at 1.5×.
* If 768² bs4 is too slow for 5 folds (25 h), fall back to 512² bs8 with the same
  70/30 sampling — the sampling is the point, not the size.

### 1.5 Augmentation

Dihedral 8× **plus arbitrary rotation about the disk centre** (the disk is round,
so this is a physically valid sample with correct limb geometry — do it on the
full image before cropping), scale 0.85–1.2, gamma 0.7–1.4, brightness/contrast,
Gaussian blur σ ∈ [0, 1.5] (seeing differs by station), additive Gaussian +
Poisson noise, smooth low-frequency multiplicative field. **Instance copy-paste**
of flattened filament cut-outs onto other on-disk locations. No mixup.

Only flips/rot90/brightness exist today. Rotation-about-centre and copy-paste are
the two that matter and neither exists.

### 1.6 Recipe

AdamW, lr 5e-5 encoder / 3e-4 decoder, wd 0.05, drop-path 0.2, cosine with
5-epoch warm-up, **EMA 0.999**, bf16 AMP, 160 epochs, batch 4–8.

Checkpoint selection on **EMA weights evaluated on the full fold val**, not
argmax over epochs on a 64-image subset (FEEDBACK item 4 — the current
`best.pt` selection is winner's-curse biased).

### 1.7 Inference

Whole 2048² image (0.1 s), 8× dihedral TTA, average probabilities across folds.
Threshold and min_area chosen on OOF with a CI, not on the public LB.

### 1.8 Acceptance

| check | threshold |
|---|---|
| pooled OOF PQ | ≥ 0.46 |
| paired CI vs `norm_disk_raw` | excludes 0 |
| SQ | ≥ 0.68 (must not regress) |
| RQ | ≥ 0.66 |
| fold spread | max − min ≤ 0.04 (else the split is too noisy, switch to month grouping) |

If pooled OOF is < 0.44 after 5 folds, stop and diagnose with
`scripts/pq_decompose.py` before touching Phase 2.

---

## 6. Phase 2 — instance re-scoring (days 5–9)

Target: **+0.03–0.06** on top of Phase 1. This is the part that implements
STRATEGY_V2 §3 and that the 0.40 crowd does not have.

The arithmetic, restated because it drives every threshold: a candidate matched
by a fraction `q` of annotator copies with mean IoU `s` raises PQ iff
`2·q·s > PQ`, i.e. at PQ 0.45 and s 0.7, **keep anything q > 0.32**. The
instance decision must therefore be made on a *calibrated probability that an
annotator would draw this blob*, not on a pixel threshold.

### 2.1 Candidate generation

Threshold the OOF ensemble map **low** (≈ 0.3) for recall, connected components,
keep ≥ 128 px. Record every candidate's mask, area, elongation, mean/max prob,
mean contrast vs its annulus, and r/R.

### 2.2 Fragment merging

For each candidate pair within ~40 px: gap length, orientation alignment of
principal axes, mean probability and mean **spine-heatmap** value along the
straight bridge, area ratio. Train a GBDT on OOF candidates labelled "both pieces
match the same GT instance in the majority of copies". Merge in decreasing
confidence order with union-find.

Fallback if time is short: merge when the spine heatmap along the bridge stays
above a threshold. That is a one-line rule and it captures most of the value,
because 58.5 % of FPs are splits of an already-matched GT.

### 2.3 Candidate scoring

Small CNN (ResNet-18, 256² crop at full resolution, channels
`[flattened image, ensemble prob, candidate mask, r/R]`) trained on OOF
candidates to regress `q = fraction of annotator copies in which this candidate
has IoU > 0.5`. Average with a GBDT on the scalar features. Keep `q̂ > τ`,
τ tuned on OOF, expected 0.30–0.35.

Why this should work: the oracle per-image threshold already buys +0.04 with a
single scalar per image; a per-instance score with image context is strictly more
expressive, and it is trained on OOF maps so it absorbs the ensemble's own
failure modes and station/epoch shift.

### 2.4 Acceptance

| check | threshold |
|---|---|
| pooled OOF PQ | ≥ Phase 1 + 0.03 |
| paired CI vs Phase 1 | excludes 0 |
| FP count | down ≥ 25 % |
| FN count | not up more than 5 % |

If merging alone (2.2) does not move PQ, the fragmentation hypothesis is wrong
and 2.3 should be re-scoped before building the CNN.

---

## 7. Phase 3 — scale the data (days 8–13)

Target: **+0.03–0.05**, highest variance. The labelled set is 707 images with
noisy instance labels; both problems are better addressed with more images than
with more architecture.

### 3.1 Label transfer (cheap, allowed, do first)

For each training file, fetch the same-site GONG frame at ±2–10 min and reuse the
annotation unchanged (solar rotation in 10 min is < 1 px at disk centre at GONG
resolution). 2–4× more training frames with real labels and genuinely different
seeing/noise.

* `src/solar/gong.py`: URL builder, downloader with 2 connections + 1 s delay,
  FITS → uint8 normalisation matching the competition images, disk geometry.
* Exclude anything within ±1 day of any test timestamp.
* Verified working today: `20220101000030Lh.fits.fz` downloads and reads as
  2048² int16.

### 3.2 Noisy-student self-training

Download 5–20 k unlabelled frames spread over 2011–2022 and all six stations,
**over-sampling 2021–22** (11.7 % of test vs 5.7 % of train). Pseudo-label with
the Phase 1+2 ensemble (after re-scoring, so pseudo-labels are consensus-like),
keep only confident instances (`q̂ > 0.6`) and confident background, train the
student from scratch on labelled + pseudo-labelled data with the full Phase 1
augmentation. One or two rounds.

This is the recipe Diercke et al. used to get past a small labelled set on GONG,
and it is the most plausible route to the consensus behaviour that scores 0.61
against individual annotators (`docs/CORRECTIONS.md` §2).

Budget: 10 k frames × 2.8 MB = 28 GB, ~4–6 h to download at the rate limit.
Training at 20 k samples/epoch is ~13 min/epoch, so 60 epochs = 13 h per round.
**Run one model, not five folds** — the point is data, not variance reduction.

### 3.3 Acceptance

| check | threshold |
|---|---|
| pooled OOF PQ | ≥ Phase 2 + 0.02 |
| 2021–22 subset PQ | improves by more than the overall PQ (that is the point) |
| outer-ring (r/R > 0.8) recall | improves |

If self-training does not beat label transfer alone, ship label transfer and drop
the student. Do not spend days 13–15 on a second round that is not working.

---

## 8. Phase 4 — ensemble and diversity (days 12–15)

* **5 folds of the main model** are already an ensemble — average their
  probabilities. This is the most reliable +0.01–0.02 available.
* **One transformer member**: PVTv2-B2 at 512² (38.7 samp/s, 9.1 GB — measured,
  and it is the only transformer family in timm 1.0.30 that is both fast and
  size-agnostic enough to run at 2048² inference). Transformers see the whole
  filament at once and fail differently from ConvNets.
* **2 seeds per fold** only if GPU time remains after Phase 3.
* Dense CRF: skip. SQ is already 0.68 and human-vs-human SQ is lower.
* Mask2Former / CondInst: not as the primary system. Nothing to separate
  (GT + CC = PQ 1.00), and thin objects are their weak point.

### Acceptance

Ensemble ≥ best single member + 0.01 with a paired CI excluding 0. If not, ship
the single best model — an ensemble that does not help is just inference cost.

---

## 9. Phase 5 — deliverables (days 15–17, non-negotiable)

30 % of the rubric, and the qualitative half is contingent on code availability.

1. **Pinned `requirements.txt`** (currently only `pyproject.toml` with ranges).
2. **One notebook** reproducing both selected submissions from the test images
   alone, no extra files.
3. **4-page report** (Overleaf template): pipeline figure, the §3 detection-rule
   derivation, OOF PQ/SQ/RQ with CIs, Dice/IoU distributions, and the
   one-to-many / many-to-one histogram the organisers said they will inspect.
4. **Disclose every GONG frame used**, with date ranges and station list.
5. **`v2/docs/`** holds one file per experiment with its figure directory.

---

## 10. Schedule and compute budget

| days | phase | GPU hours | deliverable |
|---|---|---|---|
| 1 | 0 — measurement | 1 | 5-fold split, official harness, OOF format, calibration submission |
| 2–5 | 1 — segmentation | 15 | 5-fold ConvNeXt-Small, pooled OOF ≥ 0.46 |
| 5–9 | 2 — re-scoring | 6 | merge + `q̂`, pooled OOF ≥ 0.49 |
| 8–13 | 3 — data scale | 30 | label transfer + 1 student round |
| 12–15 | 4 — ensemble | 12 | 5-fold + PVTv2-B2 |
| 15–17 | 5 — deliverables | 2 | report, notebook, requirements |

Total ≈ 66 GPU-hours against ~200 available. **The budget is not the binding
constraint; the measurement gate and the download rate limit are.** That is why
Phase 0 is a full day and why Phase 3 starts downloading on day 8, not day 11.

Parallelism: downloads and pseudo-labelling inference are CPU/network bound and
can overlap with training. Only one training job at a time (24 GB card, and
`convnext_small` at 768² already uses 11 GB).

---

## 11. Risk register and kill criteria

| risk | signal | response |
|---|---|---|
| Local→public gap is structural and does not shrink | Phase 1 re-calibration still shows ≥ 0.07 | Stop optimising local PQ; submit every 2 days and treat public as the metric, accepting the 5/day limit |
| A preprocessing/architecture choice is real but too small to measure on one fold | CI half-width ≈ 0.015 on 224 images | Decide on the 5-fold CV (≈ 1,120 files), never on a single fold. Budget for it up front. |
| Week-grouped folds too noisy | fold spread > 0.04 | Switch to month grouping (118 groups, folds 138–143) |
| Flattening hurts | Phase 1 A/B CI excludes 0 on the negative side | Drop it, keep raw + r/R |
| clDice/Lovász destabilise training | loss spikes, PQ < baseline at epoch 40 | Drop to BCE + Dice, keep EMA and sampling |
| GONG download rate-limited | < 0.3 frames/s sustained | Cut to 3 k frames, prioritise 2021–22 and the six stations |
| Self-training collapses to its own errors | student OOF < teacher OOF | Ship label transfer only |
| Phase 2 merging does not move PQ | FP count unchanged | Re-scope 2.3 before building the CNN |
| Time runs out | day 14 | Freeze the model, spend days 14–17 entirely on deliverables |

**Hard rule:** no change ships on a point estimate. Every table gets a bootstrap
CI, every A/B gets a paired per-image test. The two largest time sinks in this
project so far were both unmeasured changes.

---

## 12. What we will not do

* **Do not touch the public MAGFiLO 1.0 release annotations** or anything derived
  from the 0.55 notebook. It is a rules violation and the organisers review code.
* Do not spend time on watershed / NMS / box detectors — GT + CC = PQ 1.00, so
  separation is solved; watershed was measured at −0.075.
* Do not evaluate on a deduplicated or 64-image subset, and do not compare
  numbers without a CI. Differences < 0.02 local are noise.
* Do not tune on the public LB (2-decimal rounding, ~90 images, 5/day).
* Do not re-run the 6-epoch backbone sweep. It was truncated and is
  uninformative; `resnet34` needed 160 epochs to converge.
* Do not rebuild the DeepCrack line. It is 0.35 against the UNet's 0.44 and the
  gap is the missing decoder, which Phase 1 fixes properly.

---

## 13. Expected outcome

Gap is **0.0725**, measured (§0.4), not assumed.

| stage | pooled OOF PQ | public (gap 0.0725) |
|---|---|---|
| current honest baseline (`norm_disk_raw`) | 0.4225 | **0.35** (measured) |
| Phase 1 (3ch + ConvNeXt + sampling + loss + EMA + 5-fold) | 0.46–0.48 | 0.39–0.41 |
| + Phase 2 (merge + re-score) | 0.49–0.52 | 0.42–0.45 |
| + Phase 3 (label transfer + self-training) | 0.51–0.55 | 0.44–0.48 |
| + Phase 4 (ensemble) | 0.52–0.56 | 0.45–0.49 |

**The 0.40 public line is the real milestone** — it is top 141 of 1,125, out of
the 554-team pile-up between 0.33 and 0.36. Anything ≥ 0.45 public is, on
current evidence, the best legitimate submission in the competition. The 0.55+
cluster is the leak and is not a target.
