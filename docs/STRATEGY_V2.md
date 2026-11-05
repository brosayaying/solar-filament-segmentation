# Strategy v2 — a from-scratch plan to beat the honest leaderboard

Written 2026-10-29, 17 days before the 2026-11-15 deadline. This deliberately
ignores the existing runs/networks and re-derives the plan from the problem,
the metric, the data, and the public forum. Where a number comes from an
earlier diagnosis in this repo it is cited; everything else was re-measured.

---

## 1. What the problem actually is

**Task.** Instance-segment dark, thin, elongated filaments in 2048² grayscale
GONG H-alpha disks. Scored with Panoptic Quality at IoU ≥ 0.5, pooled over all
annotator copies of each test image (the self-evaluation notebook keys GT by
`<annotator>-<image>` and matches predictions by bare `<image>`).

**Objects are small and thin** (re-measured on the 8,199 train annotations):

| statistic | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| area (px) | 317 | 670 | 1,228 | 2,438 | 6,947 |
| bbox diagonal (px) | 46 | 74 | 112 | 179 | 362 |
| spine length (px) | 41 | 71 | 111 | 182 | 387 |
| width ≈ area / spine length (px) | 5.6 | 8.3 | **11.1** | 15.2 | 25.5 |

Filament pixels are 0.36 % of the image. A median filament is an 11 px wide,
110 px long dark ridge on a background of dark intergranular texture of similar
contrast and width. The discriminating cue is **length, coherence and
orientation**, not local intensity.

**Labels are noisy at the instance level.** 296/707 files have 2–3 annotator
copies. Between copies of the *same* image the instance count differs by a
median of 2 (mean 2.6) on ~7 instances per image. Human-vs-human PQ is 0.36;
a 3-annotator majority vote scores 0.61 against the individuals
([CORRECTIONS.md](CORRECTIONS.md)). The scorer therefore rewards predicting the
**consensus**, and the job is as much "which dark ridges would annotators draw"
(RQ) as "where exactly is the edge" (SQ).

**What is *not* the problem** (from [PQ_DIAGNOSIS.md](PQ_DIAGNOSIS.md), still valid):

* Instance separation by touching — GT mask + connected components = PQ 1.00.
* Resolution — perfect prediction on a stride-4 grid still scores 0.77.
* Mask thickness — dilate and erode both hurt.

**Where the errors are** (baseline resnet34): 58 % of FPs are *fragments of an
already-matched filament*; 57 % of FNs are *found but IoU < 0.5*; 15 % of GT
instances are not touched at all. Oracle per-image threshold gives +0.04. So the
room is in (a) connectivity/fragmentation, (b) instance-level confidence, (c)
the IoU-0.5 cliff.

## 2. What the leaderboard actually says

Public LB (1,152 teams, 2026-10-24): median 0.34, p90 0.40, p95 0.55, top 0.69.
The distribution is bimodal with nothing between 0.41 and 0.55.

The forum resolves this: the 0.55 cluster is a public notebook that **leaks the
original MAGFiLO 1.0 release annotations for the test images**
(`lamhuy8904/solar-filament-unet-segmentation-0-55`), and the 0.56–0.69 entries
are variations of the same leak. Organisers have said they are aware, that
"PQ > 0.35 is of great value", that final ranking is 70 % quantitative
(private PQ + Dice/IoU distributions + one-to-many stats) and 30 % code/report,
and that they will review code. Using the MAGFiLO release annotations violates
the "no other ground-truth metadata" rule.

**The honest frontier is ~0.41 public**, with several teams reporting 0.46–0.49
local on pooled-annotator validation and a consistent local→public drop of
~0.05 (test has proportionally more 2021–22 frames: 11.7 % vs 5.7 %, and an
unknown annotator mix). The realistic target is **0.45–0.50 public**, which
would be the top legitimate score.

Also confirmed on the forum (organiser replies):

* Unlabelled GONG / KSO / BBSO / Meudon H-alpha frames may be used for
  pretraining or self-training, excluding test timestamps ± 1 h.
* Copying *training* annotations onto same-site GONG frames a few minutes away
  is allowed.
* Magnetograms are allowed as a training-only signal. Neighbouring frames are
  **not** allowed at inference.
* ImageNet-pretrained encoders and channel replication are fine.

## 3. A metric fact that drives the design: the optimal detection rule

Let a candidate blob be matched by a fraction `q` of the annotator copies of
its image, with mean IoU `s` when matched. Adding it changes the denominator
`TP + ½FP + ½FN` by exactly `½k` (k copies) relative to omitting it, whether or
not it is matched, and the numerator by `q·k·s`. To first order it raises PQ iff

```
2 · q · s  >  PQ          ⇒   predict if  q > PQ / (2s)
```

At PQ ≈ 0.45 and s ≈ 0.7 that is **q > 0.32**: keep any blob that roughly one
annotator in three would draw. Two consequences:

1. The instance-level decision should be made on a **calibrated probability
   that an annotator would draw this blob**, not on a pixel threshold. A pixel
   threshold conflates "how dark/certain is the core" with "is this a filament".
2. For fragments: two pieces of one filament cost `1 TP + 1 FP` and a low IoU;
   merged they cost `1 TP` with the sum of the IoUs. Merging is right whenever
   the pieces belong to the same filament, and wrong only when they are truly
   two filaments — a learnable pairwise decision.

The same arithmetic says an instance at IoU 0.45 is a **double penalty**
(FN + FP = 1.0 in the denominator and 0 in the numerator), while at IoU 0.55 it
contributes +0.55 and only +1. Pushing the IoU distribution across 0.5 is worth
far more per instance than polishing 0.7 → 0.8.

## 4. The plan

Ordered by expected value per day. A/B are the core; C is the scaling play; D
is diversity only if time remains.

### A. Foundation (days 1–5) — expected local 0.40 → 0.46

**A1. Evaluation that matches the scorer.** 5-fold CV grouped by ISO week
(date grouping still puts adjacent days on both sides). Score pooled over every
annotator copy, pycocotools rasterisation, IoU ≥ 0.5 inclusive, bootstrap CI
per comparison. Save OOF probability maps for every train image — they are the
training data for stage B and the only honest place to tune thresholds. Five
folds also yield five ensemble members for free.

**A2. Preprocessing & input channels.**
Hough-circle disk detection → radial flattening (annular median profile divided
out) → normalise with on-disk statistics (mean 0.446, std 0.131, not 0.5/0.25).
Feed a 3-channel tensor: `[flattened, raw-normalised, r/R]`. The explicit
radius channel lets the network condition on limb geometry instead of inferring
it from vignetting. Measured recall collapses 0.73 → 0.28 in the outer ring,
which holds 11 % of filament area ([DISK_MASKING.md](DISK_MASKING.md)); this is
the cheapest known defect and it was never trained on.

**A3. Backbone and decoder.**
U-Net-style decoder to full resolution over a **ConvNeXt-Small (IN-22k)** or
**EfficientNetV2-M** encoder from timm, `in_chans=3` as above. Rationale: the
discriminating cue is long-range coherence of a thin dark ridge against
same-width texture, so large kernels / large effective receptive field matter
(MORDEN's single biggest ablation gain was 5×5 kernels; ConvNeXt is 7×7
depthwise). Decoder: keep a stride-1 path (concatenate the input / stem
features) so the 11 px objects are not reconstructed from stride-4 alone. Add
two auxiliary heads on the same decoder:

* **spine heatmap** (Gaussian σ ≈ 3 px along the provided centreline) — free
  supervision that encodes connectivity; also consumed by B2;
* **boundary / distance map** — optional; only if A/B tests show SQ movement.

The earlier backbone sweep was stopped at 6 epochs and is uninformative;
ResNet34 needed 160 epochs to converge, so budget 100–150 epochs here.

**A4. Training recipe.**

* Crops **768²–1024²**, sampled 70 % centred on a random annotation / 30 %
  uniformly on-disk. This fixes the "half the crops are empty, half the
  filaments are truncated" problem without giving up translation diversity.
* AdamW; lr 5e-5 encoder / 3e-4 decoder; weight decay 0.05; drop-path 0.2;
  cosine with 5-epoch warm-up; EMA of weights (0.999); bf16 AMP; batch 4–8.
* **Loss:** `BCE(soft target)` + `Lovász-hinge` + `0.3 · clDice`.
  BCE keeps the pixel map calibrated (Dice alone destroys calibration, which
  stage B needs). Lovász-hinge optimises IoU directly, i.e. SQ and the 0.5
  cliff. **clDice** (Shit et al., CVPR 2021) is a topology-preserving loss
  built for tubular structures; it penalises exactly the fragmentation that
  produces 58 % of our false positives. Spine head: focal/MSE on the heatmap.
* **Soft targets from annotator copies.** For files with k copies the pixel
  target is the mean of the k rasterised masks, and each file is sampled once
  per epoch (not once per copy, which over-weights multi-annotator files 3×).
  The earlier soft-label test ([IDEAS_TO_EXPLORE.md](IDEAS_TO_EXPLORE.md) C7)
  was a 25-epoch fine-tune at lr 1e-5 and could not reshape anything; this is
  training from init.

**A5. Augmentation** (the dataset is 707 images; this is where generalisation
to 2021–22 and to six stations comes from):

* Dihedral group (8×) **plus arbitrary rotation about the disk centre** — the
  disk is round, so rotation about its centre is a physically valid sample
  with correct limb geometry (do this on the full image before cropping).
* Random scale 0.85–1.2 (filament widths vary 2.5× across the quartiles).
* Photometric: gamma 0.7–1.4, contrast, brightness; Gaussian blur σ ∈ [0, 1.5]
  (seeing differs by station); additive Gaussian + Poisson noise; a smooth
  low-frequency multiplicative field (transparency/clouds, residual flat-field).
* **Instance copy-paste**: paste flattened filament cut-outs onto other
  on-disk locations of other images, keeping local contrast. Doubles the
  effective number of positive instances and decorrelates filament from its
  neighbourhood.
* No mixup (thin structures), no colour jitter beyond the above.

**A6. Inference.** Whole 2048² image (0.1 s), 8× dihedral TTA, average
probabilities across the 5 folds × N architectures. Threshold and min-area
chosen on OOF, expected around 0.5 once the loss is BCE-dominant (the old 0.75
sweet spot was compensating for fat cv2 masks and Dice over-confidence).

### B. Instance re-scoring — the RQ lever (days 4–9) — expected +0.03–0.06

This is the part that directly implements §3, and the part the forum's
"decoupled pipeline → 0.44+" teams have and the 0.40 crowd does not.

**B1. Candidate generation.** Threshold the ensemble map *low* (≈0.3) so recall
is high, connected components, keep anything ≥ 128 px.

**B2. Fragment merging.** For each pair of candidates within ~40 px: features =
gap length, orientation alignment of the two principal axes, mean probability
and mean spine-heatmap value along the straight bridge, areas. Train a gradient
boosted classifier on OOF candidates with label "both pieces match the same GT
instance in the majority of copies". Merge in decreasing confidence order
(union-find). Fallback heuristic if time is short: merge when the spine heatmap
along the bridge stays above a threshold.

**B3. Candidate scoring.** For each (merged) candidate, a small CNN
(ResNet-18, 256² crop at full resolution, channels `[flattened image, ensemble
prob, candidate mask, r/R]`) trained on OOF candidates to regress
`q = fraction of annotator copies in which this candidate has IoU ≥ 0.5`.
Add scalar features (area, elongation, mean/max prob, mean contrast vs annulus,
r/R) through a GBDT and average the two. Keep candidates with
`q̂ > τ`, τ tuned on OOF but expected ≈ PQ/(2·SQ) ≈ 0.30–0.35.

Why it should work: the oracle per-image threshold already buys +0.04 with a
single scalar per image; a per-instance score with image context is strictly
more expressive. It also absorbs station/epoch shifts because it is trained on
OOF maps that already contain the ensemble's failure modes.

### C. Scale the data (days 6–13) — expected +0.03–0.05, highest variance

The labelled set is tiny and the label noise is high. Both are best addressed
with more images rather than more architecture.

**C1. Label transfer (allowed, cheap).** For each training file fetch the
same-site GONG frame(s) at ±2–10 min and reuse the annotation unchanged (solar
rotation in 10 min is < 1 px at disk centre at GONG resolution). 2–4× more
training frames with real labels and genuinely different seeing/noise. Respect
the archive's rate limiting (2 connections, 1 s delay, or request trusted-IP
status via the NSO note linked on the forum). Exclude anything within ±1 day
of any test timestamp to be safe.

**C2. Noisy-student self-training.** Download 5–20 k unlabelled GONG frames
spread over 2011–2022 and all six stations (over-sample 2021–22, which is
under-represented in train and over-represented in test). Pseudo-label with the
A+B ensemble (after re-scoring, so pseudo-labels are consensus-like), keep only
confident instances (`q̂ > 0.6`) and confident background, and train the
student from scratch on labelled + pseudo-labelled data with the full A5
augmentation. One or two rounds. This is the recipe Diercke et al. used to get
past a small labelled set on GONG, and it is the most plausible route to the
consensus behaviour that scores 0.61.

**C3. (If and only if C2 is running and GPU is idle) self-supervised
pretraining** of the ConvNeXt encoder on the same corpus (MAE or DINO-style).
Lower priority: ImageNet init already works and the frozen-DINOv2 result was
poor; the gain is speculative.

### D. Diversity and polish (days 12–15)

* A second architecture family for the ensemble (SegFormer-B3 / UPerNet-Swin-T
  trained on 1024² tiles with stitched inference) — transformers see the whole
  filament at once and fail differently from ConvNets.
* Seeds: 2 seeds per fold if time allows; the ensemble is itself a consensus
  estimator and is the single most reliable +0.01–0.02 in this setting.
* Dense CRF: low priority — SQ is already above the human-vs-human SQ.
* Mask2Former / CondInst: not as the primary system (nothing to separate, thin
  objects are their weak point), at most one ensemble member.

### E. Deliverables (days 15–17, non-negotiable — 30 % of the rubric)

Pinned `requirements.txt`, a single notebook reproducing the two selected
submissions from the test images alone, and the 4-page report with the
pipeline figure, the §3 derivation, OOF PQ/SQ/RQ with CIs, Dice/IoU
distributions and the one-to-many / many-to-one histogram the organisers said
they will inspect. Disclose any GONG frames used, with date ranges.

## 5. What not to do

* Do not touch the public MAGFiLO 1.0 annotations or anything derived from the
  0.55 notebook — it is a rules violation and the organisers review code.
* Do not spend time on watershed / NMS / box detectors — oracle says
  separation is solved.
* Do not evaluate on a deduplicated or 64-image subset, and do not compare
  numbers without a CI; differences < 0.02 local are noise.
* Do not tune on the public LB (2-decimal rounding, ~90 images, 5/day).

## 6. Expected outcome

| stage | pooled OOF PQ | public (−0.04 ± 0.02) |
|---|---|---|
| current best (resnet34, 160 ep) | 0.40 | 0.34 |
| A (preproc + recipe + ensemble) | 0.45–0.47 | 0.41–0.43 |
| + B (merge + re-score) | 0.48–0.51 | 0.44–0.47 |
| + C (label transfer + self-training) | 0.50–0.54 | 0.46–0.50 |

Anything ≥ 0.45 public is, on current evidence, the best legitimate submission
in the competition.
