# Review feedback — actionable steps

Review of the pipeline in `src/solar`, the experiment docs, and the submissions,
cross-checked against the organizers' self-evaluation notebook and the
competition data page (2026-10-24). Items are ordered by expected impact.
Each has a concrete action, the evidence, and how to verify it.

---

## P0 — Fix the local metric so it measures the official ground truth

### 1. Rasterise masks with pycocotools, not `cv2.fillPoly`

**Problem.** Every mask in the repo is produced by `cv2.fillPoly` on rounded
integer vertices: training targets (`anns_to_mask`), val GT
(`anns_to_labelmap`), `scripts/annotator_agreement.py`,
`detector_metrics.py`, `embed.py`, `scripts/embed_dinov2.py`. The hidden GT is
pycocotools polygon rasterisation — the annotation `area` field equals
`pycocotools.mask.area(frPyObjects(...))` exactly on 300/300 sampled
annotations, and the organizers' notebook decodes GT from RLE.

**Measured.** On 400 random annotations, cv2 masks are **10.5 % larger** than
the official ones; median IoU between the two rasterisations is **0.88**
(p10 0.80, min 0.72). Re-scoring `runs/baseline_resnet34/val_probs` (all 232
val entries) against both GTs:

| thr | GT | PQ | SQ | RQ |
|---|---|---|---|---|
| 0.55 | cv2 (repo) | 0.4101 | 0.682 | 0.601 |
| 0.55 | pycocotools (official) | **0.3898** | 0.666 | 0.585 |
| 0.75 | cv2 | 0.4103 | 0.684 | 0.600 |
| 0.75 | pycocotools | **0.3958** | 0.672 | 0.589 |

That is −0.02 PQ from evaluation alone; the model was also *trained* on the
fat masks, so retraining should recover more. The "threshold 0.75–0.80 free
win" is partly compensating for this bloat.

**Actions.**
- [ ] Add a single `rasterise(anns, h, w, offset) -> mask/labelmap` helper in
      `data.py` built on `mask_utils.frPyObjects` → `merge` → `decode`, with
      offset handled by cropping the full-size mask (or translating polygon
      coordinates before encoding).
- [ ] Route `anns_to_mask`, `anns_to_labelmap`, `detector_metrics.py`,
      `embed.py`, `scripts/annotator_agreement.py` through it. Delete the
      duplicated `fillPoly` copies in `scripts/`.
- [ ] Cache val label maps once (`scripts/build_cache.py`) since pycocotools
      decode is slower than `fillPoly`.
- [ ] Retrain `configs/baseline.json`, re-run `python -m solar.tune`, and
      re-submit. Record old-vs-new local PQ **and** public score side by side.
- [ ] Verify: for a handful of annotations, `rasterise(...).sum() == ann["area"]`.

### 2. Evaluate against every annotator entry — remove the dedup recommendation

**Problem.** The organizers' notebook iterates GT entries keyed
`<annotator_id>-<image_id>` and matches predictions by bare `<image_id>`; the
data page says images "may be annotated multiple times by different
annotators, independently". So one prediction is scored against each annotator
variant on test, exactly as the full 232-entry val does. `README.md` and
`docs/METRIC_CEILING.md` recommend the opposite (dedup to 142), and the
explanation of the local→public gap ("val averages annotators, test uses
one") is unsupported. (`tune.py` already uses all 232 — those numbers are the
faithful ones. No config actually sets `dedup_val_files`.)

**Actions.**
- [ ] Make the full non-dedup val set the single reported metric; keep
      `dedup_val_files` only as an explicitly-labelled diagnostic.
- [ ] Rewrite the "metric ceiling" and "local vs public gap" sections of
      `README.md` / `docs/METRIC_CEILING.md` (see item 3).
- [ ] Build a local scorer that *is* the official one: write val predictions
      to a submission-format CSV, write val GT as `<annotator>-<image>_<n>`
      RLE rows, and run the notebook's `get_overlap_df` / `get_pq_score`.
      Use it as the final check before every submission.

### 3. Retract the "metric ceiling" argument and use proper uncertainty

**Problem.** Human-vs-human PQ (0.36) is not a bound on model-vs-human PQ: a
consensus predictor beats each annotator against the others. The repo's own
model already does (0.436 vs 0.372), and the public top is 0.69 against the
same kind of GT. "Stop optimising local PQ past ~0.45" is the wrong strategic
conclusion at 0.34 vs a leader at 0.69.

"0.02–0.05 differences are inside the noise band (p10 0.15 – p90 0.58)"
confuses per-image spread with the standard error of a dataset-level
aggregate (≈0.01–0.013 for n = 142–232, sd ≈ 0.16).

**Actions.**
- [ ] Add a per-image bootstrap (resample images, recompute global PQ, 1000
      reps) to `PQAccumulator`/`tune.py` and report 95 % CIs on every table.
- [ ] Use the paired per-image test (already done once for tiled-vs-512)
      for every A/B comparison; stop quoting the p10–p90 band as noise.
- [ ] Delete "Stop optimising local PQ past ~0.45" and the 0.36–0.50 ceiling
      claim from the docs; keep human-vs-human as a *reference line*, not a cap.

### 4. Fix checkpoint selection

**Problem.** `train.val_images: 64` + early return in `validate()` takes the
first 64 of a *sorted* id list: all annotator groups `0101xx`–`0103xx`,
chronologically skewed. Every `best.pt` is chosen on this subset. A 160-epoch
run takes the max over 160 noisy evaluations vs 25 for the baseline, so part
of the "+0.074" is winner's curse. Post-processing is then tuned on the full
val, which contains those 64 images.

**Actions.**
- [ ] Select the subset randomly with a fixed seed (or validate on the full
      set every k epochs; full val is ~0.1 s/image).
- [ ] Use an EMA of the weights and/or `last.pt` plus a short tail average
      instead of argmax-over-epochs; compare `best.pt` vs `last.pt` on full
      val to quantify the selection bias.
- [ ] Report full-val PQ for the *same* checkpoint used for submission, and
      say which subset selected it.

### 5. Stop inferring "transfer rates" from 2-decimal leaderboard numbers

**Problem.** "160 ep gained +0.038 locally and +0.01 publicly → 26 % transfer,
local compresses gains 4×" rests on 0.33 → 0.34 (rounded to 0.01) on a ~90
image public split.

**Actions.**
- [ ] Pull 5-decimal scores from the Kaggle "My Submissions" page/API and
      replace every rounded number in the README tables.
- [ ] State public-score noise (~±0.02 at 90 images) next to every public number.
- [ ] Re-derive the local→public ratio after items 1 and 2; expect most of
      the gap to close.

---

## P1 — Code fixes

### 6. `predict.py` is missing things the docs call "free wins"
- [ ] Add 4× flip (or 8× dihedral) TTA with probability averaging; the docs
      measured +0.008.
- [ ] Honour `cfg["eval"]["tile"]` (reuse `_predict_tiled` from `train.py`)
      so ViT checkpoints are not silently run at 2048².
- [ ] Default `--threshold` to the tuned value stored in
      `tune_results.json`, not `cfg["eval"]["threshold"]` (0.55).
- [ ] Log the per-image prediction count and list images with zero
      predictions (3–5 per submission today, all 2017–2021 — see item 14).

### 7. Align `metrics.py` with the official scorer
- [ ] `match_instances`: use strict `iou > threshold` (currently IoU == 0.5
      counts as a TP).
- [ ] Evaluate GT as independent layers rather than an overwriting label map
      (0 overlapping GT pairs found in 150 entries, so harmless today, but it
      is a latent divergence from the notebook).

### 8. Reproducibility
- [ ] Seed Python `random` (used for crops and augmentation) in `train.py`
      alongside `torch`/`numpy`; set `worker_init_fn`.
- [ ] Run the baseline with 3 seeds once so "run-to-run variance" is a
      measured number for the main pipeline, not just the 60-image U-Net.

### 9. Split leakage
- [ ] Date grouping still puts adjacent days (same filaments, different
      stations) on both sides. Add `group_by: "week"` (7–14 day window) and
      check how much local PQ drops; if it moves, use it.

---

## P2 — Research direction

### 10. Resolve the instance-identity contradiction before more two-stage work
The README says "instance separation is solved: GT + CC gives PQ 1.0, watershed
has nothing to win", yet the main in-progress effort is a CenterNet stage 1
justified by "the semantic head has no notion of instance identity". Stage 1's
mask-hit recall (0.65) equals the seg model's own instance recall
(1073 / 1641 = 0.65), so it adds nothing yet.
- [ ] Run the GT+CC test on *predicted* masks: dilate GT by the model's
      typical over-segmentation (≈1–2 px) and re-run CC. If PQ stays ~1.0,
      merges are genuinely not the problem and stage 1 should be paused.
- [ ] Decompose RQ loss on full val into {merge, split, missed, spurious}
      with `scripts/pq_decompose.py` against pycocotools GT. Pick the next
      experiment from the largest bucket, not from a hypothesis.

### 11. Act on the limb diagnosis
Recall is 0.28 in r/R 0.8–1.0 (≈11 % of filament area) because contrast
inverts under limb darkening. Diagnosed in `docs/DISK_MASKING.md`, never fixed.
- [ ] Add radial flat-fielding as a preprocessing option: fit/smooth the
      azimuthal mean intensity profile `I(r)`, divide (or subtract) it, then
      normalise. Apply identically at train and inference.
- [ ] Alternatively feed `r/R` as a second input channel so the network can
      condition on radius.
- [ ] Verify with the per-ring recall table from `scripts/disk_analysis.py`.

### 12. Standard levers not yet tried (cheap, usually +0.01–0.03 each)
- [ ] EMA weights (decay 0.999) — also fixes item 4.
- [ ] Ensemble: 3–5 seeds or folds of the same config, average probabilities.
- [ ] Pseudo-label the 180 test images with the ensemble and fine-tune.
- [ ] Multi-annotator-aware targets: for files with 2–3 annotators, use the
      mean mask (soft target) or weight pixels by agreement. This directly
      optimises the official scorer, which scores against every annotator.
- [ ] A converged backbone comparison: convnext_tiny / efficientnet_b3 /
      resnet50 at 512², 60+ epochs, lr 1e-4 for transformers. The 6-epoch
      384² sweep cannot support "bigger is not better".

### 13. Deprioritise the hand-built patch/pixel line
`scripts/patch_clf.py` → `patch_cnn.py` → `pixel_cnn.py` (≈1 k lines) re-derive
that a dense U-Net beats patch classification, which `src/solar` already
embodies with a pretrained encoder on 922 images (0.42 vs 0.277 on 60).
- [ ] Freeze this line; keep `docs/PIXEL_CNN.md` as a record but fix its
      internal inconsistencies: §3 quotes TP 40→73 / FP 90→123 / RQ 0.426 /
      SQ 0.652 (run 1) while §2's table reports 67/110/80, RQ 0.414, SQ 0.665
      (run 2). Note the 512-tile inference has no overlap (seam artefacts) and
      the "19× more supervision" ignores spatial correlation of pixel labels.
- [ ] If the small-model angle matters for the report, do it inside
      `src/solar` with `model.pretrained=false` so it shares data, metric and
      post-processing with the baseline.

### 14. Check the test distribution
- [ ] 2021–22 is 11.7 % of test vs 5.7 % of train files. Visualise the 3–5
      test images that receive zero predictions (all 2017–2021). If they are
      quiet-Sun images, fine; if they show a processing change, add
      brightness/contrast augmentation or the flat-fielding from item 11.

---

## Documentation hygiene
- [ ] `README.md`: one results table, one evaluation protocol (full val,
      pycocotools GT, bootstrap CI), 5-decimal public scores. Remove the two
      incompatible val subsets.
- [ ] Every doc that draws a conclusion from a <0.02 local difference should
      either show a CI/paired test or be marked "not significant".
- [ ] Mark `docs/METRIC_CEILING.md` as superseded with a pointer to this file.

## Suggested order
1 → 2 → 6 (TTA + tuned threshold) → retrain + submit → 4/8 → 11 → 12 (EMA,
ensemble) → 10 → 9/14. Items 3, 5 and the doc hygiene can be done alongside.
