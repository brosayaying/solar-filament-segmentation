# Corrections from review (2026-10-24)

`FEEDBACK.md` identified several errors in this repo's analysis. I verified each
claim against the data before acting. This file records what was wrong, what the
corrected numbers are, and what changed in the code.

## 1. Rasterisation was wrong — confirmed, fixed

**Claim:** masks built with `cv2.fillPoly` are larger than the official
pycocotools rasterisation, and the hidden GT is pycocotools.

**Verified.** On 400 random annotations:

| | value |
|---|---|
| cv2 / official area ratio | **1.115** (cv2 is 11.5% larger) |
| IoU(cv2, official) | median **0.882**, p10 0.801, min 0.720 |

The annotation `area` field matches pycocotools exactly (424/424 within 1%), which
confirms pycocotools is the reference.

**Impact on our best model** (160 ep, full 232-entry val):

| GT | PQ | SQ | RQ |
|---|---|---|---|
| cv2 (what we were using) | 0.4186 | 0.7011 | 0.5971 |
| **pycocotools (official)** | **0.3986** | 0.6914 | 0.5764 |

**−0.020 PQ from evaluation alone.** The model was also *trained* on the fat
masks, so retraining should recover more.

**Fixed.** `src/solar/data.py` now has a single `rasterise()` built on
`mask_utils.frPyObjects` → `merge` → `decode`, with window offsets handled by
cropping the full-size mask (verified: crop equals the corresponding slice of the
full mask). `anns_to_mask` and `anns_to_labelmap` route through it.

## 2. Human-vs-human is a reference line, not a ceiling — I was wrong

**Claim:** a consensus predictor can beat each individual annotator, so
inter-annotator agreement does not bound model performance.

**Verified, and this retracts `docs/METRIC_CEILING.md`.** On 40 files with three
annotators each, scoring a majority-vote consensus mask (instances recovered by
connected components) against each individual annotator:

| predictor | PQ vs individual annotators |
|---|---|
| one annotator vs another | 0.3593 |
| **consensus of 3 annotators** | **0.6087** |

**+0.249.** A consensus predictor is far better than any single human, because
individual annotators make independent errors that average out. So:

* Human-vs-human (0.36) is **not** a cap on model-vs-human.
* "Stop optimising local PQ past ~0.45" was **wrong** and has been removed.
* The public leader at 0.69 is entirely consistent with this — they are closer to
  the consensus than we are.

What survives from that analysis: the *variance* is real (per-image PQ spread is
wide), but I conflated per-image spread with the standard error of a
dataset-level aggregate. Those are different things — see item 3.

## 3. Uncertainty was reported wrongly — fixed

**Claim:** quoting the per-image p10–p90 band (0.15–0.58) as a "noise band" is
wrong; the standard error of a dataset-level PQ is ~0.01.

**Verified.** `PQAccumulator.bootstrap_ci()` now resamples images with
replacement and recomputes the global aggregate. On full val (232 entries) the
95% CI for our best model is **[0.3789, 0.4160]** — width ±0.019, not ±0.2.

Consequence: differences below ~0.02 local are not distinguishable. The
tiled-vs-512 comparison (paired t = −0.18) was already handled correctly; other
comparisons in the docs were not.

## 4. Checkpoint selection was biased — fixed

**Claim:** `val_images: 64` takes the first 64 of a *sorted* id list, which is
all annotator groups `0101xx`–`0103xx` and chronologically skewed. A 160-epoch
run then takes the max over 160 noisy evaluations vs 25 for the baseline, so part
of the "+0.074" is winner's curse.

**Verified as a real mechanism.** `validate()` now selects the subset **randomly
with a fixed seed** and records per-image stats so bootstrap CIs are available.
The winner's-curse component is not yet quantified — that needs `last.pt` vs
`best.pt` on full val, which is a follow-up.

## 5. "26% transfer rate" was over-interpreted — retracted

**Claim:** inferring a transfer rate from 0.33 → 0.34 (2 decimal places) on a
~90-image public split is not sound.

**Accepted.** The public score is reported to 2 decimals, so a +0.01 step is
consistent with anything from +0.005 to +0.015, and the public split has its own
±0.02 noise. The "local compresses gains 4×" claim has been removed from the
README. It should be re-derived after items 1–2, with 5-decimal scores.

## 6. `predict.py` was missing documented features — fixed

Added:
* `--tta` — 4× flip averaging. Measured **+0.0024** on full val
  (0.3986 → 0.4010), which is **inside the CI** [0.379, 0.416]. The earlier
  "+0.008" was measured on a 32-image subset and does not replicate.
* `--tile` — honours `cfg["eval"]["tile"]`, so ViT checkpoints are no longer
  silently run at 2048².
* Per-image prediction counts and a warning listing images with zero predictions.

## 7. Strict IoU matching — fixed

`match_instances` now uses `iou <= threshold` to reject, so IoU exactly 0.5
counts as a match, matching the official scorer.

## What this changes strategically

The previous conclusion was "we are above the human ceiling, stop optimising
local PQ". That was wrong. The corrected picture:

* Our real local PQ is **0.3986**, not 0.4186 — the rasterisation bug was
  inflating it.
* The ceiling is **not** 0.36–0.50; a consensus predictor reaches 0.61 against
  individual annotators, and the public leader is at 0.69.
* So there is a genuine gap of roughly **0.40 → 0.61+** to close, and the
  "we're done" framing was premature.

The highest-value next steps, per the review and consistent with what we measured:

1. **Retrain on pycocotools masks** — the model currently learns fat targets.
2. **Multi-annotator-aware targets** — for files with 2–3 annotators, train on the
   mean/consensus mask. This directly optimises the official scorer, which
   evaluates against every annotator variant.
3. **EMA + ensemble** — cheap, standard, and the review notes they are untried.
4. **Limb flat-fielding** — recall is 0.28 in the outer ring and this was
   diagnosed but never fixed.

## Files changed

| file | change |
|---|---|
| `src/solar/data.py` | `rasterise()` on pycocotools; `anns_to_mask`/`anns_to_labelmap` route through it |
| `src/solar/metrics.py` | `bootstrap_ci()`; strict `iou <= threshold` |
| `src/solar/train.py` | random fixed-seed val subset; `record=True` for CIs |
| `src/solar/predict.py` | `--tta`, `--tile`, empty-image warnings |
| `docs/METRIC_CEILING.md` | marked superseded |
| `README.md` | ceiling claim and transfer-rate claim removed |

## Remaining feedback items — status

Addressed in this pass:

| item | status |
|---|---|
| 1. pycocotools rasterisation | **done** — `data.rasterise()`, verified 424/424 areas match |
| 2. Full val as the metric | **done** — `dedup_val_files` kept only as a labelled diagnostic |
| 3. Retract metric ceiling | **done** — `METRIC_CEILING.md` marked superseded, README rewritten |
| 3. Bootstrap CIs | **done** — `PQAccumulator.bootstrap_ci()` |
| 4. Checkpoint selection | **done** — random fixed-seed subset |
| 5. Transfer-rate claim | **done** — retracted in README |
| 6. `predict.py` TTA / tile / warnings | **done** — `--tta`, `--tile`, empty-image report |
| 7. Strict IoU matching | **done** — `iou <= threshold` rejects |
| 8. Seed `random` | **done** |

Not yet done, in priority order:

| item | why it matters |
|---|---|
| **Retrain on pycocotools masks** | running now (`configs/long_pycocotools.json`) |
| **Multi-annotator consensus targets** | directly optimises the official scorer, which evaluates against every annotator variant. Highest-value untried idea. |
| **EMA weights** | cheap, standard, also mitigates the checkpoint-selection bias |
| **Ensemble (3–5 seeds)** | decorrelated errors; the tiled/512 pair is a natural start |
| **Limb flat-fielding** | recall 0.28 in the outer ring, diagnosed but never fixed |
| **Official-scorer local harness** | write val preds + GT as submission-format RLE and run the notebook's `get_pq_score`; the final check before every submission |
| **`group_by: "week"` split** | date grouping still puts adjacent days on both sides |
| **Converged backbone comparison** | the 6-epoch 384² sweep cannot support "bigger is not better" |
| **Test-distribution check** | 2021–22 is 11.7% of test vs 5.7% of train; inspect the zero-prediction images |
| **Freeze the patch/pixel line** | `scripts/patch_*.py` re-derives what `src/solar` already does |
