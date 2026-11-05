# Phase 3 — scaling the data

Written 2026-11-01. Implements §7 of [`PLAN.md`](PLAN.md) with two corrections
that came out of verifying the archive before writing code.

**Status: 3.1 (label transfer) complete and training. 3.2 (noisy student) not
started.**

---

## 1. Two corrections to the plan

### 1.1 The competition images are the GONG `hag` JPEGs, not the `haf` FITS

The plan specifies downloading `haf/YYYYMM/YYYYMMDD/<ts><station>.fits.fz` and
converting to uint8. That would have introduced a domain shift. Measured against
the competition image for `20140609195854Bh`:

| source | correlation with competition JPEG |
|---|---|
| `hag/.../<ts>.jpg` | **0.997** (median abs diff 1, best shift 0,0) |
| `haf/.../<ts>.fits.fz` + best-fit linear stretch | 0.858 |

The `haf` product is a different reduction. **Label transfer downloads JPEGs and
needs no radiometric normalisation at all** — the transferred frames are already
in the same photometric space as the training data.

### 1.2 Solar rotation is ~1.6 px at 10 min, not <1 px

The plan says "solar rotation in 10 min is < 1 px at disk centre". A feature at
heliographic longitude `L` projects to `x = R sin L`, so at disk centre
`dx = R dL` with `R = 1024 px` and `dL = 0.00916 deg/min`:

| offset | displacement at centre | at r/R = 0.8 |
|---|---|---|
| 2 min | 0.33 px | 0.20 px |
| 5 min | 0.82 px | 0.50 px |
| 10 min | 1.64 px | 0.99 px |
| 30 min | 4.91 px | 2.96 px |

Filaments are ~11 px wide, so ±10 min is safe and ±30 min is marginal. The
window used is **±2/5/10 min**, matching the plan's intent.

Verified empirically: for three files across three stations, the best-alignment
shift between the original and the ±5/±10 min frame is **(0, 0)**. The residual
correlation loss (0.90–0.93) is real solar evolution and seeing, which is exactly
the augmentation value the transfer is meant to provide.

## 2. What was built

`src/solar/gong.py` — URL builder, downloader, timestamp arithmetic, disk
geometry. `v2/scripts/build_label_transfer.py` — the transfer set.
`v2/scripts/build_phase3_dataset.py` — the leakage-safe merge.

### The transfer set

| | value |
|---|---|
| source files | 707 |
| offsets per file | ±2, ±5, ±10 min |
| frames requested | 3,794 |
| excluded for test proximity (±1 day) | 448 |
| downloaded | 2,433 |
| missing (archive gaps) | 1,361 (35.9 %) |
| source files covered | 618 / 707 |
| frames per source | min 1, median 3, max 6 |
| annotations | 28,167 |

The 35.9 % miss rate is GONG archive gaps and station downtime, not an error —
404 is the normal response for a frame that does not exist.

### The leakage rule

A transferred frame is a different observation of the *same region* minutes
later, so it carries essentially the same filaments. Training on a frame derived
from a validation file while validating on the original would leak.

`build_phase3_dataset.py` admits a transferred frame only if its `source_stem` is
in the fold's **train** list, and asserts this at the end. For fold 0:

| | count |
|---|---|
| original train entries | 930 |
| transferred admitted | 1,915 |
| transferred rejected (source is val) | 518 |
| **total train entries** | **2,845** (3.1×) |
| val entries | 224 (unchanged) |

## 3. Training

`v2/configs/phase3_fold0.json` — identical to `seg_v2.json` except the data root,
split file, run name, and 60 epochs with patience 15 (the Phase 1 runs used 160
epochs; with 3.1× the data per epoch, 60 is a comparable number of gradient
steps).

Measured: **474 steps/epoch, ~11 min/epoch**, so 60 epochs is ~11 h. Early
stopping on val PQ with `min_delta` 0.002.

The transferred frames are not in the flatten cache, so they are preprocessed on
the fly. `FilamentDataset._cached` falls through to computing rather than
failing, which is what makes this work without a cache rebuild.

## 4. Acceptance (§7.3)

| check | threshold | result |
|---|---|---|
| pooled OOF PQ | ≥ Phase 2 + 0.02 | pending |
| 2021–22 subset PQ | improves more than overall | pending |
| outer-ring (r/R > 0.8) recall | improves | pending |

The comparison is against Phase 2's **0.4406** (ensemble + `q̂` scorer), not
against Phase 1's 0.4220 — the scorer is part of the pipeline now.

## 5. What is not done

**3.2 noisy-student self-training** is not started. It needs 5–20 k unlabelled
frames (28 GB at 10 k, and only 46 GB is free), pseudo-labelling with the Phase 2
ensemble, and ~13 h per training round. The plan's own kill criterion applies:
*"If self-training does not beat label transfer alone, ship label transfer and
drop the student."* Label transfer has to be measured first.

## 6. Reproduce

```bash
.venv/bin/python v2/scripts/build_label_transfer.py --workers 6   # ~20 min
.venv/bin/python v2/scripts/build_phase3_dataset.py --fold 0
.venv/bin/python -m solar.train --config v2/configs/phase3_fold0.json
```

---

## 7. Training run (in progress, started 2026-11-01 09:50)

`v2/runs/phase3_fold0/`, config `v2/configs/phase3_fold0.json`.

Measured rate: **474 steps/epoch at ~1.6 s/it = 12.6 min train + ~1 min val =
~14 min/epoch**. 60 epochs is therefore **~14 h**, finishing around 2026-11-02
00:00.

That is consistent with Phase 1 (109 s/epoch × 160 = 4.8 h/fold) scaled by the
3.1× data increase. 60 epochs on 3.1× data is 186 Phase-1-equivalent epochs, so
the run is not undertrained relative to Phase 1.

Early curve (val PQ, 224 entries):

| epoch | loss | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| 1 | 0.6294 | 0.1487 | 0.5923 | 0.2511 | 440 | 1510 | 1114 |
| 2 | 0.4295 | 0.2741 | 0.6331 | 0.4329 | 802 | 1349 | 752 |

Phase 1 fold 0 reached PQ 0.4177 at its best epoch. Two epochs is far too early
to compare, but the trajectory is normal and there is no sign of the fp16
divergence that killed the Phase 1 A/B arms (bf16 is in use).

### What to check when it finishes

1. **Best val PQ vs Phase 1 fold 0's 0.4177.** This is a like-for-like comparison:
   same fold, same val entries, same architecture, same preprocessing. Only the
   training data differs.
2. **The 2021–22 subset.** The plan's acceptance criterion is that this improves
   *more* than the overall PQ, since that is the stated point of the phase. The
   transferred frames add 85 + 40 = 125 frames for 2021–22 against 61 original
   entries, so the relative increase there is larger than average.
3. **Outer-ring recall (r/R > 0.8).** Not yet instrumented; needs a small script.

If best val PQ does not beat 0.4177, label transfer has failed and the plan's kill
criterion applies — ship Phase 2 and drop both label transfer and the student.

### Monitoring

`v2/scripts/phase3_status.py` prints the curve, the best-so-far, the delta against
Phase 1 fold 0, and the early-stopping state. Measured epoch time from
`log.jsonl`: **831 s (13.9 min)**, so 60 epochs is ~13.9 h.

### A note on running evaluations concurrently

The first attempt to compute the Phase 1 subset baseline alongside training hit
`CUDA out of memory`: training holds 18.3 GB of 24.6 GB, and a full-resolution
2048² forward pass needs ~2.5 GB more than the 2.1 GB free. Running the eval on
CPU instead worked but competed for the 12 cores and slowed training, so it was
killed. **The baseline will be computed after training finishes.** This is worth
recording because it is the kind of thing that silently corrupts a comparison —
an eval that OOMs halfway leaves a partial accumulator that looks like a result.

### Two logging artefacts worth knowing about

**1. `log.jsonl` has a stale epoch-1 row.** The smoke test that was killed after
one epoch wrote its own epoch-1 entry before being terminated, so the file has two
epoch-1 rows: PQ 0.0861 (smoke, killed) and PQ 0.1487 (real run). Harmless —
`best.pt` is written by the real run and correctly holds epoch 6 — but any script
that reads `log.jsonl` must not assume one row per epoch. `phase3_status.py`
prints both and takes the max, which is the safe behaviour.

**2. The stdout log's epoch lines can be clobbered by tqdm.** The progress bar
writes carriage returns, so `grep '^epoch'` on the raw log misses epochs whose
line was overwritten. `phase3_status.py` t
### Disk-full incident (2026-11-01 11:20)

Writing this section failed with `No space left on device` — the volume hit 100 %
while training was running. Training itself survived (no write errors in its log,
`best.pt` intact), but the next checkpoint write would have failed.

Freed 40 GB by deleting two classes of regenerable or already-documented-unusable
data:

| deleted | size | why safe |
|---|---|---|
| `v2/runs/seg_v2/ensemble_probs/` | 5.6 GB | intermediate for the leaky ensemble measurement, which is recorded in `ensemble_results.json` |
| `runs/*/val_probs/` (7 dirs) | 34 GB | **stale GT** — all predate the pycocotools rasteriser switch and are documented as unusable in `PLAN.md` §1.1 |

Every checkpoint, `log.jsonl` and `config.json` was kept. Verified afterwards that
`runs/norm_disk_raw/best.pt` and `runs/baseline_resnet34/best.pt` still exist.

**Lesson.** The disk was at 98 % before this phase started and the plan's own
budget section noted 186 GB free — that was measured weeks earlier. A 14-hour
training run that writes a 208 MB checkpoint every time val PQ improves needs
headroom checked *at launch*, not inherited from an old measurement. This is the
same failure mode as the stale `val_probs` caches: a number that was true once and
was never re-checked.

---

## 8. Interim finding: label transfer is plateauing below Phase 1

At epoch 17 of 60, val PQ has flattened:

| ep | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|
| 13 | 0.3663 | 0.6621 | 0.5532 | 813 | 572 | 741 |
| 14 | 0.3547 | 0.6587 | 0.5385 | 903 | 897 | 651 |
| 15 | 0.3635 | 0.6536 | 0.5561 | 850 | 653 | 704 |
| 16 | 0.3610 | 0.6589 | 0.5478 | 825 | 633 | 729 |
| 17 | 0.3641 | 0.6640 | 0.5483 | 854 | 707 | 700 |

Last 6 epochs: **mean 0.3614, sd 0.0042**. Phase 1 fold 0 reached **0.4177**.
Gap: **−0.056**.

### What this is not

Three possible explanations were checked and ruled out:

1. **Not a preprocessing mismatch.** The original files come from the flatten
   cache; the transferred frames are preprocessed on the fly. Verified on
   `20140609195854Bh`: cached vs freshly computed differ by max 0.0625 grey
   levels (mean 0.012), which is float16 storage rounding, not a pipeline
   difference.
2. **Not a label-alignment failure.** Transferred frames were verified earlier:
   best-alignment shift (0,0), and the p25 grey level on GT matches the control
   exactly (−2.0 vs −2.0, −1.0 vs −1.0).
3. **Not undertrained.** 60 epochs on 3.1× data is 186 Phase-1-equivalent
   epochs, and loss is still falling (0.63 → 0.17).

### What it probably is

The transferred labels are **the same annotations applied to a different
observation**. Solar evolution over ±10 min is small but not zero, and the
seeing differs. So the model is being trained to reproduce one annotator's
polygons on frames where those polygons are slightly wrong — which teaches it to
be *less* precise about boundaries, not more.

That is consistent with the numbers: **SQ is flat (0.653–0.664) while RQ moves
(0.524–0.556)**. More data is improving instance *detection* and not instance
*drawing*, which is the opposite of what the phase was supposed to deliver.

### Decision point

The plan's kill criterion (§7.3) is: *"If self-training does not beat label
transfer alone, ship label transfer and drop the student."* The stronger version
applies here — if label transfer does not beat Phase 1, drop it too.

Training continues to epoch 60 because early stopping may still fire and the
curve could recover, but the prior is now that **label transfer will not beat
Phase 1's 0.4177**. If it does not, the correct action is to ship Phase 2
(public 0.38) and move to Phase 4 (ensemble diversity), not to spend the
remaining days on the noisy student.

---

## 9. Final result: label transfer FAILED the primary criterion

Training finished at epoch 58 (early stopping, patience 15). Best `best.pt` is
epoch 53, val PQ **0.4045**.

The curve recovered substantially from the epoch-17 plateau (0.3614) — the
interim pessimism in §8 was premature — but it still does not beat Phase 1.

| | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|
| Phase 1 fold 0 | **0.4213** | 0.6750 | 0.6241 | 1037 | 732 | 517 |
| Phase 3 fold 0 | 0.4063 | 0.6667 | 0.6095 | 991 | 707 | 563 |
| delta | **−0.0150** | −0.0083 | −0.0146 | −46 | −25 | +46 |

### Acceptance (§7.3)

| check | threshold | result | |
|---|---|---|---|
| pooled OOF PQ | ≥ Phase 2 + 0.02 (0.4606) | 0.4063 | **FAIL** |
| 2021–22 improves more than overall | — | +0.0109 vs −0.0150 | **PASS** |
| outer-ring recall improves | — | +0.0435 | **PASS** |

**Two of three pass, and the one that fails is the one that matters.** The
mechanism worked exactly as designed — the 2021–22 subset improved while
everything else regressed, and limb recall improved — but the overall level is
below Phase 1, so the phase does not ship.

### The per-year breakdown is the interesting part

| year | Phase 1 | Phase 3 | delta |
|---|---|---|---|
| 2011 | 0.3657 | 0.3182 | **−0.0475** |
| 2012 | 0.4353 | 0.4335 | −0.0017 |
| 2013 | 0.4442 | 0.4419 | −0.0023 |
| 2014 | 0.4341 | 0.4179 | −0.0163 |
| 2015 | 0.4028 | 0.4059 | +0.0031 |
| 2016 | 0.4782 | 0.4356 | **−0.0426** |
| 2017 | 0.3063 | 0.2674 | −0.0389 |
| 2018 | 0.4663 | 0.4688 | +0.0025 |
| 2021–22 | 0.3758 | 0.3867 | **+0.0109** |

The gains are exactly where the extra frames were concentrated (2021–22, 2015,
2018) and the losses are in 2011, 2016 and 2017 — years where the transferred
frames were *fewest* relative to the original count, so the model spent capacity
on data that did not represent them.

### Why it failed

The transferred labels are one annotator's polygons applied to a different
observation. Solar evolution over ±10 min is small but non-zero, and the seeing
differs, so the model is trained to reproduce polygons that are slightly wrong on
those frames. That teaches boundary imprecision, which is visible in SQ
(0.6750 → 0.6667).

The extra data did what it was supposed to for *detection* — limb recall
+0.0435, 2021–22 +0.0109 — but the boundary cost outweighed it.

### Decision

**Ship Phase 2 (public 0.38). Drop label transfer and the noisy student.**

The plan's kill criterion (§7.3) says to drop the student if it does not beat
label transfer. The stronger reading applies: label transfer itself does not beat
Phase 1, so neither ships. Spending the remaining days on a noisy student built
on top of a weaker teacher would compound the loss.

Remaining time goes to **Phase 4 (ensemble diversity)** — the PVTv2-B2 member is
the untried part, and the 5-fold averaging is already measured.

### A note on the interim call in §8

§8 predicted label transfer would not beat Phase 1, based on the epoch-17
plateau. That prediction was **correct in conclusion but wrong in reasoning** —
the curve went on to gain +0.043 after epoch 17, and the final gap (−0.015) is
much smaller than the −0.056 seen at the plateau. Judging a 60-epoch run at epoch
17 was too early. The conclusion held, but it held by luck as much as by
analysis.
