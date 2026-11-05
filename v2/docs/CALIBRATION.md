# Local → public calibration

The single most important unknown in the plan: how much PQ is lost between the
pooled-annotator local validation and the hidden test set. Every target in
[`PLAN.md`](PLAN.md) §13 depends on it.

## Measurement 1 — `norm_disk_raw` (2026-10-29)

The first submission of a checkpoint whose local number was measured under the
**current** pycocotools GT. Every earlier submission used the fat `cv2.fillPoly`
GT and the wrong normalisation (0.5 / 0.25).

| | value |
|---|---|
| checkpoint | `runs/norm_disk_raw/best.pt` (resnet34, 512², disk-norm, 160 ep, stopped at 106) |
| local PQ (full 232-entry val, pycocotools GT) | **0.4225** |
| post-processing | thr 0.65, min_area 256, close_k 15, no TTA |
| submission | `submissions/norm_disk_raw_thr65_a256_c15.csv` (1,435 filaments, 177/180 images) |
| **public PQ** | **0.35** |
| **gap** | **−0.0725** |

### Verification that the submission reproduces the local number

`predict.py` had no `close_k` support, so the tuned setting could not be
reproduced from the CLI. Added `--close-k` to `predict.py` and `close_k` to
`postprocess.prob_to_labelmap`, then checked the two code paths are
**bit-identical** on all 232 val images:

```
labelmap mismatches: 0
tune.labelmap_from_prob    : pq 0.4225  sq 0.6778  rq 0.6234  tp 1109 fp 808 fn 532
postprocess.prob_to_labelmap: pq 0.4225  sq 0.6778  rq 0.6234  tp 1109 fp 808 fn 532
```

So the 0.0725 gap is a genuine local→public difference, not a post-processing
mismatch between the tuning harness and the submission path.

### What the gap is made of

| candidate cause | evidence | verdict |
|---|---|---|
| Test set is harder (2021–22 is 11.7 % of test vs 5.7 % of train) | 3 test images get zero predictions, all 2017–2021 | **contributing** |
| Local val averages multiple annotator copies per image, which flatters a consensus-like predictor | 296/707 files have 2–3 copies; the scorer pools over copies | **likely dominant** |
| Public split is ~90 images → ±0.02 noise | 2-decimal rounding | **partly** |
| Post-processing mismatch | ruled out above (0 mismatches) | **no** |

The gap is **0.0725**, not the 0.04 ± 0.02 that `docs/STRATEGY_V2.md` §6 assumes.
That is a 1.8× larger correction and it changes the target arithmetic: a local
0.52 maps to ~0.45 public, not ~0.48.

## Where 0.35 ranks

Leaderboard 2026-10-22 (1,125 teams): median 0.34, p90 0.40, max 0.69.

| threshold | teams at or above |
|---|---|
| 0.35 (us) | 554 |
| 0.36 | 476 |
| 0.40 | 141 |
| 0.41 | 106 |
| 0.55 (leak cluster) | 93 |

**Rank ≈ 477 / 1,125.** The distribution is extremely dense between 0.33 and
0.36 — 554 teams within 0.01 of us — so small local gains will not move the rank
much until we clear ~0.40, where only 141 teams sit.

## Consequences for the plan

1. **Use gap = 0.0725, not 0.04.** Revised targets:

   | stage | pooled OOF PQ | public (gap 0.0725) |
   |---|---|---|
   | current (`norm_disk_raw`) | 0.4225 | 0.35 |
   | Phase 1 | 0.46–0.48 | 0.39–0.41 |
   | + Phase 2 | 0.49–0.52 | 0.42–0.45 |
   | + Phase 3 | 0.51–0.55 | 0.44–0.48 |
   | + Phase 4 | 0.52–0.56 | 0.45–0.49 |

2. **The 0.40 public line is the real milestone**, not 0.45. Clearing it puts us
   in the top 141 and out of the 554-team pile-up.

3. **Re-calibrate after Phase 1.** If the gap shrinks when the model is trained
   on soft/consensus targets (which is the hypothesis — a consensus-like
   predictor should transfer better), the targets move up. If it grows, the test
   set is genuinely harder and Phase 3 (2021–22 oversampling) becomes the
   highest-value phase rather than the highest-variance one.

4. **The 3 zero-prediction images are worth ~0.005 public** and are free. They
   are all 2017–2021, consistent with the known P14 issue. Phase 1's
   brightness/contrast augmentation and the 2021–22 oversampling in Phase 3 both
   target this directly.

---

## Second calibration point (2026-10-31)

The 5-fold ensemble was submitted on the test set (honest — no fold saw test
data). This gives a second (local, public) pair:

| model | local PQ | public | gap |
|---|---|---|---|
| `norm_disk_raw` resnet34 | 0.4225 | 0.35 | 0.0725 |
| 5-fold ConvNeXt-S ensemble | 0.4346* | 0.36 | 0.0746 |

\* the 0.4346 is the leaky OOF ensemble number, used here only as the local
estimate for the same set of predictions that produced the 0.36. It is not
quoted as a result anywhere else.

**The gap is stable: 0.0725 vs 0.0746, a difference of 0.002.** Two different
architectures, two different preprocessing pipelines, two different
post-processing thresholds — same gap. This strongly supports the reading that
the gap is a property of the test set (different instrument, different years,
harder filaments), not of the model.

### Practical consequence

`public ≈ local − 0.073` can now be used as a planning constant with ±0.01
confidence. Revised targets:

| stage | pooled OOF PQ | expected public |
|---|---|---|
| current | 0.4225 | 0.35 (measured) |
| Phase 1 (5-fold) | 0.4220 | 0.35 (measured, ensemble 0.36) |
| Phase 2 target | 0.45 | 0.38 |
| Phase 2 stretch | 0.48 | 0.41 |
| Phase 3 | 0.51 | 0.44 |

The 0.40 public line needs local ≈ 0.473. That is +0.05 over Phase 1 — a large
but not implausible jump if Phase 2 delivers the instance-level gains the error
anatomy suggests (58.5 % of FPs are splits, 57.1 % of FNs are found-but-IoU<0.5).

---

## Third calibration point — the gap is not constant (2026-11-01)

Phase 2 (instance re-scoring) was submitted and scored **0.38 public** against
0.4386 local.

| stage | local PQ | public | gap |
|---|---|---|---|
| Phase 1 single (`norm_disk_raw`) | 0.4225 | 0.35 | 0.0725 |
| Phase 1 5-fold ensemble | 0.4346* | 0.36 | 0.0746 |
| **Phase 2 ensemble + `q̂`** | **0.4386** | **0.38** | **0.0586** |

\* leaky OOF number, used only as the local estimate for the same predictions.

### Correction to the previous section

The previous section concluded that `public ≈ local − 0.073` was a usable
planning constant, on the grounds that it was stable across two architectures and
two preprocessing pipelines. **That conclusion was premature.** It was fitted on
two points from the same pipeline family, both of which changed only the
segmentation network.

Phase 2 changes a different component — it removes false positives rather than
improving boundaries — and the gap drops by 0.014. The public gain (+0.02) is
larger than the local gain (+0.0166).

The reading that fits: **the gap depends on what kind of improvement is being
made.** A component that suppresses false positives transfers better than one
that sharpens boundaries, because the FPs it removes are partly local-split
artefacts that do not exist on the test set. A boundary improvement, by contrast,
is measured against local GT and may not transfer at all.

### Revised guidance

* Use **0.059** as the current gap estimate, not 0.073.
* Do **not** extrapolate the gap across component types. Re-measure after each
  phase that changes a different part of the pipeline.
* The 0.40 public milestone now needs local ≈ **0.459**, not 0.473. That is
  +0.020 over Phase 2 — closer than it looked before this submission.
