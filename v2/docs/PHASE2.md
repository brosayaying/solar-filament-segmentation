# Phase 2 — instance re-scoring

Written 2026-10-31. Supersedes §6 of [`PLAN.md`](PLAN.md), whose central premise
turned out to be false. Everything below is measured on the honest Phase 1 OOF
dumps (707 files, 1,154 annotator entries, pycocotools GT).

**Headline: +0.0186 pooled OOF (0.4220 → 0.4406), paired CI [+0.0109, +0.0199].
Below the +0.03 target, but real, and it clears the FP criterion. Fragment
merging — the plan's first half — is dead.**

---

## 1. The plan's premise was wrong

`PLAN.md` §6 and `docs/PQ_DIAGNOSIS.md` both claim **58.5 % of false positives are
splits of an already-matched GT instance**. That number drove the whole design:
fragment merging was to be the first deliverable, and the per-instance scorer was
gated behind it.

Re-measured on the current OOF dumps (`v2/scripts/diagnose_fp.py`):

| FP class | count | share | what it is |
|---|---|---|---|
| `background` | 1,608 | 47.3 % | overlaps no GT at all |
| `unmatched_gt` | 1,735 | 51.0 % | overlaps a GT that no TP matched |
| `fragment` | **60** | **1.8 %** | overlaps a GT a TP already claimed |
| `duplicate` | 0 | 0.0 % | near-copy of a TP |

**Fragments are 1.8 %, not 58.5 %.** The old figure predates the pycocotools
rasteriser switch and the ConvNeXt model; it does not describe the current
pipeline.

The oracle ceilings confirm it. Removing *all* fragments is worth **+0.0015**:

| oracle (perfect removal of one class) | PQ | gain |
|---|---|---|
| remove all `fragment` | 0.4235 | **+0.0015** |
| remove all `background` | 0.4665 | +0.0445 |
| remove all `unmatched_gt` | 0.4704 | +0.0484 |
| remove all FPs | 0.5287 | +0.1067 |

So §2.2 (fragment merging) was abandoned before it was built. This was checked
two ways: the FP anatomy above, and a direct count of GT instances matched by more
than one candidate at thresholds 0.30/0.40/0.50/0.60 — **zero at every
threshold**, and 890 candidate pairs with **0 positive merge labels**.

The reason is structural: lowering the threshold *fattens* the mask until
fragments reconnect, so the low-threshold candidate set never contains the splits
that a high-threshold pipeline produces. Merging solves a problem this pipeline
does not have.

## 2. What was built instead

`src/solar/instances.py` — candidate generation, per-candidate features, per-pair
features, `q` labelling, union-find merging (kept for completeness, unused).

**No spine feature.** The plan specifies "mean spine-heatmap value along the
bridge". The `spines` array in the OOF dumps is the *ground-truth* centreline
rasterised from the annotation, not a model prediction — using it would be
leakage, since it does not exist on the test set. Bridge evidence is read from the
probability map instead.

Pipeline:

1. **Candidates** — threshold 0.30, connected components ≥ 128 px. Deliberately
   over-produces: 6,166 candidates over 695 files (8.9/file) against 5,259 TPs.
2. **Labels** — `q` = fraction of annotator copies in which the candidate has
   IoU > 0.5. Distribution: 38.5 % at `q = 0`, 58.1 % at `q ≥ 0.5`, mean 0.555.
3. **Scorers** — a GBDT on 17 scalar features, and a ResNet-18 on 4-channel
   256² crops (`[image, prob, mask, r/R]`), both cross-validated **by file**.
4. **Decision** — keep candidates with score > τ.

## 3. Results

| variant | τ | PQ | SQ | RQ | tp | fp | fn | ΔPQ |
|---|---|---|---|---|---|---|---|---|
| Phase 1 (thr .50 / area 256 / close 15) | — | 0.4220 | 0.6765 | 0.6238 | 5259 | 3403 | 2940 | — |
| GBDT | 0.35 | 0.4386 | 0.6786 | 0.6464 | 5015 | 2303 | 3184 | +0.0166 |
| CNN | 0.20 | 0.4285 | 0.6766 | 0.6333 | 5076 | 2756 | 3123 | +0.0065 |
| **blend 0.7 GBDT + 0.3 CNN** | **0.30** | **0.4406** | 0.6784 | 0.6495 | 5017 | 2233 | 3182 | **+0.0186** |

Best: **PQ 0.4406, paired per-image +0.0154, CI [+0.0109, +0.0199]** — excludes 0.

### Acceptance

| check | target | result | |
|---|---|---|---|
| pooled OOF PQ | ≥ Phase 1 + 0.03 | +0.0186 | **FAIL** |
| paired CI vs Phase 1 | excludes 0 | [+0.0109, +0.0199] | **PASS** |
| FP count | down ≥ 25 % | −34.4 % | **PASS** |
| FN count | not up > 5 % | +8.2 % | **FAIL** |

Two of four. The gain is real but undersized, and it is bought by losing 242 true
positives.

## 4. Why the CNN lost

`corr(q̂, q)`: **GBDT 0.575, CNN 0.507.** The CNN is worse, and the blend only
helps because the two make different mistakes.

This is not a bug. 6,166 crops from 695 images is small for a ResNet-18, and the
scalar features already encode most of what the crop shows — area, elongation,
mean probability, contrast against the annulus. The CNN's potential advantage is
context the scalars cannot express (does the surrounding field look like the rest
of the image?), and 15 epochs on 5k crops does not learn it. The blend weight that
wins is 0.7 GBDT, i.e. the CNN contributes a small correction.

**Do not invest further in the CNN without more data.** Phase 3 (GONG label
transfer) is the prerequisite, not more epochs.

## 5. The ceiling says the scorer is still the bottleneck

`v2/scripts/ceiling_phase2.py` applies *perfect* `q` to the same candidate set:

| rule | PQ | ΔPQ |
|---|---|---|
| Phase 1 | 0.4220 | — |
| oracle `q > 0` (keep any candidate matching GT) | **0.5028** | **+0.0808** |
| oracle `q ≥ 0.5` | 0.5002 | +0.0782 |
| predicted `q̂ > 0.35` | 0.4386 | +0.0166 |

**The candidate set is worth +0.0808; the scorer captures +0.0186 of it.** So
+0.062 remains in scoring, not in candidate generation. That is the single most
useful number Phase 2 produced: it says the approach is sound and the model is
weak, which is a data problem, not a design problem.

## 6. The decision rule

The plan's arithmetic — keep iff `2·q·s > PQ` — was implemented as an
expected-value rule:

$$\mathbb{E}[\text{gain}] = q s - \text{PQ}\,(0.5 + 0.5q)$$

It did **not** beat a fixed threshold on `q̂` (+0.0153 vs +0.0166). The reason is
that `s` is predicted with `corr = 0.648`, so the extra term adds noise as often
as signal. A fixed τ on the better-predicted quantity wins. Recorded because the
plan asserted the EV rule was the right one and it is not, empirically.

`close_k` was also swept as a possible confound (Phase 1 closes at 15 px, the
first Phase 2 run did not). It is not a confound: closing is worth −0.002 under
Phase 2, so the comparison is fair without it.

## 7. What this means for the plan

* **Phase 2 delivers +0.0186, not +0.03.** At the measured gap of 0.073 that is
  roughly **0.36 → 0.38 public** — a real move, but not the 0.40 milestone.
* **Fragment merging is cancelled.** It was 1.8 % of the problem.
* **The remaining headroom is in the scorer** (+0.062 oracle gap), and the scorer
  is data-limited. Phase 3 (GONG label transfer, then noisy-student) is now the
  highest-value phase, ahead of Phase 4.
* **The FN increase (+8.2 %) is the thing to fix next.** 242 TPs are lost to
  filtering. A scorer that ranked TPs above FPs more cleanly would fix both the
  FN criterion and part of the PQ gap simultaneously.

## 8. Artefacts

| path | contents |
|---|---|
| `src/solar/instances.py` | candidates, features, labels, merging |
| `v2/scripts/diagnose_fp.py` | FP anatomy (the 1.8 % result) |
| `v2/scripts/build_phase2_dataset.py` | 6,166 candidates + 890 pairs |
| `v2/scripts/build_phase2_crops.py` | 4-channel 256² crops (1.6 GB) |
| `v2/scripts/train_phase2_cnn.py` | ResNet-18 scorer, file-grouped CV |
| `v2/scripts/ceiling_phase2.py` | oracle ceilings |
| `v2/scripts/eval_phase2_final.py` | GBDT / CNN / blend sweep |
| `v2/runs/phase2/final.json` | all 33 configurations |
| `v2/runs/phase2/ceiling.json` | oracle numbers |
| `v2/runs/phase2/fp_anatomy.json` | per-file FP classification |

## 9. Reproduce

```bash
.venv/bin/python v2/scripts/diagnose_fp.py            # 1.8 % fragments
.venv/bin/python v2/scripts/build_phase2_dataset.py   # ~10 min
.venv/bin/python v2/scripts/build_phase2_crops.py     # ~3 min
.venv/bin/python v2/scripts/train_phase2_cnn.py       # ~11 min
.venv/bin/python v2/scripts/ceiling_phase2.py         # ~7 min
.venv/bin/python v2/scripts/eval_phase2_final.py      # ~20 min
```

---

## 10. Test-set submission (2026-11-01)

Phase 2 was run on the 180 test images and submitted. **Public 0.38**, up from
0.36 for the Phase 1 ensemble.

| stage | local PQ | public | gap |
|---|---|---|---|
| Phase 1 single (`norm_disk_raw`) | 0.4225 | 0.35 | 0.0725 |
| Phase 1 5-fold ensemble | 0.4346* | 0.36 | 0.0746 |
| **Phase 2 ensemble + `q̂` scorer** | **0.4386** | **0.38** | **0.0586** |

\* leaky OOF number, used only as the local estimate for the same predictions.

### The gap shrank, and that is the interesting part

The local→public gap had been stable at 0.073 across two different architectures
and two preprocessing pipelines (`CALIBRATION.md`), which supported reading it as
a property of the test set. Phase 2 breaks that: the gap drops to **0.0586**.

The public gain (+0.02) is *larger* than the local gain (+0.0166). The most
plausible explanation is that the instance scorer removes false positives that
were specific to the local split — threshold artefacts and marginal blobs that
happen to sit near training-set filaments — so the improvement survives the
domain shift better than the average local improvement does.

**Consequence for planning: `public ≈ local − 0.073` is no longer the right
constant.** It was fitted on two points from the same pipeline family. A
component that removes FPs transfers better than one that improves boundaries, so
the constant depends on *what kind* of improvement is being made. Treat 0.059 as
the current estimate and re-measure after Phase 3 rather than extrapolating.

### What was submitted

`v2/scripts/predict_test_phase2.py`, two stages:

1. `--stage probs` — 5-fold ensemble, probability-averaged, cached to
   `v2/runs/phase2/test_probs/` (180 maps, ~5 min)
2. `--stage submit` — candidates at thr 0.30 / area 128, 17 scalar features,
   GBDT `q̂`, keep `q̂ > 0.35`, write CSV (~1 min)

1,591 candidates → 1,093 kept (68.7%), mean 6.07 filaments/image, max 16. The
same 3 images got zero predictions as in Phase 1 (`20170708164010Mh`,
`20200310113230Ch`, `20211214174010Mh`) — all 2017–2021, the known P14 issue.

Validation: all 1,093 RLEs decode at 2048², min area 240, no nulls, no duplicate
`filament_id`, 177/180 images represented.

### Why GBDT alone and not the blend

The rank-average blend scored 0.4406 on OOF against 0.4386 for plain GBDT — a
+0.0020 difference, well inside the paired CI width. But the blend's threshold is
a *quantile*, which depends on the candidate distribution, and that distribution
differs on test (different instrument, 11.7% of test files are 2021–22 against
5.7% of train). An absolute threshold on a calibrated score transfers; a rank
threshold does not. The 0.002 was given up deliberately to remove a transfer risk
that cannot be measured locally.

The scorer itself was refit on all 707 files (`fit_phase2_scorer.py`) — the OOF
evaluation discarded its fold models, so no all-data model existed.
