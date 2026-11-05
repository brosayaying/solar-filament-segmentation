# v2 docs

One file per experiment. Figures go in `v2/docs/<EXPERIMENT>/<figure-name>.png`.

| doc | what it covers |
|---|---|
| [`PLAN.md`](PLAN.md) | **Start here.** Re-measured baseline, corrections to `STRATEGY_V2.md`, five phases, compute budget, kill criteria |
| [`PHASE0.md`](PHASE0.md) | **Phase 0 complete** — 5-fold split, official scorer, OOF dumps, bootstrap CI, and the OOF contamination bug it caught |
| [`CALIBRATION.md`](CALIBRATION.md) | **Measured local→public gap: 0.4225 → 0.35, gap 0.0725** (Phase 0.4, done) |
| `SEG_V2.md` | Phase 1: 3-channel input, ConvNeXt-Small, sampling, loss, EMA, 5-fold — *to be written* |
| `RESCORE.md` | Phase 2: fragment merging + per-instance `q̂` — *to be written* |
| `GONG.md` | Phase 3: label transfer and noisy-student self-training — *to be written* |
| `ENSEMBLE.md` | Phase 4: fold ensemble + PVTv2-B2 — *to be written* |

## Conventions

* Every result table carries a bootstrap CI (resample images, recompute the
  global aggregate, 1000 reps).
* Every A/B carries a paired per-image test on the difference.
* Report PQ, SQ, RQ, TP/FP/FN, and the paired delta vs the baseline.
* State the split, the GT rasteriser, and the post-processing parameters next to
  every number. A number without those three is not a result.

## Baseline

`runs/norm_disk_raw/best.pt`, pycocotools GT, full 232-entry val, thr 0.65 /
min_area 256 / close 15 → **PQ 0.4225** (SQ 0.678, RQ 0.623).
