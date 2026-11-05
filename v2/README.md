# v2 — artefact root for phase 2

`v2/` is an **artefact root, not a code fork**. Code stays in `src/solar/` and
evolves in place; only new docs and new runs live here.

```
v2/
  docs/                 one file per experiment, figures in docs/<EXPERIMENT>/
  runs/                 every new run: v2/runs/<experiment_name>/
  splits/               week5_seed42.json (5-fold, week-grouped)
```

## Rules

1. **All new docs go in `v2/docs/`.** One file per experiment, named after the
   experiment. Figures go in `v2/docs/<EXPERIMENT>/<figure-name>.png`.
2. **All new runs go in `v2/runs/<experiment_name>/`** — set
   `train.out_dir = "v2/runs"` in the config. Old runs in `runs/` stay put,
   read-only, and are cited by path.
3. **No number without a CI.** Every table gets a bootstrap CI; every A/B gets a
   paired per-image test. A change is only "shipped" if the paired CI excludes 0.
4. **No stale caches.** Every `runs/*/val_probs/` directory predates the
   pycocotools rasteriser switch (2026-10-27) and its GT is wrong. Do not reuse
   them; regenerate into `v2/runs/<exp>/oof/`.
5. **Never touch the public MAGFiLO 1.0 release annotations.** Rules violation,
   and the organisers review code.

## Start here

Read [`docs/PLAN.md`](docs/PLAN.md). It contains the re-measured baseline, the
corrections to `docs/STRATEGY_V2.md`, the five phases, the compute budget, and
the kill criteria.

## Baseline to beat

`runs/norm_disk_raw/best.pt` — resnet34, 512², disk-only normalisation,
pycocotools GT, full 232-entry val:

| thr | min_area | close | PQ | SQ | RQ |
|---|---|---|---|---|---|
| 0.65 | 256 | 15 | **0.4225** | 0.678 | 0.623 |

**Public score: 0.35** (submitted 2026-10-29), so the measured local→public gap
is **0.0725**. Rank ≈ 477 / 1,125. See [`docs/CALIBRATION.md`](docs/CALIBRATION.md).
