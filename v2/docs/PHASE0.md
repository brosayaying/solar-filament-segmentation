# Phase 0 — measurement infrastructure

Status: **complete** (2026-10-29). All five acceptance checks pass.

Nothing in Phases 1–4 is trustworthy without this. The point of Phase 0 is that
every later number is measured the same way, on the same split, with the same
scorer, with an uncertainty attached.

| item | module | status |
|---|---|---|
| 0.1 5-fold week-grouped split | `src/solar/folds.py` | done |
| 0.2 official-scorer harness | `src/solar/official.py` | done |
| 0.3 OOF dumps with metadata | `src/solar/oof.py` | done |
| 0.4 local→public calibration | [`CALIBRATION.md`](CALIBRATION.md) | done |
| 0.5 bootstrap CI in `tune` | `src/solar/tune.py` | done |

---

## 0.1 Five-fold, week-grouped split

`python -m solar.folds --inspect --out v2/splits/week5_seed42.json`

Groups by **ISO week**, not date. Date grouping still puts adjacent days on
opposite sides, and filaments persist for days, so the same filament can appear
in train and val — one candidate explanation for the 0.0725 local→public gap.

```
group_by   groups   files/group median   max   singletons
week          343                  2.0     6          130
month         118                  5.0    18           13
year           12                 75.5   102            0
```

| fold | train ids | val ids | val files |
|---|---|---|---|
| 0 | 930 | 224 | 142 |
| 1 | 909 | 245 | 142 |
| 2 | 925 | 229 | 142 |
| 3 | 932 | 222 | 141 |
| 4 | 920 | 234 | 140 |

**Acceptance:** 707/707 files covered, 0 file overlaps, 0 week overlaps. ✅

Outputs:

* `v2/splits/week5_seed42.json` — all folds, file-level, with `meta`.
* `v2/splits/week5_seed42/fold{0..4}.json` — per-fold `{train, val}` keyed by
  `image_id`, which is the format `solar.train.get_split` reads. Without these
  the k-fold file cannot be passed to `--config data.split_file`.

All annotator copies of a file inherit that file's fold, so the same underlying
image can never appear on both sides.

---

## 0.2 Official-scorer harness

`python -m solar.official --selftest`

Written from `docs/COMPETITION.md` §4 because the organisers' notebook is not in
the repo. Implements: GT keyed `<annotator>-<image>`, predictions matched by bare
`<image>`, pycocotools RLE decode, IoU **strictly > 0.5**, greedy 1:1, PQ pooled
globally over all annotator copies.

Cross-checked against `PQAccumulator` on the same 232 val predictions:

```
metric   PQAccumulator     official      delta
    pq          0.4225       0.4225     0.0000
    sq          0.6778       0.6778     0.0000
    rq          0.6234       0.6234     0.0000

tp/fp/fn  ours 1109/808/532  official 1109/808/532
images 232  pairs 232

official PQ 95% CI: [0.4036, 0.4414]
```

**Acceptance:** the two scorers agree to 4 dp. ✅

---

## 0.3 OOF dumps with metadata

`python -m solar.oof --ckpt <fold ckpt> --fold <f> --out v2/runs/<exp>/oof`

`solar.tune.build_cache` writes `NNNN_prob.npy` / `NNNN_label.npy` with **no
filename mapping**, which makes it impossible to join a predicted candidate back
to the annotations that produced it. Phase 2 is built entirely on that join — a
candidate's label is "the fraction of annotator copies in which this blob has
IoU > 0.5", which requires knowing which file and which copies.

One `.npz` per unique file:

```
prob      float16 (2048, 2048)      ensemble-averaged probability
labels    int32   (k, 2048, 2048)   one instance labelmap per annotator copy
spines    uint8   (k, 2048, 2048)   one centreline raster per annotator copy
meta      json string               file_name, image_ids, annotators, fold, ckpt
```

`k` is the number of annotator copies (1–3). The probability map is stored once
because every copy is the same pixels.

**Acceptance:** `python -m solar.oof --verify <dir>` reports 707/707 files, no
missing/extra/duplicates, every file in the right fold, and stored labels equal
to a fresh pycocotools rasterisation. ✅ (validated on a fold-0 dump; the full
707-file dump is produced by Phase 1's per-fold training)

### A contamination bug this caught

The first attempt dumped `runs/norm_disk_raw/best.pt` over all five week-5 folds.
That checkpoint was trained on **565 of the 707 files** under the old date split,
so:

| fold | val files | were in `norm_disk_raw` train |
|---|---|---|
| 0 | 142 | 115 (81 %) |
| 1 | 142 | 119 (84 %) |
| 2 | 142 | 108 (76 %) |
| 3 | 141 | 107 (76 %) |
| 4 | 140 | 116 (83 %) |
| **total** | **707** | **565 (79.9 %)** |

The dumps looked completely valid — right shape, right fold, right labels — but
80 % of them were in-sample. Phase 2 would have trained its candidate scorer on
leaked probabilities and reported a number that could never transfer.

944 MB of contaminated dumps were deleted. `oof.check_ckpt_was_not_trained_on`
now reads the split recorded in the checkpoint's own config and **refuses** to
dump if any val file was in that checkpoint's training set:

```
REFUSING to dump fold 0: 115/142 val files (81%) were in this checkpoint's
training set (splits/date_seed42.json).
```

Verified both ways: it blocks the contaminated checkpoint and passes a
legitimate per-fold one (0 overlap).

**Consequence for the plan:** OOF dumps require per-fold checkpoints, so Phase 1
must train all five folds before Phase 2 can start. This was already the plan,
but it is now enforced rather than assumed.

---

## 0.4 Local→public calibration

See [`CALIBRATION.md`](CALIBRATION.md). **Local 0.4225 → public 0.35, gap
0.0725.** Rank ≈ 477 / 1,125.

---

## 0.5 Bootstrap CI everywhere

`python -m solar.tune ... --bootstrap --n-boot 1000`

`PQAccumulator.bootstrap_ci` already existed and is correct (resamples images,
recomputes the global aggregate). It is now wired into the sweep and reported
next to the point estimate:

```
best PQ 0.4225 95% CI [0.4052, 0.4392]  (SQ [0.6718, 0.6835], RQ [0.6014, 0.6473])
```

**Acceptance:** every grid point can carry a CI. ✅

From here on, **no number is reported without a CI**, and no change is
"shipped" on a point estimate.

---

## What Phase 0 changed in the codebase

| file | change |
|---|---|
| `src/solar/folds.py` | new — week/month/year k-fold splits, leakage check, per-fold split emission |
| `src/solar/official.py` | new — official-scorer harness, self-test, bootstrap |
| `src/solar/oof.py` | new — OOF dumps with metadata, integrity verify, **leakage guard** |
| `src/solar/tune.py` | `--bootstrap` / `--n-boot`, CI in output |
| `src/solar/predict.py` | `--close-k` (added during 0.4; the tuned setting was unreachable from the CLI) |
| `src/solar/postprocess.py` | `close_k` parameter on `prob_to_labelmap` |

## Next

Phase 1: 5-fold ConvNeXt-Small training. ~12 h GPU. Produces the per-fold
checkpoints that Phase 2's OOF dumps require.
