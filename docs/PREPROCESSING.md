# Preprocessing and normalisation

Two changes were made to the input pipeline, both aimed at defects this repo had
measured but never fixed: off-disk black pixels polluting the normalisation
statistics, and limb darkening inverting filament contrast in the outer ring.

The short version:

* **Normalisation was wrong and is now measured.** `mean=0.5, std=0.25` was
  computed over the whole frame, of which 28% is off-disk black. Measured over
  the disk, the statistics are **mean 0.4463, std 0.1313**. The old `std` was
  1.9x too wide, compressing the real signal into a narrow band.
* **Eval-time preprocessing is not a test of preprocessing.** A model trained
  on raw input has already adapted to raw statistics. Bolting flattening or
  CLAHE on at inference measures a distribution shift, not a benefit. The only
  valid test is to retrain.

---

## 1. Why the old normalisation was wrong

`scripts/disk_stats.py --n 200` measures the statistics over three regions, on
the same 142 de-duplicated val files used everywhere else:

| region | mean | std | pixels |
|---|---|---|---|
| whole frame | 0.3266 | 0.2231 | 595,591,168 |
| **disk only** | **0.4463** | **0.1313** | 428,354,520 |
| disk, after flattening | 0.4319 | 0.1086 | 428,354,520 |

Disk area is 71.9% of each frame (range 0.638-0.798). The remaining 28% is sky
with mean 3.2/255, which pulls the global mean down by 0.12 and inflates the
variance.

Two independent errors in `mean=0.5, std=0.25`:

1. **Wrong region.** Global statistics describe a distribution the model never
   has to predict on — nothing is annotated off-disk, and `min_area=512`
   removes the stray components that do appear there.
2. **Too wide.** Even the global std (0.223) is smaller than 0.25, and the disk
   std is 0.1313. Dividing by 0.25 maps the useful intensity range into
   roughly `[-1.0, +1.0]` instead of `[-1.9, +1.9]`, throwing away about half
   the representable dynamic range in the first layer.

Per-image variation is small, so fixed constants are appropriate rather than
per-image standardisation: disk mean is 0.4471 +/- 0.0152 (range 0.413-0.478)
and disk std is 0.1295 +/- 0.0129 across files. Normalising each image to its
own mean would also erase the absolute brightness differences that distinguish
faint from bright filaments, which is information the model should be allowed
to use.

New defaults: `mean=0.4463`, `std=0.1313`.

---

## 2. Radial flattening

Limb darkening drops the disk background from 139 DN at centre to ~46 DN at the
limb. Filaments are dark-on-bright everywhere except the outer ring, where the
background is dark enough that the contrast inverts. `docs/DISK_MASKING.md`
measured the consequence: recall falls from 0.730 at r/R 0.4-0.6 to 0.281 at
r/R 0.8-1.0, across 11% of all filament area.

Radial flattening estimates the background as the median intensity in 64
concentric annuli, smooths that 1-D profile with a Gaussian of width 9 bins, and
divides. Filaments are 8-22 px wide against ~16 px annuli, and the median
ignores them because they occupy a minority of any annulus.

The measured profile on a representative image:

| r/R | background (DN) |
|---|---|
| 0.00 | 138.97 |
| 0.20 | 139.63 |
| 0.40 | 141.67 |
| 0.60 | 135.40 |
| 0.80 | 108.95 |
| 0.90 | 78.10 |
| 0.95 | 55.53 |
| 0.99 | 46.36 |

After division the disk interior is flat, which is what removes the inverted
contrast (`/tmp/flat_small.png` versus `/tmp/raw_small.png` during development).

### 2.1 A bright limb ring has to be clipped

These images carry a thin *bright* chromospheric ring near r/R 0.90-0.94. It is
real, not an artefact: 11.5% of the pixels in that annulus exceed 2x the local
background, with p99 = 218 DN against a background of 55. Dividing by the
background amplifies it about 4x, to **+7.4 sigma** after standardisation, where
it would dominate the gradient and swamp the faint filaments sitting in the same
annulus.

`radial_flatten(..., clip=2.0)` caps the output at 2x the median background.
Filaments are dark, so clipping only the upper tail cannot affect them. With the
clip, the flattened image is bounded at +3.0 sigma.

### 2.2 Cost, and why there is a cache

Full-resolution flattening costs ~520 ms per 2048² image. Across 565 train files
and 160 epochs that is roughly 12 h of recomputing a deterministic function of
the same file, so `scripts/build_preprocess_cache.py` writes the result once:

```
python scripts/build_preprocess_cache.py --preprocess flatten
```

887 images (707 train + 180 test), 9.8 GB as float16, ~6 min to build. The
dataset loads from it when `data.preprocess_cache` is set, which is **3.1x
faster** (111 ms vs 347 ms per sample) and costs 0.0019 sigma of float16
quantisation — no pixel moves by more than 0.01 sigma.

Preprocessing must run at *full* resolution and before cropping: the background
profile needs annuli spanning the whole disk, which a 512 px crop cannot supply.

---

## 3. Why the eval-time probe was invalid

`scripts/preprocess_ablation.py` ran four variants against existing checkpoints.
Its first run had the wrong normalisation (`mean=0.0, std=1.0` instead of the
checkpoint's `0.5/0.25`), which dropped the raw baseline to PQ 0.0559 against a
documented 0.4152 — every variant then looked like a large win, CLAHE
"improving" PQ by +0.10.

With the normalisation read from the run's own `config.json`, the raw baseline
returns to **0.4220** and the deltas invert:

| variant | PQ | paired vs raw | 95% CI | verdict |
|---|---|---|---|---|
| raw | 0.4220 | — | — | baseline |
| flatten | 0.4147 | −0.0048 | [−0.0137, +0.0042] | n.s. |
| clahe | 0.3226 | −0.0951 | [−0.1197, −0.0704] | **worse** |
| both | 0.3155 | −0.1044 | [−0.1276, −0.0811] | **worse** |

Raw lands at 0.4220 rather than 0.4152 because this harness scanned its own
thresholds and picked 0.70; 0.4152 is the same checkpoint at threshold 0.55.
Both are the same model.

So CLAHE applied at inference is clearly harmful, and flattening is neutral to
slightly negative. **Neither number is evidence about preprocessing**, because
both measure what happens when a raw-trained model is fed inputs it was not
trained on. The network learned to expect limb darkening; removing it is a
distribution shift.

---

## 4. What is now in place

`src/solar/preprocess.py`
: `disk_geometry`, `radial_coord`, `radial_background`, `radial_flatten`
  (with `clip`), `apply_clahe`, `preprocess`.

`src/solar/data.py`
: `parse_preprocess`, `disk_mask`, `preprocess_image`, `normalise_image`;
  `FilamentDataset` gains `preprocess`, `mask_offdisk`, `offdisk_value`,
  `preprocess_cache`. Preprocessing is applied at full resolution before
  cropping, and identically in train, val and test.

`src/solar/predict.py`
: reads `preprocess`, `mask_offdisk`, `offdisk_value` from the checkpoint's own
  config, so a model trained with flattening is always served flattened input.

`scripts/disk_stats.py`
: measures disk, global and post-flattening statistics.

`scripts/build_preprocess_cache.py`
: writes the cache.

Configs: `configs/norm_disk_raw.json` (measured normalisation, no
preprocessing) and `configs/norm_disk_flatten.json` (plus flattening, cached).
Both are 160 epochs, resnet34 UNet, threshold 0.5.

### 4.1 Off-disk masking was tried and rejected

`mask_offdisk` replaces sky pixels with a fixed value after normalisation. It is
now off, because measuring it showed it does nothing useful and costs a lot:

| sky treatment | sky value after normalisation |
|---|---|
| raw, unmasked | −3.30 σ |
| flattened, unmasked | −3.40 σ |
| masked at −3.0 | −3.00 σ |

The sky already normalises to about −3.3 σ against disk statistics, so there is
no dynamic range to reclaim, and `offdisk_value=-3.0` actually moves it *toward*
the disk by 0.30 σ. JPEG noise on the sky is 0.215 σ, invisible either way.
Meanwhile the mask costs a full-resolution build per sample and tripled epoch
time (38 s → 120 s) while doing nothing on the nine-of-ten patches that contain
only sky. The option remains available and defaults to off.

Two bugs surfaced while measuring this, both now fixed: the per-file disk mask
was cached as a 2048² boolean array (4 MB x 565 files x 8 workers ≈ 18 GB,
which pushed epoch time to 298 s from allocation pressure; only the 3-float
geometry is cached now), and `_augment` returned a variable-length tuple that
silently broke when the disk mask was `None`.

---

## 5. Threshold

`eval.threshold` is now **0.5** everywhere. This is the competition's
binarisation threshold, so tuning it on val fits a knob the leaderboard does not
expose and the gain does not transfer. The ablation above is a good illustration:
the sweep preferred 0.70 on val, worth +0.007 PQ over 0.55 there, and the
leaderboard gave +0.01 across the whole 160-epoch rerun.

`min_area=512` is still tuned, because post-processing parameters *are* ours to
choose.

---

## 6. What to run

```bash
python scripts/build_preprocess_cache.py --preprocess flatten  # once, ~6 min
python -m solar.train --config configs/norm_disk_raw.json      # A: normalisation only
python -m solar.train --config configs/norm_disk_flatten.json  # B: + radial flattening
```

Both run at ~53 s/epoch, about 2.4 h for 160 epochs. (The original run was
38.5 s/epoch, but re-timing `configs/long.json` on the current machine gives
53 s too, so the difference is host load, not the pipeline.)

A isolates the normalisation change. B adds flattening. Compare both against the
existing `runs/long_160ep_512` (PQ 0.4152 at threshold 0.55, 0.4220 at 0.70),
using the paired per-image test in `scripts/preprocess_ablation.py`, not the
aggregate alone.

Effect sizes to expect: the decoder experiment moved PQ by +0.040 with a CI of
[+0.025, +0.056] and still was not enough to matter on the leaderboard. Anything
below ~0.02 here should be treated as noise regardless of the aggregate.
