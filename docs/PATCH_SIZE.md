# Patch size comparison: 8×8 vs 16×16

Does a larger patch help the classical patch classifier?

```bash
python scripts/compare_patch_sizes.py --patch 8 16 --train-images 60 --val-images 20
```

Results in `docs/EDGE/patch_size_compare.json`.

---

## 1. Why a matched comparison was needed

The first 16×16 run used 60 training images while the headline 8×8 result used
250, so the two numbers were not comparable. `scripts/compare_patch_sizes.py`
holds the image split, the model, and every post-processing parameter fixed and
varies only the patch size.

Both runs: RandomForest (200 trees), 60 train / 20 val images, `neg_frac=0.05`,
boundary refinement swept over `alpha ∈ {1.0, 0.8, 0.6}`, `thr ∈ {0.5, 0.6, 0.7}`,
`min_area ∈ {200, 512, 1024}`.

---

## 2. Results

| patch | grid | features | train patches | patch AUC | patch F1 | **PQ** | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **8** | 256 | 226 | 151,076 | 0.9512 | 0.546 | **0.1875** | 0.649 | 0.289 | 40 | 90 | 107 |
| **16** | 128 | 802 | 38,720 | 0.9532 | 0.526 | **0.1770** | 0.633 | 0.280 | 39 | 93 | 108 |

Both selected the same post-processing: `alpha=0.6, thr=0.6, min_area=512`.

Cost: 16×16 builds features slower (142–221 s vs 157–183 s, because the
raw-pixel block grows 4×) but fits faster (37–44 s vs 96–114 s, because there
are 4× fewer patches). Timings vary run-to-run with machine load — a DL training
job was running concurrently — so treat them as indicative only. The PQ and AUC
numbers were identical across both runs.

---

## 3. Interpretation

### The classifier is equally good at both sizes

Patch AUC is **0.9512 vs 0.9532** — a 0.002 difference, i.e. identical. The
16×16 model ranks patches just as well as the 8×8 model. Patch F1 is slightly
lower (0.526 vs 0.546) but within noise at this sample size.

So the classifier is *not* the reason PQ drops. The drop comes from **SQ:
0.649 → 0.633**. Coarser patches give coarser boundaries, and boundary
refinement cannot fully recover the lost spatial resolution.

### The important finding: 16×16 is far closer to its ceiling

From `scripts/ceiling.py`, the best achievable PQ for a blocky mask at each
stride (using ground-truth labels, so this is an upper bound):

| stride | ceiling PQ | achieved PQ | fraction of ceiling |
|---|---|---|---|
| 8 | 0.6434 | 0.1875 | **29.1%** |
| 16 | 0.2815 | 0.1770 | **62.9%** |

This inverts the naive reading. The 16×16 classifier extracts **63%** of what
its representation permits; the 8×8 classifier extracts only **29%**.

The 8×8 result is higher in absolute terms purely because its ceiling is 2.3×
higher. The classifier itself is much further from optimal at 8×8.

**Implication:** if the 8×8 classifier could be brought to the same relative
efficiency as the 16×16 one, PQ would be $0.63 \times 0.6434 \approx 0.41$ —
roughly double the current best, and competitive with the DL baseline (0.36).
The headroom is in the classifier, not the patch size.

### Why 8×8 is harder to learn

At 8×8 there are 4× more patches and each carries less context: 64 raw pixels
versus 256, and the same 4-patch-wide context window covers half the physical
area. A filament is ~17 px wide, so an 8×8 patch often sees only part of one
edge with no information about which side is inside. A 16×16 patch usually
contains the full width.

That is consistent with the feature counts: 226 vs 802. The 16×16 model has
3.5× more raw-pixel evidence per decision.

---

## 4. Conclusion

**16×16 is not better in absolute PQ (0.177 vs 0.188), but it is a much more
efficient use of its representation (63% vs 29% of ceiling).**

Neither is competitive with the DL baseline (0.36) or the tiled 1024² model
(0.39). The practical recommendation is unchanged: the classical patch
classifier is a useful diagnostic and a sanity check on the data, not a
contender.

If this line were pursued further, the highest-value direction is **not** a
larger patch but closing the 8×8 efficiency gap — more training data, better
features, or a stronger model at 8×8 resolution, where the ceiling is 0.64.

---

## 5. Caveat

20 validation images, 114 GT instances. The PQ difference (0.0105) is within
run-to-run noise at this sample size; the *ceiling fractions* are the robust
part of the result, since they are computed from ground truth and do not depend
on the classifier.
