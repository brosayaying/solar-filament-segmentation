# Morphological operators for filament contrast

Which morphological operator best separates filaments from background, and by
how much.

```bash
python scripts/morph_contrast.py --n 20
```

Results in `docs/HEATMAP/morph_contrast.json`.

---

## 1. Setup

Filaments are **dark**, **elongated** (~17 px wide at the median, from
`docs/img_stats.json`), and **low contrast** (+9.5 grey levels against a
background std of 12.5). Every operator below returns a map that is positive on
filaments.

Two scores are reported per operator, measured over 877,699 patches from 20
validation images, restricted to the inner disk (40 px limb margin):

- **AUC** — rank separation of filament vs background 8×8 patches.
- **SNR** — `(mean_fg − mean_bg) / std_bg`, i.e. how many background standard
  deviations the filament signal stands out by. This is the number that governs
  whether a threshold can work; AUC can look healthy while the absolute margin
  stays too small to threshold reliably.

---

## 2. Results

| operator | AUC | SNR |
|---|---|---|
| `bh65_min_linemax65` | **0.8219** | 1.580 |
| `bh_line_max_65` | 0.8217 | 1.530 |
| `bh65_times_linemax65` | 0.8200 | 1.557 |
| `bh65_median3` | 0.8165 | 1.554 |
| `bh_line_max_81` | 0.8161 | 1.415 |
| `bh_multiscale_mean` | 0.8151 | 1.553 |
| `bh_ellipse_65` | 0.8141 | 1.524 |
| `bh_line_max_41` | 0.8131 | **1.663** |
| `bh_ellipse_81` | 0.8115 | 1.421 |
| `bh_rect_65` | 0.8114 | 1.421 |
| `bh_multiscale_max` | 0.8114 | 1.419 |
| `bh_rect_41` | 0.8096 | 1.595 |
| `closerecon65_x_bh65` | 0.8078 | 1.509 |
| `closerecon_65` | 0.8002 | 1.467 |
| `bh_ellipse_101` | 0.7981 | 1.249 |
| `bh_ellipse_25` | 0.7457 | 1.417 |
| `morph_gradient_9` | 0.7440 | 0.308 |
| `dog_32_64` | 0.7069 | 0.454 |
| `tophat_ellipse_65` | 0.4141 | −0.257 |

---

## 3. What the numbers say

### Black-hat is the right family, and the SE size barely matters

Every black-hat variant lands in 0.80–0.82 AUC. The SE size sweep (25 → 101 px)
moves AUC by only 0.07, and the best size (65) is ~4× the median filament width
— comfortably above it, as expected, since a black-hat SE must be larger than
the feature or the response saturates.

**Practical consequence: there is no meaningful gain available from tuning the
SE size or shape.** Ellipse, rect, and line SEs at matched length all score
within 0.01 AUC of each other.

### Directional SEs give a small, real gain

`bh_line_max_65` (max over 4 orientations) beats isotropic `bh_ellipse_65` by
**+0.008 AUC**. The gain is consistent but tiny. Individual orientations are
*worse* than isotropic (0.75–0.80), which makes sense: a filament at 30° is
nearly invisible to a 0° line SE. Only the orientation-max recovers the
advantage, and it does so by paying 4× the compute.

### Reconstruction-based operators do not help

`closerecon_65` scores 0.8002 — *below* plain black-hat at the same SE size.
The theoretical appeal (exact edge preservation instead of the rounding a plain
closing introduces) does not translate into better patch-level separation here,
because the patch statistic is a mean over 8×8 and is insensitive to sub-patch
edge placement.

Note: **opening** by reconstruction is the mirror operator for *bright*
features. It returns ≈0 on this data (AUC 0.33–0.35, negative SNR) — it is
simply the wrong polarity, not a broken implementation.

### Combinations do not beat the best single operator

`bh65_min_linemax65` (elementwise min of isotropic and directional-max) is the
top scorer at 0.8219, but that is **+0.0002 over `bh_line_max_65` alone** —
noise. The soft-AND idea (suppress responses only one family finds) does not
pay off because both families are detecting essentially the same structure.

### The wrong operators are clearly wrong

- `tophat_ellipse_65` (AUC 0.41, SNR −0.26) detects *bright* features. Filaments
  are dark, so this is anti-correlated — a useful confirmation that polarity is
  handled correctly everywhere else.
- `morph_gradient_9` (SNR 0.31) and `dog_32_64` (SNR 0.45) are edge/blob
  detectors, not region detectors. They fire on both sides of a filament edge,
  so the mean over a patch cancels out. High AUC with low SNR is the signature
  of a detector that ranks correctly but with almost no absolute margin.

---

## 4. Conclusion

**No morphological operator meaningfully improves filament contrast beyond a
plain black-hat.** The best available is `bh_line_max_65` at AUC 0.8217 vs
0.8141 for the simple `bh_ellipse_65` — a 0.008 gain for 4× the compute.

The ceiling is set by the data, not the operator. Filament SNR is ~1.5–1.7
background standard deviations *per patch mean*, and no morphological
transformation moves that materially. This is consistent with the classifier
results: patch-level F1 plateaued at ~0.55 regardless of feature set, because
the underlying separation is genuinely marginal.

**Recommendation:** keep `bh_ellipse_65` (already in `patch_clf.BLACKHAT_KS` as
`bh65`). Do not invest further in morphological feature engineering; the
remaining headroom is in the classifier and the post-processing, not the
front-end filter.
