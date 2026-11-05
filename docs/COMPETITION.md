# Solar Filament Segmentation Challenge 2026

Reference notes for the Kaggle competition. Everything here was pulled from the
competition pages on 2026-10-22 and cross-checked against the downloaded data.

| | |
|---|---|
| **Kaggle ref** | [`filament-segmentation-2026`](https://www.kaggle.com/competitions/filament-segmentation-2026) (id `143678`) |
| **Host** | Earth-Space AI Research (ESAIR) Lab |
| **Sponsor** | NSF National Solar Observatory (NSO) / AURA |
| **Category** | Community (IEEE BigData 2026 Cup) |
| **Prize pool** | $3,000 (top 3, score-weighted) |
| **Metric** | Panoptic Quality (PQ) |
| **Deadline** | 2026-11-15 06:00 UTC (11:59 pm CDT) |
| **Teams** | 1,124 |
| **Submissions** | 5/day, up to 2 final selections |
| **Data license** | CC BY-NC 4.0 — non-commercial / academic use only |

---

## 1. Task

Segment individual **solar filaments** in H-alpha observations of the Sun.

Filaments are dense clouds of solar material suspended by magnetic field lines
above photospheric neutral lines. They are the core of solar eruptions (CMEs,
flares, SEP storms), which makes them a critical event type for space weather
forecasting.

Any approach is allowed: classical image processing, ML, or deep networks. The
goal is pixel-accurate masks that capture the **complete extent** of each
filament including fine-scale structure, while minimizing non-filament regions.

### Why it's hard

- **Fine-scale structures** — barbs are thin, thread-like features extending from
  the filament body. Their orientation relative to the spine encodes the magnetic
  field configuration, so losing them loses physical information.
- **Background noise** — ground-based observations carry seeing artifacts and
  noise. Separating dark filament material from dark non-filament regions is
  nontrivial.
- **Structural continuity** — naive methods fragment filaments or merge nearby
  dark "islands", producing physically inconsistent masks. The metric explicitly
  penalizes both failure modes.

---

## 2. Data

### 2.1 Layout

Downloaded and extracted to the repo root (719 MB):

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/                                        707 × .jpeg
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json         47 MB
└── test/
    └── test_images/                                         180 × .jpeg
```

### 2.2 Images

- **2048 × 2048**, 8-bit **grayscale** (`PIL` mode `L`). Converted from the
  original FITS. **Do not treat as RGB** — replicate to 3 channels if a
  pretrained RGB backbone requires it.
- Filename encodes capture time and instrument: `YYYYMMDDHHMMSS<station>.jpeg`,
  e.g. `20260901165702Bh.jpeg` = 2026-09-01 16:57:02, Big Bear (Bh).
- Source: GONG H-alpha network (Big Bear, High Altitude Observatory, Learmonth,
  Udaipur, IAC, Cerro Tololo).

### 2.3 Annotations — COCO format

Standard COCO (`info`, `licenses`, `categories`, `images`, `annotations`), so
`pycocotools` / `pycocotools`-compatible loaders work directly.

**Categories** (class labels, *not* used by the metric):

| id | name | count |
|---|---|---|
| 1 | Left | 2,535 |
| 2 | Right | 2,590 |
| 3 | Unidentifiable | 3,074 |
| 4 | Ambiguous | 0 (unused) |

**Annotation fields** — note the two non-standard extras:

| field | type | notes |
|---|---|---|
| `id` | str | UUID from the annotation platform |
| `image_id` | str | `<annotator_batch>-<filename>` |
| `category_id` | int | 1–4 as above |
| `segmentation` | `[[float, ...]]` | **single** closed polygon, `[x0,y0,x1,y1,...]`, first point == last point |
| `area` | float | polygon area (not bbox area), computed via pycocoapi RLE |
| `bbox` | `[x, y, w, h]` | top-left corner + size, pixels |
| `spine` | `[float, ...]` | **non-standard** — filament centerline polyline, same encoding as segmentation, no disconnected segments |
| `iscrowd` | 0 | always 0; one polygon per filament, never grouped |

**Image fields**: `id` (str), `width`, `height`, `file_name`, `license`,
`date_captured`, plus a `url` pointing at the original GONG `.jpg`.

### 2.4 ⚠️ Duplicate images across annotators

The JSON has **1,154 image entries but only 707 unique files**. The same
observation was annotated independently by multiple annotators; the `id` prefix
disambiguates them:

```text
20111116063134Lh.jpeg → 050103-20111116063134Lh
                        050102-20111116063134Lh
                        050101-20111116063134Lh
```

296 filenames have 2–3 entries. The organizers say to **treat these as different
images** (each has its own annotation set).

> **Consequence for validation:** splitting by `image_id` leaks the same
> underlying image into train and val, inflating scores. **Split by
> `file_name`** (group-aware split) to get an honest estimate.

### 2.5 Dataset statistics (verified locally)

| | |
|---|---|
| Image entries | 1,154 |
| Unique image files | 707 |
| Annotations | 8,199 |
| Annotations / image | min 1, max 26, mean 7.1 |
| Images with ≥1 annotation | 1,154 (all) |
| Test images | 180 (no annotations released) |
| Annotator batches | 37 distinct prefixes |

---

## 3. Metric — Panoptic Quality

Replaced the original Dice score on 2026-08-07; all prior submissions were
re-scored.

$$
\mathrm{PQ}(Y, \hat{Y}) = \frac{\sum_{(y, \hat{y}) \in TP} \mathrm{IoU}(y, \hat{y})}{|TP| + 0.5\,|FP| + 0.5\,|FN|}
$$

$$
\mathrm{IoU}(y, \hat{y}) = \frac{|y \cap \hat{y}|}{|y \cup \hat{y}|}
= \frac{\sum (y \odot \hat{y})}{\sum (y \oplus \hat{y} \ominus y \odot \hat{y})}
$$

- $Y$ = ground-truth segments, $\hat{Y}$ = predicted segments.
- $TP$/$FP$/$FN$ = true-positive / false-positive / false-negative sets.
- Matching is by **actual overlap**, not index — the number of predicted
  filaments may differ from the number of ground-truth ones.
- PQ decomposes into recognition quality (RQ) × segmentation quality (SQ), which
  is why it penalizes both fragmentation (one-to-many) and over-merging
  (many-to-one).

Reference: Kirillov et al., *Panoptic Segmentation*, CVPR 2019,
[10.1109/CVPR.2019.00963](https://ieeexplore.ieee.org/abstract/document/8953237).

**Score context** (from organizer announcements):

- **PQ > 0.35 is considered "of great value"** by the organizers.
- Public leaderboard top score as of 2026-10-22: **0.69** (IONode Online);
  2nd 0.64, then a cluster at 0.62.
- The organizers have flagged leaderboard gaming — some submissions are
  exploiting the scoring system rather than solving the task. Don't chase the
  leaderboard blindly.

### Leaderboard split

The public scoreboard shows PQ on **~50% of the test images**. The remaining
images are scored privately by the organizers. The two scores should be close;
a large gap suggests overfitting to the public half.

---

## 4. Submission format

A **single CSV** for the entire test set, one row per predicted filament:

```text
filament_id,segmentation_rle
20150125172714Mh_1,"f8uSDds ... VQNC"
20150125172714Mh_2,"KHT%$HD ... 9>km"
20150125172714Mh_3,"YQNEgn1 ... BH6^"
```

### Rules

- `filament_id` = `<image_id>_<n>`, where `<image_id>` is the test image's
  filename stem (e.g. `20150125172714Mh`) and `<n>` makes rows unique. Any
  unique suffix is acceptable as long as the image id is unchanged.
- `segmentation_rle` = **RLE counts only**. No size, no quotes in the actual
  file. Size is fixed at **2048 × 2048** for every image.
- Use `pycocotools` — `annToMask`, `decodeMask`, `encodeMask` — to convert
  masks/polygons ↔ RLE losslessly.
- Predicted count per image need not match ground truth; matching is by overlap.

### Final submission (separate from the Kaggle CSV)

Added 2026-08-07. To be considered for prizes, submit via the
[Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog):

1. **Technical report** — 4 pages, using the
   [Overleaf template](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe),
   submitted as a single PDF.
2. **Code repository** — public Git link (GitHub/GitLab/Bitbucket), no access
   requests. Must contain `requirements.txt` with pinned versions and a Jupyter
   notebook illustrating the full pipeline. The committee must be able to
   reproduce predictions without any additional files.

---

## 5. Evaluation rubric

Not purely leaderboard-driven — the final ranking is a weighted judgement.

**Quantitative comparison (70%)**

- Panoptic Quality (primary)
- Distribution of Dice scores
- Distribution of IoU scores
- Distribution of one-to-many / many-to-one relations between GT and predictions

**Qualitative comparison (30%)**

- Detailed description of the full pipeline (preprocessing → final prediction,
  including architecture)
- Apparent morphology of predicted segmentations on H-alpha images
- Code quality (modularity, documentation)

> The qualitative half is **contingent on source code availability**. A strong
> PQ with an undocumented repo will lose points.

---

## 6. Rules and constraints

- **External data is allowed**, with two hard limits:
  1. Inference must use **only** the H-alpha images in `test/`.
  2. Models may **not** train on any other ground-truth metadata.
- **Team size**: the rules page says max **5**; the Kaggle API reports
  `max_team_size: 10`. Treat 5 as the binding number (rules page wins) and
  verify before merging.
- **Submissions**: 5/day; up to 2 final selections for judging.
- **Team mergers**: allowed until the merger deadline; combined submission count
  must not exceed `5 × days_running`.
- **Data use**: non-commercial and academic research/education only.
- **Winner license**: winning submission and source code must be released under
  an OSI-approved open-source license.
- **Prize conditions**: valid banking info for AURA/NSO direct transfer (US
  financial institution), full reproducible source code, and the technical
  report. Prizes split equally among verified team members.

### Prize distribution

Proportional to score, with a temperature of 2.0:

```python
def compute_prizes(scores, total_prize=3000, temperature=2.0):
    scores = np.asarray(scores, dtype=float)
    weights = np.power(scores, temperature)
    weights = weights / np.sum(weights)
    return total_prize * weights
```

---

## 7. Timeline

| Date | Event |
|---|---|
| 2026-07-10 | Competition launched |
| 2026-08-07 | Metric changed from Dice → Panoptic Quality |
| 2026-08-09 | Self-evaluation notebook released |
| 2026-08-12 | Leaderboard re-scored |
| **2026-11-15 23:59 CDT** | **Final reports and solutions due** |
| 2026-11-30 | Winners announced |
| 2026-12-14 → 17 | IEEE BigData, Phoenix AZ — winners presented |

---

## 8. Useful links

- Competition: <https://www.kaggle.com/competitions/filament-segmentation-2026>
- Self-evaluation notebook:
  <https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook>
- MAGFiLO dataset paper (Scientific Data, 2024):
  [10.1038/s41597-024-03876-y](https://doi.org/10.1038/s41597-024-03876-y)
- Panoptic Quality paper:
  [10.1109/CVPR.2019.00963](https://ieeexplore.ieee.org/abstract/document/8953237)
- SABID 2026 workshop (accepting related reports):
  <https://sites.google.com/view/sabid2026/home>
- GONG: <https://en.wikipedia.org/wiki/Global_Oscillations_Network_Group>
- NSO: <https://nso.edu/>

### Organizers

- Azim Ahmadzadeh, Ph.D. — University of Missouri–St. Louis
- Dustin J. Kempton, Ph.D. — Georgia State University
- Qin Li, Ph.D. — New Jersey Institute of Technology
- Alexei A. Pevtsov, Ph.D. — National Solar Observatory

---

## 9. Practical notes

- **Group-aware splitting is mandatory.** Split on `file_name`, not `image_id`
  (see §2.4). Otherwise validation PQ will be optimistic and misleading.
- **Grayscale input.** Images are `L` mode; adapt the first conv layer or
  replicate channels.
- **Class labels are a red herring for scoring.** PQ ignores `category_id`, but
  the labels (Left/Right/Unidentifiable) may still be useful as auxiliary
  supervision or for filtering.
- **`spine` is free extra supervision.** The centerline is provided for every
  annotation and is not used by the metric — a good auxiliary target for
  instance separation, which directly helps PQ's one-to-many/one-to-many
  penalties.
- **Instance separation matters more than raw pixel accuracy.** PQ's
  $0.5|FP| + 0.5|FN|$ denominator punishes splitting one filament into three
  almost as hard as missing it entirely. Prefer under-segmentation to
  fragmentation.
- **2048² is large.** Train on crops/patches (e.g. 512²) and stitch, or
  downsample — but remember the submission RLE is always at full 2048²
  resolution.
- **Reproducibility is scored.** Keep `requirements.txt` pinned and the pipeline
  in a notebook from early on; it's 30% of the rubric and a prize condition.
