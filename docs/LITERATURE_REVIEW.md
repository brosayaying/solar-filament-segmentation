# Literature review: what has actually worked on solar filament segmentation

Written 2026-10-27, after the decoder experiment (§21) confirmed that architecture
changes alone do not move PQ. The question this document answers: **what have
other groups done on this exact problem, and what did it buy them?**

Two of the papers below are trained on **MAGFiLO — our dataset**. That makes them
directly comparable in a way nothing else in this repo is.

---

## 1. The papers that matter

| paper | venue | dataset | what it is |
|---|---|---|---|
| **EdgeAttNet** (Solomon et al.) | arXiv 2509.02964, 2025 | **MAGFiLO** | U-Net + edge-guided attention |
| **MORDEN** (Hu, Hao et al.) | RAA 2026, arXiv 2607.24525 | MHAS/AHAS (CHASE) | modern ConvNet + DenseCRF + DBSCAN |
| **Diercke et al.** | A&A 686 A213, 2024 | ChroTel/GONG/KSO | YOLOv5 → U-Net, semi-supervised |
| **Guo et al.** | Sol. Phys. 297:104, 2022 | — | **CondInst instance segmentation** |
| **Zheng et al.** | ApJ 965:150, 2024 | CHASE | U-Net + focal loss + k-means |
| **Flat U-Net** (Zhu et al.) | ApJ 980:176, 2025 | — | ultralight channel-attention U-Net |

---

## 2. The single most important finding: preprocessing, not architecture

**EdgeAttNet trains on MAGFiLO and their preprocessing pipeline is exactly the
thing we diagnosed but never implemented.**

Their pipeline (arXiv 2509.02964 §III-A):

1. Normalise intensities to [0, 1]
2. **Locate the disk with a Hough Circle Transform**, mask off-disk to zero
3. **Radial flattening** — compute normalised radius per pixel, bin intensities
   into concentric annuli, smooth the 1-D profile, interpolate back to a 2-D
   background estimate, divide it out
4. Gaussian blur, 3×3, σ = 0.7
5. **CLAHE** (contrast-limited adaptive histogram equalisation) inside the disk mask

Step 3 is the limb flat-fielding that `docs/DISK_MASKING.md` identified as the
biggest single defect in our model and that `docs/CORRECTIONS.md` listed as
"diagnosed but never fixed". Our own measurement:

| r/R | our recall | GT pixels |
|---|---|---|
| 0.4–0.6 | **0.730** | 114,323 |
| 0.8–1.0 | **0.281** | 58,662 |

A 2.6× collapse in the outer ring, which holds 11% of all filament area, caused
by limb darkening *inverting* the filament/background contrast (+12 → −24). We
wrote down the fix and never ran it. EdgeAttNet does it as step 3 of five.

**This is the highest-confidence item in this document** because it is (a) on our
dataset, (b) fixing a defect we measured ourselves, and (c) cheap — it is image
preprocessing, no retraining of anything else.

### What EdgeAttNet's numbers say

MAGFiLO test split, 1,295 train / 45 val / 99 test, 50 epochs, lr 1e-4, **no
pretrained weights, no augmentation**:

| model | mIoU_pairwise | mIoU_multiscale | params |
|---|---|---|---|
| U-Net | 0.5724 | 0.5848 | 31.0 M |
| U-Net + MHSA (no PE) | 0.5856 | 0.6000 | 35.2 M |
| U-Net + MHSA (PE) | 0.6200 | 0.6601 | 35.4 M |
| **EdgeAttNet** | **0.6451** | **0.7032** | **22.7 M** |

Two things to note. First, the architecture gain is real but modest: +0.073
pairwise IoU over U-Net, and most of it comes from adding attention at all
(+0.048 from MHSA+PE), not from the edge prior specifically (+0.025 on top).
Second, **all four models share the preprocessing pipeline**, so these numbers
say nothing about how much the preprocessing alone is worth — but they do
establish that a plain U-Net on properly preprocessed MAGFiLO reaches 0.572
pairwise IoU, which is a useful reference point.

Caveat: these are IoU/Dice metrics, not PQ. They are not directly comparable to
our 0.4152. But the *preprocessing* is metric-agnostic.

---

## 3. The second finding: DenseCRF beats the human ground truth on edges

MORDEN's most surprising result (arXiv 2607.24525 §3.4.1). They measure average
edge gradient magnitude and orientation on mask boundaries, and compare:

| mask | gradient magnitude | orientation |
|---|---|---|
| MHAS ground truth | 250.6 | 69.6 |
| MORDEN segmentation | 217.7 | 60.9 |
| DenseCRF @ 1K | 222.2 | 62.7 |
| **DenseCRF @ 2K** | **259.0** | **72.9** |

**DenseCRF applied at 2K resolution produces sharper edges than the manual
annotations.** Their interpretation: the CNN only supplies the unary potential;
DenseCRF adds pairwise terms from the *original image* intensity and position, so
the final output is a posterior over multiple sources rather than an imitation of
the training labels.

This matters directly for us. We concluded in §21 that SQ is pinned at ~0.67 and
attributed it to the annotation ceiling. MORDEN's result says that ceiling is
beatable by post-processing that uses the original image, because the annotation
itself is blurrier than the underlying signal.

**We have never tried DenseCRF.** We tried morphological close (+0.012),
watershed (negative), hysteresis (negative), dilate/erode (negative), and
proximity merge (+0.011). DenseCRF is a different class of method — it is a
learned-free probabilistic refinement using image intensity, not a morphological
operation.

Cost: MORDEN reports 0.17 s/image for the CNN alone vs 2 s at 1K and 6.5 s at 2K
with DenseCRF. For 180 test images that is ~20 minutes. Acceptable.

---

## 4. The third finding: we are solving the wrong task for this metric

PQ is a **panoptic** metric. It rewards correctly separated instances with
accurate masks. The literature has three distinct workflows, and they are not
equivalent under PQ:

| workflow | papers | how instances are formed |
|---|---|---|
| **semantic + clustering** | Zheng 2024, MORDEN, **us** | threshold → connected components / DBSCAN / k-means |
| **detect-then-segment** | Diercke 2024 (YOLOv5 → U-Net) | boxes define instances |
| **instance segmentation** | **Guo 2022 (CondInst)** | model allocates a per-pixel identifier directly |

We are in the first camp. Guo et al. use CondInst, which "directly allocates
identifier for every filament at the pixel level" — that is exactly what PQ
measures, and it is the only workflow where the training objective matches the
evaluation metric.

MORDEN explicitly chose *against* this, and their reasoning is worth quoting
because it is a real objection:

> fragment integration based on solitary Hα images is regarded as semantically
> ambiguous, which relies exclusively on morphological characteristics... neither
> humans nor deep learning models can accurately identify which fragments belong
> to a single filament without referring to other bands, such as photospheric
> magnetic polarity inversion lines (PILs) in magnetograms or filament channels
> in 304 Å filtergrams

So the field is genuinely split. But note what this implies for **our** metric:
if fragment integration is ambiguous, then the GT itself is ambiguous, and PQ
punishes us for disagreeing with an arbitrary choice. That is consistent with
what we measured — our FP/TP ratio of 1.13 is dominated by "splits of an
already-matched GT" (`docs/PQ_DIAGNOSIS.md`: 58.5% of FPs).

**Implication:** an instance-segmentation model would not obviously fix this,
because the ambiguity is in the labels. But it would at least be optimising the
right objective, and it removes the arbitrary threshold + min_area + connectivity
choices that currently cost us ~0.05 PQ in sensitivity.

---

## 5. Cheap architectural changes with measured ablations

MORDEN's ablation table (MHAS dataset, IoU / F1) is the most useful set of
controlled numbers in the literature for this task:

| variant | IoU | F1 | delta IoU |
|---|---|---|---|
| **MORDEN** | **0.696** | 0.819 | — |
| w/o Instance Norm | 0.678 | 0.805 | **−0.018** |
| w/o large kernel (5×5) | 0.663 | 0.795 | **−0.033** |
| w/o APP (adaptive pyramid pooling) | 0.678 | 0.806 | −0.018 |
| w/o PAG (patch attention gate) | 0.688 | 0.813 | −0.008 |
| w/o branch block | 0.688 | 0.812 | −0.008 |
| w/o multihead | 0.695 | 0.818 | −0.001 |
| PPM instead of APP | 0.683 | 0.809 | −0.013 |
| ASPP instead of APP | 0.678 | 0.805 | −0.018 |

The two biggest single wins are **large kernels (+0.033)** and **Instance
Normalization instead of BatchNorm (+0.018)**. Their justification for IN is
directly relevant to our limb problem:

> IN focuses more on local contrast variations, while BN retains the global
> intensity statistics. This property is particularly suitable for solar filament
> segmentation because local intensity variations are critical, and even
> decisive, indicators on Hα images.

That is the same physical argument as radial flattening, implemented in the
normalisation layer instead of the preprocessing. **BN normalises away exactly
the local contrast that distinguishes a limb filament from limb darkening.** Our
model uses BatchNorm throughout (DeepCrack) or whatever timm ships (resnet34).

Both changes are a few lines. Neither requires retraining from scratch to test —
a 25-epoch run would show whether the direction holds.

---

## 6. What the field does *not* do

Worth recording, because it bounds what is worth trying:

- **Nobody uses PQ.** Every paper above reports IoU, Dice, F1, or accuracy. The
  competition's switch to PQ (2026-08-07) is recent enough that the literature
  has not caught up. So there is no published PQ baseline to beat, and no
  published evidence about which of these methods transfers to PQ.
- **Nobody trains on consensus/soft targets.** MORDEN and EdgeAttNet both note
  that multiple annotators exist and that annotation is imperfect, but neither
  exploits it. Our soft-label experiment (§13) is, as far as this review found,
  ahead of the literature — and it gave +0.0005.
- **Nobody uses pretrained ImageNet weights** in the MAGFiLO/CHASE papers.
  EdgeAttNet explicitly: "without the use of pretrained weights or data
  augmentation". MORDEN trains from scratch. Our resnet34 UNet uses ImageNet
  weights, which may be why it beats our from-scratch DeepCrack.
- **Nobody does test-time augmentation.** Not mentioned in any of the six.

---

## 7. Ranked recommendations

Ordered by (expected gain × confidence) / cost. Confidence is my judgement based
on how directly the evidence applies to our setup.

### Tier 1 — do these first, they are cheap and literature-validated

**1. Radial flattening + CLAHE preprocessing.**
EdgeAttNet's steps 3 and 5, on our dataset. Fixes a defect we measured ourselves
(recall 0.73 → 0.28 in the outer ring, 11% of filament area). No architecture
change, no retraining of the model class — just a different input pipeline.
*Expected: the largest single gain available. Confidence: high.*

**2. DenseCRF post-processing at full resolution.**
MORDEN shows it beats manual GT on edge metrics. We have never tried it. Attacks
SQ directly, which is where we concluded the ceiling was — and MORDEN's result
suggests that ceiling is not real.
*Expected: +0.01–0.03 PQ via SQ. Confidence: medium-high. Cost: ~20 min inference.*

**3. Instance Normalization instead of BatchNorm.**
MORDEN ablation: +0.018 IoU. Same physical argument as radial flattening.
*Expected: +0.01–0.02. Confidence: medium. Cost: one config flag + one 25-epoch run.*

### Tier 2 — worth one experiment each

**4. Large kernels (5×5) in the encoder.** MORDEN: +0.033 IoU, their biggest
single ablation win. Cheap to test on the resnet34 UNet? No — timm backbones have
fixed 3×3. Would need a custom encoder. *Confidence: medium, cost: high.*

**5. DBSCAN fragment integration instead of connected components.**
MORDEN reports 0.844 expert-judged cluster accuracy with ε=45, minPts=140 (at
1.04″/px; ours is ~1″/px so similar). We use plain 8-connectivity CC. DBSCAN can
merge fragments that CC splits, which is exactly our FP failure mode.
*Expected: +0.01–0.02 RQ. Confidence: medium. Cost: low.*

**6. Focal loss for the long-tail.** Used by Zheng 2024 and MORDEN (α=0.66,
γ=2). We use BCE+Dice. Filaments occupy <10% of the disk, and our FN are
concentrated in small/faint instances.
*Expected: small. Confidence: low-medium. Cost: low.*

### Tier 3 — structural, only if Tier 1 disappoints

**7. Instance segmentation (CondInst / Mask2Former) instead of semantic + CC.**
The only workflow whose training objective matches PQ. But MORDEN's objection is
real: fragment integration is ambiguous from Hα alone, so the labels are noisy in
exactly the way PQ punishes. *Confidence: low. Cost: very high.*

**8. Semi-supervised scale-up (Diercke).** Train a detector on the 707 labelled
files, use it to pseudo-label the unlabelled GONG archive, retrain. This is how
they got to 92% accuracy with a small labelled set. We have 707 files and no
access to the wider archive in this repo, so this is out of reach here.

---

## 8. The honest summary

The literature says our problem is **not** the architecture. Three independent
groups on this exact task converge on the same conclusion: a plain U-Net with
good preprocessing and good post-processing is competitive, and the gains come
from the input pipeline and the output refinement, not from the backbone.

We have spent this project's effort on backbones (DeepCrack, DINOv2, ConvNeXt,
Swin, EfficientNet, resnet18/34/50) and on training recipes (long schedules,
boundary loss, deep supervision, soft labels, decoders). We have spent almost
none on preprocessing — the one thing every paper above treats as a first-class
component, and the one place where we have a measured, unfixed, 2.6× defect.

That is the gap. Radial flattening is the single highest-value thing left.

### Verified absence (checked 2026-10-27)

Grepping `src/solar/` and `scripts/` for `flatten|radial|CLAHE|hough|crf|dbscan`
returns only unrelated hits (`tensor.flatten()`, and `deepcrack_fp_radial.py` /
`disk_analysis.py`, which *measure* the radial problem but do not correct it).
So:

| technique | in our pipeline? |
|---|---|
| radial flattening | **no** |
| CLAHE | **no** |
| Hough-circle disk detection | **no** (we use a threshold + largest CC) |
| DenseCRF | **no** (`pydensecrf` not installed) |
| DBSCAN | **no** (`sklearn.cluster` is available) |

All five are used by at least one of the papers above. None is in our code.

---

## References

1. Solomon V., Martens P., Liu J., Angryk R. *EdgeAttNet: Towards Barb-Aware
   Filament Segmentation.* arXiv:2509.02964 (2025). Code:
   https://github.com/dasjar/EdgeAttNet
2. Hu J.R., Hao Q., Zheng Z., Chen P.F., Li C., Meng Y. *A Modern ConvNet for
   Solar Filament Detection.* RAA, arXiv:2607.24525 (2026). Code:
   https://github.com/irisaltHu/MORDEN
3. Diercke A., Jarolim R., Kuckein C., et al. *A Universal Method for Solar
   Filament Detection from H-alpha Observations using Semi-supervised Deep
   Learning.* A&A 686, A213 (2024). arXiv:2402.15407
4. Guo X., Yang Y., Feng S., Bai X., Liang B., Dai W. *Solar-Filament Detection
   and Classification Based on Deep Learning.* Sol. Phys. 297:104 (2022).
5. Zheng Z., Hao Q., Qiu Y., Hong J., Li C., Ding M.D. *Developing an Automated
   Detection, Tracking and Analysis Method for Solar Filaments Observed by CHASE
   via Machine Learning.* ApJ 965:150 (2024). arXiv:2402.14209
6. Zhu G., Lin G., Yang X., Zeng C. *Flat U-Net: An Efficient Ultralightweight
   Model for Solar Filament Segmentation in Full-disk H-alpha Images.*
   ApJ 980:176 (2025). arXiv:2502.07259
7. Ahmadzadeh A., Adhyapak R., Chaurasiya K., et al. *A Dataset of Manually
   Annotated Filaments from H-alpha Observations* (MAGFiLO). Scientific Data
   11:1031 (2024).
8. Chalmers A., Ahmadzadeh A. *Automatic Classification of Magnetic Chirality of
   Solar Filaments from H-Alpha Observations.* arXiv:2509.18214 (2025) — first
   reproducible MAGFiLO classification baseline.
9. Krähenbühl P., Koltun V. *Efficient Inference in Fully Connected CRFs with
   Gaussian Edge Potentials.* NIPS (2011) — the DenseCRF method.
10. Tian Z., Zhang B., Chen H., Shen C. *Instance and Panoptic Segmentation Using
    Conditional Convolutions.* IEEE TPAMI 45(1) (2023) — CondInst.
