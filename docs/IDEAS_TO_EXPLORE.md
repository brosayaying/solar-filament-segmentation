# Ideas to explore — systematic register

Living document. Every idea gets an ID, a source, a status, and a measured
outcome. Nothing is closed without a number.

Status values: `todo` · `running` · `done` · `rejected` · `blocked`

Sources: `LIT` = `docs/LITERATURE_REVIEW.md`, `OWN` = our own diagnosis,
`LB` = leaderboard-driven.

---

## Tier 1 — cheap, literature-validated, highest expected value

| ID | idea | source | why | status | result |
|---|---|---|---|---|---|
| **P1** | **Radial flattening** (annular background estimate, divide out) | LIT EdgeAttNet §III-A step 3; OWN `DISK_MASKING.md` | Fixes measured recall collapse 0.73 → 0.28 in outer ring (11% of filament area). Limb darkening inverts contrast +12 → −24. | **implemented, retrain pending** | Eval-time probe invalid (see N1). Flatten is contrast-neutral-to-worse when bolted onto a raw-trained model: −0.0048 CI [−0.0137, +0.0042]. Needs a model trained on it. |
| **P2** | **CLAHE** inside disk mask | LIT EdgeAttNet §III-A step 5 | Amplifies faint filamentary signal in low-intensity regions without over-enhancing bright ones. | **rejected as standalone** | −0.0951 CI [−0.1197, −0.0704] at eval time. Clearly harmful; deprioritised. |
| **N1** | **Disk-only normalisation** | OWN `DISK_MASKING.md`; `img_stats.json` | `mean=0.5, std=0.25` were wrong twice: computed over the whole frame (28% is off-disk black, mean 3.2) and too wide. Measured disk stats are **mean 0.4463, std 0.1313**. The old std was **1.9× too large**, compressing the signal into a narrow band. | **implemented, retrain pending** | — |
| **P3** | **Hough-circle disk detection** | LIT EdgeAttNet §III-A step 2 | Our `disk_geometry` uses threshold + largest CC, which includes darkening limb as background. Hough is more robust. | todo | — |
| **P4** | **DenseCRF post-processing at full res** | LIT MORDEN §3.4.1 | Their DenseCRF@2K edges (grad 259.0) beat manual GT (250.6). Attacks SQ, which we thought was capped. Never tried. | todo | — |
| **P5** | **InstanceNorm instead of BatchNorm** | LIT MORDEN ablation (+0.018 IoU) | BN normalises away the local contrast that distinguishes limb filaments. Same physics as P1, in the norm layer. | todo | — |

## Tier 2 — one experiment each

| ID | idea | source | why | status | result |
|---|---|---|---|---|---|
| **P6** | **DBSCAN fragment integration** instead of 8-conn CC | LIT MORDEN §2.3 (ε=45, minPts=140) | Merges fragments CC splits — our dominant FP mode (58.5% of FPs are splits of matched GT). | todo | — |
| **P7** | **Focal loss** (α=0.66, γ=2) | LIT Zheng 2024, MORDEN | Long-tail: filaments <10% of disk; FN concentrated in small/faint instances. | todo | — |
| **P8** | **Large kernels (5×5)** in encoder | LIT MORDEN ablation (+0.033 IoU, their biggest win) | Needs a custom encoder; timm backbones are fixed 3×3. | todo | — |
| **P9** | **Limb-aware loss weighting** | OWN `DISK_MASKING.md` option 2 | Up-weight outer ring so the model stops ignoring it. Cheaper than P1 but treats the symptom. | todo | — |

## Tier 3 — structural, only if Tier 1 disappoints

| ID | idea | source | why | status | result |
|---|---|---|---|---|---|
| **P10** | **Instance segmentation** (CondInst / Mask2Former) | LIT Guo 2022 | Only workflow whose training objective matches PQ. But MORDEN argues fragment integration is ambiguous from Hα alone, so labels are noisy in the way PQ punishes. | todo | — |
| **P11** | **Semi-supervised scale-up** (YOLOv5 → U-Net on unlabelled GONG) | LIT Diercke 2024 (92% accuracy) | How they beat a small labelled set. Needs the wider GONG archive, not in this repo. | blocked | — |
| **P12** | **EMA weights + 3–5 seed ensemble** | OWN `CORRECTIONS.md` | Cheap, standard, decorrelates errors. Tiled/512 pair is a free start. | todo | — |

## Already tried — closed

| ID | idea | result | verdict |
|---|---|---|---|
| C1 | DeepCrack zero-shot | PQ 0.000 (TP=0, FP=921) | rejected |
| C2 | DeepCrack finetune (3 runs) | 0.3521 best | rejected — below UNet |
| C3 | DeepCrack from scratch | 0.3507 | rejected — pretraining worth nothing |
| C4 | DeepCrack + decoder | 0.3834 (+0.040 paired, CI [+0.025,+0.056]) | **real but insufficient** — still −0.032 vs UNet at 3.1× cost; SQ unmoved |
| C5 | Deep supervision on side heads | 0.3394 (−0.025) | rejected |
| C6 | BN recalibration | AUC 0.639 → 0.425 | rejected |
| C7 | Soft labels (mean over annotators) | +0.0005, CI [−0.007, +0.009] | rejected (recipe, not idea — see P13) |
| C8 | Boundary loss | PQ flat, contours better | rejected |
| C9 | Watershed splitting | negative | rejected — nothing to split |
| C10 | Morphological close k=15 | +0.012 | **shipped** |
| C11 | Proximity merge 4px | +0.011 | superseded by C10 |
| C12 | Hysteresis thresholding | −0.011 | rejected |
| C13 | Dilate / erode ±1–3 px | negative | rejected — thickness already correct |
| C14 | Tiled 1024² inference | t=−0.18 vs 512² | rejected — 6× cost, no gain |
| C15 | DINOv2 frozen encoder | 0.3409 | rejected |
| C16 | Backbone sweep (6 ep) | all < 0.27 | inconclusive — truncated at 6 epochs |
| C17 | 4× flip TTA | +0.0024, inside CI | marginal |

## Open questions that are not yet ideas

- **P13** — Soft labels *from pretrained init* rather than warm-started 25 ep at
  lr=1e-5. C7 failed but the recipe was too small a perturbation to reshape the
  target. Only 42% of train files have >1 annotator, so the ceiling is limited.
- **P14** — Why do 3–5 test images produce zero predictions? ~0.005 public PQ,
  free. Check whether they are limb-heavy or 2021–22 (11.7% of test vs 5.7% of
  train).
- **P15** — `group_by: "week"` split instead of `date`. Date grouping still puts
  adjacent days on both sides.
- **P16** — Official-scorer local harness: write val preds + GT as
  submission-format RLE and run the notebook's `get_pq_score`. Final check before
  every submission.

---

## Measurement protocol

Every idea is scored the same way so results are comparable:

- **Split**: `splits/date_seed42.json`, deduplicated val (142 unique files).
- **Metric**: PQ at IoU > 0.5, threshold × min_area swept, best reported.
- **Significance**: paired per-image test, 95% CI. A point estimate without a CI
  is not a result.
- **Baseline to beat**: resnet34 UNet 512², 160 ep → **0.4152** local / 0.34 public.
- **Report**: PQ, SQ, RQ, TP/FP/FN, and the paired delta vs baseline.

A change is only "shipped" if it improves the paired CI and does not regress SQ.
