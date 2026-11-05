# Small CNN on 16×16 patches vs the RandomForest

Does learning the features beat hand-engineering them?

```bash
python scripts/patch_cnn.py --patch 16 --ctx 48 --train-images 60 --val-images 20 --epochs 30
```

Results in `docs/EDGE/patch_cnn.json`.

---

## 1. Setup

Same train/val image split, same patch size, same patch-F1 metric as
`scripts/compare_patch_sizes.py`, so the numbers are directly comparable.

| | RandomForest | CNN |
|---|---|---|
| input | 802 hand-engineered features | 48×48 raw crop (3×3 patches) |
| model | 200 trees | 6 conv layers, width 32, ~1.2 M params |
| imbalance | negative subsampling (5%) | `pos_weight` in BCE |
| augmentation | none | dihedral (8×), re-drawn each epoch |
| normalisation | per-feature (implicit) | global mean/std from train |
| train time | 37 s (CPU) | 353 s (GPU) |

The CNN gets a 48×48 crop centred on the patch — the same 3×3-patch context the
RF's context block provides — and must learn its own filters from raw intensity.

---

## 2. Result

| model | patch AUC | patch F1 | best thr |
|---|---|---|---|
| RandomForest | 0.9532 | **0.526** | 0.70 |
| CNN | **0.9715** | 0.522 | 0.90 |

**The CNN does not improve F1.** It is 0.004 *worse* — well within noise at this
sample size. It does have higher AUC (0.9715 vs 0.9532), but as
`docs/SCOREMAP.md` shows, AUC is inflated by the 1:110 class imbalance and does
not track PQ.

Training curve: F1 rose to 0.544 at epoch 20, then oscillated between 0.49 and
0.54 through epoch 30. Train loss kept falling (0.67 → 0.29), so the model is
mildly overfitting on 60 images despite augmentation.

---

## 3. Three bugs found along the way

Worth recording because two of them produced *plausible-looking* wrong answers.

### 3.1 Patch-size mismatch corrupted the labels (fatal)

`patch_fraction()` reads the module-global `PATCH` from `patch_clf`, but
`patch_cnn.py` never called `set_patch_size`. Labels were therefore computed on
an **8×8 grid** while crops were 16×16 — the label for patch `(i, j)` described a
different region than the crop centred on it.

Symptom: loss stuck at 1.15 (predicting 0.5 everywhere), val F1 ≈ 0.03. The
model could not learn because the labels were noise.

Fix: call `pc.set_patch_size(args.patch)` before any label is computed.

### 3.2 Per-crop normalisation erased the signal

The first version standardised each crop independently to mean 0, std 1. That
destroys exactly the contrast that separates a dark filament from flat
background — both become identical to the network.

Fix: global mean/std computed from the training set (mean 120.9, std 26.1).

### 3.3 Validation OOM

219,501 val patches in a single forward pass needed 60 GB. Fix: batched
inference at 4096.

---

## 4. Interpretation

**Hand-engineered features are competitive with a learned front-end at this data
size.** That is the honest conclusion, and it is not surprising:

- 60 training images is very little for a CNN. The RF's black-hat features
  encode a strong prior (filaments are dark, elongated, ~17 px wide) that the
  CNN must rediscover from scratch.
- The RF's 802 features are multi-scale by construction. The CNN sees one
  scale (48×48) and would need depth or dilation to match.
- The CNN's higher AUC with equal F1 suggests it ranks slightly better but
  calibrates worse — its best threshold is 0.90, which is a sign of
  overconfidence from `pos_weight` on a 1:110 problem.

**Neither model is good enough to matter.** F1 ≈ 0.52 for both, against a
ceiling set by the data (per-pixel SNR 0.76, see `docs/EDGE_DETECTION.md`). The
DL segmentation baseline reaches PQ 0.36–0.39 by working at full resolution with
a pretrained backbone; a 16×16 patch classifier, learned or not, discards too
much spatial information to compete.

---

## 5. What would actually help

If this line were pursued, in order of expected value:

1. **More training images.** 60 → 700 is available. Both models were still
   improving; the CNN especially would benefit.
2. **Multi-scale input.** Feed the CNN a pyramid (16/48/144 px crops) or use
   dilated convolutions, so it sees the same scales the RF's black-hat stack
   covers.
3. **Combine rather than replace.** The RF and CNN make different errors
   (different AUC/F1 trade-offs); averaging their probabilities is the cheapest
   possible gain and was not tested here.
4. **Do not chase AUC.** Patch F1 and per-image precision/recall are the metrics
   that track PQ.

---

## 6. Caveat

20 validation images, 219,501 patches, 2,043 positives. The F1 difference
(0.004) is far inside run-to-run noise — a single seed was used for each model.
The conclusion "CNN does not clearly beat RF" is safe; the conclusion "RF is
better" is not.
