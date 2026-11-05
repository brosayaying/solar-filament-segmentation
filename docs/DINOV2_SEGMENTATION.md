# End-to-end DINOv2 for segmentation

Question: can DINOv2 be used end-to-end for the segmentation task, and does it
beat training from scratch?

Short answer: **it works, but frozen DINOv2 does not beat the from-scratch
resnet34 baseline.** It matches on boundary quality (SQ) and loses on instance
recall (RQ). The interesting part is *why*, and one real bug it exposed in the
validation path.

## Why the naive approach fails

`TimmUNet` builds its encoder with `features_only=True`. For DINOv2 that returns
three levels **all at stride 14**:

```
channels:  [384, 384, 384]
reduction: [14, 14, 14]
modules:   ['blocks.9', 'blocks.10', 'blocks.11']
```

So the "UNet" is three 1×1 convs on identical-resolution features, concatenated,
then bilinearly upsampled **14×** to full resolution. There are no fine-scale
skip connections because there is no fine scale. That is not a UNet — it is a
token classifier with an optimistic upsample, and a negative result from it would
be uninformative.

## What was built

`DinoUNet` in `src/solar/models.py`:

* **Pseudo-pyramid from block depth.** Taps blocks 2/5/8/11 (configurable via
  `layers`) and fuses them into one stride-14 map. Different depths carry
  different amounts of semantics at the same resolution.
* **Conv stem on the raw image** supplies the fine scales. This is where
  sub-token detail has to come from: a filament is ~30 px wide, i.e. only ~2
  tokens across at stride 14.
* **Decoder 14 → 7 → 4 → 2 → 1**, concatenating the stem at each step.
* **Frozen encoder by default** (`freeze_encoder: true`). Deliberate: this is the
  cheap ablation that answers "are the pretrained features worth anything here"
  before paying for full fine-tuning. `train()` is overridden so the encoder
  stays in eval mode regardless of the outer mode.

Two implementation details that cost time:

* DINOv2 has a **CLS prefix token**, so `num_prefix_tokens` must be stripped
  before reshaping tokens to a grid. Without it: `block 2: 1370 tokens != 37x37`.
* 1024 and 2048 are **not divisible by 14**. Use 518/1022/2044, or pad.

Decoder widths taper (128 → 128 → 64 → 32) because a full 128-channel
`ConvBlock` at 1024² costs ~1 GB per tensor per image; the first version OOM'd
at batch 4.

## Cost

| encoder | res | bs | peak | img/s |
|---|---|---|---|---|
| resnet34 | 512² | 8 | 5.7 GB | **48.1** |
| DinoUNet (frozen) | 518² | 8 | 4.4 GB | **40.8** |
| resnet34 | 1024² | 8 | 22.7 GB | **8.9** |
| DinoUNet (frozen) | 1022² | 8 | 16.9 GB | **7.5** |

Frozen DINOv2 is only ~15% slower than resnet34 at matched resolution, and uses
less memory. Trainable params: **1.19 M** of 23.7 M total.

## The bug this exposed: validation ran at the wrong scale

The first run reported PQ 0.03–0.05 with RQ ≈ 0.04 — the model predicted ~15
instances where there were ~417. It looked like a collapsed model. It was not.

The model was **trained on 518² crops (37×37 tokens) but validated on full
2048² images (146×146 tokens)**. ViTs are strongly scale-sensitive: pos-embed
resampling and global attention behave very differently at 4× the token count.
Measuring the same image content at three sizes:

| input | predicted area / GT area |
|---|---|
| 1022² | 1.46 |
| 2048² | **0.19** |

A 7.7× swing from resolution alone. Tiled inference at the training scale
recovered most of it:

| inference | PQ |
|---|---|
| full 2048² | 0.045 |
| tiled 1022² | 0.146 |
| tiled 518² | **0.218** |

Fix: `eval.tile` in the config, plus `_predict_tiled` in `train.py`. This
matters for **any** ViT encoder, not just this experiment — the old validation
path silently under-reports them.

A second, smaller bug: `_predict_tiled` returns `(B, H, W)` while the non-tiled
path returns `(B, 1, H, W)`, so the shared `probs[i, 0]` indexing read row 0 and
scored PQ 0.0000. Fixed by adding the channel axis.

## Result

40 epochs, 518² crops, frozen encoder, tiled-518 validation, default
postprocessing (`thr 0.55`, `min_area 512`):

| run | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|
| resnet34, 512², from scratch | **0.3619** | 0.6791 | **0.5328** | 284 | 365 | 133 |
| DINOv2 ViT-S/14 frozen + conv decoder | 0.3409 | **0.6817** | 0.5000 | 231 | 276 | 186 |

Both at their best epoch (21 and 28 respectively), same split, same
postprocessing.

### Reading the result

**SQ is identical** (0.6791 vs 0.6817). When the model does find a filament, it
delineates it just as well as the from-scratch baseline. The pretrained features
are not hurting boundary quality.

**The entire gap is RQ** (0.5328 vs 0.5000), i.e. instance detection. DINOv2
finds 231 instances where the baseline finds 284, and misses 186 vs 133. It is
more conservative: fewer false positives (276 vs 365) but more misses.

That is consistent with the embedding analysis in
[docs/EMBEDDINGS.md](EMBEDDINGS.md): at 14 px tokens, ~28% of filament pixels
have no positive token, and the model under-predicts area. The conv stem
recovers fine detail for *boundaries* but cannot recover *instances* the
stride-14 features never separated.

## What this does and does not show

**Does show:** frozen DINOv2 features are usable end-to-end and produce
competitive boundary quality with 1.19 M trainable params. The architecture
works; the scale-sensitivity bug is fixed.

**Does not show:** that DINOv2 is worse than from-scratch training in general.
The encoder was frozen, so this measures *frozen features + conv decoder*, not
DINOv2 fine-tuned. Fine-tuning could plausibly close the RQ gap, since the
encoder could adapt its features to thin structures.

## Next steps, in order of expected value

1. **Fine-tune the encoder** (`freeze_encoder: false`). This is the obvious
   untested variable. Cost: ~3× memory, so batch 4 at 518².
2. **DPT-style reassembly** instead of block-tapping: progressive upsampling
   with lateral connections, which is the standard ViT-segmentation answer.
3. **Combine with the spine detector.** The stage-1 CenterNet already reaches
   mask-hit 0.650. Seeding instances from predicted spines would attack RQ
   directly, which is exactly where this model is weak — and it does not depend
   on which segmentation backbone wins.

## Reproducing

```bash
python -m solar.train --config configs/dino_frozen.json
```

Config: `configs/dino_frozen.json` (40 epochs, 518² crops, `eval.tile: 518`).
Checkpoint: `runs/dino_frozen_518_v2/best.pt` (epoch 28, PQ 0.3409).
