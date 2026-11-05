# Long training: 160 epochs at 512²

Run: `runs/long_160ep_512` — resnet34, 512² random crops, BCE+Dice, 160 epochs,
lr 3e-4 cosine, batch 12. Same split and post-processing as every other run.

## Result

Best PQ **0.4361** at epoch 126 (SQ 0.7012, RQ 0.6219, tp 250 / fp 137 / fn 167).
Measured on the 64-image validation subset used for checkpoint selection.

| run | epochs | PQ (64-img) |
|---|---|---|
| baseline | 25 | 0.3619 |
| **long** | **160** | **0.4361** |
| boundary loss | 25 | 0.3642 |

So +0.074 from the longer schedule — the largest single gain of any experiment so
far, larger than tiled training, DINOv2, boundary loss, or all post-processing
tuning combined.

> **Caveat on comparing runs.** The 64-image subset is used for checkpoint
> selection, so these numbers are optimistic. The tiled-vs-512 table in the README
> uses the **full 232-entry val set** (1641 GT instances) and is the honest
> comparison. The two subsets differ by roughly 0.02–0.04; never compare one to
> the other.

## But the gain decomposes, and it is not what we expected

| component | ep 25 | ep 80 | ep 126 | gain over last 135 epochs |
|---|---|---|---|---|
| **SQ** | 0.6861 | 0.6940 | 0.7012 | **+0.014** |
| **RQ** | 0.5188 | 0.5653 | 0.6219 | **+0.095** |
| PQ | 0.3560 | 0.3923 | 0.4361 | +0.074 |

**SQ was essentially done at epoch 25.** 135 more epochs bought 0.014 of boundary
quality. Everything else was RQ.

### What RQ was actually learning

The predicted/GT instance count ratio over training:

| epoch | pred | GT | ratio | PQ | SQ | RQ |
|---|---|---|---|---|---|---|
| 10 | 569 | 417 | 1.36 | 0.329 | 0.681 | 0.483 |
| 25 | 485 | 417 | 1.16 | 0.356 | 0.686 | 0.519 |
| 50 | 493 | 417 | 1.18 | 0.366 | 0.694 | 0.527 |
| 80 | 464 | 417 | 1.11 | 0.392 | 0.694 | 0.565 |
| 110 | 388 | 417 | 0.93 | 0.424 | 0.705 | 0.601 |
| **126** | **387** | **417** | **0.93** | **0.436** | **0.701** | **0.622** |
| 160 | 420 | 417 | 1.01 | 0.430 | 0.700 | 0.614 |

The model was spending most of its 160 epochs learning **how many** filaments to
predict, not **how well to delineate** them. It started at 1.36× too many and
converged on the correct count.

This reframes the boundary-loss result. That experiment raised IoU>0.5 coverage
63.6% → 69.4% — real contour improvement — but pushed the count to 1.38× and PQ
went *down*. Both results point at the same thing: **count control dominates
contour precision** for this metric.

## Plateau is genuine

| window | mean PQ | std |
|---|---|---|
| 121–140 | 0.4266 | 0.0051 |
| 141–160 | 0.4270 | 0.0036 |

Difference +0.0004, well within noise. RQ flattened around epoch 126 and the
cosine schedule had decayed to zero by 160. More epochs at this configuration
will not help.

## Tiled 1024², warm-started, 60 epochs

`runs/tiled1024_long` — warm-started from `runs/long_160ep_512/best.pt`, 1024²
tiled quadrants, 60 epochs, lr 2e-4, ~211 s/epoch (3.5 h).

Best PQ **0.4343** at epoch 37 (64-image subset). Essentially identical to the
512² 160-epoch run's 0.4361.

### Warm start works, and saves the count-calibration phase

| | epoch 1 PQ |
|---|---|
| tiled-1024 from scratch | 0.1046 |
| tiled-1024 warm-started | **0.3831** |

The warm start skips the ~40 epochs the 512² run spent learning the instance
count. That is the main practical value of `train.init_ckpt`.

### But tiled does not beat 512² once measured honestly

Both runs plateaued, so I re-evaluated every checkpoint on the **deduplicated
validation set** (142 entries, one annotation per unique file — see
[METRIC_CEILING.md](METRIC_CEILING.md)):

| run | PQ | SQ | RQ | tp | fp | fn |
|---|---|---|---|---|---|---|
| 512², 25 ep | 0.3768 | 0.6935 | 0.5434 | 601 | 583 | 427 |
| 512², 160 ep | 0.4152 | 0.6996 | 0.5934 | 586 | 361 | 442 |
| tiled 1024², 12 ep | 0.3942 | 0.6935 | 0.5684 | 596 | 473 | 432 |
| **tiled 1024², warm + 60 ep** | **0.4195** | 0.6968 | 0.6021 | 612 | 393 | 416 |

Tiled is nominally ahead by 0.004, but a paired per-image test says that is noise:

| | value |
|---|---|
| mean per-image difference (tiled − 512) | **−0.0016** |
| std | 0.1005 |
| stderr | 0.0084 |
| **t** | **−0.18** |
| tiled better on | 67 / 142 images |

**t = −0.18.** The two configurations are indistinguishable. Tiled costs 6× the
epoch time (211 s vs 35 s) for no measurable gain.

### Human baseline on the same set

| | PQ | SQ | RQ |
|---|---|---|---|
| human vs human (61 imgs with a 2nd annotator) | 0.3719 | 0.6505 | 0.5718 |
| **our best model** | **0.4195** | **0.6968** | **0.6021** |

Still above the human-to-human line, on the cleaner metric too.

## What the two long runs together establish

1. **Long training helps, once.** 25 → 160 epochs: 0.3768 → 0.4152 on dedup val
   (+0.038). Real, and the largest single gain available.
2. **It saturates.** Both runs plateaued with last-10-epoch std of 0.003–0.004.
3. **Resolution does not help.** Tiled 1024² is statistically identical to 512²
   despite 6× the compute. This is the third independent line of evidence against
   the resolution hypothesis (see the stride-quantisation ceiling and the width
   breakdown above).
4. **We are above the human ceiling.** 0.4195 vs 0.3719 human-to-human.

## Where the remaining headroom is

SQ is 0.701. The quantisation ceiling (GT mask downsampled to a stride-4 grid,
then connected components) is **0.822**. So there is ~0.12 of genuine SQ headroom
that 135 epochs did not touch — the model is not limited by time, it is limited by
what it can resolve.

Note also that 0.822 is the stride-4 ceiling, and stride-2 gives 0.912. Since the
encoder already emits stride-2 features, the ceiling for this architecture is
closer to 0.9. Being stuck at 0.701 with that much room means it is a
representation/capacity problem, not a resolution or schedule problem.

## Next steps, ranked

### 1. Warm start (implemented)

`train.init_ckpt` now loads weights from a checkpoint before training. The
160-epoch study showed ~40 epochs go purely into learning the instance count, a
cost worth paying once. Every experiment below should warm-start from
`runs/long_160ep_512/best.pt`.

Measured effect: tiled-1024 from scratch reached PQ 0.1046 at epoch 1; warm-started
from the 512² checkpoint it starts at **0.3831**.

### 2. Tiled 1024² + long schedule — running

`configs/tiled1024_long.json`, warm-started, 60 epochs, ~198 s/epoch (~3.3 h).

Rationale: tiled-1024 reached 0.4300 on the **full val set** in only 12 epochs,
while 512² needed 160 epochs to reach a comparable number on the harder 64-image
subset. Tiled is much stronger per unit training and attacks SQ directly — each
filament is resolved with 2× the pixels.

Known weakness: only 4 fixed tile positions per image, so low translation
diversity. If it plateaus early, overlapping crops (stride 512, 9 positions) is
the fix.

### 3. Higher-capacity decoder

If (2) plateaus near 0.45, the next lever is capacity: `decoder_dim` 128 → 256, or
a deeper decoder. SQ flat since epoch 25 is the argument — the decoder is not
extracting more from the encoder features, no matter how long it trains.

### 4. Stop doing these

* More epochs at 512² (saturated, measured).
* Tiled 1024² (no gain, 6× cost, measured).
* Post-processing tuning (exhausted, max +0.007).
* Watershed / instance separation (oracle says nothing to win).
* Boundary loss alone (improves SQ proxy, hurts count, net negative).
* Chasing local PQ past ~0.45 (above the human ceiling; see METRIC_CEILING.md).

## Reproducing

```bash
python -m solar.train --config configs/long.json
python -m solar.train --config configs/tiled1024_long.json   # warm-started
```
