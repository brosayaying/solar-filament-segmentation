# Fold-0 A/B — radial flattening vs raw

Status: **complete, verdict INCONCLUSIVE** (2026-10-30 06:24).

Two attempts were invalidated before this one:

1. **fp16 divergence** — both arms went NaN mid-training.
2. **Early-stopping bug** — `best.pt` held a worse epoch than the true maximum,
   which biased the comparison against the raw arm by 0.0020 PQ and flipped the
   verdict from "inconclusive" to "flatten wins".

See [Attempt 1](#attempt-1-fp16-divergence-invalid) and
[Attempt 2](#attempt-2-early-stopping-bug-invalid) below.

The experiment `docs/PREPROCESSING.md` never actually ran. That doc measured
flattening by feeding flattened input to a *raw-trained* checkpoint, which
measures distribution shift, not the benefit of preprocessing. The retrain config
(`configs/norm_disk_flatten.json`) existed but was never executed. This is that
retrain, done properly.

## Design

Two arms, identical in everything except `data.preprocess` and the normalisation
constants that go with it.

| | raw | flatten |
|---|---|---|
| backbone | `convnext_small.fb_in22k_ft_in1k` | same |
| `stem_ch` | 32 | 32 |
| `in_chans` | 1 | 1 |
| patch / batch | 512² / 12 | 512² / 12 |
| epochs | 160 (early stop, patience 30) | same |
| lr / wd | 3e-4 / 1e-4, cosine, 2 warm-up | same |
| loss | BCE + Dice | same |
| AMP | **bf16** | same |
| `preprocess` | none | `flatten` (cached) |
| `mean` / `std` | 0.4463 / 0.1313 | **0.4254 / 0.0809** |
| split | `v2/splits/week5_seed42/fold0.json` | same |
| val | full fold (224 entries / 142 files) | same |

### Why ConvNeXt-Small and not resnet34

The A/B answers a preprocessing question, but it must be answered on the network
we intend to ship. Testing on `resnet34` and then switching to ConvNeXt would
leave the result uninformative for the final model — and a `resnet34` fold-0
checkpoint is throwaway work, whereas a ConvNeXt fold-0 checkpoint is a real
ensemble member if that arm wins.

It also settles a question the repo never answered: the 6-epoch backbone sweep
was truncated and uninformative, so we do not actually know whether ConvNeXt
beats resnet34 at convergence.

### Why the stride-1 stem path

`convnext_small` exposes features at reductions `[4, 8, 16, 32]` — its finest
feature is **stride 4**. `resnet34` has `[2, 4, 8, 16, 32]`, so it has a stride-2
feature. A median filament is 11 px wide, so reconstructing it from stride-4
features alone discards most of the boundary signal. This is the same failure
that capped DeepCrack at PQ 0.35 (no decoder, best head at stride 8) against the
UNet's 0.44.

`TimmUNet(stem_ch=32)` adds a 3×3 conv on the *input*, concatenated into the
final decoder block, so fine detail is refined rather than hallucinated.
`stem_ch=0` is the default and keeps every existing checkpoint loadable
(verified: `load_state_dict` on `runs/norm_disk_raw/best.pt` gives 0 missing and
0 unexpected keys).

### Why the flatten arm uses different normalisation

`normalise_image` standardises with `(x - mean) / std`, so the constants must
describe the arm's own input distribution. Measured on the flatten cache
(80 files, 242.6 M disk pixels):

| input | disk mean | disk std |
|---|---|---|
| raw | 0.4463 | 0.1313 |
| flattened | **0.4254** | **0.0809** |

Using the raw constants on flattened input would leave the signal compressed into
a narrower band than intended — the same class of error as the original
`0.5 / 0.25` bug, which was 1.9× too wide.

### Early stopping

On **validation PQ**, not train loss. Train loss is the wrong signal: on
`runs/norm_disk_raw` it was flat (sd 0.006 over the last 40 epochs) while val PQ
was still climbing and peaked at epoch 100 of 106. Stopping on loss would have
quit ~60 epochs early.

Val PQ is noisy (sd 0.007), so patience is 30 evaluations with `min_delta`
0.002 — roughly 4× the measured noise, and long enough that a genuine late
improvement is not cut off.

## Cost

Measured **109 s/epoch**, not the ~47 s the step benchmark implied — that
benchmark measured training steps only and forgot validation:

| component | cost |
|---|---|
| 155 train steps @ ~0.50 s | 78 s |
| 224 val entries @ 2048², ~0.14 s | 31 s |

So 160 epochs ≈ **4.8 h per arm, ≈ 9.7 h for both** (less if early stopping
fires). Validation runs every epoch because early stopping needs a reliable
signal; `val_every=2` would save only 1.4 h and halve the stopping resolution,
which is not a good trade.

## Attempt 1: fp16 divergence (invalid)

The first run (18:38, fp16 AMP) reported "raw wins, Δ −0.0154", which was **not a
valid result**. Both arms had diverged to NaN mid-training:

| arm | first NaN | loss before | best PQ before NaN |
|---|---|---|---|
| raw | epoch 23 | 0.1737 | 0.3772 @ ep 18 |
| flatten | epoch 17 | 0.1730 | 0.3618 @ ep 8 |

Both went from a healthy loss of 0.17 to NaN at ~2.9e-4 LR, despite
`GradScaler`. The scorer then compared two mostly-dead models' best-epoch
checkpoints, and early stopping had fired on the flat NaN tail — the "verdict"
was an artefact.

Cause: fp16 autocast. ConvNeXt's LayerNorms are the usual suspect — fp16's 5-bit
exponent cannot represent their activation range, and once a layer overflows the
scaler backs off but the weights are already poisoned. `torch.amp.GradScaler`
does not protect against a forward overflow that has already produced inf/NaN
activations inside a residual branch.

**Fix:** `train.amp_dtype = "bf16"` (now the default). bf16 has fp32's exponent
range, needs no scaler, and is the standard choice for modern conv/transformer
backbones. The plan called for bf16 all along; the training loop had hardcoded
fp16.

Verified stable: a 30-epoch bf16 run passed epoch 23 cleanly and reached
**PQ 0.4178 by epoch 26** — already above the fp16 run's entire peak of 0.3772.

Also added:

* **non-finite loss guard** — a non-finite batch is skipped rather than applied,
  because applying it turns every weight into NaN and the run never recovers.
  Skipped steps are logged as `skipped_steps`.
* **non-finite val PQ guard** — raises `SystemExit` and marks `diverged: true`,
  instead of silently training a dead model for another 100 epochs.

Both guards exist because the original failure was **silent**: the log showed
`loss nan` and `PQ 0.0000` for 25 consecutive epochs and nothing complained.

## Scoring

`python v2/scripts/score_ab_fold0.py`

Both arms see the same 142 validation files, so the comparison is a **paired
per-image test** on the difference, not two independent aggregates. Reports the
mean per-image PQ delta with a 95 % t-interval.

Post-processing is fixed at thr 0.65 / min_area 256 / close 15 — the setting
chosen on the full-val sweep in Phase 0.4 — so the comparison isolates
preprocessing rather than re-tuning a knob per arm.

## Result — inconclusive

| arm | PQ | SQ | RQ | tp | fp | fn | best epoch |
|---|---|---|---|---|---|---|---|
| raw | 0.4041 | 0.6667 | 0.6061 | 1027 | 808 | 527 | 61 (bug; true 85) |
| flatten | 0.4198 | 0.6762 | 0.6208 | 1029 | 732 | 525 | 110 |

Paired per-image test, n = 224, flatten − raw:

| | delta | 95 % CI | p |
|---|---|---|---|
| as run | +0.0154 | [+0.0008, +0.0299] | 0.039 |
| **bias-corrected** | **+0.0134** | **[−0.0012, +0.0279]** | **0.08** |

**Verdict: inconclusive.** The as-run CI excludes zero by 0.0008, which is not a
result — it is the early-stopping bug (§Attempt 2) handicapping the raw arm by
exactly 0.0020 PQ. Correcting for it puts the CI across zero.

### The experiment was underpowered anyway

Per-image PQ sd is 0.1107, so the standard error on 224 images is 0.0074 and the
CI half-width is 0.0146. To detect a 0.015 effect at 80 % power needs **428
images — 1.9× what one fold provides**. A 0.010 effect needs 963 images (4.3×).

So even without the bug, one fold could not have resolved this. The honest
reading is: **flattening is worth somewhere between −0.001 and +0.028 PQ, and
this experiment cannot narrow it further.**

### What the arms do agree on

The direction is consistent and the mechanism is visible in the counts:

* **FP: 808 → 732 (−9.4 %)** with TP essentially unchanged (1027 → 1029).
  Flattening removes spurious detections without losing real ones.
* **SQ: 0.6667 → 0.6762**, **RQ: 0.6061 → 0.6208**. Both improve.
* The raw arm early-stopped at epoch 91; flatten ran to 140. Flattening gives the
  optimiser a longer useful runway, which is consistent with removing a
  contrast-inversion nuisance the network would otherwise spend capacity on.

That is a coherent story, but a coherent story is not a significant result.

## Decision

**Do not ship flattening on this evidence.** The decision rule was written before
the result and it says a CI including 0 means drop it. An 11 GB cache and
~0.5 s/image must earn their place.

But the FP reduction is real and the direction is consistent, so this is not a
closed question — it is an unresolved one. Two options, in order of preference:

1. **Re-run the A/B across all 5 folds** (≈ 5× the images, ≈ 1,120 val files).
   That reaches the 428-image threshold for 80 % power at 0.015. Cost: ~10 h per
   arm, ~20 h total — affordable, and it produces five usable ensemble members
   either way.
2. **Fold it into Phase 1 as a 2-arm comparison on the full 5-fold CV** rather
   than a separate experiment. Same cost, no extra wall-clock, and the answer
   arrives with the Phase 1 result instead of before it.

Option 2 is what the plan should do. The A/B was worth running once to learn that
the effect is small and that one fold cannot see it; repeating it as a standalone
experiment would be spending 20 h to re-learn that.

## Attempt 2: early-stopping bug (invalid)

The first bf16 run reported "flatten WINS, CI [+0.0008, +0.0299]". That verdict
did not survive inspection.

`best_pq` was updated only when the improvement exceeded `min_delta`:

```python
if metrics["pq"] > best_pq + min_delta:   # BUG
    best_pq = metrics["pq"]
    torch.save(..., "best.pt")
```

So `best_pq` tracked a *gated* best, not the true maximum. On the raw arm:

| | epoch | val PQ |
|---|---|---|
| saved to `best.pt` | 61 | 0.4057 |
| true maximum | 85 | 0.4077 |

The gap is 0.0020 — exactly `min_delta`, so the improvement never cleared the
gate and epoch 85 was never saved. The raw arm was scored with a checkpoint
0.0020 PQ worse than the best it actually trained, which is the same order as the
effect being measured.

**Fix:** separate the two notions of "best".

* `best_pq` — the true maximum; this is what `best.pt` must hold.
* `gate_pq` — the last value that counted as progress, used only for patience.

Verified against the real logs: the fixed logic saves epoch 85 (0.4077) for raw
and epoch 110 (0.4177) for flatten, i.e. the true maximum in both arms.

**Consequence:** the raw arm's epoch-85 weights were never written to disk, so
the as-run comparison cannot be repaired. The bias-corrected numbers above are an
estimate of what the result would have been, not a re-measurement.

## Reproduce

```bash
tmux new -s ab -d 'bash v2/scripts/run_ab_fold0.sh'
tmux attach -t ab          # watch
tail -f v2/runs/ab_fold0.log
```
