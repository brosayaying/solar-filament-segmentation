# Phase 4 — Ensemble diversity (PVTv2-B2)

**Status:** training complete 2026-11-04. Ensemble evaluation **paused** — the
GPU was needed for other work. Resume instructions at the end.

## Results so far

All five folds trained and early-stopped:

| fold | best PQ | @ epoch | stopped at |
|---|---|---|---|
| 0 | 0.3906 | 133 | 160 (limit) |
| 1 | 0.3906 | 85 | 91 |
| 2 | 0.3828 | 64 | 94 |
| 3 | 0.3827 | 70 | 70 |
| 4 | 0.3845 | 88 | 93 |

**PVTv2 solo is 0.3906 against ConvNeXt's 0.4177 — worse by 0.027.** That was
the expected outcome and is not the failure condition. PVTv2 is a diversity
member; it earns its place only if averaging it with ConvNeXt beats ConvNeXt
alone.

### Preliminary ensemble measurement (197 of 707 files)

The OOF dump was interrupted partway through fold 1, so this covers fold 0 in
full plus 55 files of fold 1. It is enough to see the shape of the answer, not
enough to conclude.

| variant | PQ | SQ | RQ |
|---|---|---|---|
| ConvNeXt (Phase 1) | 0.4162 | 0.6746 | 0.6170 |
| PVTv2 (Phase 4) | 0.3879 | 0.6678 | 0.5809 |
| ensemble (avg) | 0.4098 | 0.6719 | 0.6099 |

Paired bootstrap over 197 files, 2000 resamples:

```
ensemble vs ConvNeXt   -0.0113   CI [-0.0239, +0.0001]   P(>0) = 0.026
PVTv2    vs ConvNeXt   -0.0316   CI [-0.0475, -0.0159]   P(>0) = 0.000
```

The ensemble is **worse** than ConvNeXt alone, and the CI nearly excludes zero
on the negative side. The mechanism is visible in the correlation:

> **mean ConvNeXt/PVTv2 prediction correlation = 0.9132**

That is the whole story. The hypothesis was that a transformer would fail
differently from a CNN on thin low-contrast structures. It does not — at the
pixel level the two agree 91% of the time. Averaging two predictors that agree
that closely cannot reduce bias, and PVTv2's own bias is worse, so the average
lands between them and below the better one.

This is the same lesson as Phase 2's CNN scorer, which also lost to a GBDT on
scalars: architectural difference is not the same thing as error diversity.

**Caveat:** 197 files is 28% of the set and the fold-1 portion is a
non-random subset (the first 55 stems alphabetically, which are chronologically
skewed). The number needs to be re-measured on all 707 before it is quoted. But
a correlation of 0.91 is not a sampling artefact, and it is the quantity that
predicts the outcome.

## Resuming

```bash
# 1. finish the OOF dumps (skips files already written)
bash v2/scripts/dump_phase4_oof.sh 1 2 3 4

# 2. re-measure on the full set
python v2/scripts/eval_phase4.py
```

State at pause: fold 0 complete (142/142), fold 1 at 114/142, folds 2–4 not
started. `dump_fold` skips any `.npz` that already exists, so re-running
continues rather than restarting.

## Why this phase exists

Phase 2's five-fold ConvNeXt-Small ensemble bought **+0.0186** pooled OOF
(0.4220 → 0.4406). That is real, but it is small for five models, and the reason
is visible in the correlation: all five folds share an architecture, a
preprocessing pipeline, an augmentation policy and a loss. They make the *same*
mistakes, so averaging them mostly reduces variance, not bias.

An ensemble earns more when its members fail differently. PVTv2-B2 is a
pyramid vision transformer: self-attention gives every output position a path to
every input position at every stage, so a filament is seen as a whole shape
rather than through a stack of local receptive fields. On thin, elongated,
low-contrast structures — exactly this dataset — that is a genuinely different
inductive bias, not a re-parameterisation of the same one.

The hypothesis is therefore narrow and falsifiable:

> Adding PVTv2-B2 members to the ConvNeXt ensemble raises pooled OOF PQ by at
> least **+0.01** over the best single member, with a paired bootstrap CI that
> excludes zero.

If it does not, the correct action is to ship the single best model. An
ensemble that does not help is just inference cost.

## What was built

| File | Purpose |
|---|---|
| `v2/configs/phase4_pvt_fold{0..4}.json` | Five fold configs, `pvt_v2_b2` |
| `v2/scripts/run_phase4.sh` | Resumable sequential runner |
| `v2/scripts/phase4_status.py` | Cross-fold progress and ETA |
| `src/solar/train.py` | `--resume` support (see below) |

### Architecture notes

PVTv2-B2 exposes a four-level pyramid with channels `[64, 128, 320, 512]` and
reductions `[4, 8, 16, 32]`. The finest level has **stride 4**, not stride 2, so
`TimmUNet` needs `stem_ch=32` to bridge the gap between the input resolution and
the first feature map. Without it the decoder's skip connections are misaligned
by a factor of two and the model trains to a much lower ceiling. Verified
directly:

```
feature channels: [64, 128, 320, 512]
reductions:       [4, 8, 16, 32]
shapes: [(1,64,128,128), (1,128,64,64), (1,320,32,32), (1,512,16,16)]
```

27.0M parameters, `decoder_dim=128`.

### Training settings

Identical to Phase 1 fold 0 except for the backbone, so that any difference in
PQ is attributable to the architecture rather than to a tuning change:

| Setting | Value |
|---|---|
| epochs | 160 |
| batch size | 12 |
| lr | 3e-4, cosine, 3 warmup epochs |
| amp | bf16 (never fp16 — see below) |
| patch size | 512, 2 samples/image |
| val | every epoch, all 224 val images |
| early stop | patience 30, min_delta 0.002 |

**bf16, not fp16.** fp16 diverged to NaN on ConvNeXt-Small in an earlier run.
bf16 has fp32's exponent range, so it needs no gradient scaler and cannot
underflow the same way. The scaler is therefore disabled and `scaler` state is
absent from the checkpoint — the resume code guards for that.

## Resumability

The user needs the GPU for other work, so the run must survive being killed at
an arbitrary epoch and continue rather than restart.

`last.pt` is written **every epoch** and holds the full training state:

```
['bad_evals', 'best_epoch', 'best_pq', 'cfg', 'epoch',
 'gate_pq', 'model', 'opt', 'scaler', 'sched']
```

Restoring only the weights would not be a continuation. AdamW would restart with
zero second moments — a large effective step on every parameter — and the cosine
schedule would jump back to peak LR. That exact failure mode is already in this
project's history: a warm start at lr=1e-4 went from PQ 0.352 to **0.0000** in
six epochs. So the optimizer moments, the scheduler position, the epoch counter
and the early-stopping state are all restored.

### The bug this caught

The first version of the resume block was placed *before* the AMP scaler was
constructed, and referenced it:

```
UnboundLocalError: cannot access local variable 'scaler' where it is not
associated with a value
```

The block has to run after `model`, `opt`, `sched` and `scaler` all exist,
because it restores into all four. It was moved to immediately after the scaler
is created.

### Verification

A 2-epoch run was extended to 4 epochs and resumed:

```
resumed v2/runs/phase4_resume_test/last.pt: epoch 2 -> 3,
        best 0.1065 @ ep 2, gate 0.1065, bad_evals 0
done. best PQ=0.2879 @ epoch 4
```

It resumed at epoch **3**, not 1, and carried `best_pq` forward. The log shows
four distinct epochs with no repeats:

```
ep 1: loss 0.7081 PQ 0.0271 lr 1.51e-04
ep 2: loss 0.5601 PQ 0.1065 lr 3.00e-04
ep 3: loss 0.4674 PQ 0.2642 lr 1.50e-04
ep 4: loss 0.4143 PQ 0.2879 lr 0.00e+00
```

The LR column is the useful part: epoch 3 resumes at 1.50e-04, which is where
the cosine schedule had reached, not back at the 3.00e-04 peak. The scheduler
state was genuinely restored.

### How to resume

```bash
# after a kill, or after the GPU is free again
bash v2/scripts/run_phase4.sh          # all folds
bash v2/scripts/run_phase4.sh 2 3      # specific folds
```

The runner skips any fold whose `last.pt` records `early_stopped` or
`epoch >= epochs`, and resumes the rest. It checks for `best.pt` after each fold
rather than trusting `$?`, because `tee` masks the exit code.

## Cost

Measured at **147 s/epoch** on fold 0, of which roughly 30 s is training and
115 s is validation. Validation is CPU-bound: 224 full 2048² images at batch
size 1, each going through connected-component extraction and PQ matching.

This is not a regression. Phase 3 ran at 710 s/epoch on 3.1× the training data
with the same validation cost, so the ratio is as expected.

At 147 s/epoch, one fold to 160 epochs is ~6.5 h and five folds are ~32 h. Early
stopping at patience 30 will cut this substantially — Phase 3 stopped at epoch
58 of 160.

**Validation is deliberately left at full resolution and full val set.** Scoring
a subsample would make Phase 4's PQ incomparable to Phase 1's 0.4177, and the
whole point of this phase is a like-for-like comparison.

## Acceptance criteria

Fixed before the run, so the result cannot be rationalised afterwards:

1. **Primary:** ensemble of 5 ConvNeXt + 5 PVTv2 folds ≥ best single member
   **+ 0.01** pooled OOF PQ, with a paired bootstrap CI excluding zero.
2. **Secondary:** PVTv2 members are not redundant — mean pairwise prediction
   correlation with the ConvNeXt members is materially below the ConvNeXt-to-
   ConvNeXt correlation.
3. **Tertiary:** per-year PQ does not regress in any year by more than 0.01.

If (1) fails, ship the single best model and record the negative result. Phase 3
already established that this project reports failures rather than burying them.

## Monitoring

```bash
python v2/scripts/phase4_status.py     # all folds, best PQ, ETA
tmux attach -t p4                      # live
tail -f v2/runs/phase4/run.log         # runner-level
```

## Risks

| Risk | Mitigation |
|---|---|
| PVTv2 underperforms ConvNeXt | Expected is possible; it is a diversity member, not a replacement. Judged on ensemble gain, not solo PQ. |
| 32 h wall clock | Resumable; early stopping should cut it. Can run folds 0–2 only and still test the hypothesis. |
| GPU needed elsewhere | Kill the tmux session; `last.pt` is written every epoch, so at most one epoch is lost. |
| Disk | 248 GB free at launch. Each fold writes ~430 MB (best + last). |
