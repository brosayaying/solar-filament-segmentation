#!/usr/bin/env bash
# Phase 1: 5-fold ConvNeXt-Small segmentation, radial flattening, bf16.
#
# Trains folds 1-4 and reuses the fold-0 checkpoint from the A/B, which was
# trained on exactly this config (fold0.json, flatten, bf16, stem 32) and whose
# best.pt holds the true maximum (epoch 110, PQ 0.4177) -- the early-stopping bug
# only affected the raw arm.
#
#   tmux new -s seg -d 'bash v2/scripts/run_seg_v2.sh'
#   tmux attach -t seg
#
# Resumable: a fold whose last.pt exists is skipped, so re-running after an ssh
# disconnect or a crash continues where it stopped.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=v2/runs/seg_v2.log
mkdir -p v2/runs
: > "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== Phase 1: 5-fold seg_v2 (flatten, bf16) ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)"

# Fold 0 already exists from the A/B. Move it into place rather than retraining.
if [ -d v2/runs/ab_fold0_flatten ] && [ ! -d v2/runs/seg_v2_fold0 ]; then
    log "reusing A/B fold-0 checkpoint -> v2/runs/seg_v2_fold0"
    cp -r v2/runs/ab_fold0_flatten v2/runs/seg_v2_fold0
fi

for fold in 0 1 2 3 4; do
    run="seg_v2_fold${fold}"
    split="v2/splits/week5_seed42/fold${fold}.json"

    if [ -f "v2/runs/${run}/last.pt" ]; then
        log "SKIP fold ${fold}: v2/runs/${run}/last.pt exists"
        continue
    fi

    log "--- fold ${fold}: training (split ${split}) ---"
    $PY -m solar.train --config v2/configs/seg_v2.json \
        --set "data.split_file=${split}" "train.run_name=${run}" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "FAILED fold ${fold} (exit $rc) -- see $LOG"
        exit $rc
    fi
    log "--- fold ${fold}: done, $(grep -o 'best PQ=[0-9.]* @ epoch [0-9]*' "$LOG" | tail -1) ---"
done

log "=== all folds trained; dumping OOF ==="
for fold in 0 1 2 3 4; do
    $PY -m solar.oof --ckpt "v2/runs/seg_v2_fold${fold}/best.pt" --fold "$fold" \
        --out v2/runs/seg_v2/oof >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "FAILED oof fold ${fold} (exit $rc)"
        exit $rc
    fi
done

log "=== verifying OOF ==="
$PY -m solar.oof --verify v2/runs/seg_v2/oof >> "$LOG" 2>&1
log "=== Phase 1 complete ==="
