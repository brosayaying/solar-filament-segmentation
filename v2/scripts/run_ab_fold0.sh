#!/usr/bin/env bash
# Fold-0 A/B: radial flattening vs raw, everything else identical.
#
# Runs the two arms sequentially on one GPU. Each arm is 160 epochs on fold 0
# (930 train ids / 142 val files) at 512^2, ~51 s/epoch => ~2.3 h per arm.
#
#   tmux new -s ab -d 'bash v2/scripts/run_ab_fold0.sh'
#   tmux attach -t ab
#
# Survives an ssh disconnect because tmux owns the process, not the shell.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=v2/runs/ab_fold0.log

mkdir -p v2/runs
: > "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== fold-0 A/B start ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)"

for arm in raw flatten; do
    cfg="v2/configs/ab_fold0_${arm}.json"
    run="ab_fold0_${arm}"

    if [ -f "v2/runs/${run}/last.pt" ]; then
        log "SKIP ${arm}: v2/runs/${run}/last.pt already exists"
        continue
    fi

    log "--- ${arm}: training 160 epochs on fold 0 ---"
    $PY -m solar.train --config "$cfg" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "FAILED ${arm} (exit $rc) -- see $LOG"
        exit $rc
    fi
    log "--- ${arm}: done, best PQ $(grep -o 'best PQ=[0-9.]*' "$LOG" | tail -1) ---"
done

log "=== both arms trained; scoring ==="
$PY v2/scripts/score_ab_fold0.py >> "$LOG" 2>&1
log "=== A/B complete ==="
