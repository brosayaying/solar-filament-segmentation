#!/usr/bin/env bash
# Phase 4: train a PVTv2-B2 member on each fold, resumably.
#
# Why a transformer member at all: the five ConvNeXt folds are highly correlated
# (same architecture, same data, same preprocessing), which is why averaging them
# bought only +0.01 -- inside the leaderboard's noise. An ensemble helps when its
# members make *uncorrelated* errors. PVTv2-B2 sees the whole filament through
# attention rather than a local receptive field, so it fails differently.
#
# Resumability: `train.py --resume` restores model, optimizer moments, scheduler
# position, epoch counter and early-stopping state from `last.pt`, which is
# written every epoch. Killing this script and re-running it continues where it
# left off. A fold whose `last.pt` records `early_stopped` is skipped entirely.
#
# Usage:
#   bash v2/scripts/run_phase4.sh              # all folds, resumable
#   bash v2/scripts/run_phase4.sh 0 1          # specific folds
#   tmux new -d -s p4 'bash v2/scripts/run_phase4.sh'
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOGDIR=v2/runs/phase4
mkdir -p "$LOGDIR"

FOLDS=("$@")
if [ ${#FOLDS[@]} -eq 0 ]; then
  FOLDS=(0 1 2 3 4)
fi

for f in "${FOLDS[@]}"; do
  RUN="v2/runs/phase4_pvt_fold${f}"
  LOG="${LOGDIR}/train_fold${f}.log"

  # Skip a fold that already finished. `early_stopped` is set by train.py when
  # patience runs out; a fold that reached the epoch limit has epoch == epochs.
  if [ -f "${RUN}/last.pt" ]; then
    DONE=$($PY - "$RUN/last.pt" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ck.get("cfg", {})
epochs = cfg.get("train", {}).get("epochs", 0)
ep = ck.get("epoch", 0)
print("yes" if (ck.get("early_stopped") or ep >= epochs) else "no")
EOF
)
    if [ "$DONE" = "yes" ]; then
      echo "fold $f: already complete, skipping"
      continue
    fi
    echo "fold $f: resuming from ${RUN}/last.pt"
  else
    echo "fold $f: starting fresh"
  fi

  $PY -u -m solar.train \
      --config "v2/configs/phase4_pvt_fold${f}.json" \
      --resume 2>&1 | tee -a "$LOG"

  # tee masks the exit code; check the checkpoint instead of $?.
  if [ ! -f "${RUN}/best.pt" ]; then
    echo "fold $f: FAILED (no best.pt). Stopping so the failure is visible."
    exit 1
  fi
  echo "fold $f: done"
done

echo "all requested folds complete"
