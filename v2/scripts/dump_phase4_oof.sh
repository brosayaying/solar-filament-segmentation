#!/usr/bin/env bash
# Dump Phase 4 (PVTv2-B2) out-of-fold probabilities, one fold at a time.
#
# Why this exists: Phase 4's acceptance criterion is an *ensemble* gain, and the
# only honest way to measure that is pooled OOF -- every file scored by the fold
# that did not train on it. The checkpoints alone cannot answer the question.
#
# `--split` must be the PARENT split file (`week5_seed42.json`), not the
# per-fold file. `solar.oof` calls `expand_folds`, which reads `folds["folds"]`;
# the per-fold files only have `train`/`val` keys and raise `KeyError: 'folds'`.
#
# Resumable: `dump_fold` skips any file whose `.npz` already exists, so
# re-running this after an interruption continues rather than restarting.
#
# Usage:
#   bash v2/scripts/dump_phase4_oof.sh          # all folds
#   bash v2/scripts/dump_phase4_oof.sh 1 2      # specific folds
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
OUT=v2/runs/phase4/oof
mkdir -p "$OUT"

FOLDS=("$@")
if [ ${#FOLDS[@]} -eq 0 ]; then
  FOLDS=(0 1 2 3 4)
fi

for f in "${FOLDS[@]}"; do
  echo "=== fold $f ==="
  $PY -u -m solar.oof \
      --ckpt "v2/runs/phase4_pvt_fold${f}/best.pt" \
      --fold "$f" \
      --split v2/splits/week5_seed42.json \
      --out "$OUT" \
      --num-workers 6 2>&1 | tail -4

  # tee/tail mask the exit code; check the artefact instead.
  n=$(ls "$OUT/fold$f" 2>/dev/null | wc -l)
  if [ "$n" -lt 140 ]; then
    echo "fold $f: FAILED (only $n files). Stopping so the failure is visible."
    exit 1
  fi
  echo "fold $f: $n files"
done

echo "all requested folds dumped"
