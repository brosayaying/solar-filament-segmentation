#!/usr/bin/env bash
# Sequential backbone sweep. Each run is short so ideas can be compared quickly.
#
#   ./scripts/sweep.sh resnet18 convnext_tiny efficientnet_b0 mit_b0
#
# Results land in runs/<backbone>_sweep/log.jsonl; compare with:
#   ./scripts/compare.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
EPOCHS="${EPOCHS:-6}"
PATCH="${PATCH:-384}"

FAILED=()
for bb in "$@"; do
  echo "=== $bb ==="
  # No `|| true`: a bad backbone name must fail the sweep loudly rather than
  # silently producing no run directory.
  if ! $PY -m solar.train \
      --config configs/quick.json \
      --set "model.backbone=$bb" \
            "train.epochs=$EPOCHS" \
            "data.patch_size=$PATCH" \
            "train.run_name=sweep_$bb" \
      2>&1 | grep -E '^(epoch|model|done|.*Error)' ; then
    FAILED+=("$bb")
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED backbones: ${FAILED[*]}"
  exit 1
fi
