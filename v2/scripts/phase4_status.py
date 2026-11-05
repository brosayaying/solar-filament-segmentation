"""Print Phase 4 progress across all five PVTv2 folds.

    python v2/scripts/phase4_status.py

Reads ``v2/runs/phase4_pvt_fold*/log.jsonl``, which is the authoritative record:
one JSON object per completed epoch, written after validation. The stdout logs
under ``v2/runs/phase4/`` are only a fallback, because tqdm's carriage returns
can clobber the epoch line.

Two things this deliberately does *not* do:

* It does not assume one row per epoch. A killed smoke test can leave a stale
  epoch-1 row behind, so the curve is de-duplicated by taking the last row for
  each epoch number.
* It does not compare against Phase 1's 0.4177 as if it were the target. PVTv2
  is a *diversity* member, not a replacement: a fold that lands below ConvNeXt
  can still earn its place in the ensemble. The number that matters is the
  ensemble's, which is measured in ``eval_phase4.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "v2/runs"
N_FOLDS = 5

# Phase 1 fold 0 (ConvNeXt-Small), for orientation only.
PHASE1_BEST = 0.4177


def load_curve(run: Path) -> list[dict]:
    """Return one row per epoch, last write wins."""
    path = run / "log.jsonl"
    if not path.exists():
        return []
    by_epoch: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_epoch[int(d["epoch"])] = d
    return [by_epoch[e] for e in sorted(by_epoch)]


def fold_state(run: Path) -> str:
    """Describe a fold from its checkpoint, which is the ground truth."""
    last = run / "last.pt"
    if not last.exists():
        return "not started"
    try:
        import torch

        ck = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a corrupt ckpt is a real state
        return f"unreadable last.pt ({exc.__class__.__name__})"
    ep = int(ck.get("epoch", 0))
    epochs = int(ck.get("cfg", {}).get("train", {}).get("epochs", 0))
    if ck.get("early_stopped"):
        return f"early stopped at {ep}/{epochs}"
    if epochs and ep >= epochs:
        return f"finished {ep}/{epochs}"
    return f"in progress {ep}/{epochs}"


def main() -> None:
    print(f"{'fold':<6}{'state':<26}{'best PQ':>9}{'@ep':>6}{'last':>8}{'s/ep':>8}")
    print("-" * 63)

    bests: list[float] = []
    for f in range(N_FOLDS):
        run = RUNS / f"phase4_pvt_fold{f}"
        curve = load_curve(run)
        state = fold_state(run)
        if not curve:
            print(f"{f:<6}{state:<26}{'-':>9}{'-':>6}{'-':>8}{'-':>8}")
            continue
        best = max(curve, key=lambda r: r["val_pq"])
        last = curve[-1]
        secs = [r.get("epoch_time_s") for r in curve if r.get("epoch_time_s")]
        spd = f"{sum(secs) / len(secs):.0f}" if secs else "-"
        print(
            f"{f:<6}{state:<26}{best['val_pq']:>9.4f}{best['epoch']:>6}"
            f"{last['val_pq']:>8.4f}{spd:>8}"
        )
        bests.append(best["val_pq"])

    if not bests:
        print("\nno folds have produced an epoch yet")
        return

    print("-" * 63)
    print(f"best single fold: {max(bests):.4f}   (Phase 1 fold 0: {PHASE1_BEST:.4f})")

    # Rough ETA from the folds that are actually running.
    for f in range(N_FOLDS):
        run = RUNS / f"phase4_pvt_fold{f}"
        curve = load_curve(run)
        if not curve:
            continue
        secs = [r.get("epoch_time_s") for r in curve if r.get("epoch_time_s")]
        if not secs:
            continue
        avg = sum(secs) / len(secs)
        done = len(curve)
        total = 160
        remaining = (total - done) * avg
        print(
            f"fold {f}: {done}/{total} epochs, {avg:.0f}s/epoch, "
            f"~{remaining / 3600:.1f}h remaining if it runs to {total}"
        )


if __name__ == "__main__":
    main()
