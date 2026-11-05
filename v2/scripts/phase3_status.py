"""Print the Phase 3 training curve and the comparison against Phase 1.

    python v2/scripts/phase3_status.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "v2/runs/phase3/train_fold0.log"
RUN = ROOT / "v2/runs/phase3_fold0"

# Phase 1 fold 0, for the like-for-like comparison.
PHASE1_BEST = 0.4177
PHASE1_EPOCH = 110


def main() -> None:
    if not LOG.exists():
        raise SystemExit(f"no log at {LOG}")

    # ``log.jsonl`` is the authoritative record: it is written once per epoch and
    # is never overwritten by a progress bar. The stdout log is only used as a
    # fallback, because tqdm's carriage returns can clobber the epoch line.
    rows = []
    meta = RUN / "log.jsonl"
    if meta.exists():
        for line in meta.read_text().strip().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({
                "epoch": int(d["epoch"]), "loss": float(d["train_loss"]),
                "pq": float(d["val_pq"]), "sq": float(d["val_sq"]),
                "rq": float(d["val_rq"]), "tp": int(d["val_tp"]),
                "fp": int(d["val_fp"]), "fn": int(d["val_fn"]),
                "secs": float(d.get("epoch_time_s", 0)),
            })

    if not rows:
        text = LOG.read_text(errors="ignore").replace("\r", "\n")
        for m in re.finditer(
            r"^epoch (\d+): loss=([\d.]+) PQ=([\d.]+) SQ=([\d.]+) RQ=([\d.]+) "
            r"\(tp=(\d+) fp=(\d+) fn=(\d+)\)",
            text,
            re.M,
        ):
            rows.append({
                "epoch": int(m.group(1)), "loss": float(m.group(2)),
                "pq": float(m.group(3)), "sq": float(m.group(4)),
                "rq": float(m.group(5)), "tp": int(m.group(6)),
                "fp": int(m.group(7)), "fn": int(m.group(8)), "secs": 0.0,
            })

    if not rows:
        raise SystemExit("no completed epochs yet")

    print(f"{'ep':>4} {'loss':>7} {'PQ':>7} {'SQ':>7} {'RQ':>7} "
          f"{'tp':>5} {'fp':>5} {'fn':>5}  {'vs P1':>7}")
    print("-" * 62)
    for r in rows[-15:]:
        print(f"{r['epoch']:>4} {r['loss']:>7.4f} {r['pq']:>7.4f} "
              f"{r['sq']:>7.4f} {r['rq']:>7.4f} {r['tp']:>5} {r['fp']:>5} "
              f"{r['fn']:>5}  {r['pq'] - PHASE1_BEST:>+7.4f}")

    best = max(rows, key=lambda r: r["pq"])
    print()
    print(f"epochs done      : {len(rows)}")
    print(f"best so far      : PQ {best['pq']:.4f} at epoch {best['epoch']}")
    print(f"Phase 1 fold 0   : PQ {PHASE1_BEST:.4f} at epoch {PHASE1_EPOCH}")
    print(f"delta            : {best['pq'] - PHASE1_BEST:+.4f}")

    # Early-stopping state, mirroring train.py's logic.
    gate = rows[0]["pq"]
    since = 0
    for r in rows[1:]:
        if r["pq"] > gate + 0.002:
            gate = r["pq"]
            since = 0
        else:
            since += 1
    print(f"epochs since gate: {since} / 15 (patience)")

    secs = [r["secs"] for r in rows if r["secs"] > 0]
    if secs:
        avg = sum(secs) / len(secs)
        left = max(0, 60 - len(rows))
        print(f"epoch time       : {avg:.0f} s  ->  {left} epochs left "
              f"= {avg * left / 3600:.1f} h")

    ck = RUN / "best.pt"
    if ck.exists():
        import torch

        d = torch.load(ck, map_location="cpu", weights_only=False)
        print(f"best.pt          : epoch {d.get('epoch')} pq {d.get('pq', 0):.4f}")


if __name__ == "__main__":
    main()
