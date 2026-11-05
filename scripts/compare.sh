#!/usr/bin/env bash
# Print the best validation PQ for every run under runs/.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python - <<'PY'
import json
from pathlib import Path

rows = []
for log in sorted(Path("runs").glob("*/log.jsonl")):
    best = None
    for line in log.read_text().splitlines():
        d = json.loads(line)
        if "val_pq" in d and (best is None or d["val_pq"] > best["val_pq"]):
            best = d
    if best:
        rows.append((best["val_pq"], log.parent.name, best["epoch"],
                     best.get("val_sq", 0), best.get("val_rq", 0)))

rows.sort(reverse=True)
print(f"{'run':<28} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'ep':>4}")
print("-" * 58)
for pq, name, ep, sq, rq in rows:
    print(f"{name:<28} {pq:>7.4f} {sq:>7.4f} {rq:>7.4f} {ep:>4}")
PY
