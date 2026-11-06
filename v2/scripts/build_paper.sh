#!/usr/bin/env bash
# Build the technical report PDF.
#
# Uses tectonic rather than a TeX Live install: it is a single self-contained
# binary that fetches only the packages the document actually needs, so the
# build works on a machine with no LaTeX toolchain and no sudo.
#
#   bash v2/scripts/build_paper.sh
#
# The figures are copied from v2/results/figures/ rather than referenced
# directly, so the paper directory is self-contained and can be uploaded to
# Overleaf as-is.
set -eu

cd "$(dirname "$0")/../.." || exit 1
PAPER=v2/results/report/paper
FIGS=v2/results/figures

mkdir -p "$PAPER/figures"
cp "$FIGS"/*.png "$PAPER/figures/"

# Prefer a tectonic on PATH; fall back to the copy this repo downloaded.
TECTONIC=$(command -v tectonic || echo /tmp/tectonic)
if [ ! -x "$TECTONIC" ]; then
  echo "tectonic not found. Install it with:"
  echo "  curl -sL -o /tmp/tectonic.tar.gz \\"
  echo "    https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz"
  echo "  tar xzf /tmp/tectonic.tar.gz -C /tmp"
  exit 1
fi

cd "$PAPER"
"$TECTONIC" -X compile paper.tex

# The competition requires 4 pages. Check it rather than assuming.
cd - >/dev/null
.venv/bin/python - <<'EOF'
from pypdf import PdfReader

r = PdfReader("v2/results/report/paper/paper.pdf")
n = len(r.pages)
print(f"\npaper.pdf: {n} pages")
if n > 4:
    raise SystemExit(f"OVER THE 4-PAGE LIMIT ({n} pages)")
print("within the 4-page limit")
EOF
