#!/usr/bin/env bash
# Build the technical report PDF in the competition's official template.
#
# The official template is `acmart` with the `sigconf` option, distributed as
# kaggle_comp_report_fi_segm_2026.zip. It lives in
# v2/results/report/paper_official/ with the fixed text preserved verbatim:
#
#   - the subtitle "A Solution to the Solar Filament Segmentation Challenge 2026"
#   - the abstract's opening sentence
#   - the introduction's opening sentence
#   - all three Acknowledgment paragraphs
#   - \bibliographystyle{ACM-Reference-Format} and \bibliography{main}
#
# `\guideStylefalse` is set, so no gray guidance text appears in the PDF.
#
#   bash v2/scripts/build_paper.sh
#
# Uses tectonic rather than a TeX Live install: a single self-contained binary
# that fetches only the packages the document needs, so the build works without
# sudo and without a full LaTeX distribution.
set -eu

cd "$(dirname "$0")/../.." || exit 1
PAPER=v2/results/report/paper_official
FIGS=v2/results/figures

mkdir -p "$PAPER/figures"
cp "$FIGS"/*.png "$PAPER/figures/"

TECTONIC=$(command -v tectonic || echo /tmp/tectonic)
if [ ! -x "$TECTONIC" ]; then
  echo "tectonic not found. Install it with:"
  echo "  curl -sL -o /tmp/tectonic.tar.gz \\"
  echo "    https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz"
  echo "  tar xzf /tmp/tectonic.tar.gz -C /tmp"
  exit 1
fi

cd "$PAPER"
"$TECTONIC" -X compile main.tex

# The competition allows 4 pages EXCLUDING the Acknowledgment and References.
# Check the body length rather than assuming it.
cd - >/dev/null
.venv/bin/python - <<'EOF'
from pypdf import PdfReader

r = PdfReader("v2/results/report/paper_official/main.pdf")
n = len(r.pages)

# Find the page where the Acknowledgment starts; everything from there on is
# excluded from the limit.
body_pages = n
for i, page in enumerate(r.pages, 1):
    text = page.extract_text() or ""
    if "ACKNOWLEDGMENT" in text.upper():
        body_pages = i
        break

print(f"\nmain.pdf: {n} pages total, body ends on page {body_pages}")
if body_pages > 4:
    raise SystemExit(
        f"BODY OVER THE 4-PAGE LIMIT ({body_pages} pages before Acknowledgment)"
    )
print("body within the 4-page limit (Acknowledgment and References excluded)")
EOF
