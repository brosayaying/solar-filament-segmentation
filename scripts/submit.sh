#!/usr/bin/env bash
# Submit a CSV to the competition via the Kaggle API.
#
#   ./scripts/submit.sh submissions/baseline.csv
#
# Kaggle's submit endpoint requires a two-step flow: request a resumable upload
# URL, PUT the file (with an explicit Content-Length, otherwise the recorded size
# is 0), then submit referencing the returned blob token.
set -euo pipefail

cd "$(dirname "$0")/.."
FILE="${1:?usage: submit.sh <csv>}"
COMP="${COMP:-filament-segmentation-2026}"

TOKEN=$(grep '^KAGGLE_API_KEY=' .env | cut -d= -f2- | tr -d '"'"'"' \r')
SIZE=$(stat -c '%s' "$FILE")

echo "file: $FILE ($SIZE bytes)"

# 1. Request a resumable upload slot.
RESP=$(curl -s -X POST \
  "https://www.kaggle.com/api/v1/competitions/submissions/upload/$COMP" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"fileName\": \"$(basename "$FILE")\", \"contentLength\": $SIZE}")

BLOB=$(echo "$RESP" | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin)["token"])')
URL=$(echo "$RESP" | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin)["createUrl"])')

# 2. Upload the bytes.
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X PUT "$URL" \
  -H 'Content-Type: text/csv' \
  -H "Content-Length: $SIZE" \
  --data-binary "@$FILE")
echo "upload: HTTP $CODE"
[ "$CODE" = "200" ] || { echo "upload failed"; exit 1; }

# 3. Submit.
curl -s -X POST \
  "https://www.kaggle.com/api/v1/competitions/submissions/submit/$COMP" \
  -H "Authorization: Bearer $TOKEN" \
  -F "blobFileTokens=$BLOB" \
  -w '\nHTTP:%{http_code}\n'
