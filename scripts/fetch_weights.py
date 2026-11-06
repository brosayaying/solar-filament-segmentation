"""Download the trained weights needed to reproduce the submission.

Why a script instead of committing the weights
----------------------------------------------
Each fold checkpoint is 198 MB. GitHub rejects any file over 100 MB, and Git LFS
has a 1 GB free quota that five checkpoints would exhaust. So the weights live as
release assets and this script fetches them.

What gets downloaded
--------------------
    seg_v2_fold{0..4}/best.pt   five ConvNeXt-Small UNet folds, 198 MB each
    gbdt_qhat.joblib            Phase 2 candidate scorer, 915 KB

Total ~993 MB. The five folds are needed because the submitted predictions
average all five probability maps; the GBDT is needed because the submission
re-ranks candidates with it.

Usage
-----
    python scripts/fetch_weights.py                  # into ./weights
    python scripts/fetch_weights.py --out /tmp/w     # elsewhere
    python scripts/fetch_weights.py --check          # verify only, no download

The script is idempotent: files already present with the right size are skipped,
so an interrupted download resumes cheaply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "brosayaying/solar-filament-segmentation"
TAG = "weights-v1"

# name -> (release asset name, expected size in bytes)
#
# Sizes are recorded so a truncated download is caught rather than silently
# producing a checkpoint that fails to load with a confusing error.
ASSETS: dict[str, tuple[str, int]] = {
    "seg_v2_fold0/best.pt": ("seg_v2_fold0_best.pt", 208_130_377),
    "seg_v2_fold1/best.pt": ("seg_v2_fold1_best.pt", 208_130_377),
    "seg_v2_fold2/best.pt": ("seg_v2_fold2_best.pt", 208_130_377),
    "seg_v2_fold3/best.pt": ("seg_v2_fold3_best.pt", 208_130_377),
    "seg_v2_fold4/best.pt": ("seg_v2_fold4_best.pt", 208_130_377),
    "gbdt_qhat.joblib": ("gbdt_qhat.joblib", 936_785),
}


def asset_url(name: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{TAG}/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r  {dest.name}: {pct:5.1f}%  "
                      f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)", end="")
    print()
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights", help="destination directory")
    ap.add_argument("--check", action="store_true",
                    help="verify existing files, do not download")
    ap.add_argument("--force", action="store_true", help="re-download")
    args = ap.parse_args()

    out = Path(args.out)
    missing, wrong = [], []

    for rel, (asset, size) in ASSETS.items():
        dest = out / rel
        if dest.exists() and not args.force:
            have = dest.stat().st_size
            if have == size:
                print(f"ok      {rel}  ({have / 1e6:.0f} MB)")
                continue
            print(f"size    {rel}  have {have}, want {size}")
            wrong.append(rel)
        else:
            missing.append(rel)

    if args.check:
        if missing or wrong:
            print(f"\n{len(missing)} missing, {len(wrong)} wrong size")
            raise SystemExit(1)
        print("\nall weights present")
        return

    for rel in missing + wrong:
        asset, size = ASSETS[rel]
        dest = out / rel
        print(f"fetching {rel} <- {asset}")
        try:
            download(asset_url(asset), dest)
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"  HTTP {exc.code} for {asset_url(asset)}\n"
                f"  The release '{TAG}' may not exist yet. Create it with:\n"
                f"    gh release create {TAG} --repo {REPO} <files>"
            ) from exc
        got = dest.stat().st_size
        if got != size:
            raise SystemExit(f"  {rel}: got {got} bytes, expected {size}")

    # Record hashes so a reviewer can verify what they downloaded.
    manifest = {
        rel: {"sha256": sha256(out / rel), "bytes": (out / rel).stat().st_size}
        for rel in ASSETS
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out / 'manifest.json'}")
    print("all weights present")


if __name__ == "__main__":
    main()
