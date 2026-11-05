"""GONG archive access for Phase 3 label transfer and self-training.

Two things were verified before writing this, and both change the plan:

**1. The competition images are the GONG ``hag`` JPEGs, not the ``haf`` FITS.**
Downloading ``https://gong2.nso.edu/HA/hag/YYYYMM/YYYYMMDD/<ts><station>.jpg``
for a training file and comparing against the competition JPEG gives median
absolute difference **1**, correlation **0.997** at zero shift. The ``haf``
``.fits.fz`` product is a different reduction (correlation 0.86 against the
competition image after a best-fit linear stretch), so using it would introduce a
domain shift that the JPEG route avoids entirely. **Label transfer therefore
downloads JPEGs and needs no radiometric normalisation.**

**2. Solar rotation over the offset window is ~1.6 px at 10 min, not <1 px.**
A feature at heliographic longitude ``L`` projects to ``x = R sin L``, so at disk
centre ``dx = R dL`` with ``R = 1024 px`` and ``dL = 0.00916 deg/min``:

| offset | displacement at centre | at r/R = 0.8 |
|---|---|---|
| 2 min | 0.33 px | 0.20 px |
| 5 min | 0.82 px | 0.50 px |
| 10 min | 1.64 px | 0.99 px |
| 30 min | 4.91 px | 2.96 px |

Filaments are ~11 px wide, so 10 min is safe and 30 min is marginal. The default
window is therefore **±2–10 min**, matching the plan.

    from solar.gong import frame_url, download_frame, nearby_timestamps
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

HAG = "https://gong2.nso.edu/HA/hag/{ym}/{ymd}/{ts}.jpg"

# GONG station codes. The competition images use Bh, Ch, Lh, Mh, Th, Uh.
STATIONS = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")


def parse_stem(stem: str) -> tuple[datetime, str]:
    """Split ``20140609195854Bh`` into ``(datetime, station)``."""
    if len(stem) < 16:
        raise ValueError(f"not a GONG timestamp: {stem!r}")
    ts, station = stem[:14], stem[14:]
    return datetime.strptime(ts, "%Y%m%d%H%M%S"), station


def frame_url(dt: datetime, station: str) -> str:
    """URL of the ``hag`` JPEG for a given time and station."""
    return HAG.format(
        ym=dt.strftime("%Y%m"), ymd=dt.strftime("%Y%m%d"),
        ts=dt.strftime("%Y%m%d%H%M%S") + station,
    )


def nearby_timestamps(
    stem: str, offsets_min: tuple[int, ...] = (-10, -5, -2, 2, 5, 10)
) -> list[tuple[str, int]]:
    """Same-station timestamps at the given minute offsets.

    Same station only: different stations have different plate scale, limb
    darkening and seeing, so a label transferred across stations is not the same
    label. Returns ``[(stem, offset_minutes), ...]``.
    """
    dt, station = parse_stem(stem)
    out = []
    for off in offsets_min:
        t = dt + timedelta(minutes=off)
        out.append((t.strftime("%Y%m%d%H%M%S") + station, off))
    return out


def download_frame(
    stem: str, dest_dir: Path, timeout: int = 60, retries: int = 2
) -> Path | None:
    """Download one ``hag`` JPEG. Returns the path, or ``None`` if unavailable.

    GONG returns 404 for frames that do not exist (gaps in the archive, station
    downtime), which is common enough that it must not be an error.
    """
    import urllib.error
    import urllib.request

    dt, station = parse_stem(stem)
    out = dest_dir / f"{stem}.jpg"
    if out.exists() and out.stat().st_size > 10_000:
        return out

    url = frame_url(dt, station)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            if len(data) < 10_000:
                return None
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            return out
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries:
                return None
            time.sleep(1.0 + attempt)
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.0 + attempt)
    return None


def load_frame(path: Path) -> np.ndarray | None:
    """Read a GONG JPEG as a 2048^2 uint8 grayscale array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (2048, 2048):
        return None
    return img


def disk_geometry(img: np.ndarray, thresh: int = 20) -> tuple[float, float, float]:
    """Estimate ``(cy, cx, radius)`` of the solar disk.

    The disk is the bright region; the corners are black. A centroid plus an
    area-equivalent radius is accurate to well under a pixel for a full-disk
    frame and is far cheaper than a circle fit.
    """
    mask = img > thresh
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 1024.0, 1024.0, 1024.0
    cy, cx = float(ys.mean()), float(xs.mean())
    radius = float(np.sqrt(len(ys) / np.pi))
    return cy, cx, radius
