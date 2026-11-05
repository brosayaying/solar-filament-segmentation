"""Where are filaments on the solar disk? Density heatmap + disk masking.

Two outputs:

1. ``heatmap.png`` -- filament pixel density in *solar-disk coordinates*.
   Every image is resampled so the disk centre maps to the origin and the disk
   radius maps to 1.0, then the masks are accumulated. Without this
   normalisation the heatmap would just show where the disk happens to sit in
   the frame, since the disk centre wanders by up to ~100 px between
   observations and the radius varies from ~927 to ~1033 px.

2. ``disk_mask.png`` -- the disk/limb/off-disk partition used by the classical
   pipeline, so the masking is visible rather than assumed.

    python scripts/heatmap.py --n all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar.data import anns_to_mask, build_ann_index, load_coco  # noqa: E402

ROOT = Path("MAGFiLO_1.0_Kaggle_2026")
OUT = Path("docs/HEATMAP")
cv2.setNumThreads(4)

# Normalised output grid: 2.2 disk radii across, so the limb and a little
# off-disk margin are both visible.
GRID = 512
EXTENT = 2.2


def disk_geometry(img: np.ndarray, thr: int = 25) -> tuple[float, float, float]:
    """Return ``(cy, cx, r)`` of the solar disk: largest bright blob."""
    m = (img > thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        h, w = img.shape
        return h / 2.0, w / 2.0, min(h, w) / 2.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cy, cx = cent[k]
    r = float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi))
    return float(cy), float(cx), r


def to_disk_coords(mask: np.ndarray, cy: float, cx: float, r: float) -> np.ndarray:
    """Resample a full-frame mask into the normalised disk frame.

    Output pixel ``(u, v)`` corresponds to normalised disk coordinates
    ``((u - GRID/2) * EXTENT/GRID, (v - GRID/2) * EXTENT/GRID)``, so the disk
    centre lands at the grid centre and the limb at radius ``GRID/EXTENT``.
    """
    scale = r * EXTENT / GRID  # image px per output px
    # Maps output (dst) coordinates -> input (src) coordinates, hence
    # WARP_INVERSE_MAP below.
    M = np.array(
        [
            [scale, 0.0, cx - (GRID / 2.0) * scale],
            [0.0, scale, cy - (GRID / 2.0) * scale],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        mask.astype(np.float32),
        M,
        (GRID, GRID),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderValue=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="all")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coco = load_coco(ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {i["id"]: i for i in coco["images"]}

    files = sorted({img["file_name"] for img in coco["images"]})
    if args.n != "all":
        files = files[: int(args.n)]

    by_fn: dict[str, list[str]] = {}
    for iid, meta in images.items():
        by_fn.setdefault(meta["file_name"], []).append(iid)

    # Accumulators in the normalised disk frame.
    fg = np.zeros((GRID, GRID), np.float64)  # filament pixel count
    disk = np.zeros((GRID, GRID), np.float64)  # disk pixel count (denominator)
    n_inst = np.zeros((GRID, GRID), np.float64)  # filament *instance* count

    radii: list[float] = []
    centres: list[tuple[float, float]] = []
    example: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for k, fn in enumerate(files):
        img = cv2.imread(str(ROOT / "train/train_images" / fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(fn)
        h, w = img.shape

        mask = np.zeros((h, w), np.uint8)
        for iid in by_fn[fn]:
            mask |= anns_to_mask(ann_index.get(iid, []), h, w)

        cy, cx, r = disk_geometry(img)
        radii.append(r)
        centres.append((cy, cx))

        fg += to_disk_coords(mask, cy, cx, r)
        disk += to_disk_coords((img > 25).astype(np.uint8), cy, cx, r)

        # Instance density: count each filament once at its centroid, so a few
        # huge filaments do not dominate the way they do in the pixel sum.
        for iid in by_fn[fn]:
            for ann in ann_index.get(iid, []):
                bx, by, bw, bh = ann["bbox"]
                pt = np.zeros((h, w), np.uint8)
                cv2.circle(pt, (int(bx + bw / 2), int(by + bh / 2)), 1, 1, -1)
                n_inst += to_disk_coords(pt, cy, cx, r)

        if example is None and mask.any():
            example = (img.copy(), mask.copy(), np.array([cy, cx, r]))

        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(files)}", flush=True)

    # Density = filament pixels / disk pixels, per normalised location.
    density = np.divide(fg, disk, out=np.zeros_like(fg), where=disk > 0)

    # ---- heatmap ---------------------------------------------------------- #
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(density, cmap="inferno", extent=[-EXTENT / 2, EXTENT / 2] * 2)
    axes[0].set_title("Filament pixel density\n(filament px / disk px)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(n_inst, cmap="viridis", extent=[-EXTENT / 2, EXTENT / 2] * 2)
    axes[1].set_title(f"Filament instance centroids\n(n={int(n_inst.sum())})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # Radial profile: density vs normalised disk radius.
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    rr = np.hypot(yy - GRID / 2, xx - GRID / 2) / (GRID / EXTENT)
    bins = np.linspace(0, 1.1, 45)
    prof = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (rr >= lo) & (rr < hi) & (disk > 0)
        prof.append(density[sel].mean() if sel.any() else np.nan)
    axes[2].plot(0.5 * (bins[:-1] + bins[1:]), prof, "o-")
    axes[2].set_xlabel("normalised disk radius (1.0 = limb)")
    axes[2].set_ylabel("mean filament density")
    axes[2].set_title("Radial distribution")
    axes[2].grid(alpha=0.3)
    axes[2].axvline(1.0, color="r", ls="--", lw=1)

    for ax in axes[:2]:
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="cyan", lw=1.2, ls="--"))
        ax.set_xlabel("x / R_sun")
        ax.set_ylabel("y / R_sun")

    fig.tight_layout()
    fig.savefig(out / "heatmap.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out / 'heatmap.png'}")

    # ---- disk masking illustration ---------------------------------------- #
    if example is not None:
        img, mask, (cy, cx, r) = example
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        rad = np.hypot(yy - cy, xx - cx)

        inner = rad < (r - 40)
        limb = (rad >= (r - 40)) & (rad < r)
        offdisk = rad >= r

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[limb] = (0.45 * vis[limb] + 0.55 * np.array([0, 165, 255])).astype(np.uint8)
        vis[offdisk] = (0.25 * vis[offdisk]).astype(np.uint8)
        vis[mask > 0] = (0, 0, 255)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("raw H-alpha")
        axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title("masked: red=filament, orange=limb ring, dim=off-disk")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "disk_mask.png", dpi=130)
        plt.close(fig)
        print(f"wrote {out / 'disk_mask.png'}")

    # ---- numbers ---------------------------------------------------------- #
    stats = {
        "n_files": len(files),
        "disk_radius_px": {
            "median": float(np.median(radii)),
            "p05": float(np.percentile(radii, 5)),
            "p95": float(np.percentile(radii, 95)),
        },
        "disk_centre_px": {
            "y_median": float(np.median([c[0] for c in centres])),
            "y_p05": float(np.percentile([c[0] for c in centres], 5)),
            "y_p95": float(np.percentile([c[0] for c in centres], 95)),
            "x_median": float(np.median([c[1] for c in centres])),
            "x_p05": float(np.percentile([c[1] for c in centres], 5)),
            "x_p95": float(np.percentile([c[1] for c in centres], 95)),
        },
        "filament_density_by_radius": {
            f"{0.5 * (lo + hi):.2f}": (None if np.isnan(v) else float(v))
            for (lo, hi), v in zip(zip(bins[:-1], bins[1:]), prof)
        },
        "frac_filament_px_inside_limb": float(
            fg[rr < 1.0].sum() / max(1e-9, fg.sum())
        ),
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=2)

    print(f"\ndisk radius px: median={stats['disk_radius_px']['median']:.1f} "
          f"[{stats['disk_radius_px']['p05']:.1f}, {stats['disk_radius_px']['p95']:.1f}]")
    print(f"disk centre px: y={stats['disk_centre_px']['y_median']:.1f} "
          f"[{stats['disk_centre_px']['y_p05']:.1f}, {stats['disk_centre_px']['y_p95']:.1f}]  "
          f"x={stats['disk_centre_px']['x_median']:.1f} "
          f"[{stats['disk_centre_px']['x_p05']:.1f}, {stats['disk_centre_px']['x_p95']:.1f}]")
    print(f"filament pixels inside limb: {100 * stats['frac_filament_px_inside_limb']:.2f}%")
    print(f"wrote {out / 'stats.json'}")


if __name__ == "__main__":
    main()
