"""Plot the DINOv2 vs SAM embedding comparison from ``metrics.json``.

    python scripts/plot_embed_compare.py --dir docs/embed_compare

Produces ``compare.png``: per-image paired bars for the four metrics, plus a
scatter of token recall against filament prevalence (the interesting failure
mode is small filaments, where coarse tokens cannot resolve the structure).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/embed_compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = Path(args.dir)
    rows = json.load(open(d / "metrics.json"))
    summary = json.load(open(d / "summary.json"))
    kinds = list(summary)

    # Align per-image values across backbones.
    images = sorted({r["image"] for r in rows})
    by = {(r["image"], r["backbone"]): r for r in rows}

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # --- per-image bars ---------------------------------------------------- #
    metrics = [
        ("auc", "linear-probe AUC", (0.5, 1.02)),
        ("knn_fg", "1-NN filament agreement", (0.0, 1.0)),
        ("recall", "token recall (filament px covered)", (0.0, 1.0)),
        ("precision", "token precision", (0.0, 1.0)),
    ]
    colors = {"dinov2": "#1f77b4", "sam": "#ff7f0e"}
    x = np.arange(len(images))
    width = 0.8 / max(1, len(kinds))

    for ax, (key, label, ylim) in zip(axes.ravel()[:4], metrics):
        for i, kind in enumerate(kinds):
            vals = [by.get((im, kind), {}).get(key, np.nan) for im in images]
            ax.bar(x + i * width, vals, width, label=kind, color=colors.get(kind))
        ax.set_xticks(x + width * (len(kinds) - 1) / 2)
        ax.set_xticklabels([im[:10] for im in images], rotation=90, fontsize=6)
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        ax.axhline(0.5, ls=":", c="k", lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # --- recall vs prevalence ---------------------------------------------- #
    ax = axes.ravel()[4]
    for kind in kinds:
        sub = [r for r in rows if r["backbone"] == kind]
        ax.scatter(
            [100 * r["prevalence"] for r in sub],
            [r["recall"] for r in sub],
            s=40,
            alpha=0.8,
            label=kind,
            color=colors.get(kind),
        )
    ax.set_xlabel("filament token prevalence (%)")
    ax.set_ylabel("token recall")
    ax.set_title("recall vs filament prevalence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- summary text ------------------------------------------------------ #
    ax = axes.ravel()[5]
    ax.axis("off")
    lines = [f"{'':14s}" + "".join(f"{k:>10s}" for k in kinds)]
    for key, label in [
        ("auc_mean", "AUC"),
        ("knn_fg_mean", "1-NN fg"),
        ("recall_mean", "recall"),
        ("precision_mean", "precision"),
        ("tokens_per_image", "tokens/img"),
        ("seconds_per_image", "sec/img"),
    ]:
        vals = "".join(f"{summary[k][key]:10.3f}" for k in kinds)
        lines.append(f"{label:14s}{vals}")
    ax.text(
        0.0,
        0.9,
        "\n".join(lines),
        family="monospace",
        fontsize=11,
        va="top",
        transform=ax.transAxes,
    )
    ax.set_title("aggregate (mean over images)", loc="left")

    fig.suptitle("Frozen-backbone token embeddings: DINOv2 ViT-S/14 vs SAM ViT-B/16", fontsize=13)
    fig.tight_layout()
    out = Path(args.out) if args.out else d / "compare.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
