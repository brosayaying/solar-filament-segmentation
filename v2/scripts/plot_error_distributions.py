"""Figures for the two rubric criteria that were never plotted.

    Distribution of Dice scores
    Distribution of IoU scores
    Distribution of one-to-many / many-to-one relations

Reads ``v2/results/error_distributions.json`` (summary) and
``v2/results/error_distributions.npz`` (raw values), and writes PNGs to
``v2/results/figures/``.

Why three populations per metric
--------------------------------
A single histogram of "IoU" is ambiguous, because there are three different
populations and they tell different stories:

  matched        only the instances that were found and matched. This is the
                 boundary-quality view, and it is the one that looks good.
  best_per_gt    every ground-truth instance, with missed ones scoring 0. This
                 is the honest per-object view.
  best_per_pred  every prediction, with false positives scoring 0.

Plotting only the first would flatter the system by hiding the misses. Plotting
only the second would hide that boundaries are good when the instance is found.
The report needs both, so both are drawn.

    python v2/scripts/plot_error_distributions.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "v2/results/error_distributions.json"
RAW = ROOT / "v2/results/error_distributions.npz"
FIG = ROOT / "v2/results/figures"

COLORS = {
    "matched": "#1f77b4",
    "best_per_gt": "#d62728",
    "best_per_pred": "#ff7f0e",
}
LABELS = {
    "matched": "matched pairs (SQ view)",
    "best_per_gt": "per ground-truth (misses = 0)",
    "best_per_pred": "per prediction (FPs = 0)",
}
VIEWS = ("matched", "best_per_gt", "best_per_pred")


def load() -> tuple[dict, dict[str, np.ndarray]]:
    if not SRC.exists() or not RAW.exists():
        raise SystemExit(
            "missing inputs; run v2/scripts/error_distributions.py first"
        )
    summary = json.loads(SRC.read_text())
    with np.load(RAW) as z:
        raw = {k: z[k] for k in z.files}
    return summary, raw


def plot_metric(summary: dict, raw: dict, metric: str, out: Path) -> None:
    """Histogram + survival curve for one metric across three populations."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    # --- left: normalised histogram ------------------------------------
    # Density, not counts: the populations differ by ~60% in size (5259 vs
    # 8662) and raw counts would make the smaller one look unimportant.
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    for view in VIEWS:
        vals = raw[f"{metric}_{view}"]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=2,
                color=COLORS[view], label=f"{LABELS[view]}  (n={vals.size:,})")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("density")
    ax.set_title(f"{metric.upper()} distribution")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- right: survival curve -----------------------------------------
    # "What fraction of instances reach at least this quality?" This is the
    # question a reader actually asks, and it is not readable off a histogram.
    ax = axes[1]
    thresholds = np.linspace(0, 1, 101)
    for view in VIEWS:
        vals = raw[f"{metric}_{view}"]
        if vals.size == 0:
            continue
        frac = [(vals >= t).mean() for t in thresholds]
        ax.plot(thresholds, frac, color=COLORS[view], linewidth=2,
                label=LABELS[view])
    # Mark the metric's own matching threshold.
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(0.52, 0.55, "0.5 = match\nthreshold", fontsize=7, va="top")
    ax.set_xlabel(f"{metric.upper()} threshold")
    ax.set_ylabel("fraction at or above")
    ax.set_title(f"How often is {metric.upper()} good enough?")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    t = summary["totals"]
    fig.suptitle(
        f"{metric.upper()} — {t['n_gt']:,} GT, {t['n_pred']:,} predictions, "
        f"{t['n_tp']:,} matched  "
        f"(recall {t['recall']:.3f}, precision {t['precision']:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_relations(summary: dict, out: Path) -> None:
    """One-to-many and many-to-one counts across overlap thresholds."""
    thresholds = summary["params"]["relation_thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)

    for ax, key, title in (
        (axes[0], "one_to_many", "One-to-many: one GT split by k predictions"),
        (axes[1], "many_to_one", "Many-to-one: one prediction merges k GT"),
    ):
        zero = [summary[key][str(t)]["_summary"]["frac_zero"] for t in thresholds]
        one = [summary[key][str(t)]["_summary"]["frac_one"] for t in thresholds]
        multi = [summary[key][str(t)]["_summary"]["frac_multi"]
                 for t in thresholds]

        x = np.arange(len(thresholds))
        w = 0.26
        ax.bar(x - w, zero, w, label="k = 0 (missed)", color="#999999")
        ax.bar(x, one, w, label="k = 1 (clean 1:1)", color="#2ca02c")
        ax.bar(x + w, multi, w, label="k > 1 (fragmented / merged)",
               color="#d62728")

        ax.set_xticks(x)
        ax.set_xticklabels([f"IoU>{t}" for t in thresholds])
        ax.set_ylabel("fraction of instances")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

        # The k>1 bars are the ones the metric punishes and they are small
        # enough to be invisible without a label.
        for xi, m in zip(x, multi):
            if m > 0.001:
                ax.text(xi + w, m + 0.01, f"{100 * m:.1f}%",
                        ha="center", fontsize=7)

    fig.suptitle(
        "Instance relations between predictions and ground truth "
        "(shipped post-processing: thr 0.50 / area 256 / close 15)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    summary, raw = load()
    FIG.mkdir(parents=True, exist_ok=True)
    plot_metric(summary, raw, "iou", FIG / "iou_distribution.png")
    plot_metric(summary, raw, "dice", FIG / "dice_distribution.png")
    plot_relations(summary, FIG / "instance_relations.png")


if __name__ == "__main__":
    main()
