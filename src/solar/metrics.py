"""Panoptic Quality (PQ) for instance segmentation.

    PQ = sum(IoU of TP) / (|TP| + 0.5|FP| + 0.5|FN|)

Matching is by overlap (IoU > 0.5), one-to-one, greedy by descending IoU. The
number of predicted instances need not equal the number of ground-truth ones.

Pairwise IoU is computed with a single ``np.bincount`` over the combined label
map, which is ~100x faster than looping over instance pairs at 2048^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def pairwise_iou(pred: np.ndarray, gt: np.ndarray, n_pred: int, n_gt: int) -> np.ndarray:
    """IoU matrix between two instance label maps (0 = background)."""
    combined = pred.astype(np.int64) * (n_gt + 1) + gt.astype(np.int64)
    counts = np.bincount(combined.ravel(), minlength=(n_pred + 1) * (n_gt + 1))
    counts = counts.reshape(n_pred + 1, n_gt + 1)

    inter = counts[1:, 1:].astype(np.float64)
    pred_area = counts[1:, :].sum(axis=1).astype(np.float64)
    gt_area = counts[:, 1:].sum(axis=0).astype(np.float64)
    union = pred_area[:, None] + gt_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_instances(
    iou: np.ndarray, threshold: float = 0.5
) -> tuple[int, float]:
    """Greedy one-to-one matching. Returns ``(n_true_positives, sum_of_iou)``."""
    if iou.size == 0:
        return 0, 0.0

    n_pred, n_gt = iou.shape
    matched_pred = np.zeros(n_pred, dtype=bool)
    matched_gt = np.zeros(n_gt, dtype=bool)

    tp = 0
    iou_sum = 0.0
    for flat in np.argsort(iou, axis=None)[::-1]:
        i, j = divmod(int(flat), n_gt)
        if iou[i, j] <= threshold:
            break
        if matched_pred[i] or matched_gt[j]:
            continue
        matched_pred[i] = True
        matched_gt[j] = True
        tp += 1
        iou_sum += float(iou[i, j])
    return tp, iou_sum


@dataclass
class PQAccumulator:
    """Accumulates PQ statistics across a dataset.

    PQ is aggregated from global sums rather than averaged per image, matching
    the standard panoptic evaluation protocol.

    ``threshold`` is the **IoU matching threshold** (0.5 in the official
    protocol), not a probability threshold. It was once passed the probability
    threshold by mistake, which matched instances at IoU>0.45 and inflated PQ.
    """

    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    n_images: int = 0
    per_image: list[dict] = field(default_factory=list)

    def update(self, pred: np.ndarray, gt: np.ndarray, record: bool = False) -> dict:
        n_pred = int(pred.max()) if pred.size else 0
        n_gt = int(gt.max()) if gt.size else 0

        if n_pred == 0 and n_gt == 0:
            stats = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
        elif n_pred == 0:
            stats = {"tp": 0, "fp": 0, "fn": n_gt, "iou_sum": 0.0}
        elif n_gt == 0:
            stats = {"tp": 0, "fp": n_pred, "fn": 0, "iou_sum": 0.0}
        else:
            iou = pairwise_iou(pred, gt, n_pred, n_gt)
            tp, iou_sum = match_instances(iou, self.threshold)
            stats = {"tp": tp, "fp": n_pred - tp, "fn": n_gt - tp, "iou_sum": iou_sum}

        self.tp += stats["tp"]
        self.fp += stats["fp"]
        self.fn += stats["fn"]
        self.iou_sum += stats["iou_sum"]
        self.n_images += 1
        if record:
            self.per_image.append(stats)
        return stats

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denom if denom > 0 else 0.0

    @property
    def rq(self) -> float:
        """Recognition quality: TP / (TP + 0.5FP + 0.5FN)."""
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def sq(self) -> float:
        """Segmentation quality: mean IoU over matched pairs."""
        return self.iou_sum / self.tp if self.tp > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "pq": self.pq,
            "sq": self.sq,
            "rq": self.rq,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_images": self.n_images,
        }

    def bootstrap_ci(
        self, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
    ) -> dict[str, tuple[float, float]]:
        """Per-image bootstrap CI for PQ/SQ/RQ.

        Resamples *images* with replacement and recomputes the global aggregate,
        which is the correct uncertainty for a dataset-level PQ. Quoting the
        per-image p10-p90 spread as a "noise band" is wrong: that is the spread
        of individual images, not the standard error of the mean, which is
        roughly sd/sqrt(n) and therefore ~0.01 for n=142-232.

        Requires ``update(..., record=True)`` to have been used.
        """
        if not self.per_image:
            raise ValueError("call update(..., record=True) to enable bootstrap")
        rng = np.random.default_rng(seed)
        n = len(self.per_image)
        tp = np.array([s["tp"] for s in self.per_image], np.float64)
        fp = np.array([s["fp"] for s in self.per_image], np.float64)
        fn = np.array([s["fn"] for s in self.per_image], np.float64)
        iou = np.array([s["iou_sum"] for s in self.per_image], np.float64)

        out: dict[str, tuple[float, float]] = {}
        for key in ("pq", "sq", "rq"):
            vals = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                t, f, m, i = tp[idx].sum(), fp[idx].sum(), fn[idx].sum(), iou[idx].sum()
                denom = t + 0.5 * f + 0.5 * m
                if key == "pq":
                    vals[b] = i / denom if denom > 0 else 0.0
                elif key == "rq":
                    vals[b] = t / denom if denom > 0 else 0.0
                else:
                    vals[b] = i / t if t > 0 else 0.0
            out[key] = (
                float(np.percentile(vals, 100 * alpha / 2)),
                float(np.percentile(vals, 100 * (1 - alpha / 2))),
            )
        return out
