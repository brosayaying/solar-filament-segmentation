"""Stage 1 training: filament bounding-box detector.

    python -m solar.train_detector --config configs/detector.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2

from .config import load_config
from .data import load_coco
from .detector import build_detector
from .detector_data import DetectionDataset, collate_detection, decode_boxes, decode_points
from .detector_loss import detector_loss
from .detector_metrics import mask_hit_stats, match_points
from .train import get_split


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of [x, y, w, h] boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(ax1[:, None], bx1[None, :])
    iy2 = np.minimum(ay1[:, None], by1[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match(iou: np.ndarray, thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns (tp, fp, fn)."""
    if iou.size == 0:
        return 0, 0, int(iou.shape[1])
    used_p = np.zeros(iou.shape[0], bool)
    used_g = np.zeros(iou.shape[1], bool)
    tp = 0
    for flat in np.argsort(iou, axis=None)[::-1]:
        i, j = divmod(int(flat), iou.shape[1])
        if iou[i, j] < thr:
            break
        if used_p[i] or used_g[j]:
            continue
        used_p[i] = used_g[j] = True
        tp += 1
    return tp, int(iou.shape[0]) - tp, int(iou.shape[1]) - tp


def match_points(pred: np.ndarray, gt: np.ndarray, radius: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching of predicted points to GT points by distance.

    A prediction counts as a true positive if it lands within ``radius`` pixels
    of an unmatched ground-truth spine midpoint.
    """
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0

    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    used_p = np.zeros(len(pred), bool)
    used_g = np.zeros(len(gt), bool)
    tp = 0
    for flat in np.argsort(d, axis=None):
        i, j = divmod(int(flat), len(gt))
        if d[i, j] > radius:
            break
        if used_p[i] or used_g[j]:
            continue
        used_p[i] = used_g[j] = True
        tp += 1
    return tp, int(len(pred) - tp), int(len(gt) - tp)


def match_points_legacy(pred: np.ndarray, gt: np.ndarray, radius: float) -> tuple[int, int, int]:
    """Deprecated: kept only so old call sites keep working."""
    return match_points(pred, gt, radius)


@torch.no_grad()
def validate_detector(
    model, loader, device, score_thr: float, max_images: int | None, nms_dist: float = 40.0
) -> dict:
    """Validate stage 1.

    Reports both the conventional point F1 (radius 24 px) and mask-hit recall
    (a predicted point inside the filament's own mask). Mask-hit is the number
    that matters for stage 2, since any seed on the filament is usable.
    """
    model.eval()
    tp = fp = fn = 0
    hit = miss = pred_on = pred_off = 0
    seen = 0
    point_mode = not getattr(model, "with_size", True)
    radius = 24.0

    for batch in loader:
        pred = model(batch["image"].to(device, non_blocking=True))
        if point_mode:
            pts = decode_points(pred, threshold=score_thr, nms_dist=nms_dist)
        else:
            pts = decode_boxes(pred, threshold=score_thr)

        for i, pb in enumerate(pts):
            if point_mode:
                gt = batch["gt_centers"][i]
                t, p, n = match_points(pb, gt, radius)
                st = mask_hit_stats(pb, batch["anns"][i], 2048, 2048)
                hit += st["hit"]
                miss += st["miss"]
                pred_on += st["pred_on_filament"]
                pred_off += st["pred_off"]
            else:
                gt = batch["gt_boxes"][i]
                t, p, n = greedy_match(box_iou(pb, gt), 0.5)
            tp, fp, fn = tp + t, fp + p, fn + n
            seen += 1
            if max_images and seen >= max_images:
                break
        if max_images and seen >= max_images:
            break

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
    if point_mode:
        out["mask_hit_recall"] = hit / max(1, hit + miss)
        out["mask_hit"] = hit
        out["mask_miss"] = miss
        out["pred_on_filament"] = pred_on
        out["pred_off"] = pred_off
    return out


class ValDetectionDataset(DetectionDataset):
    """Validation: full image, no augmentation, returns GT boxes."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["augment"] = False
        kwargs["foreground_bias"] = 0.0
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        meta = self.images[image_id]
        img = cv2.imread(str(self.img_dir / meta["file_name"]), cv2.IMREAD_GRAYSCALE)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        boxes = self._boxes(image_id)
        gt = np.stack(boxes) if boxes else np.zeros((0, 4), np.float32)
        centers = self._centers(image_id)
        gt_c = np.stack(centers) if centers else np.zeros((0, 2), np.float32)
        return {
            "image": torch.from_numpy(x).unsqueeze(0),
            "gt_boxes": gt,
            "gt_centers": gt_c,
            "anns": self.ann_index.get(image_id, []),
            "image_id": image_id,
        }


def collate_val_detection(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_centers": [b["gt_centers"] for b in batch],
        "anns": [b["anns"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg["data"]["seed"])
    np.random.seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["data"]["root"])
    coco = load_coco(root / cfg["data"]["ann_file"])
    split = get_split(cfg, coco)

    dcfg = cfg["detector"]
    train_ds = DetectionDataset(
        root,
        coco,
        split["train"],
        patch_size=dcfg["patch_size"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        foreground_bias=dcfg["foreground_bias"],
        with_size=dcfg.get("with_size", True),
    )
    val_ds = ValDetectionDataset(
        root,
        coco,
        split["val"],
        patch_size=dcfg["patch_size"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        with_size=dcfg.get("with_size", True),
    )

    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=dcfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_detection,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(2, nw // 2),
        collate_fn=collate_val_detection,
        pin_memory=True,
    )

    model = build_detector(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"detector: {dcfg['backbone']} | {n_params/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=dcfg["lr"], weight_decay=dcfg["weight_decay"])
    steps = len(train_loader) * dcfg["epochs"]
    warm = len(train_loader) * dcfg["warmup_epochs"]

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_name = dcfg.get("run_name") or f"det_{dcfg['backbone']}_{int(time.time())}"
    out_dir = Path(cfg["train"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    log_path = out_dir / "log.jsonl"

    best_f1 = -1.0
    for epoch in range(1, dcfg["epochs"] + 1):
        model.train()
        running = 0.0
        parts = {"heatmap": 0.0, "offset": 0.0}
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{dcfg['epochs']}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            target = {
                "heatmap": batch["heatmap"].to(device, non_blocking=True),
                "offset": batch["offset"].to(device, non_blocking=True),
                "mask": batch["mask"].to(device, non_blocking=True),
            }
            if "size" in batch:
                target["size"] = batch["size"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred = model(images)
                loss, comps = detector_loss(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += float(loss.detach())
            for k in comps:
                parts[k] = parts.get(k, 0.0) + comps[k]
            pbar.set_postfix(loss=f"{float(loss.detach()):.3f}")

        n = max(1, len(train_loader))
        record = {
            "epoch": epoch,
            "train_loss": running / n,
            **{f"loss_{k}": v / n for k, v in parts.items()},
        }

        if epoch % dcfg["val_every"] == 0:
            m = validate_detector(
                model,
                val_loader,
                device,
                dcfg["score_threshold"],
                dcfg["val_images"],
                dcfg.get("nms_dist", 40.0),
            )
            record.update({f"val_{k}": v for k, v in m.items()})
            print(
                f"epoch {epoch}: loss={record['train_loss']:.3f} "
                f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                f"maskHit={m.get('mask_hit_recall', 0):.3f} "
                f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})"
            )
            if m.get("mask_hit_recall", m["f1"]) > best_f1:
                best_f1 = m.get("mask_hit_recall", m["f1"])
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "f1": best_f1},
                           out_dir / "best.pt")
        else:
            print(f"epoch {epoch}: loss={record['train_loss']:.3f}")

        record["epoch_time_s"] = round(time.time() - t0, 1)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "last.pt")
    print(f"done. best F1={best_f1:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
