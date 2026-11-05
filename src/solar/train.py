"""Training entry point.

    python -m solar.train --config configs/base.json
    python -m solar.train --set model.backbone=convnext_tiny train.epochs=5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .crop_data import PositiveCropDataset
from .data import FilamentDataset, collate_train, collate_val, load_coco, make_split
from .losses import bce_dice_loss
from .metrics import PQAccumulator
from .models import build_model
from .losses import build_loss
from .postprocess import prob_to_labelmap


def get_split(cfg: dict, coco: dict) -> dict:
    """Load a cached split, or create and cache it."""
    path = Path(cfg["data"]["split_file"])
    if path.exists():
        with open(path) as fh:
            return json.load(fh)

    split = make_split(
        coco,
        group_by=cfg["data"]["group_by"],
        val_frac=cfg["data"]["val_frac"],
        seed=cfg["data"]["seed"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(split, fh, indent=2)
    return split


@torch.no_grad()
def _predict_tiled(model, image: torch.Tensor, tile: int) -> np.ndarray:
    """Run the model over ``tile``-sized crops and stitch the probabilities.

    Needed for ViT encoders: they are strongly scale-sensitive, so a model
    trained on 518 px crops degrades badly when fed a full 2048 px image
    (measured: 7.7x less predicted area, PQ 0.045 vs 0.218). Evaluating at the
    training scale is what makes the validation number meaningful.
    """
    _, h, w = image.shape
    prob = np.zeros((h, w), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            crop = image[:, y0:y1, x0:x1].unsqueeze(0)
            out = model(crop)
            if isinstance(out, dict):
                out = out["logits"]
            elif isinstance(out, tuple):
                out = out[0]
            p = torch.sigmoid(out).float().cpu().numpy()[0, 0]
            prob[y0:y1, x0:x1] = p[: y1 - y0, : x1 - x0]
    return prob


@torch.no_grad()
def validate(model, loader, cfg, device, max_images: int | None = None) -> dict:
    """Score the model on the validation loader.

    ``max_images`` subsamples for speed during training. The subset is chosen
    **randomly with a fixed seed**, not by taking the head of the loader: the
    val ids are sorted, so the first 64 are all annotator groups ``0101xx``-
    ``0103xx`` and chronologically skewed. Selecting checkpoints on that subset
    biases every ``best.pt``.
    """
    model.eval()
    # PQAccumulator.threshold is the IoU *matching* threshold, fixed at 0.5 by
    # the official scorer. It is NOT the probability threshold. Passing
    # cfg["eval"]["threshold"] here silently matched instances at IoU>0.45,
    # inflating every in-training PQ and therefore every best.pt selection.
    acc = PQAccumulator(threshold=0.5)
    tile = cfg["eval"].get("tile")

    n_total = len(loader.dataset)
    if max_images and max_images < n_total:
        rng = np.random.default_rng(cfg["data"].get("seed", 42))
        keep = set(rng.choice(n_total, size=max_images, replace=False).tolist())
    else:
        keep = None

    seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if tile:
            # Keep the (B, 1, H, W) layout so the indexing below is shared.
            probs = np.stack([_predict_tiled(model, im, tile) for im in images])[:, None]
        else:
            logits = model(images)
            if isinstance(logits, dict):
                # Deep supervision returns {"logits", "sides"}; score the fused map.
                logits = logits["logits"]
            elif isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits).float().cpu().numpy()

        for i, label in enumerate(batch["label"]):
            if keep is not None and seen not in keep:
                seen += 1
                continue
            pred = prob_to_labelmap(
                probs[i, 0],
                threshold=cfg["eval"]["threshold"],
                min_area=cfg["eval"]["min_area"],
            )
            acc.update(pred, label.numpy(), record=True)
            seen += 1
    return acc.summary()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <out_dir>/<run_name>/last.pt if it exists. Restores "
             "model, optimizer, scheduler, epoch counter and early-stopping "
             "state, so an interrupted run continues rather than restarting.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg["data"]["seed"])
    np.random.seed(cfg["data"]["seed"])
    # `random` drives crop offsets and augmentation choices in data.py, so it
    # must be seeded too or runs are not reproducible.
    random.seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["data"]["root"])
    coco = load_coco(root / cfg["data"]["ann_file"])
    split = get_split(cfg, coco)

    print(f"split: {split['meta']}")

    train_ds = FilamentDataset(
        root,
        coco,
        split["train"],
        mode="train",
        patch_size=cfg["data"]["patch_size"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        samples_per_image=cfg["data"]["samples_per_image"],
        tiled=cfg["data"]["tiled"],
        spine_thickness=cfg["data"].get("spine_thickness", 3),
        preprocess=cfg["data"].get("preprocess"),
        mask_offdisk=cfg["data"].get("mask_offdisk", False),
        offdisk_value=cfg["data"].get("offdisk_value", -3.0),
        preprocess_cache=cfg["data"].get("preprocess_cache"),
    )
    if cfg["data"].get("sampler", "uniform") == "positive_crop":
        # Positive-biased + limb-aware crops. See solar/crop_data.py for why
        # uniform cropping cannot fix the limb false positives on its own.
        train_ds = PositiveCropDataset(
            root,
            coco,
            split["train"],
            patch_size=cfg["data"]["patch_size"],
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
            samples_per_image=cfg["data"]["samples_per_image"],
            p_positive=cfg["data"].get("p_positive", 0.5),
            p_limb=cfg["data"].get("p_limb", 0.2),
            limb_width=cfg["data"].get("limb_width", 24),
            soft_labels=cfg["data"].get("soft_labels", False),
        )
        print(
            f"sampler: positive_crop "
            f"(p_pos={cfg['data'].get('p_positive', 0.5)}, "
            f"p_limb={cfg['data'].get('p_limb', 0.2)}, "
            f"soft_labels={cfg['data'].get('soft_labels', False)})"
        )

    val_ds = FilamentDataset(
        root,
        coco,
        split["val"],
        mode="val",
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        dedup_files=cfg["data"].get("dedup_val_files", False),
        preprocess=cfg["data"].get("preprocess"),
        mask_offdisk=cfg["data"].get("mask_offdisk", False),
        offdisk_value=cfg["data"].get("offdisk_value", -3.0),
        preprocess_cache=cfg["data"].get("preprocess_cache"),
    )

    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_train,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=max(2, nw // 2),
        collate_fn=collate_val,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch = cfg["model"].get("arch", "unet")
    label = arch if arch != "unet" else cfg["model"]["backbone"]
    print(f"model: {label} | {n_params/1e6:.1f}M params")
    seg_loss = build_loss(cfg)
    print(f"loss: {cfg['train'].get('loss', 'bce_dice')}")

    # Warm start. The 160-epoch study (docs/LONG_TRAINING.md) showed ~40 of those
    # epochs go purely into learning the right instance count, a cost worth paying
    # once. `init_ckpt` loads weights only; the schedule restarts.
    init_ckpt = cfg["train"].get("init_ckpt")
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        state = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"warm start: {init_ckpt} (ep {ck.get('epoch')}, PQ {ck.get('pq', float('nan')):.4f})"
              f" missing={len(missing)} unexpected={len(unexpected)}")
        if missing or unexpected:
            print(f"  NOTE: missing={missing[:4]} unexpected={unexpected[:4]}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["train"]["warmup_epochs"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # A checkpoint whose own schedule already annealed to ~0 has settled weights.
    # Restarting cosine at peak LR applies a full-size step to a converged model,
    # which is how a warm start destroys a good checkpoint: measured once at
    # lr=1e-4, PQ collapsed 0.352 -> 0.0000 within six epochs. Warn loudly.
    if init_ckpt:
        prev_lr = ck.get("cfg", {}).get("train", {}).get("lr")
        if prev_lr is not None and cfg["train"]["lr"] >= prev_lr:
            print(
                f"WARNING: warm start at lr={cfg['train']['lr']:g} but the checkpoint "
                f"was trained with initial lr={prev_lr:g}. If that schedule had already "
                f"annealed, this is a large step for a converged model and PQ may "
                f"collapse. Consider train.lr <= {prev_lr / 3:g}."
            )

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    # `last.pt` is written every epoch and holds the full training state, not
    # just weights. Without the optimizer moments and the scheduler position a
    # resumed run is not a continuation: AdamW restarts with zero second
    # moments (a large effective step) and the cosine schedule jumps back to
    # peak LR, which is the same failure mode that destroyed a warm start at
    # lr=1e-4 (PQ 0.352 -> 0.0000 in six epochs).
    #
    # This block must run *after* the optimizer, scheduler and scaler exist,
    # because it restores into all three.
    start_epoch = 1
    best_pq = -1.0
    best_epoch = 0
    gate_pq = -1.0
    bad_evals = 0

    # AMP dtype. bf16 needs no gradient scaler (it has fp32's exponent range),
    # and fp16 diverged to NaN on ConvNeXt-Small -- see config.py.
    amp_dtype = str(cfg["train"].get("amp_dtype", "bf16")).lower()
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    torch_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and torch_dtype is torch.float16
    )
    print(f"amp: {amp_dtype} (scaler {'on' if scaler.is_enabled() else 'off'})")

    # ------------------------------------------------------------------ #
    # Resume (must come after model/opt/sched/scaler exist)
    # ------------------------------------------------------------------ #
    resume_path = None
    if args.resume:
        run_name_early = cfg["train"]["run_name"]
        if run_name_early:
            cand = Path(cfg["train"]["out_dir"]) / run_name_early / "last.pt"
            if cand.exists():
                resume_path = cand

    if resume_path is not None:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        if "scaler" in ck and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_pq = float(ck.get("best_pq", -1.0))
        best_epoch = int(ck.get("best_epoch", 0))
        gate_pq = float(ck.get("gate_pq", -1.0))
        bad_evals = int(ck.get("bad_evals", 0))
        print(
            f"resumed {resume_path}: epoch {ck.get('epoch')} -> {start_epoch}, "
            f"best {best_pq:.4f} @ ep {best_epoch}, "
            f"gate {gate_pq:.4f}, bad_evals {bad_evals}"
        )
        if start_epoch > cfg["train"]["epochs"]:
            print(
                f"nothing to do: already at epoch {start_epoch - 1} of "
                f"{cfg['train']['epochs']}"
            )
            return
    elif args.resume:
        print("--resume given but no last.pt found; starting from scratch")

    run_name = cfg["train"]["run_name"] or f"{cfg['model']['backbone']}_{int(time.time())}"
    out_dir = Path(cfg["train"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    log_path = out_dir / "log.jsonl"

    # Two separate notions of "best", which must not be conflated:
    #
    #   best_pq    the true maximum seen. This is what best.pt must hold.
    #   gate_pq    the last value that counted as *progress* for early stopping.
    #
    # An earlier version used a single variable updated only when the improvement
    # exceeded min_delta. That silently saved a worse checkpoint: on the fold-0
    # A/B raw arm it kept epoch 61 (PQ 0.4057) instead of epoch 85 (PQ 0.4077),
    # because 0.4077 - 0.4057 = 0.0020 was not > min_delta = 0.002. The reported
    # "best" was then wrong by exactly the min_delta, and the checkpoint used for
    # scoring was not the best one trained.
    #
    # Both are initialised above, before the resume block, so that a resumed run
    # continues the same early-stopping decision rather than restarting patience.
    # Early stopping on val PQ. See config.py for why train loss is the wrong
    # signal: it plateaus long before PQ does.
    patience = int(cfg["train"].get("early_stop_patience", 0) or 0)
    min_delta = float(cfg["train"].get("early_stop_min_delta", 0.0) or 0.0)

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        skipped = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['train']['epochs']}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            spines = batch["spine"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch_dtype):
                out = model(images)
                side_logits = None
                if isinstance(out, dict):
                    # DeepCrack with deep supervision: fused + five side heads.
                    logits, side_logits = out["logits"], out["sides"]
                    spine_logits = None
                elif isinstance(out, tuple):
                    logits, spine_logits = out
                else:
                    logits, spine_logits = out, None

                loss = seg_loss(logits, masks)
                if side_logits is not None:
                    # Each side head gets the same target as the fused output.
                    # Averaged rather than summed so side_weight is comparable
                    # across the five heads and does not scale with depth count.
                    side_loss = sum(seg_loss(s, masks) for s in side_logits) / len(side_logits)
                    loss = loss + cfg["train"].get("side_weight", 0.0) * side_loss
                if spine_logits is not None:
                    loss = loss + cfg["train"]["spine_weight"] * bce_dice_loss(
                        spine_logits, spines, 0.5, 0.5
                    )

            if not torch.isfinite(loss):
                # A non-finite loss means the step is already poisoned; applying
                # it would turn every weight into NaN and the run would never
                # recover. Skip the step and report it rather than silently
                # training a dead model for another 100 epochs.
                skipped += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_loader) - skipped)
        record = {"epoch": epoch, "train_loss": train_loss, "lr": sched.get_last_lr()[0]}
        if skipped:
            record["skipped_steps"] = skipped
            print(f"  WARNING: skipped {skipped} non-finite steps this epoch")

        if epoch % cfg["train"]["val_every"] == 0:
            metrics = validate(model, val_loader, cfg, device, cfg["train"]["val_images"])
            record.update({f"val_{k}": v for k, v in metrics.items()})
            if not np.isfinite(metrics["pq"]):
                print(f"epoch {epoch}: val PQ is non-finite -- model has diverged")
                record["diverged"] = True
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
                raise SystemExit(
                    "training diverged (non-finite val PQ). Check amp_dtype, lr, "
                    "and grad_clip."
                )
            print(
                f"epoch {epoch}: loss={train_loss:.4f} "
                f"PQ={metrics['pq']:.4f} SQ={metrics['sq']:.4f} RQ={metrics['rq']:.4f} "
                f"(tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})"
            )
            # Checkpoint on the true maximum, independent of min_delta.
            if metrics["pq"] > best_pq:
                best_pq = metrics["pq"]
                best_epoch = epoch
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "pq": best_pq},
                    out_dir / "best.pt",
                )

            # Patience counts only *meaningful* improvements, so that noise-level
            # wobble does not reset the counter forever.
            if metrics["pq"] > gate_pq + min_delta:
                gate_pq = metrics["pq"]
                bad_evals = 0
            else:
                bad_evals += 1
                if patience and bad_evals >= patience:
                    record["early_stop"] = True
                    record["epoch_time_s"] = round(time.time() - t0, 1)
                    with open(log_path, "a") as fh:
                        fh.write(json.dumps(record) + "\n")
                    print(
                        f"early stop at epoch {epoch}: no val PQ improvement "
                        f"> {min_delta:g} for {bad_evals} evals "
                        f"(best {best_pq:.4f} @ ep {best_epoch})"
                    )
                    # Save before breaking, so a resumed run sees the final
                    # state and exits immediately rather than re-running the
                    # epochs that already triggered the stop.
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "scaler": scaler.state_dict() if scaler.is_enabled() else {},
                            "cfg": cfg,
                            "epoch": epoch,
                            "best_pq": best_pq,
                            "best_epoch": best_epoch,
                            "gate_pq": gate_pq,
                            "bad_evals": bad_evals,
                            "early_stopped": True,
                        },
                        out_dir / "last.pt",
                    )
                    break
        else:
            print(f"epoch {epoch}: loss={train_loss:.4f}")

        record["epoch_time_s"] = round(time.time() - t0, 1)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        # Full training state, written every epoch so an interrupted run can be
        # resumed exactly. Weights alone are not enough: AdamW's moments and the
        # scheduler position are part of the state.
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else {},
                "cfg": cfg,
                "epoch": epoch,
                "best_pq": best_pq,
                "best_epoch": best_epoch,
                "gate_pq": gate_pq,
                "bad_evals": bad_evals,
            },
            out_dir / "last.pt",
        )

    print(f"done. best PQ={best_pq:.4f} @ epoch {best_epoch} -> {out_dir}")


if __name__ == "__main__":
    main()
