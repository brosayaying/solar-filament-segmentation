"""Paired per-image test: does the decoder beat the no-decoder model?

Both checkpoints are scored on the same 142 val files with the same
post-processing, so a paired test on per-image PQ is the right comparison.
The point estimate (+0.031) is meaningless without it: per-image PQ has a
standard deviation around 0.10, so the standard error on 142 images is ~0.008.

    python scripts/deepcrack_decoder_paired.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar.data import anns_to_labelmap, build_ann_index, load_coco  # noqa: E402
from solar.deepcrack import DeepCrack  # noqa: E402
from solar.metrics import PQAccumulator  # noqa: E402
from solar.postprocess import prob_to_labelmap  # noqa: E402

cv2.setNumThreads(4)

THR = 0.55
MIN_AREA = 512


def load_model(path: Path, decoder: bool, device: torch.device) -> DeepCrack:
    m = DeepCrack(in_chans=1, ngf=64, pretrained=False, decoder=decoder)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model", ck))
    return m.to(device).eval()


def per_image_pq(model, ids, images, ann_index, data_root, device) -> np.ndarray:
    out = []
    for iid in ids:
        im = images[iid]
        img = cv2.imread(
            str(data_root / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE
        )
        x = (
            torch.from_numpy(img.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            p = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
        g = anns_to_labelmap(ann_index[iid], im["height"], im["width"])
        acc = PQAccumulator()
        acc.update(prob_to_labelmap(p, threshold=THR, min_area=MIN_AREA), g)
        out.append(acc.summary()["pq"])
    return np.array(out)


def main() -> None:
    data_root = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    coco = load_coco(data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ann_index = build_ann_index(coco)
    images = {im["id"]: im for im in coco["images"]}
    split = json.loads((ROOT / "splits/date_seed42.json").read_text())

    seen: set[str] = set()
    ids = []
    for iid in split["val"]:
        fn = images[iid]["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        ids.append(iid)
    ids = ids[:142]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    a = load_model(ROOT / "runs/deepcrack_ft_256_long/best.pt", decoder=False, device=device)
    b = load_model(ROOT / "runs/deepcrack_decoder_512/best.pt", decoder=True, device=device)

    pa = per_image_pq(a, ids, images, ann_index, data_root, device)
    pb = per_image_pq(b, ids, images, ann_index, data_root, device)

    d = pb - pa
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se
    # 95% CI via normal approximation; n=142 is large enough.
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se

    print(f"n images            : {len(d)}")
    print(f"no-decoder mean PQ  : {pa.mean():.4f}")
    print(f"decoder    mean PQ  : {pb.mean():.4f}")
    print(f"paired diff         : {d.mean():+.4f}")
    print(f"95% CI              : [{lo:+.4f}, {hi:+.4f}]")
    print(f"std / stderr        : {d.std(ddof=1):.4f} / {se:.4f}")
    print(f"t                   : {t:.2f}")
    print(f"decoder better on   : {(d > 0).sum()}/{len(d)} images")
    print(f"decoder worse on    : {(d < 0).sum()}/{len(d)} images")
    print(f"tied                : {(d == 0).sum()}/{len(d)} images")
    print()
    if lo > 0:
        print("VERDICT: gain is real (CI excludes 0).")
    elif hi < 0:
        print("VERDICT: decoder is worse (CI excludes 0).")
    else:
        print("VERDICT: not resolved at this sample size (CI includes 0).")


if __name__ == "__main__":
    main()
