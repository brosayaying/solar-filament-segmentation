# Third-party components and data licensing

The code in this repository is MIT-licensed; see [`LICENSE`](LICENSE).

## Dependencies

This repository depends on, but does not redistribute:

| Package | License |
|---|---|
| PyTorch | BSD-3-Clause |
| timm | Apache-2.0 |
| OpenCV | Apache-2.0 |
| NumPy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| pycocotools | BSD-2-Clause |

Pinned versions are in [`requirements.txt`](requirements.txt).

## Trained weights

The checkpoints published as release assets are derived from ConvNeXt-Small
initialised with ImageNet weights (`convnext_small.fb_in22k_ft_in1k`, available
through timm under Apache-2.0).

## Competition data — NOT covered by the MIT license

The MAGFiLO dataset is **not** included in this repository and is **not**
covered by the MIT license above. It is distributed by the competition
organisers under **CC BY-NC 4.0** (non-commercial, academic use only).

Download it from
<https://www.kaggle.com/competitions/filament-segmentation-2026>.

## GONG data

The competition images are GONG H-alpha observations, courtesy of the NSF
National Solar Observatory (NSO) Integrated Synoptic Program, managed by the
National Solar Observatory, which is operated by the Association of Universities
for Research in Astronomy (AURA), Inc. under a cooperative agreement with the
National Science Foundation and with contribution from the National Oceanic and
Atmospheric Administration.

The GONG network of instruments is hosted by the Big Bear Solar Observatory,
High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar
Observatory, Instituto de Astrofísica de Canarias, and Cerro Tololo Interamerican
Observatory.
