# Report and submission artefacts

## The submission PDF

**`paper_official/main.pdf`** — this is the technical report to submit.

It is built from the competition's official template (`acmart` with the
`sigconf` option), with all fixed text preserved verbatim. Four pages, body
within the limit (Acknowledgment and References excluded).

Rebuild with:

```bash
bash v2/scripts/build_paper.sh
```

The script fails if the body ever exceeds four pages.

## What else is here

| Path | What it is |
|---|---|
| `paper_official/` | The report source (`main.tex`, `preamble.tex`, `main.bib`) and figures |
| `kaggle_comp_report_fi_segm_2026.zip` | The official template, as distributed |
| `report_draft.md` / `.html` | Machine-generated draft from `notebooks/report_draft.ipynb` |

`report_draft.*` is **not** the submission. It is a Markdown rendering produced
from `v2/results/summary.json`, useful for checking that the numbers in the
paper match the measurements. The paper is the PDF.

## Where the numbers come from

Every figure in the report traces to a script:

| Claim | Source |
|---|---|
| PQ, SQ, RQ per phase | `v2/runs/seg_v2/oof_results.json`, `v2/runs/phase2/final.json` |
| Paired bootstrap CIs | `v2/runs/phase2/acceptance.json` |
| Oracle ceiling | `v2/runs/phase2/ceiling.json` |
| False-positive anatomy | `v2/runs/phase2/fp_anatomy.json` |
| Dice/IoU distributions, relations | `v2/results/error_distributions.json` |
| Phase 3 (label transfer) | `v2/runs/phase3/eval_phase3_fold0.json` |
| Phase 4 (PVTv2 ensemble) | `v2/runs/phase4/eval_phase4.json` |

`v2/results/summary.json` consolidates these, and `notebooks/report_draft.ipynb`
asserts twelve consistency checks against them before rendering anything.
