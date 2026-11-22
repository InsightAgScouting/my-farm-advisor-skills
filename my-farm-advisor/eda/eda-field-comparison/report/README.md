# Assignment 2 report artifact

One-time submission PDF assembled from the rendered outputs of the
`eda-field-comparison` subskill in the parent folder. It is **not** reusable
subskill code: nothing under `src/field_eda/` imports it, the validator does not
read it, and rebuilding requires the runtime data.

- `assignment-2-eda-report.pdf` — 8-page report committed for the instructor.
  Scope, grower locations, field counts and layers, comparison levels, what the
  statistical figures and correlations show, what the maps show, limitations,
  and the soil-out-of-scope statement.
- `build_report.py` — regenerates the PDF from
  `${DATA_PIPELINE_DATA_ROOT}/data-pipeline/shared/eda/assignment-2/`
  (`panel.csv` plus the rendered PNGs).

## Rebuild

```bash
export DATA_PIPELINE_DATA_ROOT=$HOME/my-farm-advisor-runtime
cd my-farm-advisor/eda/eda-field-comparison
$DATA_PIPELINE_DATA_ROOT/data-pipeline/.venv/bin/python report/build_report.py
```

Add `--preview-dir /tmp/report-pages` to also write one PNG per page for review.
The PDF is kept under 1 MB (no Git LFS): figures are embedded at 95 dpi and
text stays vector.

## Boundaries

- Report assembly is outside the reusable subskill; `src/` remains write-free.
- Figures are read, never regenerated, so the report cannot drift from the
  rendered gallery. Tables are re-derived from `panel.csv`.
- Soil analysis is not part of this assignment and the report says so; the
  subskill never reads SSURGO.
