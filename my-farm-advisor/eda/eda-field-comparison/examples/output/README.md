# Assignment 2 gallery (committed snapshots)

Curated PNG snapshots of the figures `examples/build_panel.py` renders, kept in
Git so the results can be reviewed without a runtime. They are **snapshots, not
the write path**: the script itself still writes to
`${DATA_PIPELINE_DATA_ROOT}/data-pipeline/shared/eda/assignment-2/`, and nothing
here is read by `src/field_eda/`.

## Regenerate

```bash
export DATA_PIPELINE_DATA_ROOT=$HOME/my-farm-advisor-runtime
$DATA_PIPELINE_DATA_ROOT/data-pipeline/.venv/bin/python \
    examples/build_panel.py --no-maps      # statistical figures
$DATA_PIPELINE_DATA_ROOT/data-pipeline/.venv/bin/python \
    examples/build_panel.py                # adds M1/M2/M3 (needs basemap tiles)
```

Copy the PNGs you want to refresh from the runtime folder into this one.

## What is here

| Group | Files |
| --- | --- |
| Coverage | `coverage_all.png` |
| Boundaries | `a1_all.png` |
| NDVI | `n1_il-r1.png`, `n2_all.png`, `n3_all.png` |
| Cropland | `c1_{illinois,iowa,nebraska,delaware}.png`, `c2_all.png`, `c3_all.png` |
| Weather | `w1_all.png`, `w2_all.png`, `w3_all.png` |
| Cross-checks | `x4_all.png`, `x5_all.png`, `x6_all.png`, `x7_all.png` |
| Maps | `m1_delaware.png`, `m2_delaware.png`, `m3_delaware.png` |

Every file above is under the repo's 1 MB asset threshold, so no Git LFS is
needed.

## What is deliberately not here

- **The nine large map plates** — `m1/m2/m3` for Illinois, Iowa and Nebraska run
  2.2–3.2 MB each. They stay in the runtime and are viewable through the report.
- **The CSVs** — `panel.csv` carries per-field `lat`/`lon`, so it is kept out of
  the repository. `composition.csv` and `coverage.csv` live beside it in the
  runtime folder.
- **The full-resolution report figures** — `report/assignment-2-eda-report.pdf`
  embeds them at 95 dpi, which keeps the artifact under 1 MB.

Figures here passed the repo's five-point review before being committed; if you
regenerate one, re-check it against `../../../AGENTS.md` and say so in the
commit message.
