# Local Instructions

## Purpose

This folder owns the Assignment 2 field-level EDA subskill: one field-year
panel and static figures that compare boundaries, CDL, weather, and NDVI at
within-field, within-grower, and across-grower levels.

## Safe edit scope

Edits stay in this folder and its children. Do not change parent `SKILL.md`,
`../INDEX.md`, sibling EDA workflows, or root policy from a subskill task
unless the routing entry is explicitly part of the request.

## Read nearby docs first

Read `GUIDE.md` first. It carries the coverage table, the three-level mapping,
the full figure catalogue (including the Stage B/C outputs not built yet), and
the semantics that must not drift.

## Non-negotiable boundaries

- **No soil.** This subskill never reads `soil/`, never imports SSURGO
  helpers, and never adds soil columns to the panel.
- **No writes in library code.** `src/field_eda/` returns figures and frames;
  only `examples/` or a one-off report script writes PNG/CSV. Report assembly
  is not part of this subskill.
- **No HTML, no dashboard, no CLI.** Static matplotlib PNGs only.
- **Runtime data stays in the runtime.** Never write under the skill checkout;
  preview outputs go to `${DATA_PIPELINE_DATA_ROOT}/data-pipeline/shared/eda/`.

## Runtime contract

- `DATA_PIPELINE_DATA_ROOT` is required and points at the runtime root
  (`.../my-farm-advisor-runtime`). The panel reads
  `${DATA_PIPELINE_DATA_ROOT}/data-pipeline/growers/...` and
  `.../shared/cdl/rasters/`.
- Use the runtime venv for the example:
  `$DATA_PIPELINE_DATA_ROOT/data-pipeline/.venv/bin/python`.
- The panel derives CDL crop names from `crop_code` using
  `../../soil/cdl-cropland/src/cdl_codes.py`; do not reintroduce stored
  `crop_name` strings for IL/IA.
- NDVI composites are annual means, not peaks. If a true temporal peak is
  added later, it needs a new metric and a new caption — do not rename p95.

## Local validation

From the repository root:

```bash
./scripts/validate.sh
```

For a behaviour check, run the example with `--no-maps` first (panel and
statistical figures, no network), then once with maps to confirm the basemap
degrade path. A skipped NDVI composite must produce nulls plus a `notes`
entry, not an exception.

## Local-delta-only reminder

This nested AGENTS.md only records instructions that differ from the parent or
root files. Do not duplicate root-wide asset, vendor, or validation policy
here except this pointer to `../../../AGENTS.md`.
