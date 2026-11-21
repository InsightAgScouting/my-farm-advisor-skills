# Field Comparison EDA (Assignment 2)

Compare the same fields across boundaries, CDL/cropland, weather, and NDVI at
three levels — within a field over time, across fields inside a grower, and
across growers. Static Python figures only: no dashboard, no HTML, no writes
inside the reusable code.

This is the Assignment 2 workflow. Soil (SSURGO) is deliberately **out of
scope**; this subskill never reads `soil/`.

## Coverage first

The dataset has **34 fields x 5 years (2021–2025) = 170 field-years**.

| Layer | Coverage | Why |
|---|---|---|
| Field boundaries | 34/34 | complete |
| CDL 2021–2025 | 34/34 | complete |
| Weather 2021–2025 | 34/34 | complete, per-field daily POWER records |
| NDVI | **20/34** | NE 10 + Marvel 4 + IL `r1–r3` + IA `r1–r3`; IL/IA `r4–r10` have no satellite ingest |

NDVI is a **subset category**, not a gate. The panel records the gap as a
null plus a note, and every NDVI figure carries a "covers N of 34 fields"
strip. The greenness chapter comes **after** the crop chapter, so the reader
knows what the greenness describes before noticing what is missing.

## Three comparison levels — what covers what

| Level | CDL | Weather | NDVI | Data that makes it possible |
|---|---|---|---|---|
| **Within a field** | **C1** per-field composition small multiples | **W1** cumulative GDD, one line per year | **N1** NDVI distribution per year (violins) | The year column: same footprint, five seasons |
| **Across fields in a grower** | **C2** field x year dominant-crop heatmap; **X5** area vs crop purity | **W2** per-field season GDD and rainfall dot plot | N2 (crop class pooled across growers) | Multiple `field_id` rows sharing grower and year; per-field geometry |
| **Across growers** | **C3** crop-share and Shannon diversity by grower | **W3** monthly climatology; **X6** GDD x rainfall ellipses | **N3** annual-mean NDVI climatology; **X4** NDVI vs GDD/rainfall | Grower label plus the same measurement per farm |

**Where a level cannot apply:** geometry is static. Acreage, perimeter, and
compactness cannot be compared *within* a field across years — there is one
boundary per field. The substitute is a cross-field comparison (C1/C2/X5 use
area as the axis), and the panel repeats each field's geometry on every year
row so the value is visibly constant rather than recomputed differently.
Weather at the within-field level is only a *time* comparison (W1); the
spatial source is too coarse to say anything about position inside a field.

**Maps** are the cross-cutting view: **M1** grower plate (2025 CDL fill),
**M2** crop flip 2021→2025, **M3** NDVI with its coverage strip.

## Figure catalogue

Stage A (built): `coverage`, `N1`, `C1`, `W1`, `M1`.
Stage B (built): `N2`, `N3`, `X4`, `C2`, `C3`, `X5`.
Stage C (next): `W2`, `W3`, `X6`, `M2`, `M3`.

| ID | Level | Plot | Story |
|---|---|---|---|
| N1 | within field | annual-mean NDVI distribution per year, violins | the field's own greenness baseline and each season's departure |
| N2 | across fields | NDVI by CDL crop class, one point per field-year | whether the label shows up as greenness; grower confounding stays visible |
| N3 | across growers | grower x year NDVI climatology | growers side by side, with the coverage strip |
| X4 | fields + growers | NDVI vs season GDD and vs rainfall, ρ per crop | how much of the greenness is season, and what is left |
| C1 | within field | 100 % stacked composition by year, per field | each field's own rotation, not the farm average |
| C2 | across fields | field x year dominant-crop heatmap | synchronised Corn Belt rotation vs Delaware patchwork |
| C3 | across growers | crop share + Shannon diversity distribution | two-crop system vs double-crop/vegetable system |
| X5 | fields + growers | field area vs crop purity, ρ overall and per grower | small parcels are less pure; the pooled and within-grower ρ disagree, which bounds the confidence cap |
| W1 | within field | cumulative GDD by day-of-year, one line per year | interannual spread and season length on the same ground |
| W2 | across fields | season GDD and rainfall per field within a grower | how much fields inside one grower actually differ |
| W3 | across growers | monthly temperature/rainfall climatology | NE drier and flashier, IL/IA wetter, DE different timing |
| X6 | fields + growers | season GDD vs rainfall with grower ellipses | growers separate in climate space |
| M1 | map | grower plate, imagery + boundaries + CDL 2025 fill | where the fields are and what each carried |
| M2 | map | 2021→2025 crop flip | rotation as geography |
| M3 | map | NDVI year with coverage strip | greenness patterns and where imagery does not reach |

## API

```python
from field_eda import (
    build_panel, coverage_strip, n1_ndvi_distribution, n2_ndvi_by_crop,
    n3_ndvi_climatology, c1_crop_composition, c2_dominant_crop_heatmap,
    c3_crop_share_diversity, w1_cumulative_gdd, x4_ndvi_climate,
    x5_area_vs_purity, m1_grower_map, save,
)

panel = build_panel()                 # 170 rows, needs DATA_PIPELINE_DATA_ROOT
print(panel.coverage())               # grower x layer field counts
strip = coverage_strip(panel)
save(strip, "coverage.png")           # writing is the caller's job
```

- `build_panel(growers=..., years=..., layers=..., data_root=..., season=...)`
  returns `FieldPanel(panel, composition, daily_weather, growers, years,
  data_root)`. Missing layers become nulls plus `notes`; nothing raises for a
  missing NDVI composite or a short CDL table.
- Every figure returns `FigureResult(figure_id, figure, stats, story,
  coverage_note, warnings)`. `stats` carries `n_fields` and `n_field_years`.
  The library never saves, prints, or assembles a report.
- `m1_grower_map` imports `lib/field_frame.py` and `lib/basemap.py` from the
  runtime (`<root>/data-pipeline/src/scripts/lib`) so EDA maps share the
  skill's frame and imagery conventions. Statistical figures have no runtime
  library dependency.

Run the reviewer example against the runtime:

```bash
export DATA_PIPELINE_DATA_ROOT=$HOME/my-farm-advisor-runtime
cd my-farm-advisor/eda/eda-field-comparison
$DATA_PIPELINE_DATA_ROOT/data-pipeline/.venv/bin/python examples/build_panel.py
```

Outputs go to
`${DATA_PIPELINE_DATA_ROOT}/data-pipeline/shared/eda/assignment-2/` and stay
uncommitted. `--no-maps` skips the basemap download; `--growers Illinois,Iowa`
limits the run.

## Semantics that must not drift

- **NDVI composites are annual means of every valid scene**, not temporal
  peaks. `ndvi_p95` is the greenest *area* of that annual mean, not the
  greenest *day*. Captions say "annual-mean NDVI".
- **Weather is coarse.** Fields inside a grower do **not** share one reading:
  Nebraska has three distinct daily series, Illinois four, Iowa four, Marvel
  one. The spread is small (roughly 20–90 GDD within a grower versus hundreds
  between growers). W2 must state the fraction, not the earlier "one POWER
  cell" shorthand.
- **CDL names come from `crop_code`** via `soil/cdl-cropland/src/cdl_codes.py`.
  The stored `crop_name` strings in the IL/IA tables predate the code fix and
  are wrong for several classes (for example 28 is Oats, 36 is Alfalfa); the
  panel never trusts them.
- **Growing season** defaults to day-of-year 60–304 (1 Mar – 31 Oct) for GDD
  and rainfall totals. Calendar-year weather stays in `daily_weather` for
  anyone who needs it.
- **Crop purity** is the dominant share among planted classes only. Grass,
  fallow, forest, and development stay in the composition figures but would
  otherwise make a small produce parcel look pure simply because it is ringed
  by grass.
- **Every NDVI figure carries its denominator.** N1/N2/N3 and X4 print the
  grower `n/N` footer; N3 also draws the field count as a bar strip below the
  climatology. A missing composite is a gap, never an interpolation.
- **N2 pools rare classes.** Classes with fewer than four field-years with
  imagery are drawn as "Other crop classes" with their n, not as a class
  estimate. One point is one field-year, and the caption says the repeats are
  not independent.
- **C3 shares are planted-only and renormalised to 100 %.** Non-crop covers are
  removed before averaging field-year shares; "Other planted" is the rest of
  the planted classes outside the top five. Shannon H uses the same planted
  subset as `crop_purity`.
- **X5 aggregates to one point per field.** Area is static, so the five
  year-rows are averaged instead of counted as 170 observations. The pooled ρ
  and the per-grower ρ are both printed because they disagree: the pooled
  positive value is mostly the Delaware-vs-Corn-Belt contrast, while within a
  grower the slope is flat or negative.
- **`NPK`/yield/economics are out of scope.** So are IL/IA label rewrites and
  any ingest beyond NDVI composites.

## Map conventions

- Single grower plates (IL/IA/NE) are north-up local UTM (`force_rot=0.0`):
  rotating a county-sized group to fit the page would leave north sideways for
  no gain. Metre axes, boxed north arrow, imagery under the field polygons.
- Elongated plans (Delaware: 1.5 km x 7.3 km of 5–12 ha parcels) use per-field
  small multiples, each with the field-local auto-rotated frame. One page
  frame would shrink the parcels to specks.
- CDL is masked to the field polygons exactly (no halo). Off-field land use is
  already visible in the imagery and would dominate a small-parcel legend.
