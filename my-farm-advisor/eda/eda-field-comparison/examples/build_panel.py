#!/usr/bin/env python3
"""Stage A/B/C example: build the field-year panel and render its figures.

Run against the Assignment 2 runtime::

    export DATA_PIPELINE_DATA_ROOT=$HOME/my-farm-advisor-runtime
    <runtime-venv>/bin/python examples/build_panel.py

Nothing here belongs to the reusable subskill. The script is the reviewer's
handle: it prints panel shape, coverage, and per-figure stats, and writes PNGs
to ``<runtime>/data-pipeline/shared/eda/assignment-2/`` (never committed).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from field_eda import (  # noqa: E402
    DEFAULT_GROWERS,
    a1_field_acreage,
    build_panel,
    c1_crop_composition,
    c2_dominant_crop_heatmap,
    c3_crop_share_diversity,
    coverage_strip,
    m1_grower_map,
    m2_crop_flip_map,
    m3_ndvi_map,
    n1_ndvi_distribution,
    n2_ndvi_by_crop,
    n3_ndvi_climatology,
    save,
    w1_cumulative_gdd,
    w2_within_grower_weather,
    w3_monthly_climatology,
    x4_ndvi_climate,
    x5_area_vs_purity,
    x6_gdd_rainfall,
    x7_area_vs_diversity,
)
from field_eda.style import GROWER_ORDER  # noqa: E402


def _default_out() -> Path:
    root = os.environ.get("DATA_PIPELINE_DATA_ROOT")
    if not root:
        raise SystemExit("DATA_PIPELINE_DATA_ROOT is not set")
    return Path(root).expanduser() / "data-pipeline" / "shared" / "eda" / "assignment-2"


def _print_result(result) -> None:
    print(f"\n[{result.figure_id}] {result.story}")
    scalars = {k: v for k, v in result.stats.items() if isinstance(v, (int, float, str))}
    print("  stats:", scalars)
    if result.coverage_note:
        print("  coverage:", result.coverage_note)
    for warning in result.warnings:
        print("  WARNING:", warning)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Output folder (runtime, uncommitted)")
    parser.add_argument("--growers", default=",".join(g.label for g in DEFAULT_GROWERS))
    parser.add_argument("--no-maps", action="store_true", help="Skip M1/M2/M3 (no basemap download)")
    parser.add_argument("--no-figures", action="store_true", help="Panel/QA only")
    args = parser.parse_args()

    wanted = [g.strip() for g in args.growers.split(",") if g.strip()]
    specs = tuple(g for g in DEFAULT_GROWERS if g.label in wanted)
    if not specs:
        raise SystemExit(f"No known growers in {wanted}")

    out = args.out or _default_out()
    out.mkdir(parents=True, exist_ok=True)

    panel = build_panel(growers=specs)
    frame = panel.panel
    print("Panel:", frame.shape[0], "rows x", frame.shape[1], "columns")
    print("Growers:", ", ".join(sorted(frame["grower"].unique())))
    print("Fields:", frame["field_id"].nunique(), "| field-years:", len(frame))
    print("Years:", ", ".join(map(str, panel.years)))
    print("\nCoverage (fields with data / fields):")
    print(panel.coverage().pivot(index="grower", columns="layer", values="n_fields").to_string())
    notes = frame.loc[frame["notes"].fillna("") != "", ["field_id", "notes"]].drop_duplicates()
    if not notes.empty:
        print("\nNotes recorded in the panel (first 10):")
        for row in notes.head(10).itertuples():
            print(f"  {row.field_id}: {row.notes}")

    frame.to_csv(out / "panel.csv", index=False)
    panel.composition.to_csv(out / "composition.csv", index=False)
    panel.coverage().to_csv(out / "coverage.csv", index=False)
    print(f"\nWrote panel.csv, composition.csv, coverage.csv to {out}")

    if args.no_figures:
        return 0

    results = [coverage_strip(panel)]
    results.append(a1_field_acreage(panel))
    n1_field = (
        frame[frame["grower"] == "Illinois"].sort_values("field_id")["field_id"].iloc[0]
        if "Illinois" in set(frame["grower"])
        else frame.sort_values("field_id")["field_id"].iloc[0]
    )
    results.append(n1_ndvi_distribution(panel, field_id=n1_field))
    for grower in GROWER_ORDER:
        if grower in set(frame["grower"]):
            results.append(c1_crop_composition(panel, grower))
    results.append(w1_cumulative_gdd(panel))
    results.append(n2_ndvi_by_crop(panel))
    results.append(n3_ndvi_climatology(panel))
    results.append(x4_ndvi_climate(panel))
    results.append(c2_dominant_crop_heatmap(panel))
    results.append(c3_crop_share_diversity(panel))
    results.append(x5_area_vs_purity(panel))
    results.append(w2_within_grower_weather(panel))
    results.append(w3_monthly_climatology(panel))
    results.append(x6_gdd_rainfall(panel))
    results.append(x7_area_vs_diversity(panel))
    if not args.no_maps:
        for grower in GROWER_ORDER:
            if grower in set(frame["grower"]):
                results.append(m1_grower_map(panel, grower))
                results.append(m2_crop_flip_map(panel, grower))
                results.append(m3_ndvi_map(panel, grower))

    print()
    for result in results:
        name = f"{result.figure_id}_{result.stats.get('grower', 'all')}".lower().replace(" ", "-")
        if result.figure_id == "N1":
            name = f"n1_{result.stats['field_id']}"
        path = save(result, str(out / f"{name}.png"))
        _print_result(result)
        print("  wrote:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
