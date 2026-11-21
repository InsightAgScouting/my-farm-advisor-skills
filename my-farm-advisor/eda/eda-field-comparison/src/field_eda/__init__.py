"""Field-level EDA for cross-grower comparison (Assignment 2 subskill).

Public surface::

    from field_eda import build_panel, coverage_strip, n1_ndvi_distribution

The panel is the contract. Figures and maps consume it and return
``FigureResult`` objects; nothing in this package writes files or assembles a
report. See ``GUIDE.md`` for the intended chapter order and coverage caveats.
"""

from .figures import (
    FigureResult,
    c1_crop_composition,
    c2_dominant_crop_heatmap,
    c3_crop_share_diversity,
    coverage_strip,
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
)
from .maps import m1_grower_map, m2_crop_flip_map, m3_ndvi_map
from .metrics import (
    cumulative_gdd,
    frost_windows,
    ndvi_stats,
    season_summary,
    shape_metrics,
)
from .panel import (
    DEFAULT_GROWERS,
    DEFAULT_LAYERS,
    DEFAULT_YEARS,
    FieldPanel,
    GrowerSpec,
    build_panel,
    crop_group,
    crop_name,
)

__all__ = [
    "DEFAULT_GROWERS",
    "DEFAULT_LAYERS",
    "DEFAULT_YEARS",
    "FieldPanel",
    "FigureResult",
    "GrowerSpec",
    "build_panel",
    "c1_crop_composition",
    "c2_dominant_crop_heatmap",
    "c3_crop_share_diversity",
    "coverage_strip",
    "crop_group",
    "crop_name",
    "cumulative_gdd",
    "frost_windows",
    "m1_grower_map",
    "m2_crop_flip_map",
    "m3_ndvi_map",
    "n1_ndvi_distribution",
    "n2_ndvi_by_crop",
    "n3_ndvi_climatology",
    "ndvi_stats",
    "save",
    "season_summary",
    "shape_metrics",
    "w1_cumulative_gdd",
    "w2_within_grower_weather",
    "w3_monthly_climatology",
    "x4_ndvi_climate",
    "x5_area_vs_purity",
    "x6_gdd_rainfall",
]
