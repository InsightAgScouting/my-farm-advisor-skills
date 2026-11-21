"""Build the tidy field-year panel used by every EDA figure.

One row per ``(grower, farm, field, year)``. The four growers in the
Assignment 2 dataset give 34 fields x 5 years = 170 rows. Layers are
*additive*: a missing weather file, CDL table, or NDVI composite becomes a
null plus a note, never an exception. That is what lets one panel serve
categories with different coverage.

The subskill never reads ``soil/`` — SSURGO is out of scope by design.

Runtime layout (the documented data-pipeline contract)::

    ${DATA_PIPELINE_DATA_ROOT}/data-pipeline/
      growers/<grower>/farms/<farm>/boundary/field_boundaries.geojson
      growers/<grower>/farms/<farm>/derived/tables/*_cdl_<y0>_<y1>_full_composition.csv
      growers/<grower>/farms/<farm>/fields/<field>/weather/daily_weather.csv
      growers/<grower>/farms/<farm>/fields/<field>/derived/features/ndvi_year_<year>_composite.tif
      shared/cdl/rasters/CDL_<year>_<state_fips>.tif
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .metrics import (
    SEASON_END_DOY,
    SEASON_START_DOY,
    composition_metrics,
    ndvi_stats,
    season_summary,
    shape_metrics,
)

# Single-source CDL legend from the soil/cdl-cropland subtree. Importing the
# legend is deliberate: stored crop_name strings in the IL/IA tables predate
# the code fix and are wrong for a few classes, so names are rebuilt from
# crop_code and the stored string is never trusted.
_CDL_SRC = Path(__file__).resolve().parents[4] / "soil" / "cdl-cropland" / "src"
if str(_CDL_SRC) not in sys.path:
    sys.path.insert(0, str(_CDL_SRC))
from cdl_codes import CDL_CLASS_NAMES, ROTATION_EXCLUDED_CODES  # noqa: E402

METRIC_CRS = "EPSG:5070"  # CONUS Albers equal-area; areas in m²

CROP_GROUPS: dict[str, tuple[int, ...]] = {
    "Corn": (1, 12, 13),
    "Soybeans": (5,),
    "Double crop": (26, 254, 233),
    "Small grains": (21, 22, 23, 24, 25, 27, 28, 29, 30, 31),
    "Vegetables": (48, 49, 50, 51, 53, 54, 206, 207, 208, 216, 219, 222, 227),
    "Hay/Pasture": (36, 37, 176),
    "Fallow/Idle": (61,),
    "Forest": (63, 141, 142, 143, 152),
    "Developed": (82, 121, 122, 123, 124, 131),
    "Water/Wetland": (83, 87, 92, 111, 112, 190, 195),
}
_CODE_TO_GROUP = {code: name for name, codes in CROP_GROUPS.items() for code in codes}


def crop_group(code: int) -> str:
    """Broad crop group for a CDL class code (``Other`` when unmapped)."""
    return _CODE_TO_GROUP.get(int(code), "Other")


def crop_name(code: int) -> str:
    """CDL class name for a code, independent of any stored label."""
    return CDL_CLASS_NAMES.get(int(code), f"Class {int(code)}")


@dataclass(frozen=True)
class GrowerSpec:
    """A grower folder, its farm folder, and the label used in figures."""

    slug: str
    farm: str
    label: str
    state: str


DEFAULT_GROWERS: tuple[GrowerSpec, ...] = (
    GrowerSpec("illinois-grower", "illinois-farm", "Illinois", "IL"),
    GrowerSpec("iowa-grower", "iowa-farm", "Iowa", "IA"),
    GrowerSpec("nebraska-grower", "nebraska-farm", "Nebraska", "NE"),
    GrowerSpec("marvel-farms", "main-farm", "Delaware", "DE"),
)

DEFAULT_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
DEFAULT_LAYERS: tuple[str, ...] = ("boundaries", "cdl", "weather", "ndvi")


def default_data_root() -> Path:
    """``${DATA_PIPELINE_DATA_ROOT}/data-pipeline`` or a clear error."""
    raw = os.environ.get("DATA_PIPELINE_DATA_ROOT")
    if not raw:
        raise RuntimeError(
            "DATA_PIPELINE_DATA_ROOT is not set. Export it to the runtime root "
            "(for example $HOME/my-farm-advisor-runtime) before building a panel."
        )
    return Path(raw).expanduser() / "data-pipeline"


@dataclass
class FieldPanel:
    """Tidy panel plus the tables and coverage the figures need."""

    panel: pd.DataFrame
    composition: pd.DataFrame
    daily_weather: pd.DataFrame
    growers: tuple[GrowerSpec, ...]
    years: tuple[int, ...]
    data_root: Path

    @property
    def field_ids(self) -> list[str]:
        return sorted(self.panel["field_id"].unique())

    def grower(self, label: str) -> GrowerSpec:
        for spec in self.growers:
            if spec.label == label:
                return spec
        raise KeyError(f"Unknown grower label {label!r}; have {[g.label for g in self.growers]}")

    def coverage(self) -> pd.DataFrame:
        """Field counts per grower and layer (the coverage strip's source)."""
        rows = []
        for spec in self.growers:
            sub = self.panel[self.panel["grower"] == spec.label]
            n_fields = sub["field_id"].nunique()
            if n_fields == 0:
                continue
            per_field = sub.groupby("field_id")
            checks = {
                "boundaries": pd.Series(True, index=per_field.size().index),
                "CDL": per_field["crop_code"].apply(lambda s: s.notna().any()),
                "weather": per_field["gdd_total"].apply(lambda s: s.notna().any()),
                "NDVI": per_field["ndvi_mean"].apply(lambda s: s.notna().any()),
            }
            for layer, present in checks.items():
                n = int(present.sum())
                rows.append(
                    {
                        "grower": spec.label,
                        "layer": layer,
                        "n_fields": n,
                        "n_total": n_fields,
                        "pct": 100.0 * n / n_fields,
                    }
                )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _farm_dir(data_root: Path, spec: GrowerSpec) -> Path:
    return data_root / "growers" / spec.slug / "farms" / spec.farm


def _load_boundaries(farm_dir: Path) -> gpd.GeoDataFrame:
    path = farm_dir / "boundary" / "field_boundaries.geojson"
    if not path.exists():
        return gpd.GeoDataFrame(columns=["field_id", "geometry"], crs="EPSG:4326")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _load_composition(farm_dir: Path, years: tuple[int, ...]) -> pd.DataFrame:
    """Farm CDL composition for the requested window, or an empty frame."""
    tables = farm_dir / "derived" / "tables"
    window = f"{min(years)}_{max(years)}"
    candidates = sorted(tables.glob(f"*_cdl_{window}_full_composition.csv"))
    if not candidates:  # fall back to annual tables with the same schema
        parts = [tables / f"*_{year}_cdl.csv" for year in years]
        frames = [pd.read_csv(p) for pattern in parts for p in sorted(tables.glob(pattern.name))]
        if not frames:
            return pd.DataFrame(
                columns=["field_id", "year", "crop_code", "pct", "crop", "crop_group", "is_rotation_crop"]
            )
        return _normalize_composition(pd.concat(frames, ignore_index=True), years)
    return _normalize_composition(pd.read_csv(candidates[0]), years)


def _normalize_composition(raw: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["field_id", "year", "crop_code", "pct", "crop", "crop_group", "is_rotation_crop"]
        )
    df = raw[raw["year"].isin(years)].copy()
    df["crop_code"] = pd.to_numeric(df["crop_code"], errors="coerce")
    df = df[df["crop_code"].notna()].copy()
    df["crop_code"] = df["crop_code"].astype(int)
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce").fillna(0.0)
    df["crop"] = df["crop_code"].map(crop_name)
    df["crop_group"] = df["crop_code"].map(crop_group)
    df["is_rotation_crop"] = ~df["crop_code"].isin(ROTATION_EXCLUDED_CODES)
    return df.sort_values(["field_id", "year", "pct"], ascending=[True, True, False]).reset_index(drop=True)


def _load_weather(farm_dir: Path, field_id: str) -> pd.DataFrame:
    path = farm_dir / "fields" / field_id / "weather" / "daily_weather.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


def _ndvi_path(farm_dir: Path, field_id: str, year: int) -> Path:
    return (
        farm_dir
        / "fields"
        / field_id
        / "derived"
        / "features"
        / f"ndvi_year_{year}_composite.tif"
    )


def _weather_series_id(daily: pd.DataFrame) -> str:
    """Short fingerprint of a field's daily T2M + precipitation series.

    Two fields with the same id share every weather value in the cache. This
    is how the panel knows that intra-grower weather differences are a coarse
    grid artefact rather than independent measurements.
    """
    if daily.empty:
        return ""
    arr = np.column_stack(
        [
            pd.to_numeric(daily.sort_values("date")["T2M"], errors="coerce").to_numpy(),
            pd.to_numeric(daily.sort_values("date")["PRECTOTCORR"], errors="coerce").to_numpy(),
        ]
    )
    return hashlib.md5(np.round(arr, 4).tobytes()).hexdigest()[:8]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_panel(
    growers: tuple[GrowerSpec, ...] | list[GrowerSpec] | None = None,
    years: tuple[int, ...] | list[int] = DEFAULT_YEARS,
    layers: tuple[str, ...] | list[str] = DEFAULT_LAYERS,
    data_root: str | Path | None = None,
    season: tuple[int, int] = (SEASON_START_DOY, SEASON_END_DOY),
) -> FieldPanel:
    """Build the field-year panel from the runtime cache.

    Parameters
    ----------
    growers
        Defaults to the four Assignment 2 growers.
    years
        Calendar years to include. The IL/IA tables also hold 2019–2020; the
        default window matches the weather and NDVI cache (2021–2025).
    layers
        Any of ``boundaries``, ``cdl``, ``weather``, ``ndvi``. Passing a
        smaller set is for focused analyses and tests. ``boundaries`` is
        always read because it defines the rows.
    data_root
        Runtime base; defaults to ``${DATA_PIPELINE_DATA_ROOT}/data-pipeline``.
    season
        Growing-season day-of-year window for weather totals.
    """
    root = Path(data_root).expanduser() if data_root else default_data_root()
    specs = tuple(growers) if growers else DEFAULT_GROWERS
    years = tuple(int(y) for y in years)
    layers = tuple(layers)
    if "boundaries" not in layers:
        raise ValueError("'boundaries' is required: it defines the panel rows")
    start_doy, end_doy = season

    composition_frames: list[pd.DataFrame] = []
    weather_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    notes_by_field: dict[str, list[str]] = {}

    for spec in specs:
        farm_dir = _farm_dir(root, spec)
        if not farm_dir.exists():
            raise FileNotFoundError(
                f"Farm folder not found for {spec.label}: {farm_dir}. "
                "Check DATA_PIPELINE_DATA_ROOT and the grower/farm slugs."
            )
        boundaries = _load_boundaries(farm_dir)
        if boundaries.empty:
            continue
        metrics = boundaries.to_crs(METRIC_CRS)
        centroid_wgs = gpd.GeoSeries(metrics.geometry.centroid, crs=METRIC_CRS).to_crs(
            "EPSG:4326"
        )

        comp = _load_composition(farm_dir, years) if "cdl" in layers else pd.DataFrame()
        if not comp.empty:
            composition_frames.append(comp)

        for pos, (_, boundary) in enumerate(boundaries.iterrows()):
            field_id = str(boundary["field_id"])
            notes: list[str] = []
            shape = shape_metrics(metrics.geometry.iloc[pos])
            lat = float(centroid_wgs.iloc[pos].y)
            lon = float(centroid_wgs.iloc[pos].x)
            county = boundary.get("county")

            daily = _load_weather(farm_dir, field_id) if "weather" in layers else pd.DataFrame()
            weather_by_year: dict[int, dict] = {}
            if daily.empty:
                notes.append("weather: no cached daily file")
                series_id = ""
            else:
                daily = daily.assign(field_id=field_id)
                daily["year"] = pd.to_datetime(daily["date"]).dt.year
                daily = daily[daily["year"].isin(years)]
                weather_frames.append(daily)
                series_id = _weather_series_id(daily)
                summary = season_summary(daily, start_doy=start_doy, end_doy=end_doy)
                weather_by_year = {int(r["year"]): r for _, r in summary.iterrows()}

            field_comp = comp[comp["field_id"] == field_id] if not comp.empty else pd.DataFrame()
            if "cdl" in layers and field_comp.empty:
                notes.append("CDL: no composition rows")
            elif "cdl" in layers and field_comp["year"].nunique() < len(years):
                missing = sorted(set(years) - set(field_comp["year"].unique()))
                notes.append(f"CDL: no rows for {', '.join(map(str, missing))}")

            ndvi_paths = (
                {year: _ndvi_path(farm_dir, field_id, year) for year in years}
                if "ndvi" in layers
                else {}
            )
            missing_ndvi = [year for year, path in ndvi_paths.items() if not path.exists()]
            if "ndvi" in layers and missing_ndvi:
                if len(missing_ndvi) == len(years):
                    notes.append("NDVI: no composites for any year")
                else:
                    notes.append("NDVI: no composite for " + ", ".join(map(str, missing_ndvi)))

            for year in years:
                year_comp = field_comp[field_comp["year"] == year] if not field_comp.empty else pd.DataFrame()
                cdl = composition_metrics(year_comp)
                weather = weather_by_year.get(year, {})
                row: dict[str, object] = {
                    "grower": spec.label,
                    "state": spec.state,
                    "farm": spec.farm,
                    "field_id": field_id,
                    "county": county,
                    "year": year,
                    "lat": lat,
                    "lon": lon,
                    "weather_series": series_id,
                    **shape,
                    "crop_code": cdl["crop_code"],
                    "crop": cdl["crop"],
                    "crop_group": cdl["crop_group"],
                    "dominant_pct": cdl["dominant_pct"],
                    "crop_purity": cdl["crop_purity"],
                    "n_crop_classes": cdl["n_crop_classes"],
                    "shannon": cdl["shannon"],
                    "shannon_evenness": cdl["shannon_evenness"],
                }
                for key in (
                    "gdd_total",
                    "precip_mm",
                    "tmean_c",
                    "tmax_mean_c",
                    "tmin_mean_c",
                    "solar_mj_m2",
                    "days",
                    "last_spring_frost_doy",
                    "first_fall_frost_doy",
                    "frost_free_days",
                ):
                    row[key] = weather.get(key, math.nan)

                ndvi_path = ndvi_paths.get(year)
                if ndvi_path is not None and ndvi_path.exists():
                    row.update(ndvi_stats(ndvi_path))
                    row["has_ndvi"] = True
                    row["ndvi_path"] = str(ndvi_path)
                else:
                    row.update(
                        {
                            "ndvi_mean": math.nan,
                            "ndvi_p05": math.nan,
                            "ndvi_p50": math.nan,
                            "ndvi_p95": math.nan,
                            "ndvi_valid_frac": math.nan,
                            "ndvi_pixels": 0,
                            "has_ndvi": False,
                            "ndvi_path": None,
                        }
                    )
                rows.append(row)
            notes_by_field[field_id] = notes

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("Panel is empty: no boundaries matched the requested growers.")
    panel["notes"] = panel["field_id"].map(
        lambda fid: "; ".join(dict.fromkeys(notes_by_field.get(fid, [])))
    )
    panel = panel.sort_values(["grower", "field_id", "year"]).reset_index(drop=True)

    composition = (
        pd.concat(composition_frames, ignore_index=True)
        if composition_frames
        else pd.DataFrame(
            columns=["field_id", "year", "crop_code", "pct", "crop", "crop_group", "is_rotation_crop"]
        )
    )
    daily_weather = (
        pd.concat(weather_frames, ignore_index=True)
        if weather_frames
        else pd.DataFrame(columns=["field_id", "date", "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"])
    )
    for frame in (composition, daily_weather):
        if "grower" not in frame.columns and not frame.empty:
            lookup = panel[["field_id", "grower"]].drop_duplicates()
            frame["grower"] = frame["field_id"].map(lookup.set_index("field_id")["grower"])

    return FieldPanel(
        panel=panel,
        composition=composition,
        daily_weather=daily_weather,
        growers=specs,
        years=years,
        data_root=root,
    )
