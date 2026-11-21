"""Reusable metrics for the field EDA subskill.

Every function is pure: it takes a dataframe, a geometry, or a raster path and
returns numbers. No file is written and no figure is drawn here.

Conventions
-----------
* GDD is the daily-mean method on the cached POWER record:
  ``max(0, (T2M_MAX + T2M_MIN) / 2 - base)``. Base temperature is 10 °C.
* The default *season* is day-of-year 60–304 (1 March – 31 October). The
  cached weather files are calendar years; restricting to the season keeps
  winter chill out of "growing season" totals. Every function takes the
  window as an argument so a caller can widen it explicitly.
* NDVI composites on disk are **annual means of every valid scene**, not
  temporal peaks. ``ndvi_stats`` therefore reports the spatial distribution
  of that annual mean: p95 is the field's greenest *area*, not its greenest
  *day*. Figure captions must say "annual mean NDVI", never "peak".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

GDD_BASE_C = 10.0
SEASON_START_DOY = 60
SEASON_END_DOY = 304


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------
def prepare_daily_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Add year / doy / daily GDD to a cached ``daily_weather.csv`` frame."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["year"] = out["date"].dt.year
    out["doy"] = out["date"].dt.dayofyear
    tmean = (out["T2M_MAX"] + out["T2M_MIN"]) / 2.0
    out["gdd"] = (tmean - GDD_BASE_C).clip(lower=0.0)
    return out


def cumulative_gdd(
    daily: pd.DataFrame,
    start_doy: int = SEASON_START_DOY,
    end_doy: int = SEASON_END_DOY,
) -> pd.DataFrame:
    """Cumulative GDD by day-of-year, one row per field-year-day in season."""
    df = prepare_daily_weather(daily)
    season = df[(df["doy"] >= start_doy) & (df["doy"] <= end_doy)].copy()
    season["gdd_cum"] = season.groupby("year")["gdd"].cumsum()
    return season


def frost_windows(daily: pd.DataFrame) -> dict[str, float]:
    """Median last spring freeze DOY and first autumn freeze DOY.

    A freeze is ``T2M_MIN <= 0``. Spring is day-of-year <= 182, autumn after.
    Returns NaN when a year never freezes or the record is too short.
    """
    df = prepare_daily_weather(daily)
    last_spring: list[int] = []
    first_fall: list[int] = []
    for _, g in df.groupby("year"):
        frozen = g[g["T2M_MIN"] <= 0.0]
        if frozen.empty:
            continue
        spring = frozen[frozen["doy"] <= 182]
        autumn = frozen[frozen["doy"] > 182]
        if not spring.empty:
            last_spring.append(int(spring["doy"].max()))
        if not autumn.empty:
            first_fall.append(int(autumn["doy"].min()))
    return {
        "last_spring_frost_doy": float(np.median(last_spring)) if last_spring else math.nan,
        "first_fall_frost_doy": float(np.median(first_fall)) if first_fall else math.nan,
    }


def season_summary(
    daily: pd.DataFrame,
    start_doy: int = SEASON_START_DOY,
    end_doy: int = SEASON_END_DOY,
) -> pd.DataFrame:
    """One row per calendar year: GDD, rainfall, means, frost windows.

    ``days`` counts rows inside the season window, so a short record is
    visible instead of silently averaging over fewer days.
    """
    df = prepare_daily_weather(daily)
    frost = frost_windows(df)
    season = df[(df["doy"] >= start_doy) & (df["doy"] <= end_doy)]
    rows = []
    for year, g in season.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "gdd_total": float(g["gdd"].sum()),
                "precip_mm": float(g["PRECTOTCORR"].sum()),
                "tmean_c": float(g["T2M"].mean()),
                "tmax_mean_c": float(g["T2M_MAX"].mean()),
                "tmin_mean_c": float(g["T2M_MIN"].mean()),
                "solar_mj_m2": float(g["ALLSKY_SFC_SW_DWN"].sum()),
                "days": int(len(g)),
                **frost,
            }
        )
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    out["frost_free_days"] = out["first_fall_frost_doy"] - out["last_spring_frost_doy"]
    return out


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------
def shape_metrics(projected_geometry) -> dict[str, float]:
    """Area, perimeter and two compactness measures from a metric geometry.

    ``compactness`` is the Polsby-Popper score (4πA/P²; 1 = circle).
    ``rect_fill`` is A / area of the minimum rotated rectangle, which stays
    meaningful for the long thin strips common in Midwest section lines.
    """
    area_m2 = float(projected_geometry.area)
    perim_m = float(projected_geometry.length)
    compactness = (
        4.0 * math.pi * area_m2 / (perim_m**2) if perim_m > 0 else math.nan
    )
    mrr = projected_geometry.minimum_rotated_rectangle
    rect_fill = area_m2 / float(mrr.area) if mrr.area > 0 else math.nan
    return {
        "area_ha": area_m2 / 10_000.0,
        "area_acres": area_m2 / 4046.8564224,
        "perimeter_m": perim_m,
        "compactness": compactness,
        "rect_fill": rect_fill,
    }


# --------------------------------------------------------------------------
# CDL composition
# --------------------------------------------------------------------------
def shannon_diversity(percentages: pd.Series) -> float:
    """Shannon H over class shares (percent values accepted, any total)."""
    values = pd.to_numeric(percentages, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return math.nan
    p = values / values.sum()
    return float(-(p * np.log(p)).sum())


def composition_metrics(comp: pd.DataFrame) -> dict[str, object]:
    """Dominant class and diversity for one field-year composition table.

    ``crop_purity`` is the dominant share among *planted* classes only
    (``is_rotation_crop``). Non-crop covers — grass, fallow, forest, water —
    stay in the composition figures but would otherwise make a small produce
    parcel look perfectly pure just because it is ringed by grass.
    """
    if comp.empty:
        return {
            "crop_code": math.nan,
            "crop": None,
            "crop_group": None,
            "dominant_pct": math.nan,
            "crop_purity": math.nan,
            "n_crop_classes": 0,
            "shannon": math.nan,
            "shannon_evenness": math.nan,
        }
    total = comp["pct"].sum()
    dominant = comp.loc[comp["pct"].idxmax()]
    planted = comp[comp.get("is_rotation_crop", True)]
    planted_pct = float(planted["pct"].max()) if not planted.empty else math.nan
    h = shannon_diversity(planted["pct"])
    n_planted = int(planted["crop_code"].nunique())
    return {
        "crop_code": int(dominant["crop_code"]),
        "crop": dominant["crop"],
        "crop_group": dominant["crop_group"],
        "dominant_pct": float(100.0 * dominant["pct"] / total) if total else math.nan,
        "crop_purity": planted_pct,
        "n_crop_classes": n_planted,
        "shannon": h,
        "shannon_evenness": (h / math.log(n_planted)) if n_planted > 1 else 0.0,
    }


# --------------------------------------------------------------------------
# NDVI rasters
# --------------------------------------------------------------------------
def ndvi_values(path: str | Path) -> np.ndarray:
    """Valid pixels of one annual-mean NDVI composite, as a 1-D float array."""
    import rasterio

    with rasterio.open(path) as ds:
        band = ds.read(1, masked=True)
    return np.asarray(band.compressed(), dtype="float64")


def ndvi_stats(path: str | Path) -> dict[str, float]:
    """Spatial summary of one annual-mean NDVI composite."""
    import rasterio

    with rasterio.open(path) as ds:
        band = ds.read(1, masked=True)
        total = int(ds.width * ds.height)
    values = np.asarray(band.compressed(), dtype="float64")
    if values.size == 0:
        return {
            "ndvi_mean": math.nan,
            "ndvi_p05": math.nan,
            "ndvi_p50": math.nan,
            "ndvi_p95": math.nan,
            "ndvi_valid_frac": 0.0,
            "ndvi_pixels": 0,
        }
    return {
        "ndvi_mean": float(values.mean()),
        "ndvi_p05": float(np.percentile(values, 5)),
        "ndvi_p50": float(np.percentile(values, 50)),
        "ndvi_p95": float(np.percentile(values, 95)),
        "ndvi_valid_frac": float(values.size / total) if total else 0.0,
        "ndvi_pixels": int(values.size),
    }
