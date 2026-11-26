from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

try:  # package import (``import cdl_cropland.cdl_reporting``)
    from .cdl_codes import CDL_CODES, ROTATION_EXCLUDED_CODES
except ImportError:  # flat import (skill ``src`` directory on sys.path)
    from cdl_codes import CDL_CODES, ROTATION_EXCLUDED_CODES

# A field-year only counts as rotation evidence when its dominant crop class
# covers a meaningful share of the pixels that actually fall inside the field.
# 30 m CDL against a 6 ha produce parcel is 60-70 pixels, so a handful of
# misclassified pixels must not drive a "high" rotation confidence.
MIN_ROTATION_DOMINANT_PCT = 10.0
MIN_ROTATION_DOMINANT_PIXELS = 3
HIGH_CONFIDENCE_MIN_YEARS = 5
MEDIUM_CONFIDENCE_MIN_YEARS = 3
PATCHY_DOMINANT_PCT = 30.0
CONTEXTUAL_DOMINANT_PCT = 50.0

CONTEXT_OPERATIONAL = "cdl-operational"
CONTEXT_CONTEXTUAL = "cdl-contextual"


def filter_cdl_categories(crop_mix: pd.DataFrame) -> pd.DataFrame:
    if crop_mix.empty:
        return crop_mix.copy()
    df = crop_mix.copy()
    if "crop_name" not in df.columns:
        return df
    crop_names = pd.Series(df["crop_name"], dtype="object").fillna("").astype(str).str.strip()
    valid_mask = ~crop_names.eq("No Data") & ~crop_names.str.startswith("Code_")
    return df.loc[valid_mask].copy()


def filter_crop_history_window(crop_mix: pd.DataFrame, window_years: int = 5) -> pd.DataFrame:
    if crop_mix.empty or window_years <= 0 or "year" not in crop_mix.columns:
        return crop_mix.copy()
    df = crop_mix.copy()
    years = pd.Series(pd.to_numeric(df["year"], errors="coerce"), dtype="float64")
    if years.isna().all():
        return df
    max_year = int(years.dropna().max())
    min_year = max_year - window_years + 1
    return df.loc[years >= min_year].copy()


def _predict_next_crop(crop_names: list[str], current_crop: str) -> tuple[str | None, str]:
    if not crop_names:
        return None, "none"
    followers = [
        crop_names[idx + 1]
        for idx, crop_name in enumerate(crop_names[:-1])
        if crop_name == current_crop and idx + 1 < len(crop_names)
    ]
    if not followers:
        if len(crop_names) >= 2 and crop_names[-1] != crop_names[-2]:
            return crop_names[-2], "low"
        return current_crop, "low"
    counts = Counter(followers)
    best_count = max(counts.values())
    best_choices = sorted(name for name, count in counts.items() if count == best_count)
    choice = best_choices[0]
    confidence = "high" if best_count >= 2 else "medium"
    return choice, confidence


_CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_CONFIDENCE_BY_RANK = {rank: label for label, rank in _CONFIDENCE_RANK.items()}


def _rotation_evidence_caps(evidence: dict[str, object]) -> tuple[int, list[str]]:
    """Return the maximum defensible confidence rank and the reasons for it.

    The transition heuristic only looks at crop sequences. It cannot know that
    a sequence came from two usable years or from a handful of 30 m pixels
    inside a small produce parcel, so the evidence caps are applied on top.
    """
    usable_years = int(evidence.get("usable_years") or 0)
    reasons: list[str] = []
    cap_rank = _CONFIDENCE_RANK["high"]

    if usable_years <= 0:
        cap_rank = _CONFIDENCE_RANK["none"]
        reasons.append("no year has usable CDL crop history")
    elif usable_years == 1:
        cap_rank = _CONFIDENCE_RANK["none"]
        reasons.append("only 1 year of usable CDL crop history is available")
    elif usable_years < MEDIUM_CONFIDENCE_MIN_YEARS:
        cap_rank = _CONFIDENCE_RANK["low"]
        reasons.append(f"only {usable_years} years of usable CDL crop history are available")
    elif usable_years < HIGH_CONFIDENCE_MIN_YEARS:
        cap_rank = _CONFIDENCE_RANK["medium"]
        reasons.append(f"only {usable_years} years of usable CDL crop history are available")

    median_pct = evidence.get("median_dominant_pct")
    if median_pct is not None and float(median_pct) < PATCHY_DOMINANT_PCT:
        if cap_rank > _CONFIDENCE_RANK["low"]:
            cap_rank = _CONFIDENCE_RANK["low"]
        reasons.append(
            "the dominant CDL class covers a median "
            f"{float(median_pct):.0f}% of the field, so parcel-level CDL support is patchy"
        )

    if evidence.get("latest_year_usable") is False and cap_rank > _CONFIDENCE_RANK["medium"]:
        cap_rank = _CONFIDENCE_RANK["medium"]
        reasons.append("the most recent year has no usable CDL crop class")

    return cap_rank, reasons


def _rotation_context(usable_years: int, median_pct: float | None) -> str:
    if median_pct is None or usable_years < MEDIUM_CONFIDENCE_MIN_YEARS:
        return CONTEXT_CONTEXTUAL
    if float(median_pct) < CONTEXTUAL_DOMINANT_PCT:
        return CONTEXT_CONTEXTUAL
    return CONTEXT_OPERATIONAL


def build_crop_rotation_outlook(
    crop_names: list[str], *, evidence: dict[str, object] | None = None
) -> dict[str, str]:
    """Build the heuristic outlook for a crop-only sequence.

    Args:
        crop_names: Dominant crop classes per year, oldest first, with
            non-crop land cover already removed.
        evidence: Optional field-level evidence used to cap confidence, with
            keys ``usable_years``, ``total_years``, ``median_dominant_pct`` and
            ``latest_year_usable``. When omitted the historical
            sequence-only behaviour is preserved.
    """
    if not crop_names:
        return {
            "predicted_next_crop": "Unknown",
            "predicted_following_crop": "Unknown",
            "rotation_confidence": "none",
            "rotation_outlook": "Insufficient crop history for a heuristic rotation outlook.",
            "rotation_context": CONTEXT_CONTEXTUAL,
        }

    current_crop = crop_names[-1]
    next_crop, next_confidence = _predict_next_crop(crop_names, current_crop)
    following_crop, following_confidence = _predict_next_crop(
        crop_names + ([next_crop] if next_crop else []), next_crop or current_crop
    )
    combined_confidence = min(
        _CONFIDENCE_RANK[next_confidence], _CONFIDENCE_RANK[following_confidence]
    )

    cap_rank = _CONFIDENCE_RANK["high"]
    reasons: list[str] = []
    if evidence is not None:
        cap_rank, reasons = _rotation_evidence_caps(evidence)
    confidence_rank = min(combined_confidence, cap_rank)
    confidence_label = _CONFIDENCE_BY_RANK[confidence_rank]

    next_label = next_crop or "Unknown"
    following_label = following_crop or next_label
    reason_text = "; ".join(reasons)
    if confidence_label == "none":
        outlook = "Insufficient crop history for a heuristic rotation outlook"
        outlook = f"{outlook}: {reason_text}." if reason_text else f"{outlook}."
    elif confidence_label == "high":
        outlook = (
            f"Heuristic outlook: {next_label} next, then {following_label}, "
            "based on the recent rotation pattern."
        )
    else:
        detail = reason_text or "the recent sequence is short or noisy"
        outlook = (
            f"Heuristic outlook: {next_label} next, then {following_label}; "
            f"confidence is {confidence_label} because {detail}."
        )

    median_pct = None if evidence is None else evidence.get("median_dominant_pct")
    usable_years = 0 if evidence is None else int(evidence.get("usable_years") or 0)

    return {
        "predicted_next_crop": next_label,
        "predicted_following_crop": following_label,
        "rotation_confidence": confidence_label,
        "rotation_outlook": outlook,
        "rotation_context": _rotation_context(usable_years, median_pct),
    }


def extract_crop_composition(
    fields: gpd.GeoDataFrame, cdl_path: str | Path, year: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with rasterio.open(cdl_path) as src:
        fields_proj = fields.to_crs(src.crs)
        for _, field in fields_proj.iterrows():
            stats = zonal_stats(field.geometry, cdl_path, categorical=True)
            counts = stats[0] if stats else {}
            total = int(sum(counts.values()))
            if total <= 0:
                rows.append(
                    {
                        "field_id": field["field_id"],
                        "year": year,
                        "crop_code": 0,
                        "crop_name": "No Data",
                        "pixel_count": 0,
                        "pct": 0.0,
                    }
                )
                continue
            for crop_code, pixel_count in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            ):
                rows.append(
                    {
                        "field_id": field["field_id"],
                        "year": year,
                        "crop_code": int(crop_code),
                        "crop_name": CDL_CODES.get(int(crop_code), f"Code_{crop_code}"),
                        "pixel_count": int(pixel_count),
                        "pct": round(float(pixel_count) / total * 100.0, 2),
                    }
                )
    return filter_cdl_categories(pd.DataFrame(rows))


def _pct_col(df: pd.DataFrame) -> str:
    if "pct" in df.columns:
        return "pct"
    if "dominant_pct" in df.columns:
        return "dominant_pct"
    return "pct"


def _usable_rotation_rows(
    rows: list[dict[str, object]], pct_col: str | None
) -> list[dict[str, object]]:
    """Filter field-year rows down to usable crop-rotation evidence."""
    usable: list[dict[str, object]] = []
    for row in rows:
        try:
            crop_code = int(row.get("crop_code", 0))
        except (TypeError, ValueError):
            continue
        if crop_code in ROTATION_EXCLUDED_CODES:
            continue
        if pct_col is not None:
            pct_value = row.get(pct_col)
            if pct_value is None or float(pct_value) < MIN_ROTATION_DOMINANT_PCT:
                continue
        pixel_value = row.get("pixel_count")
        if pixel_value is not None and float(pixel_value) < MIN_ROTATION_DOMINANT_PIXELS:
            continue
        usable.append(row)
    return usable


def summarize_crop_history(crop_mix: pd.DataFrame, window_years: int = 5) -> pd.DataFrame:
    if crop_mix.empty:
        return pd.DataFrame()
    crop_mix = filter_cdl_categories(crop_mix)
    if crop_mix.empty:
        return pd.DataFrame()
    crop_mix = filter_crop_history_window(crop_mix, window_years=window_years)
    pct = _pct_col(crop_mix)
    if pct in crop_mix.columns:
        dominant = crop_mix.sort_values(["field_id", "year", pct], ascending=[True, True, False])
        dominant = dominant.groupby(["field_id", "year"], as_index=False).first()
        pct_col: str | None = pct
    else:  # odd-shaped frame without composition percentages (e.g. example CSV)
        dominant = crop_mix.groupby(["field_id", "year"], as_index=False).first()
        pct_col = None
    sequences = []
    for field_id, group in dominant.groupby("field_id"):
        ordered = group.sort_values("year")
        rows = ordered.to_dict("records")
        crop_names = [row["crop_name"] for row in rows]
        usable_rows = _usable_rotation_rows(rows, pct_col)
        usable_names = [str(row["crop_name"]) for row in usable_rows]
        transitions = [f"{crop_names[i]} → {crop_names[i + 1]}" for i in range(len(crop_names) - 1)]
        latest_row = rows[-1] if rows else None
        latest_year_usable = bool(
            latest_row is not None
            and any(row.get("year") == latest_row.get("year") for row in usable_rows)
        )
        usable_pcts = [
            float(row[pct_col])
            for row in usable_rows
            if pct_col is not None and row.get(pct_col) is not None
        ]
        evidence = {
            "usable_years": len(usable_rows),
            "total_years": len(rows),
            "median_dominant_pct": median(usable_pcts) if usable_pcts else None,
            "latest_year_usable": latest_year_usable,
        }
        outlook = build_crop_rotation_outlook(usable_names, evidence=evidence)
        sequences.append(
            {
                "field_id": field_id,
                "rotation_sequence": " -> ".join(str(name) for name in crop_names),
                "rotation_count": len(transitions),
                "rotation_patterns": "; ".join(sorted(set(transitions))),
                "history_years": int(len(ordered)),
                "history_start_year": int(ordered["year"].min()),
                "history_end_year": int(ordered["year"].max()),
                "rotation_usable_years": len(usable_rows),
                "crop_diversity": int(ordered["crop_name"].nunique()),
                "corn_years": int((ordered["crop_name"] == "Corn").sum()),
                "soybean_years": int((ordered["crop_name"] == "Soybeans").sum()),
                **outlook,
            }
        )
    return pd.DataFrame(sequences)


def plot_crop_mix_stacked_100(ax, crop_mix: pd.DataFrame, title: str = "Crop composition by year"):
    if crop_mix.empty:
        ax.text(0.5, 0.5, "No CDL data", ha="center", va="center")
        ax.set_axis_off()
        return ax
    crop_mix = filter_cdl_categories(crop_mix)
    if crop_mix.empty:
        ax.text(0.5, 0.5, "No CDL data", ha="center", va="center")
        ax.set_axis_off()
        return ax
    pct = _pct_col(crop_mix)
    pivot = crop_mix.pivot_table(
        index="year", columns="crop_name", values=pct, aggfunc="sum"
    ).fillna(0)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(pivot.columns))))
    bottom = np.zeros(len(pivot))
    for idx, crop_name in enumerate(pivot.columns):
        values = pivot[crop_name].to_numpy(dtype=float)
        ax.bar(pivot.index.astype(str), values, bottom=bottom, label=crop_name, color=colors[idx])
        bottom += values
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent of field")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    return ax
