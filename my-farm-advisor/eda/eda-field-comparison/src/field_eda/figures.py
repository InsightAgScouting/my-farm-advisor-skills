"""Statistical figures for the field EDA subskill (Stage A: N1, C1, W1 + strip).

Every function returns a :class:`FigureResult` and writes nothing. The caller
decides where a PNG goes. ``stats`` always carries the field and field-year
counts behind the figure so a caption cannot silently overclaim coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import cumulative_gdd, ndvi_values  # noqa: E402
from .panel import FieldPanel  # noqa: E402
from .style import CDL_RGB, GROWER_COLORS, GROWER_ORDER, apply_style  # noqa: E402

NDVI_TOTAL_FIELDS = 34  # Assignment 2 full dataset; kept as the declared base


@dataclass
class FigureResult:
    """A drawn figure plus the numbers and prose a caption needs."""

    figure_id: str
    figure: plt.Figure
    stats: dict[str, Any]
    story: str
    coverage_note: str | None = None
    warnings: list[str] = field(default_factory=list)


def save(result: FigureResult, path: str, dpi: int = 200) -> str:
    """Write a result to PNG. Only the example/report layer calls this."""
    result.figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def _ndvi_coverage_note(panel: FieldPanel) -> str:
    n = panel.panel.groupby("field_id")["ndvi_mean"].apply(lambda s: s.notna().any()).sum()
    total = max(NDVI_TOTAL_FIELDS, panel.panel["field_id"].nunique())
    return f"NDVI covers {int(n)} of {total} fields"


def _field_area(panel: FieldPanel, field_id: str) -> float:
    areas = panel.panel.loc[panel.panel["field_id"] == field_id, "area_ha"].dropna()
    return float(areas.iloc[0]) if not areas.empty else math.nan


def _crop_color(code: int, name: str) -> str:
    rgb = CDL_RGB.get(int(code))
    if rgb is not None:
        return "#%02x%02x%02x" % rgb
    palette = plt.get_cmap("tab20")
    return palette((int(code) % 20) / 20.0)


# --------------------------------------------------------------------------
# Coverage strip
# --------------------------------------------------------------------------
def coverage_strip(panel: FieldPanel) -> FigureResult:
    """Grower x layer field coverage, annotated as ``n/N``.

    This is the honest header for the whole report: two layers cover all 34
    fields and NDVI covers a subset. Every later figure repeats its own n.
    """
    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        cov = panel.coverage()
        growers = [g for g in GROWER_ORDER if g in set(cov["grower"])]
        layers = ["boundaries", "CDL", "weather", "NDVI"]
        matrix = np.full((len(growers), len(layers)), np.nan)
        labels = [["" for _ in layers] for _ in growers]
        for _, row in cov.iterrows():
            if row["grower"] not in growers:
                continue
            i, j = growers.index(row["grower"]), layers.index(row["layer"])
            matrix[i, j] = row["pct"]
            labels[i][j] = f"{int(row['n_fields'])}/{int(row['n_total'])}"

        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        cmap = plt.get_cmap("Greens").copy()
        cmap.set_bad("#f3f4f6")
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(len(growers)):
            for j in range(len(layers)):
                if matrix[i, j] == 0:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, facecolor="#f3f4f6",
                            edgecolor="white", hatch="///", linewidth=0,
                        )
                    )
                ax.text(
                    j, i, labels[i][j], ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if matrix[i, j] >= 60 else "#111827",
                )
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(growers)))
        ax.set_yticklabels(growers)
        ax.set_xlabel("Layer")
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_title("What each layer covers, by grower (fields with data / fields)")
        fig.text(
            0.01, -0.02,
            "NDVI is a subset: Illinois/Iowa fields r4–r10 have no satellite ingest "
            "and are drawn as gaps, not interpolated.",
            fontsize=7.2, color="#4b5563",
        )
        fig.tight_layout()
    stats = {
        "n_fields": int(panel.panel["field_id"].nunique()),
        "n_field_years": int(len(panel.panel)),
        "by_layer": cov.to_dict(orient="records"),
    }
    return FigureResult(
        figure_id="coverage",
        figure=fig,
        stats=stats,
        story=(
            "Three layers cover every field; NDVI covers the fields that actually "
            "have imagery. Showing the gap is part of the finding — it bounds "
            "what the greenness chapter can claim."
        ),
    )


# --------------------------------------------------------------------------
# N1 — within field: NDVI distribution per year
# --------------------------------------------------------------------------
def n1_ndvi_distribution(
    panel: FieldPanel,
    grower: str | None = None,
    field_id: str | None = None,
) -> FigureResult:
    """Annual-mean NDVI distribution for one field, one violin per year.

    Within-field level: the same ground, five seasons. The composite is an
    annual mean of valid scenes, so the distribution is *spatial* (pixels in
    the field) and the p95 marks the field's greenest areas, not its greenest
    day.
    """
    frame = panel.panel
    if grower is not None:
        frame = frame[frame["grower"] == grower]
    if field_id is None:
        counts = frame.groupby("field_id")["ndvi_mean"].apply(lambda s: s.notna().sum())
        if counts.empty or counts.max() == 0:
            raise ValueError("No NDVI composites available for the requested selection")
        field_id = str(counts.sort_values(ascending=False).index[0])
    sub = frame[frame["field_id"] == field_id].sort_values("year")
    years = [int(y) for y in sub.loc[sub["ndvi_path"].notna(), "year"]]
    if not years:
        raise ValueError(f"{field_id} has no NDVI composites")

    datasets, medians, p95s = [], [], []
    for year in years:
        path = sub.loc[sub["year"] == year, "ndvi_path"].iloc[0]
        vals = ndvi_values(path)
        datasets.append(vals)
        medians.append(float(np.median(vals)))
        p95s.append(float(np.percentile(vals, 95)))

    label = sub["grower"].iloc[0]
    color = GROWER_COLORS.get(label, "#374151")
    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        parts = ax.violinplot(
            datasets, positions=years, widths=0.78, showextrema=False, showmedians=False
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("white")
            body.set_alpha(0.55)
        ax.scatter(years, medians, color=color, s=28, zorder=3, label="field median")
        ax.scatter(years, p95s, marker="^", color="#111827", s=22, zorder=3,
                   label="p95 (greenest area)")
        ax.set_xticks(years)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual-mean NDVI (pixel distribution)")
        ax.set_title(f"{field_id} — greenness distribution per season ({label})")
        ax.legend(loc="upper right")
        ax.text(
            0.01, 0.02,
            "Each violin is the spread of pixels in one annual mean composite, "
            "not daily observations.",
            transform=ax.transAxes, fontsize=7.2, color="#4b5563",
        )
        fig.tight_layout()

    stats = {
        "field_id": field_id,
        "grower": label,
        "years": years,
        "median_by_year": dict(zip(map(str, years), np.round(medians, 3))),
        "p95_by_year": dict(zip(map(str, years), np.round(p95s, 3))),
        "n_fields": 1,
        "n_field_years": len(years),
    }
    return FigureResult(
        figure_id="N1",
        figure=fig,
        stats=stats,
        story=(
            f"{field_id} keeps the same footprint for five seasons, so the violins "
            "show how far each year's canopy departed from that field's own "
            "baseline — the one comparison static geometry cannot make."
        ),
        coverage_note=_ndvi_coverage_note(panel),
    )


# --------------------------------------------------------------------------
# C1 — within field: crop composition by year, small multiples
# --------------------------------------------------------------------------
def c1_crop_composition(panel: FieldPanel, grower: str) -> FigureResult:
    """100 % stacked crop composition by year, one panel per field.

    Within-field level for CDL: each field's own rotation, not the farm
    average. Small multiples keep every field's sequence visible.
    """
    comp = panel.composition
    comp = comp[(comp["grower"] == grower) & comp["pct"].notna()]
    if comp.empty:
        raise ValueError(f"No CDL composition rows for {grower}")
    sub = panel.panel[panel.panel["grower"] == grower]
    fields = sorted(sub["field_id"].unique())
    years = sorted(int(y) for y in panel.years)
    # Shared class legend and colours for this grower.
    class_totals = comp.groupby(["crop_code", "crop"])["pct"].sum().reset_index()
    class_totals = class_totals.sort_values("pct", ascending=False)
    classes = [(int(r.crop_code), str(r.crop)) for r in class_totals.itertuples()]
    colour = {code: _crop_color(code, name) for code, name in classes}

    ncols = 2 if len(fields) > 1 else 1
    nrows = math.ceil(len(fields) / ncols)
    fig_h = 2.5 * nrows + 1.1
    legend_rows = max(1, math.ceil(len(classes) / 5))
    legend_in = 0.30 + 0.22 * legend_rows
    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(9.6, fig_h),
            sharex=True, sharey=True, squeeze=False,
        )
        for ax, field_id in zip(axes.ravel(), fields):
            field_comp = comp[comp["field_id"] == field_id]
            pivot = (
                field_comp.pivot_table(index="year", columns="crop_code", values="pct", aggfunc="sum")
                .reindex(index=years)
                .fillna(0.0)
            )
            bottom = np.zeros(len(years))
            for code in [c for c, _ in classes]:
                if code not in pivot.columns:
                    continue
                values = pivot[code].to_numpy(dtype=float)
                ax.bar(
                    years, values, bottom=bottom, width=0.78,
                    color=colour[code], edgecolor="white", linewidth=0.5,
                )
                bottom += values
            latest = max(years)
            latest_comp = field_comp[field_comp["year"] == latest] if not field_comp.empty else field_comp
            dominant = None if latest_comp.empty else latest_comp.loc[latest_comp["pct"].idxmax()]
            area = _field_area(panel, field_id)
            short = field_id.split("-", 1)[1] if "-" in field_id else field_id
            if dominant is None:
                title = f"{short} — {area:.0f} ha"
            else:
                title = f"{short} — {area:.0f} ha\n{latest}: {dominant['crop']} {dominant['pct']:.0f}%"
            ax.set_title(title, fontsize=8.2, loc="left")
            ax.set_ylim(0, 100)
            ax.set_xticks(years)
            ax.tick_params(axis="x", labelrotation=0, labelsize=7.5)
            ax.grid(axis="x", visible=False)
        for ax in axes.ravel()[len(fields):]:
            ax.set_axis_off()
        for ax in axes[:, 0]:
            ax.set_ylabel("Percent of field")
        for ax in axes[-1, :]:
            ax.set_xlabel("Year")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=colour[c]) for c, _ in classes]
        fig.legend(
            handles, [n for _, n in classes], ncol=5,
            loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=7.2,
            title="CDL class", title_fontsize=8,
        )
        fig.suptitle(
            f"{grower} — each field's own rotation (100 % stacked CDL composition)",
            fontsize=11, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=(0, legend_in / fig_h + 0.01, 1, 0.975))

    stats = {
        "grower": grower,
        "n_fields": len(fields),
        "n_field_years": int(sub[sub["crop_code"].notna()]["year"].count()),
        "fields": fields,
        "classes": [n for _, n in classes],
    }
    return FigureResult(
        figure_id="C1",
        figure=fig,
        stats=stats,
        story=(
            "Read across a row of years: Corn Belt fields alternate corn and "
            "soybeans almost in lockstep, while Delaware parcels mix double-crop "
            "small grain, hay, and vegetables. The farm average would hide both."
        ),
    )


# --------------------------------------------------------------------------
# W1 — within field: cumulative GDD by year
# --------------------------------------------------------------------------
def w1_cumulative_gdd(
    panel: FieldPanel,
    field_by_grower: dict[str, str] | None = None,
    start_doy: int = 60,
    end_doy: int = 304,
) -> FigureResult:
    """Cumulative growing-degree-days by day-of-year, one line per season.

    One representative field per grower. Within-grower weather differences are
    a coarse-grid artefact and are handled by W2; the point here is the same
    ground's interannual spread.
    """
    daily = panel.daily_weather
    if daily.empty:
        raise ValueError("No daily weather in the panel; rebuild with layers including 'weather'")
    growers = [g.label for g in panel.growers if g.label in set(panel.panel["grower"])]
    picks = dict(field_by_grower or {})
    for grower in growers:
        if grower not in picks:
            sub = panel.panel[(panel.panel["grower"] == grower) & panel.panel["gdd_total"].notna()]
            if not sub.empty:
                picks[grower] = str(sub["field_id"].iloc[0])

    ncols = 2
    nrows = math.ceil(len(picks) / ncols)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 3.4 * nrows), squeeze=False)
        totals: dict[str, dict[str, float]] = {}
        handles: list = []
        for ax, (grower, field_id) in zip(axes.ravel(), picks.items()):
            field_daily = daily[daily["field_id"] == field_id]
            cum = cumulative_gdd(field_daily, start_doy=start_doy, end_doy=end_doy)
            cmap = plt.get_cmap("YlGn")
            years = sorted(cum["year"].unique())
            totals[field_id] = {}
            for i, year in enumerate(years):
                g = cum[cum["year"] == year].sort_values("doy")
                ax.plot(
                    g["doy"], g["gdd_cum"], lw=1.8,
                    color=cmap(0.25 + 0.6 * i / max(1, len(years) - 1)),
                    label=str(int(year)),
                )
                totals[field_id][str(int(year))] = float(g["gdd_cum"].iloc[-1])
            if not handles:
                handles = list(ax.get_lines())
            ax.set_title(f"{grower} — {field_id}", loc="left")
            ax.set_xlabel("Day of year")
            ax.set_ylabel("Cumulative GDD (°C-day)")
            ax.set_xlim(start_doy - 2, end_doy + 4)
        for ax in axes.ravel()[len(picks):]:
            ax.set_axis_off()
        fig.legend(
            handles, [h.get_label() for h in handles], ncol=5,
            loc="upper center", bbox_to_anchor=(0.5, 0.935), title="Year",
            title_fontsize=8, fontsize=8,
        )
        fig.suptitle(
            "Growing-degree-day accumulation (base 10 °C, mean method; 1 Mar – 31 Oct)",
            fontsize=11, fontweight="bold", y=0.985,
        )
        fig.text(
            0.01, 0.005,
            "One representative field per grower. Fields inside a grower differ by only "
            "a fraction of the between-grower spread; those small differences are W2.",
            fontsize=7.2, color="#4b5563",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.90))

    stats = {
        "representative_fields": picks,
        "season_gdd_totals": totals,
        "season": {"start_doy": start_doy, "end_doy": end_doy},
        "n_fields": len(picks),
        "n_field_years": sum(len(v) for v in totals.values()),
    }
    return FigureResult(
        figure_id="W1",
        figure=fig,
        stats=stats,
        story=(
            "The slope is the season: Nebraska climbs fastest and finishes "
            "highest, Illinois/Iowa accumulate more slowly but steadily, and "
            "Delaware's line has a different shape again. One field per grower "
            "is enough because fields inside a grower share a coarse weather grid."
        ),
    )
