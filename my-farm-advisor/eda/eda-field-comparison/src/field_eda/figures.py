"""Statistical figures for the field EDA subskill (Stages A and B).

Stage A: coverage strip, N1, C1, W1.
Stage B: N2, N3, X4, C2, C3, X5.

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
import pandas as pd  # noqa: E402

from .metrics import cumulative_gdd, ndvi_values  # noqa: E402
from .panel import FieldPanel  # noqa: E402
from .style import (  # noqa: E402
    CDL_RGB,
    GROWER_COLORS,
    GROWER_ORDER,
    apply_style,
    text_color_for,
)


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
        coverage_note = _coverage_footer(panel, fig, y=-0.035)
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
        coverage_note=coverage_note,
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


# --------------------------------------------------------------------------
# Stage B helpers
# --------------------------------------------------------------------------
def _ndvi_frame(panel: FieldPanel) -> pd.DataFrame:
    """Field-years with an annual-mean NDVI composite, or a clear error."""
    frame = panel.panel[panel.panel["ndvi_mean"].notna()].copy()
    if frame.empty:
        raise ValueError("No NDVI composites in the panel; rebuild with layers including 'ndvi'")
    return frame


def _field_short(field_id: str) -> str:
    """Drop the state/prefix segment so row labels stay readable."""
    return field_id.split("-", 1)[1] if "-" in field_id else field_id


def _spearman(x, y) -> tuple[float, int]:
    """Spearman rho and pair count without pulling in scipy.

    Ranks are computed by pandas, so ties are handled the same way as
    ``scipy.stats.spearmanr``. Fewer than four complete pairs returns NaN:
    a rank correlation on one or two points is not a statistic.
    """
    pair = pd.DataFrame(
        {"x": pd.to_numeric(pd.Series(x), errors="coerce"),
         "y": pd.to_numeric(pd.Series(y), errors="coerce")}
    ).dropna()
    if len(pair) < 4 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return math.nan, int(len(pair))
    return float(pair["x"].rank().corr(pair["y"].rank())), int(len(pair))


def _coverage_footer(panel: FieldPanel, fig: plt.Figure, y: float = -0.02) -> str:
    """One-line NDVI coverage strip for the bottom of every NDVI figure."""
    cov = panel.coverage()
    nd = cov[cov["layer"] == "NDVI"].set_index("grower")
    parts: list[str] = []
    n_sum = t_sum = 0
    for grower in GROWER_ORDER:
        if grower not in nd.index:
            continue
        n, total = int(nd.loc[grower, "n_fields"]), int(nd.loc[grower, "n_total"])
        parts.append(f"{grower} {n}/{total}")
        n_sum += n
        t_sum += total
    text = "NDVI coverage: " + " · ".join(parts) + f" — {n_sum}/{t_sum} fields"
    fig.text(0.01, y, text, fontsize=7.2, color="#4b5563")
    return text


_N2_TICKS = {
    "Dbl Crop Barley/Soybeans": "Dbl crop\nBarley/Soy",
    "Dbl Crop WinWht/Soybeans": "Dbl crop\nWht/Soy",
}
_C2_ABBREV = {
    1: "C",
    5: "S",
    26: "W/S",
    254: "B/S",
    48: "Mel",
    229: "Let",
    176: "P",
    61: "Fal",
    63: "For",
    82: "Dev",
    36: "Alf",
    24: "Wh",
}


# --------------------------------------------------------------------------
# N2 — across fields: NDVI by CDL crop class
# --------------------------------------------------------------------------
def n2_ndvi_by_crop(panel: FieldPanel) -> FigureResult:
    """Annual-mean NDVI by CDL class, one point per field-year.

    Class labels come from ``crop_code`` (not the stored name string). Classes
    with at least four field-years are drawn separately; the two one-off
    lettuce/durum and watermelon field-years are pooled and labelled as such.
    """
    nd = _ndvi_frame(panel)
    counts = nd["crop"].value_counts()
    major = [str(c) for c in counts.index if counts[c] >= 4]
    nd["class"] = nd["crop"].where(nd["crop"].isin(major), "Other crop classes")
    medians = nd.groupby("class")["ndvi_mean"].median()
    class_order = sorted(major, key=lambda c: float(medians[c]), reverse=True)
    if (nd["class"] == "Other crop classes").any():
        class_order.append("Other crop classes")

    positions = np.arange(len(class_order))
    datasets = [
        nd.loc[nd["class"] == name, "ndvi_mean"].to_numpy(dtype=float)
        for name in class_order
    ]
    rng = np.random.default_rng(7)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, ax = plt.subplots(figsize=(8.4, 4.9))
        ax.boxplot(
            datasets,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#e5e7eb", edgecolor="#9ca3af", linewidth=0.8),
            whiskerprops=dict(color="#9ca3af", linewidth=0.8),
            capprops=dict(color="#9ca3af", linewidth=0.8),
            medianprops=dict(color="#111827", linewidth=1.4),
        )
        for grower in GROWER_ORDER:
            sub = nd[nd["grower"] == grower]
            if sub.empty:
                continue
            xpos = [class_order.index(name) for name in sub["class"]]
            jitter = rng.uniform(-0.17, 0.17, size=len(sub))
            ax.scatter(
                np.asarray(xpos) + jitter,
                sub["ndvi_mean"],
                s=26,
                color=GROWER_COLORS[grower],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.82,
                zorder=3,
                label=grower,
            )

        labels = []
        for name in class_order:
            n = len(datasets[class_order.index(name)])
            if name == "Other crop classes":
                labels.append(f"Other\n(2 classes, n={n})")
            else:
                labels.append(f"{_N2_TICKS.get(name, name)}\n(n={n})")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7.6)
        ax.set_xlabel("CDL class (n = field-years with imagery)")
        ax.set_ylabel("Annual-mean NDVI (field mean)")
        ax.set_title("Greenness by what the field was labelled")
        ax.margins(x=0.03)
        fig.legend(
            loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4,
            title="Grower", title_fontsize=8, fontsize=8,
        )
        fig.text(
            0.01, 0.005,
            "One point per field-year; a field contributes up to five points and they are "
            "not independent. Boxes span the field-year quartiles.",
            fontsize=7.2, color="#4b5563",
        )
        coverage_note = _coverage_footer(panel, fig, y=-0.045)
        fig.tight_layout(rect=(0, 0.10, 1, 1.0))

    cell = {
        name: {
            "median": round(float(medians[name]), 3),
            "n_field_years": int(len(datasets[i])),
        }
        for i, name in enumerate(class_order)
    }
    top = class_order[0]
    stats = {
        "classes": cell,
        "highest_median_class": top,
        "n_fields": int(nd["field_id"].nunique()),
        "n_field_years": int(len(nd)),
    }
    return FigureResult(
        figure_id="N2",
        figure=fig,
        stats=stats,
        story=(
            f"{top} has the highest class median ({cell[top]['median']:.2f}), but Corn "
            "and Soybeans overlap almost completely. The CDL label sets the rotation; "
            "the season and the grower set how green that label actually looks."
        ),
        coverage_note=coverage_note,
        warnings=(
            ["Other crop classes pools two one-off field-years; it is shown for "
             "completeness, not as a class estimate."]
            if "Other crop classes" in class_order else []
        ),
    )


# --------------------------------------------------------------------------
# N3 — across growers: NDVI climatology with the coverage denominator
# --------------------------------------------------------------------------
def n3_ndvi_climatology(panel: FieldPanel) -> FigureResult:
    """Grower x year annual-mean NDVI, with a coverage strip underneath.

    The top panel is the cross-grower comparison; the bottom strip is the
    denominator (fields with imagery out of all fields in that grower-year) so
    a Delaware mean over four parcels is never read like a Nebraska mean over
    ten.
    """
    nd = _ndvi_frame(panel)
    summary = (
        nd.groupby(["grower", "year"])["ndvi_mean"]
        .agg(mean="mean", lo="min", hi="max", n="size")
        .reset_index()
    )
    totals = (
        panel.panel.groupby(["grower", "year"])["field_id"].nunique().rename("total").reset_index()
    )
    summary = summary.merge(totals, on=["grower", "year"], how="left")
    growers = [g for g in GROWER_ORDER if g in set(summary["grower"])]
    years = sorted(int(y) for y in panel.years)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig = plt.figure(figsize=(8.8, 6.4))
        gs = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.0))
        ax = fig.add_subplot(gs[0])
        axn = fig.add_subplot(gs[1], sharex=ax)

        for grower in growers:
            sub = summary[summary["grower"] == grower].sort_values("year")
            color = GROWER_COLORS[grower]
            ax.fill_between(
                sub["year"], sub["lo"], sub["hi"], color=color, alpha=0.11, linewidth=0
            )
            ax.plot(
                sub["year"], sub["mean"], marker="o", markersize=5.5,
                linewidth=2.0, color=color, label=grower,
            )
            last = sub.iloc[-1]
            ax.annotate(
                f"{last['mean']:.2f}", (last["year"], last["mean"]),
                xytext=(6, 0), textcoords="offset points", fontsize=7.4,
                color=color, va="center", fontweight="bold",
            )

        ax.set_ylabel("Annual-mean NDVI (grower mean)")
        ax.set_title("Greenness by grower and year, with its denominator below")
        ax.set_xlim(min(years) - 0.4, max(years) + 0.8)
        ax.tick_params(labelbottom=False)
        ax.legend(loc="lower left", ncol=2, title="Grower (fields in group)", title_fontsize=8)

        width = 0.20
        for i, grower in enumerate(growers):
            sub = summary[summary["grower"] == grower].set_index("year")
            color = GROWER_COLORS[grower]
            for j, year in enumerate(years):
                if year not in sub.index:
                    continue
                row = sub.loc[year]
                x = year + (i - (len(growers) - 1) / 2) * width
                axn.bar(x, row["n"], width=width * 0.92, color=color, alpha=0.85)
                axn.text(
                    x, row["n"] + 0.35, f"{int(row['n'])}/{int(row['total'])}",
                    ha="center", va="bottom", fontsize=6.4, color="#374151",
                )
        axn.set_ylabel("Fields with NDVI")
        axn.set_xlabel("Year")
        axn.set_xticks(years)
        axn.set_ylim(0, 11.6)
        axn.set_yticks([0, 5, 10])
        axn.grid(axis="x", visible=False)

        fig.text(
            0.01, 0.012,
            "Bands are the field-year min–max inside each grower; lines are grower means. "
            "Annual means of valid scenes, not peak greenness.\nIllinois/Iowa r4–r10 have no "
            "imagery and are absent from both the mean and the bar.",
            fontsize=7.2, color="#4b5563",
        )
        coverage_note = _coverage_footer(panel, fig, y=-0.035)
        fig.tight_layout(rect=(0, 0.075, 1, 1.0))
        fig.subplots_adjust(hspace=0.10)

    stats = {
        "grower_year_means": {
            grower: {
                str(int(r.year)): {
                    "mean": round(float(r.mean), 3),
                    "n": int(r.n),
                    "total": int(r.total),
                }
                for r in summary[summary["grower"] == grower].itertuples()
            }
            for grower in growers
        },
        "n_fields": int(nd["field_id"].nunique()),
        "n_field_years": int(len(nd)),
    }
    return FigureResult(
        figure_id="N3",
        figure=fig,
        stats=stats,
        story=(
            "Delaware sits highest and swings hardest, Nebraska holds the narrowest range, "
            "and Illinois/Iowa drift downward through 2025. The bars make the comparison "
            "honest: Delaware's line is four parcels, Nebraska's is ten."
        ),
        coverage_note=coverage_note,
    )


# --------------------------------------------------------------------------
# X4 — NDVI vs season climate, with rho per crop
# --------------------------------------------------------------------------
def x4_ndvi_climate(panel: FieldPanel) -> FigureResult:
    """Annual-mean NDVI against season GDD and rainfall, rho per crop group.

    Two scatter panels plus a compact rho table. GDD and rainfall co-vary, so
    the coefficients are descriptive: they say how much of the greenness tracks
    the season inside a crop, not what causes it.
    """
    nd = _ndvi_frame(panel)
    nd = nd[nd["gdd_total"].notna() & nd["precip_mm"].notna()].copy()
    groups = ["Corn", "Soybeans", "Double crop"]
    present = [g for g in groups if (nd["crop_group"] == g).sum() >= 4]

    rows: list[tuple[str, int, float, float]] = []
    rho_gdd_all, n_all = _spearman(nd["gdd_total"], nd["ndvi_mean"])
    rho_rain_all, _ = _spearman(nd["precip_mm"], nd["ndvi_mean"])
    rows.append(("All fields", n_all, rho_gdd_all, rho_rain_all))
    for group in present:
        sub = nd[nd["crop_group"] == group]
        rg, n = _spearman(sub["gdd_total"], sub["ndvi_mean"])
        rr, _ = _spearman(sub["precip_mm"], sub["ndvi_mean"])
        rows.append((group, n, rg, rr))

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig = plt.figure(figsize=(11.6, 4.7))
        gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.62))
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        axt = fig.add_subplot(gs[2])
        axt.set_axis_off()

        for ax, xcol, xlabel, title in (
            (ax1, "gdd_total", "Season GDD (°C-day)", "(a) Greenness vs heat"),
            (ax2, "precip_mm", "Season rainfall (mm)", "(b) Greenness vs water"),
        ):
            for grower in GROWER_ORDER:
                sub = nd[nd["grower"] == grower]
                if sub.empty:
                    continue
                ax.scatter(
                    sub[xcol], sub["ndvi_mean"], s=28, alpha=0.75,
                    color=GROWER_COLORS[grower], edgecolor="white", linewidth=0.4,
                    label=grower,
                )
            ax.set_xlabel(xlabel)
            ax.set_title(title, loc="left")
            ax.margins(x=0.04, y=0.06)
        ax1.set_ylabel("Annual-mean NDVI (field mean)")
        ax2.tick_params(labelleft=False)
        ax1.legend(loc="upper left", ncol=2, fontsize=7.4, title="Grower", title_fontsize=7.6)

        header = f"{'Crop group':<12}{'n':>4}{'ρ GDD':>8}{'ρ rain':>8}"
        lines = [header, "-" * len(header)]
        for name, n, rg, rr in rows:
            lines.append(f"{name:<12}{n:>4}{rg:>+8.2f}{rr:>+8.2f}")
        axt.text(
            0.0, 1.0, "Spearman rank correlation\n(field-years with imagery)",
            transform=axt.transAxes, va="top", ha="left", fontsize=8.2, fontweight="bold",
        )
        axt.text(
            0.0, 0.83, "\n".join(lines), transform=axt.transAxes, va="top", ha="left",
            fontsize=7.8, family="monospace",
        )
        axt.text(
            0.0, 0.30,
            "ρ is not causal:\nheat and rainfall move\ntogether, and the sample\n"
            "repeats each field five\ntimes.",
            transform=axt.transAxes, va="top", ha="left", fontsize=7.2, color="#4b5563",
        )
        fig.suptitle(
            "Does the season explain the greenness? Rank correlation inside each crop",
            fontsize=11, fontweight="bold", y=1.02,
        )
        coverage_note = _coverage_footer(panel, fig, y=-0.06)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        fig.subplots_adjust(wspace=0.22)

    rho_table = {
        name: {"n": n, "rho_gdd": None if math.isnan(rg) else round(rg, 3),
               "rho_rain": None if math.isnan(rr) else round(rr, 3)}
        for name, n, rg, rr in rows
    }
    soy = rho_table.get("Soybeans")
    if soy and soy["rho_gdd"] is not None:
        sign = "positive" if soy["rho_gdd"] > 0 else "negative"
        crop_story = (
            f"Soybeans show a {sign} rank association with season GDD "
            f"(ρ = {soy['rho_gdd']:+.2f}); splitting by crop keeps that from being read "
            "as a whole-farm climate effect."
        )
    else:
        crop_story = (
            "No crop group has enough field-years for a stable rank correlation in this "
            "selection; the pooled row is the only summary."
        )
    stats = {
        "rho": rho_table,
        "crop_groups": present,
        "n_fields": int(nd["field_id"].nunique()),
        "n_field_years": int(len(nd)),
    }
    return FigureResult(
        figure_id="X4",
        figure=fig,
        stats=stats,
        story=(
            f"{crop_story} Most of the spread is field and year, which is what N3 and N2 "
            "show directly."
        ),
        coverage_note=coverage_note,
        warnings=["GDD and rainfall co-vary; the two panels are not independent evidence."],
    )


# --------------------------------------------------------------------------
# C2 — across fields: dominant crop heatmap
# --------------------------------------------------------------------------
def c2_dominant_crop_heatmap(panel: FieldPanel) -> FigureResult:
    """Field x year dominant-crop heatmap, one panel per grower.

    Every cell is the CDL class with the largest share in that field-year. The
    point is the pattern across cells: synchronised corn/soy alternation in the
    Corn Belt versus a Delaware patchwork of double crops and produce.
    """
    frame = panel.panel[panel.panel["crop_code"].notna()]
    if frame.empty:
        raise ValueError("No CDL rows in the panel")
    growers = [g for g in GROWER_ORDER if g in set(frame["grower"])]
    years = sorted(int(y) for y in panel.years)
    code_name = (
        frame.dropna(subset=["crop_code"])
        .drop_duplicates("crop_code")
        .set_index("crop_code")["crop"]
        .to_dict()
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), squeeze=False)
        for ax, grower in zip(axes.ravel(), growers):
            sub = frame[frame["grower"] == grower]
            fields = sorted(sub["field_id"].unique())
            rgb = np.ones((len(fields), len(years), 3), dtype=float)
            for i, field_id in enumerate(fields):
                for j, year in enumerate(years):
                    row = sub[(sub["field_id"] == field_id) & (sub["year"] == year)]
                    if row.empty:
                        rgb[i, j] = (0.96, 0.96, 0.96)
                        continue
                    code = int(row["crop_code"].iloc[0])
                    rgb[i, j] = np.asarray(CDL_RGB.get(code, (204, 204, 204))) / 255.0
            ax.imshow(rgb, aspect="auto", interpolation="nearest")

            for i, field_id in enumerate(fields):
                area = _field_area(panel, field_id)
                ax.text(
                    -0.62, i, f"{_field_short(field_id)} · {area:.0f} ha",
                    ha="right", va="center", fontsize=7.4,
                )
            for i, field_id in enumerate(fields):
                for j, year in enumerate(years):
                    row = sub[(sub["field_id"] == field_id) & (sub["year"] == year)]
                    if row.empty:
                        ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#6b7280")
                        continue
                    code = int(row["crop_code"].iloc[0])
                    label = _C2_ABBREV.get(code, str(code_name.get(code, code))[:3])
                    ax.text(
                        j, i, label, ha="center", va="center", fontsize=7.6,
                        fontweight="bold", color=text_color_for(code),
                    )
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels([str(y) for y in years], fontsize=8.0)
            ax.set_yticks([])
            ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(fields), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.4)
            ax.tick_params(which="minor", length=0)
            ax.tick_params(axis="x", length=0)
            ax.set_title(f"{grower} — {len(fields)} fields", loc="left")
        for ax in axes.ravel()[len(growers):]:
            ax.set_axis_off()

        classes_present = sorted(
            {int(c) for c in frame["crop_code"].dropna().unique()},
            key=lambda c: -int((frame["crop_code"] == c).sum()),
        )
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc="#%02x%02x%02x" % CDL_RGB.get(c, (204, 204, 204)))
            for c in classes_present
        ]
        labels = [
            f"{code_name.get(c, f'Class {c}')} ({_C2_ABBREV[c]})" if c in _C2_ABBREV
            else str(code_name.get(c, f"Class {c}"))
            for c in classes_present
        ]
        fig.legend(
            handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01),
            fontsize=7.2,
            title="Dominant CDL class (cell label in brackets)\n"
                  "A cell is the largest share that year, not purity",
            title_fontsize=7.8,
        )
        fig.suptitle(
            "Every field, every season: the class that covered the most ground",
            fontsize=11, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=(0.07, 0.16, 1, 0.96))

    stats = {
        "growers": growers,
        "n_fields": int(frame["field_id"].nunique()),
        "n_field_years": int(len(frame)),
        "classes": [str(code_name.get(c, c)) for c in classes_present],
    }
    return FigureResult(
        figure_id="C2",
        figure=fig,
        stats=stats,
        story=(
            "Illinois, Iowa, and Nebraska read as two-colour stripes: corn and soybeans "
            "trade places year to year. Delaware never settles — barley/soy, wheat/soy, "
            "soybeans, and produce rotate on different clocks inside one farm."
        ),
    )


# --------------------------------------------------------------------------
# C3 — across growers: crop share and diversity
# --------------------------------------------------------------------------
def c3_crop_share_diversity(panel: FieldPanel) -> FigureResult:
    """Mean planted-class share by grower and Shannon diversity per field-year.

    Shares are renormalised to planted CDL classes only, so grass, fallow,
    forest, and development cannot make a grower look diversified. The right
    panel shows the distribution behind the mean, not just the average.
    """
    comp = panel.composition.copy()
    if comp.empty:
        raise ValueError("No CDL composition rows in the panel")
    comp = comp[comp["is_rotation_crop"]].copy()
    totals = comp.groupby(["grower", "field_id", "year"])["pct"].transform("sum")
    comp = comp[totals > 0].copy()
    comp["share"] = comp["pct"] / totals[totals > 0] * 100.0

    class_totals = comp.groupby("crop")["pct"].sum().sort_values(ascending=False)
    top = [str(c) for c in class_totals.index[:5]]
    comp["class"] = comp["crop"].where(comp["crop"].isin(top), "Other planted")
    class_order = top + (["Other planted"] if (comp["class"] == "Other planted").any() else [])

    # A class absent from a field-year contributes zero to that grower's mean,
    # so divide the summed field-year shares by the grower's field-year count
    # instead of averaging only the rows where the class appears.
    per_field_year = (
        comp.groupby(["grower", "field_id", "year", "class"])["share"].sum().reset_index()
    )
    n_field_years = (
        comp[["grower", "field_id", "year"]].drop_duplicates().groupby("grower").size()
    )
    share = (
        per_field_year.groupby(["grower", "class"])["share"].sum()
        .unstack(fill_value=0.0)
        .div(n_field_years, axis=0)
        .reindex(columns=class_order, fill_value=0.0)
    )
    growers = [g for g in GROWER_ORDER if g in share.index]
    share = share.loc[growers]

    code_by_name = (
        comp.drop_duplicates("crop").set_index("crop")["crop_code"].to_dict()
    )
    color = {
        name: (_crop_color(int(code_by_name[name]), name) if name in code_by_name else "#9ca3af")
        for name in class_order
    }
    color["Other planted"] = "#9ca3af"
    legend_names = {
        "Dbl Crop Barley/Soybeans": "Dbl crop barley/soy",
        "Dbl Crop WinWht/Soybeans": "Dbl crop wheat/soy",
    }

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11.4, 4.7), gridspec_kw={"width_ratios": (1.15, 1.0)}
        )

        ypos = np.arange(len(growers))
        left = np.zeros(len(growers))
        for name in class_order:
            values = share[name].to_numpy(dtype=float)
            ax1.barh(ypos, values, left=left, height=0.62, color=color[name],
                     edgecolor="white", linewidth=0.8)
            for i, value in enumerate(values):
                if value >= 9:
                    ax1.text(
                        left[i] + value / 2, i, f"{value:.0f}%", ha="center", va="center",
                        fontsize=7.4, color=text_color_for(int(code_by_name.get(name, 82)))
                        if name in code_by_name else "white",
                    )
            left += values
        ax1.set_yticks(ypos)
        ax1.set_yticklabels(
            [f"{g} ({int(panel.panel.loc[panel.panel['grower'] == g, 'field_id'].nunique())} fields)"
             for g in growers]
        )
        ax1.invert_yaxis()
        ax1.set_xlim(0, 100)
        ax1.set_xlabel("Mean share of planted CDL classes (%)")
        ax1.set_title("What each grower plants", loc="left")
        ax1.grid(axis="y", visible=False)

        positions = np.arange(len(growers))
        data = [
            panel.panel.loc[(panel.panel["grower"] == g) & panel.panel["shannon"].notna(), "shannon"]
            .to_numpy(dtype=float)
            for g in growers
        ]
        box = ax2.boxplot(
            data, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
            medianprops=dict(color="#111827", linewidth=1.4),
            whiskerprops=dict(color="#6b7280", linewidth=0.8),
            capprops=dict(color="#6b7280", linewidth=0.8),
        )
        for patch, grower in zip(box["boxes"], growers):
            patch.set_facecolor(GROWER_COLORS[grower])
            patch.set_alpha(0.22)
            patch.set_edgecolor(GROWER_COLORS[grower])
        rng = np.random.default_rng(11)
        for i, (grower, values) in enumerate(zip(growers, data)):
            jitter = rng.uniform(-0.14, 0.14, size=len(values))
            ax2.scatter(
                i + jitter, values, s=12, color=GROWER_COLORS[grower],
                alpha=0.60, edgecolor="white", linewidth=0.3, zorder=3,
            )
        ax2.set_xticks(positions)
        ax2.set_xticklabels(
            [f"{g}\n(n={len(values)})" for g, values in zip(growers, data)], fontsize=7.6
        )
        ax2.set_xlabel("Grower (field-years)")
        ax2.set_ylabel("Shannon H (planted classes)")
        ax2.set_title("How mixed each field-year is", loc="left")
        ax2.set_ylim(0, max(1.7, float(np.nanmax(np.concatenate(data))) * 1.15))
        ax2.grid(axis="x", visible=False)

        handles = [plt.Rectangle((0, 0), 1, 1, fc=color[name]) for name in class_order]
        fig.legend(
            handles, [legend_names.get(name, name) for name in class_order],
            ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.045), fontsize=7.4,
            title="Planted CDL class", title_fontsize=8,
        )
        fig.suptitle(
            "Two-crop system vs mixed produce: share and diversity by grower",
            fontsize=11, fontweight="bold", y=1.02,
        )
        fig.text(
            0.01, 0.008,
            "Shares are mean field-year shares of planted CDL classes only; non-crop covers "
            "are excluded before renormalising to 100 %. Other planted = planted classes "
            "outside the top five.",
            fontsize=7.0, color="#4b5563",
        )
        fig.tight_layout(rect=(0, 0.20, 1, 0.97))
        fig.subplots_adjust(wspace=0.30)

    max_shannon = float(np.nanmax(np.concatenate(data))) if data else math.nan
    stats = {
        "mean_share_pct": {
            grower: {name: round(float(share.loc[grower, name]), 1) for name in class_order}
            for grower in growers
        },
        "median_shannon": {
            grower: round(float(np.nanmedian(values)), 3) for grower, values in zip(growers, data)
        },
        "max_shannon": round(max_shannon, 3),
        "n_fields": int(panel.panel["field_id"].nunique()),
        "n_field_years": int(len(panel.panel)),
    }
    return FigureResult(
        figure_id="C3",
        figure=fig,
        stats=stats,
        story=(
            "Illinois, Iowa, and Nebraska are effectively two-class systems: corn and "
            "soybeans split the planted area and Shannon H sits near zero. Delaware's "
            "double crops and produce push both the share and the diversity distribution "
            "somewhere the Corn Belt never goes."
        ),
    )


# --------------------------------------------------------------------------
# X5 — field area vs crop purity
# --------------------------------------------------------------------------
def x5_area_vs_purity(panel: FieldPanel) -> FigureResult:
    """Field area (log) against mean crop purity, one point per field.

    Area is static, so the five year-rows are averaged rather than repeated:
    thirty-four footprints, not 170 pseudo-observations. Pooled and per-grower
    rank correlations are both printed because they disagree.
    """
    fields = (
        panel.panel.dropna(subset=["crop_purity"])
        .groupby(["grower", "field_id"])
        .agg(area_ha=("area_ha", "first"), purity=("crop_purity", "mean"), years=("year", "size"))
        .reset_index()
    )
    if fields.empty:
        raise ValueError("No crop purity rows in the panel")
    growers = [g for g in GROWER_ORDER if g in set(fields["grower"])]
    rho_all, n_all = _spearman(fields["area_ha"], fields["purity"])
    rho_by_grower = {}
    for grower in growers:
        sub = fields[fields["grower"] == grower]
        rho, n = _spearman(sub["area_ha"], sub["purity"])
        rho_by_grower[grower] = {"rho": None if math.isnan(rho) else round(rho, 3), "n": n}

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        for grower in growers:
            sub = fields[fields["grower"] == grower]
            ax.scatter(
                sub["area_ha"], sub["purity"], s=42, alpha=0.82,
                color=GROWER_COLORS[grower], edgecolor="white", linewidth=0.5, label=grower,
            )
        logx = np.log10(fields["area_ha"].to_numpy(dtype=float))
        coef = np.polyfit(logx, fields["purity"].to_numpy(dtype=float), 1)
        grid = np.linspace(logx.min(), logx.max(), 50)
        ax.plot(10**grid, np.polyval(coef, grid), color="#6b7280", linewidth=1.3,
                linestyle=(0, (4, 3)), zorder=1, label="pooled trend")

        ax.set_xscale("log")
        ticks = [5, 10, 25, 50, 100, 200, 400]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(4, 460)
        ax.set_ylim(25, 104)
        ax.set_xlabel("Field area (ha, log scale)")
        ax.set_ylabel("Crop purity (% of planted pixels in dominant class)")
        ax.set_title("Small parcels are less pure — but mostly between growers")

        lines = [f"Pooled   ρ = {rho_all:+.2f}  (n={n_all})", ""]
        for grower in growers:
            entry = rho_by_grower[grower]
            value = "  n/a" if entry["rho"] is None else f"{entry['rho']:+.2f}"
            lines.append(f"{grower:<9} ρ = {value}  (n={entry['n']})")
        ax.text(
            0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=7.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d1d5db", alpha=0.92),
        )
        fig.text(
            0.01, 0.005,
            "One point per field (mean purity over 5 years). 30 m CDL pixels straddle the edge "
            "of a 5–12 ha parcel,\nso a single mixed pixel can move purity by tens of percent.",
            fontsize=7.0, color="#4b5563",
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055),
            ncol=5, fontsize=7.6, title="Grower", title_fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.15, 1, 1.0))

    stats = {
        "rho_pooled": None if math.isnan(rho_all) else round(rho_all, 3),
        "rho_by_grower": rho_by_grower,
        "n_fields": int(len(fields)),
        "n_field_years": int(panel.panel["crop_purity"].notna().sum()),
        "area_range_ha": [round(float(fields["area_ha"].min()), 2),
                          round(float(fields["area_ha"].max()), 2)],
    }
    return FigureResult(
        figure_id="X5",
        figure=fig,
        stats=stats,
        story=(
            "Pooled across growers, purity rises with parcel size: Delaware's 5–12 ha "
            "parcels sit far below the 200–380 ha Corn Belt fields. Inside a grower the "
            "slope is flat or negative (Delaware's four parcels give −0.80), so the pooled "
            "pattern is mostly a contrast between farms, not a resolution law. That contrast "
            "is why purity gets a size-aware confidence cap."
        ),
        warnings=["Per-grower rho for Delaware rests on 4 fields; read it as indicative only."],
    )
