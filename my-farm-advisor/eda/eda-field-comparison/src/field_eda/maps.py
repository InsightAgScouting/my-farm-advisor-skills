"""Maps for the field EDA subskill (Stages A and C: M1, M2, M3).

The map layer is intentionally thin: it borrows the skill's locked field-frame
and basemap helpers from the data-pipeline runtime instead of re-implementing
them. That keeps one map convention across reports and EDA (local-metre axes,
boxed true-north arrow, rotation callout, imagery under every footprint).

Runtime import contract
-----------------------
``DATA_PIPELINE_DATA_ROOT`` must point at the runtime root. ``lib/field_frame``
and ``lib/basemap`` are imported from ``<root>/data-pipeline/src/scripts/lib``
so the EDA map cannot drift from the renderers that produced the gallery.

Frame choice
------------
A grower plate covers many scattered fields across tens of kilometres, unlike
the single-field posters. It therefore uses ``force_rot=0.0`` (north-up local
UTM, origin at the SW corner of the whole plate): rotating a county-sized
group to fit the page would leave north pointing sideways for no gain. The
per-field posters keep their auto-rotation because one footprint's long edge
is meaningful.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .figures import FigureResult, ndvi_coverage_text  # noqa: E402
from .panel import FieldPanel, crop_name  # noqa: E402
from .style import CDL_RGB, apply_style  # noqa: E402

STATE_FIPS = {"IL": "17", "IA": "19", "NE": "31", "DE": "10"}

# M2 endpoint classes. "Same" is deliberately one class: the interesting part
# is that a five-year window returns the starting class on most Corn Belt
# fields even though the intermediate seasons alternate.
FLIP_ORDER = ("Corn → Soy", "Soy → Corn", "Same", "Other change", "No CDL")
FLIP_COLORS = {
    "Corn → Soy": "#2b8cbe",
    "Soy → Corn": "#e6a817",
    "Same": "#8f9e98",
    "Other change": "#b83280",
    "No CDL": "#f3f4f6",
}
_M2_ABBREV = {
    1: "C", 5: "S", 12: "SC", 4: "Sor", 24: "Wh", 26: "W/S", 36: "Alf",
    42: "DB", 48: "Mel", 61: "Fal", 63: "For", 82: "Dev", 176: "P",
    229: "Let", 254: "B/S",
}


def _runtime_lib(panel: FieldPanel):
    lib = panel.data_root / "src" / "scripts" / "lib"
    if not lib.exists():
        raise FileNotFoundError(
            f"Runtime lib not found at {lib}. DATA_PIPELINE_DATA_ROOT must point "
            "at a runtime installed by data-pipeline/scripts/install.sh."
        )
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))
    import basemap  # noqa: WPS433
    import field_frame  # noqa: WPS433

    return basemap, field_frame


def _cdl_rgba(codes: np.ma.MaskedArray, alpha: float = 0.68) -> np.ndarray:
    """RGBA image from a masked CDL class grid (unknown classes go grey)."""
    filled = codes.filled(-1)
    rgba = np.zeros((*filled.shape, 4), dtype="float64")
    for code, (r, g, b) in CDL_RGB.items():
        mask = filled == code
        if mask.any():
            rgba[mask] = (r / 255.0, g / 255.0, b / 255.0, alpha)
    other = (filled > 0) & (rgba[..., 3] == 0.0)
    if other.any():
        rgba[other] = (0.78, 0.78, 0.78, alpha * 0.85)
    return rgba


def _legend_classes(panel: FieldPanel, grower: str, year: int) -> list[int]:
    """Classes occupying the field interiors that year, largest share first."""
    comp = panel.composition
    comp = comp[(comp["grower"] == grower) & (comp["year"] == year)]
    if comp.empty:
        return []
    totals = comp.groupby("crop_code")["pct"].sum().sort_values(ascending=False)
    return [int(c) for c in totals.index]


def _grower_boundaries(panel: FieldPanel, spec):
    """Read one grower's field boundaries in WGS84 (or the file's own CRS)."""
    import geopandas as gpd

    path = (
        panel.data_root / "growers" / spec.slug / "farms" / spec.farm
        / "boundary" / "field_boundaries.geojson"
    )
    boundaries = gpd.read_file(path)
    if boundaries.crs is None:
        boundaries = boundaries.set_crs("EPSG:4326")
    return boundaries


def _plate_extent(f, target_res_m: float, margin_frac: float = 0.08) -> tuple[tuple, int, int]:
    """Extent, rows, cols for an all-fields plate in the pinned frame."""
    tb = f.geometry.total_bounds
    width, height = float(tb[2]), float(tb[3])
    margin = margin_frac * max(width, height)
    nr = int((height + 2 * margin) / target_res_m) + 1
    nc = int((width + 2 * margin) / target_res_m) + 1
    ext = (
        -margin, -margin + nc * target_res_m,
        -margin, -margin + nr * target_res_m,
    )
    return ext, nr, nc


def _crop_abbrev(code) -> str:
    try:
        return _M2_ABBREV.get(int(code), str(int(code)))
    except (TypeError, ValueError):
        return "—"


def _flip_class(from_code, to_code) -> str:
    if from_code is None or to_code is None:
        return "No CDL"
    a, b = int(from_code), int(to_code)
    if a == 1 and b == 5:
        return "Corn → Soy"
    if a == 5 and b == 1:
        return "Soy → Corn"
    if a == b:
        return "Same"
    return "Other change"


def _small_multiple_frames(fields, field_frame, panel_w_in: float, panel_h_in: float):
    """Per-field frames plus their extent aspect (height / width).

    Elongated-plan plates put one field in each panel, but a row of a short wide
    parcel next to a tall one would leave a large blank band under the wide
    panel. Pre-building the frames lets the caller size each grid row from the
    tallest panel it holds.
    """
    frames: dict[str, Any] = {}
    aspects: list[float] = []
    for _, field in fields.iterrows():
        fid = str(field["field_id"])
        one = fields[fields["field_id"] == field["field_id"]]
        fr = field_frame.frame_for(one, panel_h_in=panel_h_in, panel_w_in=panel_w_in)
        tb = fr.geometry.total_bounds
        width_m = max(float(tb[2] - tb[0]), 1e-6)
        height_m = max(float(tb[3] - tb[1]), 1e-6)
        frames[fid] = fr
        aspects.append(height_m / width_m)
    return frames, aspects


def _row_height_ratios(aspects: list[float], ncols: int) -> list[float]:
    """Row height ratios from the tallest extent in each row (plus title space)."""
    ratios: list[float] = []
    for start in range(0, len(aspects), ncols):
        chunk = aspects[start:start + ncols]
        ratios.append(max(chunk) + 0.09)
    return ratios


def m1_grower_map(
    panel: FieldPanel,
    grower: str,
    year: int = 2025,
    target_res_m: float = 18.0,
    basemap_res_m: float = 30.0,
) -> FigureResult:
    """One grower plate: imagery, bold boundaries, CDL class fill inside them.

    Where the fields are, how they sit in the landscape, and what each carried
    in the latest year. CDL comes from the state raster, masked to the field
    polygons (exact, no halo): off-field land use would dominate the legend on
    small produce parcels and is already visible in the imagery.
    """
    spec = panel.grower(grower)
    basemap, field_frame = _runtime_lib(panel)
    import rasterio
    from rasterio.warp import Resampling

    boundaries = _grower_boundaries(panel, spec)
    state_fips = STATE_FIPS.get(spec.state)
    if state_fips is None and "state_fips" in boundaries.columns:
        state_fips = str(int(boundaries["state_fips"].iloc[0])).zfill(2)
    cdl_path = panel.data_root / "shared" / "cdl" / "rasters" / f"CDL_{year}_{state_fips}.tif"
    if not cdl_path.exists():
        raise FileNotFoundError(
            f"CDL {year} raster missing for {spec.label}: {cdl_path}. "
            "Run the shared CDL initializer for this year/state."
        )

    f = field_frame.frame_for(boundaries, panel_h_in=6.2, panel_w_in=7.4, force_rot=0.0)
    plate_w = float(f.geometry.total_bounds[2])
    plate_h = float(f.geometry.total_bounds[3])
    margin_m = 0.08 * max(plate_w, plate_h)

    with rasterio.open(cdl_path) as ds:
        codes, ext = field_frame.reproject_into_frame(
            ds,
            f,
            dst_res=target_res_m,
            resampling=Resampling.nearest,
            dst_dtype="int16",
            nodata=-1,
            polygon_mask=True,
            mask_buffer_m=0.0,
            margin_m=margin_m,
        )
    ny, nx = codes.shape
    ext_w = float(ext[1] - ext[0])
    ext_h = float(ext[3] - ext[2])

    # Very elongated plans (Delaware: 1.5 km x 7.8 km of 5–12 ha parcels)
    # cannot be one page-sized north-up plate without shrinking the fields to
    # specks, so they use per-field panels instead.
    if max(ext_w / ext_h, ext_h / ext_w) > 1.6:
        return _m1_small_multiples(
            panel, spec, boundaries, cdl_path, year,
            target_res_m=target_res_m, basemap_res_m=basemap_res_m,
            basemap=basemap, field_frame=field_frame,
        )

    classes = _legend_classes(panel, grower, year)
    legend_rows = max(1, math.ceil(len(classes) / 5))
    legend_in = 0.56 + 0.24 * legend_rows
    title_in = 0.80

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        map_w_in = 7.0
        map_h_in = map_w_in * ext_h / ext_w
        if map_h_in > 7.4:
            map_h_in = 7.4
            map_w_in = map_h_in * ext_w / ext_h
        fig_h = map_h_in + title_in + legend_in
        fig, ax = plt.subplots(figsize=(map_w_in + 0.5, fig_h))
        drawn = basemap.draw(ax, f, ext, ny, nx, target_res_m=basemap_res_m)
        warnings: list[str] = []
        if not drawn:
            warnings.append("Imagery basemap unavailable (offline); showing layers only.")
            ax.set_facecolor("#f3f4f6")

        ax.imshow(_cdl_rgba(codes), extent=ext, origin="upper", zorder=1)
        f.geometry.boundary.plot(ax=ax, color="#111827", linewidth=1.0, zorder=3)
        points = f.geometry.representative_point()
        field_ids = (
            f.geometry["field_id"].astype(str)
            if "field_id" in f.geometry
            else f.geometry.index.astype(str)
        )
        rank = {
            k: i for i, k in enumerate(
                sorted(range(len(field_ids)),
                       key=lambda j: (-points.iloc[j].y, points.iloc[j].x))
            )
        }
        for i, (field_id, (x, y)) in enumerate(zip(field_ids, zip(points.x, points.y))):
            short = field_id.split("-", 1)[1] if "-" in field_id else field_id
            ax.annotate(
                short,
                (x, y), ha="center", va="center", fontsize=6.4, color="#111827",
                xytext=(0, 16 if rank[i] % 2 == 0 else -16), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
                zorder=5,
            )

        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel("metres east in field frame")
        ax.set_ylabel("metres north in field frame")
        field_frame.decorate(
            ax, f, callout=False, arrow_xy=(0.93, 0.08), arrow_r=0.09, arrow_fs=13
        )

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc="#%02x%02x%02x" % CDL_RGB[c], ec="none")
            for c in classes
            if c in CDL_RGB
        ]
        labels = [crop_name(c) for c in classes if c in CDL_RGB]
        if handles:
            fig.legend(
                handles, labels, ncol=5, loc="lower center",
                bbox_to_anchor=(0.5, 0.0), fontsize=6.8,
                title=f"CDL {year} class inside field boundaries", title_fontsize=7.4,
            )
        total_ha = float(boundaries.to_crs("EPSG:5070").area.sum() / 10_000.0)
        fig.suptitle(
            f"{spec.label} — {len(boundaries)} fields, {total_ha:,.0f} ha\n"
            f"CDL {year} class fill under bold field boundaries",
            fontsize=10.5, x=0.02, ha="left", y=0.985,
        )
        fig.subplots_adjust(
            left=0.08, right=0.99,
            top=1.0 - title_in / fig_h, bottom=legend_in / fig_h,
        )

    stats: dict[str, Any] = {
        "grower": spec.label,
        "year": year,
        "n_fields": int(len(boundaries)),
        "total_ha": round(total_ha, 1),
        "classes_present": [crop_name(c) for c in classes],
        "basemap_drawn": bool(drawn),
    }
    return FigureResult(
        figure_id="M1",
        figure=fig,
        stats=stats,
        story=(
            f"{spec.label}'s fields are not interchangeable parcels: the outlines show "
            f"how the {total_ha:,.0f} ha is split, while the {year} CDL fill shows what "
            "each block carried. Imagery underneath keeps the field shapes honest "
            "against roads, waterways, and neighbouring land."
        ),
        warnings=warnings,
    )


def _m1_small_multiples(
    panel: FieldPanel,
    spec,
    boundaries,
    cdl_path: Path,
    year: int,
    target_res_m: float,
    basemap_res_m: float,
    basemap,
    field_frame,
) -> FigureResult:
    """Per-field panels for grower plates too elongated for one page frame.

    Each panel keeps the field-local frame contract (auto-rotation, metre
    axes, boxed north arrow); the shared legend makes the CDL classes
    comparable across panels.
    """
    import rasterio
    from rasterio.warp import Resampling

    fields = boundaries.sort_values("field_id").reset_index(drop=True)
    field_ids = [str(v) for v in fields["field_id"]]
    areas = dict(
        zip(field_ids, fields.to_crs("EPSG:5070").area / 10_000.0)
    )
    classes = _legend_classes(panel, spec.label, year)
    comp = panel.composition
    comp = comp[(comp["grower"] == spec.label) & (comp["year"] == year)]
    dominant: dict[str, tuple[str, float]] = {}
    for fid in field_ids:
        sub = comp[comp["field_id"] == fid]
        if not sub.empty:
            row = sub.loc[sub["pct"].idxmax()]
            dominant[fid] = (str(row["crop"]), float(row["pct"]))

    ncols = 2
    nrows = math.ceil(len(fields) / ncols)
    legend_rows = max(1, math.ceil(len(classes) / 5))
    legend_in = 0.56 + 0.24 * legend_rows
    panel_w_in, panel_h_in = 3.4, 2.55
    fig_w = ncols * panel_w_in + 0.6
    fig_h = nrows * (panel_h_in + 0.55) + legend_in + 0.45
    frames, aspects = _small_multiple_frames(fields, field_frame, panel_w_in, panel_h_in)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h), squeeze=False,
            gridspec_kw={"height_ratios": _row_height_ratios(aspects, ncols)},
        )
        warnings: list[str] = []
        for ax, (_, field) in zip(axes.ravel(), fields.iterrows()):
            field_id = str(field["field_id"])
            short = field_id.split("-", 1)[1] if "-" in field_id else field_id
            f = frames[field_id]
            width = float(f.geometry.total_bounds[2])
            height = float(f.geometry.total_bounds[3])
            margin = 0.08 * max(width, height)
            with rasterio.open(cdl_path) as ds:
                codes, ext = field_frame.reproject_into_frame(
                    ds, f, dst_res=target_res_m,
                    resampling=Resampling.nearest, dst_dtype="int16", nodata=-1,
                    polygon_mask=True, mask_buffer_m=0.0, margin_m=margin,
                )
            if not basemap.draw(ax, f, ext, codes.shape[0], codes.shape[1], target_res_m=basemap_res_m):
                if "Imagery basemap unavailable (offline); showing layers only." not in warnings:
                    warnings.append("Imagery basemap unavailable (offline); showing layers only.")
                ax.set_facecolor("#f3f4f6")
            ax.imshow(_cdl_rgba(codes, alpha=0.72), extent=ext, origin="upper", zorder=1)
            f.geometry.boundary.plot(ax=ax, color="#111827", linewidth=1.0, zorder=3)
            # GeoPandas stamps projected axes with "Easting [metre]" itself;
            # clear it so only the outer panels carry the frame wording.
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect("equal")
            ax.set_anchor("N")
            field_frame.decorate(ax, f, callout=False, arrow_xy=(0.90, 0.87),
                                 arrow_r=0.10, arrow_fs=11)
            if f.rotated:
                ax.text(0.01, 0.99, f"rotated {f.rot:+.0f}°", transform=ax.transAxes,
                        fontsize=5.8, style="italic", color="#374151", va="top",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
            label = f"{short} — {areas[field_id]:.0f} ha"
            if field_id in dominant:
                crop, pct = dominant[field_id]
                label += f"\n{year}: {crop} {pct:.0f}%"
            ax.set_title(label, fontsize=8, loc="left")
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(fields):]:
            ax.set_axis_off()
        for ax in axes[:, 0]:
            ax.set_ylabel("metres north", fontsize=7)
        for ax in axes[-1, :]:
            ax.set_xlabel("metres east", fontsize=7)

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc="#%02x%02x%02x" % CDL_RGB[c], ec="none")
            for c in classes
            if c in CDL_RGB
        ]
        labels = [crop_name(c) for c in classes if c in CDL_RGB]
        if handles:
            fig.legend(
                handles, labels, ncol=5, loc="lower center",
                bbox_to_anchor=(0.5, 0.0), fontsize=6.8,
                title=f"CDL {year} class inside field boundaries", title_fontsize=7.4,
            )
        total_ha = float(sum(areas.values()))
        fig.suptitle(
            f"{spec.label} — {len(fields)} fields, {total_ha:,.0f} ha\n"
            "Fields are far apart, so each panel is its own field-local frame",
            fontsize=10.5, x=0.02, ha="left", y=0.99,
        )
        fig.subplots_adjust(
            left=0.07, right=0.99, top=1.0 - 0.8 / fig_h,
            bottom=legend_in / fig_h, hspace=0.24, wspace=0.16,
        )

    return FigureResult(
        figure_id="M1",
        figure=fig,
        stats={
            "grower": spec.label,
            "year": year,
            "n_fields": int(len(fields)),
            "total_ha": round(total_ha, 1),
            "classes_present": [crop_name(c) for c in classes],
            "per_field_dominant": {k: v[0] for k, v in dominant.items()},
            "layout": "small multiples (elongated plan)",
        },
        story=(
            f"{spec.label}'s {len(fields)} parcels are strung across the landscape, so "
            "one page-scale frame would shrink them to specks. Each panel keeps its "
            "own field-local frame; together they show the same thing as the county "
            "plates — where the ground is and what it carried."
        ),
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# M2 — 2021 -> 2025 dominant-crop flip
# --------------------------------------------------------------------------
def m2_crop_flip_map(
    panel: FieldPanel,
    grower: str,
    year_from: int = 2021,
    year_to: int = 2025,
    target_res_m: float = 18.0,
    basemap_res_m: float = 30.0,
) -> FigureResult:
    """Where the dominant crop at the two endpoints agrees or changes.

    Corn Belt fields alternate corn and soybeans annually. A window of five
    years therefore lands on the same parity at both ends on most fields, so a
    sea of "Same" is the expected signature of the rotation — not evidence
    that nothing moved. The intermediate seasons are C2's job.
    """
    spec = panel.grower(grower)
    basemap, field_frame = _runtime_lib(panel)
    boundaries = _grower_boundaries(panel, spec)
    sub = panel.panel[
        (panel.panel["grower"] == grower)
        & panel.panel["year"].isin([year_from, year_to])
    ]
    codes = {
        year: sub[sub["year"] == year].set_index("field_id")["crop_code"].to_dict()
        for year in (year_from, year_to)
    }
    names = {
        year: sub[sub["year"] == year].set_index("field_id")["crop"].to_dict()
        for year in (year_from, year_to)
    }
    f = field_frame.frame_for(boundaries, panel_h_in=6.2, panel_w_in=7.4, force_rot=0.0)
    ext, nr, nc = _plate_extent(f, target_res_m)
    ext_w, ext_h = float(ext[1] - ext[0]), float(ext[3] - ext[2])
    if max(ext_w / ext_h, ext_h / ext_w) > 1.6:
        return _m2_small_multiples(
            panel, spec, boundaries, codes, names, year_from, year_to,
            target_res_m, basemap_res_m, basemap, field_frame,
        )

    field_ids = [str(v) for v in f.geometry["field_id"]]
    flips = {
        fid: _flip_class(codes[year_from].get(fid), codes[year_to].get(fid))
        for fid in field_ids
    }
    counts = pd.Series(flips).value_counts()
    classes = [c for c in FLIP_ORDER if counts.get(c, 0) > 0]
    legend_in = 1.12
    title_in = 0.78
    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        map_w_in = 7.0
        map_h_in = map_w_in * ext_h / ext_w
        if map_h_in > 6.8:
            map_h_in = 6.8
            map_w_in = map_h_in * ext_w / ext_h
        fig_h = map_h_in + title_in + legend_in
        fig, ax = plt.subplots(figsize=(map_w_in + 0.5, fig_h))
        drawn = basemap.draw(ax, f, ext, nr, nc, target_res_m=basemap_res_m)
        warnings: list[str] = []
        if not drawn:
            warnings.append("Imagery basemap unavailable (offline); showing layers only.")
            ax.set_facecolor("#f3f4f6")

        colors = [FLIP_COLORS[flips[fid]] for fid in field_ids]
        f.geometry.plot(
            ax=ax, color=colors, edgecolor="white", linewidth=0.0,
            alpha=0.80, zorder=2,
        )
        f.geometry.boundary.plot(ax=ax, color="#111827", linewidth=1.0, zorder=3)
        points = f.geometry.representative_point()
        # Rank labels top-to-bottom, not by row order: lexicographic field ids
        # put r1 and r4 next to each other on the plate while their indices are
        # both even, which used to stack their callouts.
        rank = {
            k: i for i, k in enumerate(
                sorted(range(len(field_ids)),
                       key=lambda j: (-points.iloc[j].y, points.iloc[j].x))
            )
        }
        for i, (fid, (x, y)) in enumerate(zip(field_ids, zip(points.x, points.y))):
            short = fid.split("-", 1)[1] if "-" in fid else fid
            seq = f"{_crop_abbrev(codes[year_from].get(fid))}→{_crop_abbrev(codes[year_to].get(fid))}"
            ax.annotate(
                f"{short}\n{seq}",
                (x, y), ha="center", va="center", fontsize=6.0, color="#111827",
                xytext=(0, 16 if rank[i] % 2 == 0 else -16), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.84),
                zorder=5,
            )
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel("metres east in field frame")
        ax.set_ylabel("metres north in field frame")
        field_frame.decorate(
            ax, f, callout=False, arrow_xy=(0.93, 0.08), arrow_r=0.09, arrow_fs=13
        )

        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=FLIP_COLORS[c], ec="none")
            for c in classes
        ]
        labels = [f"{c} ({int(counts.get(c, 0))})" for c in classes]
        fig.legend(
            handles, labels, ncol=len(classes), loc="lower center",
            bbox_to_anchor=(0.5, 0.035), fontsize=7.2,
            title=f"Dominant CDL class in {year_from} → {year_to} (field count)",
            title_fontsize=7.8,
        )
        n_same = int(counts.get("Same", 0))
        fig.suptitle(
            f"{spec.label} — endpoint crop change, {year_from} → {year_to}\n"
            f"{n_same} of {len(field_ids)} fields return to the same dominant class",
            fontsize=10.5, x=0.02, ha="left", y=0.985,
        )
        fig.text(
            0.01, 0.005,
            "\"Same\" is an endpoint statement, not a static rotation: corn/soy fields "
            "alternate annually, so a five-year window preserves parity. See C2 for the "
            "intermediate seasons.",
            fontsize=6.9, color="#4b5563",
        )
        fig.subplots_adjust(
            left=0.08, right=0.99, top=1.0 - title_in / fig_h,
            bottom=(legend_in + 0.12) / fig_h,
        )

    stats = {
        "grower": spec.label,
        "year_from": year_from,
        "year_to": year_to,
        "n_fields": int(len(field_ids)),
        "counts": {c: int(counts.get(c, 0)) for c in FLIP_ORDER},
        "per_field": {
            fid: {
                "from": None if codes[year_from].get(fid) is None else int(codes[year_from][fid]),
                "to": None if codes[year_to].get(fid) is None else int(codes[year_to][fid]),
                "from_name": names[year_from].get(fid),
                "to_name": names[year_to].get(fid),
                "class": flips[fid],
            }
            for fid in field_ids
        },
        "basemap_drawn": bool(drawn),
    }
    n_changed = len(field_ids) - n_same
    return FigureResult(
        figure_id="M2",
        figure=fig,
        stats=stats,
        story=(
            f"{n_same} of {spec.label}'s {len(field_ids)} fields carry the same dominant class "
            f"at both endpoints; {n_changed} changed. On the Corn Belt plates that match is "
            "the rotation's parity, not stasis — the intermediate seasons swap corn and "
            "soybeans. Delaware's endpoint chain (barley/soy, wheat/soy, produce) is the "
            "place the map actually diverges."
        ),
        warnings=warnings + [
            "A five-year endpoint window preserves corn/soy parity; do not read 'Same' as "
            "an unchanged rotation."
        ],
    )


def _m2_small_multiples(
    panel: FieldPanel,
    spec,
    boundaries,
    codes,
    names,
    year_from: int,
    year_to: int,
    target_res_m: float,
    basemap_res_m: float,
    basemap,
    field_frame,
) -> FigureResult:
    """Per-field endpoint-change panels for an elongated grower plan."""
    fields = boundaries.sort_values("field_id").reset_index(drop=True)
    field_ids = [str(v) for v in fields["field_id"]]
    areas = dict(zip(field_ids, fields.to_crs("EPSG:5070").area / 10_000.0))
    flips = {
        fid: _flip_class(codes[year_from].get(fid), codes[year_to].get(fid))
        for fid in field_ids
    }
    counts = pd.Series(flips).value_counts()
    classes = [c for c in FLIP_ORDER if counts.get(c, 0) > 0]
    ncols = 2
    nrows = math.ceil(len(fields) / ncols)
    panel_w_in, panel_h_in = 3.4, 2.55
    fig_w = ncols * panel_w_in + 0.6
    fig_h = nrows * (panel_h_in + 0.55) + 1.35
    frames, aspects = _small_multiple_frames(fields, field_frame, panel_w_in, panel_h_in)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h), squeeze=False,
            gridspec_kw={"height_ratios": _row_height_ratios(aspects, ncols)},
        )
        warnings: list[str] = []
        for ax, (_, field) in zip(axes.ravel(), fields.iterrows()):
            fid = str(field["field_id"])
            short = fid.split("-", 1)[1] if "-" in fid else fid
            f = frames[fid]
            ext, nr, nc = _plate_extent(f, target_res_m)
            if not basemap.draw(ax, f, ext, nr, nc, target_res_m=basemap_res_m):
                if "Imagery basemap unavailable (offline); showing layers only." not in warnings:
                    warnings.append("Imagery basemap unavailable (offline); showing layers only.")
                ax.set_facecolor("#f3f4f6")
            f.geometry.plot(
                ax=ax, color=[FLIP_COLORS[flips[fid]]], edgecolor="white",
                linewidth=0.0, alpha=0.82, zorder=2,
            )
            f.geometry.boundary.plot(ax=ax, color="#111827", linewidth=1.0, zorder=3)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect("equal")
            ax.set_anchor("N")
            ax.set_xlabel("")
            ax.set_ylabel("")
            field_frame.decorate(
                ax, f, callout=False, arrow_xy=(0.90, 0.87), arrow_r=0.10, arrow_fs=11
            )
            if f.rotated:
                ax.text(
                    0.01, 0.99, f"rotated {f.rot:+.0f}°", transform=ax.transAxes,
                    fontsize=5.8, style="italic", color="#374151", va="top",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                )
            seq = (
                f"{_crop_abbrev(codes[year_from].get(fid))}→"
                f"{_crop_abbrev(codes[year_to].get(fid))}"
            )
            ax.set_title(f"{short} — {areas[fid]:.0f} ha\n{year_from}→{year_to}: {seq}",
                         fontsize=8, loc="left")
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(fields):]:
            ax.set_axis_off()
        for ax in axes[:, 0]:
            ax.set_ylabel("metres north", fontsize=7)
        for ax in axes[-1, :]:
            ax.set_xlabel("metres east", fontsize=7)

        handles = [plt.Rectangle((0, 0), 1, 1, fc=FLIP_COLORS[c], ec="none") for c in classes]
        labels = [f"{c} ({int(counts.get(c, 0))})" for c in classes]
        fig.legend(
            handles, labels, ncol=len(classes), loc="lower center",
            bbox_to_anchor=(0.5, 0.025), fontsize=7.2,
            title=f"Dominant CDL class in {year_from} → {year_to} (field count)",
            title_fontsize=7.8,
        )
        fig.suptitle(
            f"{spec.label} — endpoint crop change, {year_from} → {year_to}\n"
            "Fields are far apart, so each panel is its own field-local frame",
            fontsize=10.5, x=0.02, ha="left", y=0.99,
        )
        fig.text(
            0.01, 0.004,
            "\"Same\" is an endpoint statement, not a static rotation; the intermediate "
            "seasons are C2. Delaware's double-crop chain is where the endpoints diverge.",
            fontsize=6.9, color="#4b5563",
        )
        fig.subplots_adjust(
            left=0.07, right=0.99, top=1.0 - 0.95 / fig_h,
            bottom=0.15, hspace=0.30, wspace=0.16,
        )

    return FigureResult(
        figure_id="M2",
        figure=fig,
        stats={
            "grower": spec.label,
            "year_from": year_from,
            "year_to": year_to,
            "n_fields": int(len(fields)),
            "counts": {c: int(counts.get(c, 0)) for c in FLIP_ORDER},
            "per_field": {
                fid: {
                    "from": None if codes[year_from].get(fid) is None else int(codes[year_from][fid]),
                    "to": None if codes[year_to].get(fid) is None else int(codes[year_to][fid]),
                    "from_name": names[year_from].get(fid),
                    "to_name": names[year_to].get(fid),
                    "class": flips[fid],
                }
                for fid in field_ids
            },
            "layout": "small multiples (elongated plan)",
        },
        story=(
            f"None of {spec.label}'s {len(fields)} parcels keeps one crop across the window; "
            "the endpoint classes trace a double-crop/produce chain rather than a corn/soy "
            "alternation. The per-field panels keep each small parcel large enough to read."
        ),
        warnings=warnings + [
            "A five-year endpoint window preserves corn/soy parity where that rotation "
            "applies; do not read 'Same' as an unchanged rotation."
        ],
    )


# --------------------------------------------------------------------------
# M3 — annual-mean NDVI map with its gaps
# --------------------------------------------------------------------------
def _ndvi_grid(
    field_frame,
    f,
    paths: dict[str, str],
    target_res_m: float,
    margin_frac: float = 0.08,
):
    """Mosaic per-field NDVI composites into the frame, masked to each polygon."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    from rasterio.warp import Resampling

    ext, nr, nc = _plate_extent(f, target_res_m, margin_frac)
    transform = Affine(target_res_m, 0, ext[0], 0, -target_res_m, ext[3])
    grid = np.full((nr, nc), np.nan, dtype="float64")
    for _, row in f.geometry.iterrows():
        fid = str(row["field_id"])
        path = paths.get(fid)
        if not path:
            continue
        with rasterio.open(path) as ds:
            arr, _ext = field_frame.reproject_into_frame(
                ds, f, dst_res=target_res_m, resampling=Resampling.bilinear,
                margin_m=margin_frac * max(float(f.geometry.total_bounds[2]),
                                           float(f.geometry.total_bounds[3])),
            )
        if arr.shape != (nr, nc):
            raise RuntimeError(f"NDVI grid for {fid} is {arr.shape}, expected {(nr, nc)}")
        vals = np.ma.filled(arr.astype("float64"), np.nan)
        inside = geometry_mask(
            [row.geometry.__geo_interface__], out_shape=(nr, nc),
            transform=transform, invert=True,
        )
        ok = inside & np.isfinite(vals)
        grid[ok] = vals[ok]
    return grid, ext, nr, nc


def m3_ndvi_map(
    panel: FieldPanel,
    grower: str,
    year: int = 2025,
    target_res_m: float = 10.0,
    basemap_res_m: float = 30.0,
) -> FigureResult:
    """One grower's annual-mean NDVI with a fixed scale and visible gaps.

    Fields without a composite are hatched, never interpolated or dropped from
    the plate. The 0.2–0.7 scale is fixed across growers so a greener plate
    cannot be a scale artefact.
    """
    spec = panel.grower(grower)
    basemap, field_frame = _runtime_lib(panel)
    boundaries = _grower_boundaries(panel, spec)
    sub = panel.panel[(panel.panel["grower"] == grower) & (panel.panel["year"] == year)]
    paths = {
        str(r.field_id): str(r.ndvi_path)
        for r in sub.itertuples()
        if getattr(r, "ndvi_path", None) and isinstance(r.ndvi_path, str)
    }
    f = field_frame.frame_for(boundaries, panel_h_in=6.2, panel_w_in=7.4, force_rot=0.0)
    ext, nr, nc = _plate_extent(f, target_res_m)
    ext_w, ext_h = float(ext[1] - ext[0]), float(ext[3] - ext[2])
    if max(ext_w / ext_h, ext_h / ext_w) > 1.6:
        return _m3_small_multiples(
            panel, spec, boundaries, paths, year, target_res_m,
            basemap_res_m, basemap, field_frame, sub,
        )

    field_ids = [str(v) for v in f.geometry["field_id"]]
    missing = [fid for fid in field_ids if fid not in paths]
    grid, ext, nr, nc = _ndvi_grid(field_frame, f, paths, target_res_m)
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    title_in = 0.75
    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        map_w_in = 7.0
        map_h_in = map_w_in * ext_h / ext_w
        if map_h_in > 6.6:
            map_h_in = 6.6
            map_w_in = map_h_in * ext_w / ext_h
        fig_h = map_h_in + title_in + 1.25
        fig, ax = plt.subplots(figsize=(map_w_in + 0.5, fig_h))
        drawn = basemap.draw(ax, f, ext, nr, nc, target_res_m=basemap_res_m)
        warnings: list[str] = []
        if not drawn:
            warnings.append("Imagery basemap unavailable (offline); showing layers only.")
            ax.set_facecolor("#f3f4f6")
        im = ax.imshow(
            np.ma.masked_invalid(grid), extent=ext, origin="upper", cmap=cmap,
            vmin=0.2, vmax=0.7, zorder=1, interpolation="nearest",
        )
        if missing:
            f.geometry[f.geometry["field_id"].astype(str).isin(missing)].plot(
                ax=ax, facecolor="#d1d5db", edgecolor="#6b7280", hatch="////",
                linewidth=0.0, alpha=0.50, zorder=2,
            )
        f.geometry.boundary.plot(ax=ax, color="#111827", linewidth=1.0, zorder=3)
        points = f.geometry.representative_point()
        rank = {
            k: i for i, k in enumerate(
                sorted(range(len(field_ids)),
                       key=lambda j: (-points.iloc[j].y, points.iloc[j].x))
            )
        }
        for i, (fid, (x, y)) in enumerate(zip(field_ids, zip(points.x, points.y))):
            short = fid.split("-", 1)[1] if "-" in fid else fid
            ax.annotate(
                short, (x, y), ha="center", va="center", fontsize=6.2, color="#111827",
                xytext=(0, 16 if rank[i] % 2 == 0 else -16), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.84),
                zorder=5,
            )
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel("metres east in field frame")
        ax.set_ylabel("metres north in field frame")
        field_frame.decorate(
            ax, f, callout=False, arrow_xy=(0.93, 0.08), arrow_r=0.09, arrow_fs=13
        )
        cax = fig.add_axes([0.30, 0.105, 0.40, 0.018])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Annual-mean NDVI, fixed 0.2–0.7 scale", fontsize=7.4)
        cb.ax.tick_params(labelsize=7)
        if missing:
            fig.legend(
                handles=[plt.Rectangle((0, 0), 1, 1, fc="#d1d5db", ec="#6b7280", hatch="////")],
                labels=[f"No {year} composite ({len(missing)} fields)"],
                loc="lower center", bbox_to_anchor=(0.5, 0.045), fontsize=7.2,
                frameon=False,
            )
        n_present = len(field_ids) - len(missing)
        subtitle = (
            f"{n_present}/{len(field_ids)} fields have a composite; hatched = no imagery"
            if missing else f"all {len(field_ids)} fields have a composite"
        )
        fig.suptitle(
            f"{spec.label} — annual-mean NDVI, {year}\n{subtitle}",
            fontsize=10.5, x=0.02, ha="left", y=0.985,
        )
        fig.text(0.01, 0.006, ndvi_coverage_text(panel), fontsize=7.0, color="#4b5563")
        fig.subplots_adjust(
            left=0.08, right=0.99, top=1.0 - title_in / fig_h, bottom=0.20,
        )

    stats = {
        "grower": spec.label,
        "year": year,
        "n_fields": int(len(field_ids)),
        "n_present": int(len(field_ids) - len(missing)),
        "n_missing": int(len(missing)),
        "missing_fields": missing,
        "per_field": {
            str(r.field_id): {
                "ndvi_mean": None if pd.isna(r.ndvi_mean) else round(float(r.ndvi_mean), 3),
                "valid_frac": None if pd.isna(r.ndvi_valid_frac) else round(float(r.ndvi_valid_frac), 3),
            }
            for r in sub.itertuples()
        },
        "basemap_drawn": bool(drawn),
    }
    means = [v["ndvi_mean"] for v in stats["per_field"].values() if v["ndvi_mean"] is not None]
    spread = f"{min(means):.2f}–{max(means):.2f}" if means else "n/a"
    return FigureResult(
        figure_id="M3",
        figure=fig,
        stats=stats,
        story=(
            f"{stats['n_present']} of {spec.label}'s {len(field_ids)} fields have a {year} "
            f"composite (means {spread}). The fixed 0.2–0.7 scale makes the plate comparable "
            "with the other growers; the hatched parcels are the coverage gap, kept on the "
            "map rather than cropped away."
        ),
        warnings=warnings,
    )


def _m3_small_multiples(
    panel: FieldPanel,
    spec,
    boundaries,
    paths: dict[str, str],
    year: int,
    target_res_m: float,
    basemap_res_m: float,
    basemap,
    field_frame,
    sub,
) -> FigureResult:
    """Per-field NDVI panels for an elongated grower plan."""
    fields = boundaries.sort_values("field_id").reset_index(drop=True)
    field_ids = [str(v) for v in fields["field_id"]]
    areas = dict(zip(field_ids, fields.to_crs("EPSG:5070").area / 10_000.0))
    means = sub.set_index("field_id")["ndvi_mean"].to_dict()
    valids = sub.set_index("field_id")["ndvi_valid_frac"].to_dict()
    ncols = 2
    nrows = math.ceil(len(fields) / ncols)
    panel_w_in, panel_h_in = 3.4, 2.55
    fig_w = ncols * panel_w_in + 0.6
    fig_h = nrows * (panel_h_in + 0.55) + 1.55
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    missing = [fid for fid in field_ids if fid not in paths]
    frames, aspects = _small_multiple_frames(fields, field_frame, panel_w_in, panel_h_in)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h), squeeze=False,
            gridspec_kw={"height_ratios": _row_height_ratios(aspects, ncols)},
        )
        warnings: list[str] = []
        last_im = None
        for ax, (_, field) in zip(axes.ravel(), fields.iterrows()):
            fid = str(field["field_id"])
            short = fid.split("-", 1)[1] if "-" in fid else fid
            f = frames[fid]
            ext, nr, nc = _plate_extent(f, target_res_m)
            if not basemap.draw(ax, f, ext, nr, nc, target_res_m=basemap_res_m):
                if "Imagery basemap unavailable (offline); showing layers only." not in warnings:
                    warnings.append("Imagery basemap unavailable (offline); showing layers only.")
                ax.set_facecolor("#f3f4f6")
            grid, ext, nr, nc = _ndvi_grid(
                field_frame, f, {fid: paths[fid]} if fid in paths else {}, target_res_m
            )
            last_im = ax.imshow(
                np.ma.masked_invalid(grid), extent=ext, origin="upper", cmap=cmap,
                vmin=0.2, vmax=0.7, zorder=1, interpolation="nearest",
            )
            if fid in missing:
                f.geometry.plot(
                    ax=ax, facecolor="#d1d5db", edgecolor="#6b7280", hatch="////",
                    linewidth=0.0, alpha=0.50, zorder=2,
                )
            f.geometry.boundary.plot(ax=ax, color="#111827", linewidth=1.0, zorder=3)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect("equal")
            ax.set_anchor("N")
            ax.set_xlabel("")
            ax.set_ylabel("")
            field_frame.decorate(
                ax, f, callout=False, arrow_xy=(0.90, 0.87), arrow_r=0.10, arrow_fs=11
            )
            if f.rotated:
                ax.text(
                    0.01, 0.99, f"rotated {f.rot:+.0f}°", transform=ax.transAxes,
                    fontsize=5.8, style="italic", color="#374151", va="top",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                )
            if fid in paths and not pd.isna(means.get(fid, np.nan)):
                label = (
                    f"{short} — {areas[fid]:.0f} ha\n"
                    f"mean {float(means[fid]):.2f} · valid {float(valids[fid]):.0%}"
                )
            else:
                label = f"{short} — {areas[fid]:.0f} ha\nno {year} composite"
            ax.set_title(label, fontsize=8, loc="left")
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(fields):]:
            ax.set_axis_off()
        for ax in axes[:, 0]:
            ax.set_ylabel("metres north", fontsize=7)
        for ax in axes[-1, :]:
            ax.set_xlabel("metres east", fontsize=7)

        if last_im is not None:
            cax = fig.add_axes([0.30, 0.095, 0.40, 0.017])
            cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
            cb.set_label("Annual-mean NDVI, fixed 0.2–0.7 scale", fontsize=7.4)
            cb.ax.tick_params(labelsize=7)
        n_present = len(field_ids) - len(missing)
        fig.suptitle(
            f"{spec.label} — annual-mean NDVI, {year}\n"
            f"{n_present}/{len(field_ids)} fields have a composite; each panel is field-local",
            fontsize=10.5, x=0.02, ha="left", y=0.99,
        )
        fig.text(0.01, 0.005, ndvi_coverage_text(panel), fontsize=7.0, color="#4b5563")
        fig.subplots_adjust(
            left=0.07, right=0.99, top=1.0 - 0.95 / fig_h,
            bottom=0.13, hspace=0.30, wspace=0.16,
        )

    stats = {
        "grower": spec.label,
        "year": year,
        "n_fields": int(len(field_ids)),
        "n_present": int(len(field_ids) - len(missing)),
        "n_missing": int(len(missing)),
        "missing_fields": missing,
        "per_field": {
            str(r.field_id): {
                "ndvi_mean": None if pd.isna(r.ndvi_mean) else round(float(r.ndvi_mean), 3),
                "valid_frac": None if pd.isna(r.ndvi_valid_frac) else round(float(r.ndvi_valid_frac), 3),
            }
            for r in sub.itertuples()
        },
        "layout": "small multiples (elongated plan)",
    }
    return FigureResult(
        figure_id="M3",
        figure=fig,
        stats=stats,
        story=(
            f"{stats['n_present']} of {spec.label}'s {len(field_ids)} parcels have a {year} "
            "composite. The per-field panels keep each 5–12 ha parcel readable at the fixed "
            "0.2–0.7 scale; any missing composite would be hatched in place."
        ),
        warnings=warnings,
    )
