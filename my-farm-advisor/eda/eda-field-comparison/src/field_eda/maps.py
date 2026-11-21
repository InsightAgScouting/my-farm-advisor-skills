"""Maps for the field EDA subskill (Stage A: M1).

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

from .figures import FigureResult  # noqa: E402
from .panel import FieldPanel, crop_name  # noqa: E402
from .style import CDL_RGB, apply_style  # noqa: E402

STATE_FIPS = {"IL": "17", "IA": "19", "NE": "31", "DE": "10"}


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
    import geopandas as gpd
    import rasterio
    from rasterio.warp import Resampling

    farm_dir = panel.data_root / "growers" / spec.slug / "farms" / spec.farm
    boundary_path = farm_dir / "boundary" / "field_boundaries.geojson"
    boundaries = gpd.read_file(boundary_path)
    if boundaries.crs is None:
        boundaries = boundaries.set_crs("EPSG:4326")

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
    legend_in = 0.34 + 0.24 * legend_rows
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
        for i, (field_id, (x, y)) in enumerate(zip(field_ids, zip(points.x, points.y))):
            short = field_id.split("-", 1)[1] if "-" in field_id else field_id
            ax.annotate(
                short,
                (x, y), ha="center", va="center", fontsize=6.4, color="#111827",
                xytext=(0, 9 if i % 2 == 0 else -9), textcoords="offset points",
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
    legend_in = 0.34 + 0.24 * legend_rows
    panel_w_in, panel_h_in = 3.3, 2.9
    fig_w = ncols * panel_w_in + 0.6
    fig_h = nrows * (panel_h_in + 0.55) + legend_in + 0.5

    with plt.style.context("seaborn-v0_8-whitegrid"):
        apply_style()
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h), squeeze=False
        )
        warnings: list[str] = []
        for ax, (_, field) in zip(axes.ravel(), fields.iterrows()):
            field_id = str(field["field_id"])
            short = field_id.split("-", 1)[1] if "-" in field_id else field_id
            one = fields[fields["field_id"] == field_id]
            f = field_frame.frame_for(one, panel_h_in=panel_h_in, panel_w_in=panel_w_in)
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
            left=0.07, right=0.99, top=1.0 - 0.55 / fig_h,
            bottom=legend_in / fig_h, hspace=0.18, wspace=0.16,
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
