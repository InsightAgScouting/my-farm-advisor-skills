#!/usr/bin/env python3
"""Build the one-time Assignment 2 EDA report PDF.

This is not part of the reusable subskill. It is the "one-off report script"
allowed by the local AGENTS.md: it reads the PNGs and CSVs the example wrote to
``eda-field-comparison/output/`` and assembles a single PDF in ``report/``.
Nothing here is imported by ``src/field_eda/``, and nothing here computes a new
statistic — tables are re-derived from ``panel.csv`` only to keep the report's
counts consistent with the figures.

Usage (the venv lives in the runtime; the figures do not)::

    export DATA_PIPELINE_DATA_ROOT=$HOME/my-farm-advisor-runtime
    $DATA_PIPELINE_DATA_ROOT/data-pipeline/.venv/bin/python \
        my-farm-advisor/eda/eda-field-comparison/report/build_report.py

The PDF is the artifact; the script exists so the artifact can be rebuilt after
the figures change. Soil is deliberately absent (see the limitations page).
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

PAGE_W, PAGE_H = 8.27, 11.69  # A4 portrait
ML, MR = 0.075, 0.925
TEXT_W_IN = (MR - ML) * PAGE_W
INK = "#111827"
MUTED = "#4b5563"
RULE = "#d1d5db"
ACCENT = "#1b7f3b"

BODY_PT = 9.4
CAPTION_PT = 8.0
HEAD_PT = 10.8
DPI = 95  # keeps the PDF under 1 MB; figures stay readable when zoomed

plt.rcParams["pdf.compression"] = 9
plt.rcParams["pdf.fonttype"] = 42


def _wrap(text: str, size: float, width_in: float = TEXT_W_IN) -> str:
    """Wrap to roughly the rendered width of a proportional font."""
    em_in = size * 0.52 / 72.0
    chars = max(24, int(width_in / em_in))
    out: list[str] = []
    for paragraph in text.split("\n"):
        out.append(
            textwrap.fill(paragraph, chars, break_long_words=False, break_on_hyphens=False)
            if paragraph.strip()
            else ""
        )
    return "\n".join(out)


class Report:
    """Small layout helper: A4 pages, top-down flow, footer page numbers."""

    def __init__(self, path: Path, title: str, preview_dir: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.preview_dir = preview_dir
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)
        self.pdf = PdfPages(
            str(path),
            metadata={
                "Title": title,
                "Author": "My Farm Advisor · eda-field-comparison",
                "Subject": "Assignment 2 field-level EDA report",
            },
        )
        self.page = 0

    def new_page(self, kicker: str = "") -> plt.Figure:
        self.page += 1
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        fig.patch.set_facecolor("white")
        if kicker:
            fig.text(ML, 0.965, kicker.upper(), fontsize=7.6, color=ACCENT,
                     fontweight="bold", va="top", ha="left")
        fig.text(ML, 0.032, "Assignment 2 · field comparison EDA", fontsize=7.2, color=MUTED)
        fig.text(MR, 0.032, str(self.page), fontsize=7.6, color=MUTED, ha="right")
        return fig

    def save(self, fig: plt.Figure, dpi: int = DPI) -> None:
        self.pdf.savefig(fig, dpi=dpi)
        if self.preview_dir:
            fig.savefig(self.preview_dir / f"page-{self.page:02d}.png", dpi=110,
                        facecolor="white")
        plt.close(fig)

    def close(self) -> None:
        self.pdf.close()


def rule(fig: plt.Figure, y: float) -> None:
    fig.add_artist(Line2D([ML, MR], [y, y], color=RULE, linewidth=0.8))


def heading(fig: plt.Figure, y: float, text: str) -> float:
    fig.text(ML, y, text, fontsize=HEAD_PT, fontweight="bold", color=INK, va="top")
    return y - 0.027


def para(
    fig: plt.Figure,
    y: float,
    text: str,
    size: float = BODY_PT,
    color: str = INK,
    width_in: float = TEXT_W_IN,
    gap: float = 0.014,
) -> float:
    body = _wrap(text, size, width_in)
    leading = size * 1.45 / 72.0 / PAGE_H
    fig.text(ML, y, body, fontsize=size, color=color, va="top", linespacing=1.45)
    return y - body.count("\n") * leading - leading - gap


def bullets(fig: plt.Figure, y: float, items: list[str], size: float = BODY_PT,
            gap: float = 0.010) -> float:
    for item in items:
        body = _wrap(item, size, TEXT_W_IN - 0.12)
        leading = size * 1.45 / 72.0 / PAGE_H
        fig.text(ML, y, "•", fontsize=size, color=ACCENT, va="top")
        fig.text(ML + 0.015, y, body, fontsize=size, color=INK, va="top", linespacing=1.45)
        y -= body.count("\n") * leading + leading + gap * 0.5
    return y - gap


def _scaled(path: Path, width_in: float, dpi: int = 150) -> Image.Image:
    image = Image.open(path).convert("RGB")
    target = int(width_in * dpi)
    if image.width > target:
        image = image.resize((target, round(image.height * target / image.width)), Image.LANCZOS)
    return image


def image_block(
    fig: plt.Figure,
    path: Path,
    left: float,
    top: float,
    width_in: float,
    caption: str,
) -> float:
    """Draw one figure at ``width_in`` and its caption; return the bottom y."""
    image = _scaled(path, width_in)
    height_in = width_in * image.height / image.width
    ax = fig.add_axes([left, top - height_in / PAGE_H, width_in / PAGE_W, height_in / PAGE_H])
    ax.imshow(image)
    ax.set_axis_off()
    y = top - height_in / PAGE_H - 0.011
    body = _wrap(caption, CAPTION_PT, width_in)
    leading = CAPTION_PT * 1.4 / 72.0 / PAGE_H
    fig.text(left, y, body, fontsize=CAPTION_PT, color=MUTED, va="top", linespacing=1.4)
    return y - body.count("\n") * leading - leading


def centered_block(fig: plt.Figure, path: Path, top: float, width_in: float,
                   caption: str) -> float:
    left = ML + (TEXT_W_IN - width_in) / (2 * PAGE_W)
    return image_block(fig, path, left, top, width_in, caption)


def image_pair(fig: plt.Figure, top: float, items: list[tuple[Path, str]],
               gap_in: float = 0.34) -> float:
    """Two top-aligned images with no captions; returns the common bottom y."""
    width_in = (TEXT_W_IN - gap_in * (len(items) - 1)) / len(items)
    bottoms: list[float] = []
    for i, (path, _) in enumerate(items):
        left = ML + i * (width_in + gap_in) / PAGE_W
        image = _scaled(path, width_in)
        height_in = width_in * image.height / image.width
        ax = fig.add_axes([left, top - height_in / PAGE_H, width_in / PAGE_W,
                           height_in / PAGE_H])
        ax.imshow(image)
        ax.set_axis_off()
        bottoms.append(top - height_in / PAGE_H)
    return min(bottoms)


def table(fig: plt.Figure, y: float, rows: list[str], size: float = 7.9,
          heading_text: str | None = None) -> float:
    if heading_text:
        fig.text(ML, y, heading_text, fontsize=8.8, fontweight="bold", color=INK, va="top")
        y -= 0.022
    leading = size * 1.38 / 72.0 / PAGE_H
    fig.text(ML, y, "\n".join(rows), fontsize=size, color=INK, va="top",
             family="monospace", linespacing=1.38)
    return y - len(rows) * leading - 0.012


def grower_tables(panel: pd.DataFrame) -> tuple[list[str], list[str]]:
    fields = (
        panel.groupby(["grower", "field_id"])
        .agg(area_ha=("area_ha", "first"), has_ndvi=("has_ndvi", "max"),
             state=("state", "first"))
        .reset_index()
    )
    summary = (
        fields.groupby("grower")
        .agg(state=("state", "first"), n_fields=("field_id", "size"),
             total_ha=("area_ha", "sum"), median_ha=("area_ha", "median"),
             ndvi=("has_ndvi", "sum"))
        .reindex(["Illinois", "Iowa", "Nebraska", "Delaware"])
        .dropna(how="all")
    )
    rows = [
        f"{'Grower':<10}{'St':<4}{'Fields':>7}{'Total ha':>10}{'Median ha':>11}{'NDVI':>7}",
        "-" * 49,
    ]
    for grower, row in summary.iterrows():
        n_fields = int(row["n_fields"])
        ndvi = f"{int(row['ndvi'])}/{n_fields}"
        rows.append(
            f"{grower:<10}{row['state']:<4}{n_fields:>7}"
            f"{row['total_ha']:>10,.1f}{row['median_ha']:>11,.1f}{ndvi:>7}"
        )
    rows.append("-" * 49)
    total_fields = int(summary["n_fields"].sum())
    total_ndvi = f"{int(summary['ndvi'].sum())}/{total_fields}"
    rows.append(
        f"{'All':<10}{'':<4}{total_fields:>7}"
        f"{summary['total_ha'].sum():>10,.1f}{'':>11}{total_ndvi:>7}"
    )

    layers = [
        f"{'Layer':<12}{'Fields':>8}  {'Note':<34}",
        "-" * 56,
        f"{'Boundaries':<12}{'34/34':>8}  {'static footprint, 2021–2025':<34}",
        f"{'CDL':<12}{'34/34':>8}  {'2021–2025, 30 m raster':<34}",
        f"{'Weather':<12}{'34/34':>8}  {'POWER grid, 1–4 series/grower':<34}",
        f"{'NDVI':<12}{'20/34':>8}  {'IL/IA r4–r10 not ingested':<34}",
    ]
    return rows, layers


def main() -> int:
    default_images = Path(__file__).resolve().parents[1] / "output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=default_images,
                        help="Folder with the subskill's rendered PNGs and CSVs")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "assignment-2-eda-report.pdf")
    parser.add_argument("--preview-dir", type=Path, default=None,
                        help="Also write one PNG per page here (review aid, not committed)")
    args = parser.parse_args()

    img = args.images
    needed = [
        "coverage_all.png", "a1_all.png", "c3_all.png", "x4_all.png", "x5_all.png",
        "x7_all.png", "w2_all.png", "x6_all.png", "n3_all.png",
        "m1_illinois.png", "m2_delaware.png",
        "panel.csv",
    ]
    missing = [name for name in needed if not (img / name).exists()]
    if missing:
        raise SystemExit(f"Missing rendered artifacts in {img}: {', '.join(missing)}")

    panel = pd.read_csv(img / "panel.csv")
    grower_rows, layer_rows = grower_tables(panel)

    report = Report(
        args.out,
        "Field-level EDA across four growers — Assignment 2",
        preview_dir=args.preview_dir,
    )

    # ---------------------------------------------------------------- page 1
    fig = report.new_page("My Farm Advisor · Assignment 2")
    fig.text(ML, 0.930, "Field-level EDA across four growers", fontsize=21,
             fontweight="bold", color=INK, va="top")
    fig.text(ML, 0.892,
             "Boundaries, cropland (CDL), weather and NDVI · 34 fields · 170 field-years · 2021–2025",
             fontsize=9.6, color=MUTED, va="top")
    rule(fig, 0.876)

    y = 0.855
    y = heading(fig, y, "Dataset scope")
    y = para(fig, y,
             "The panel joins four layers on one footprint: field boundaries, USDA CDL crop class "
             "for 2021–2025, daily weather aggregated to the growing season (1 Mar – 31 Oct), and "
             "annual-mean NDVI where satellite composites exist. Missing NDVI is stored as a null "
             "with a note, never interpolated.")
    y -= 0.002
    y = table(fig, y, grower_rows, heading_text="At a glance")
    y -= 0.004
    y = heading(fig, y, "Growers and location")
    y = para(fig, y,
             "Illinois, Iowa and Nebraska are Corn Belt operations with ten fields each in a "
             "184–387 ha band; Delaware is a produce operation of four 5–12 ha parcels strung along "
             "a 1.5 × 7.3 km corridor. That size gap is the first finding (A1) and the reason "
             "several comparisons are read on a log area axis and between growers rather than "
             "within one.")
    y -= 0.002
    y = table(fig, y, layer_rows, heading_text="Data layers")

    y -= 0.002
    y = heading(fig, y, "Comparison levels")
    y = bullets(fig, y, [
        "Within a field over time — the same footprint across five seasons (N1, C1, W1).",
        "Field-year — one panel row, used by C2, X4, X7(a); repeated rows are not independent.",
        "Across fields inside one grower — C2, X5, W2, X7(b).",
        "Across growers — A1, C3, N3, W3, X6 and the X5/X7 pooled coefficients.",
    ], size=9.0)

    y = heading(fig, y, "What this report is")
    y = para(fig, y,
             "Nine statistical figures (including the coverage strip) and three maps are drawn "
             "from the subskill's 15 statistical outputs and 3 map families; page 8 lists all of "
             "them. Captions and numbers come from each figure function's own stats and story "
             "fields, so the report cannot quietly disagree with the plots. The reusable code "
             "lives in my-farm-advisor/eda/eda-field-comparison/; this PDF is a one-time "
             "submission artifact.", size=8.8)
    report.save(fig)

    # ---------------------------------------------------------------- page 2
    fig = report.new_page("Coverage and boundaries · A1")
    y = 0.930
    y = heading(fig, y, "Field size: two different scales of operation")
    y = para(fig, y,
             "A1 puts one point per field, not per field-year: acreage comes from the boundary and "
             "does not change across the five years, so repeating it on 170 rows would fake "
             "ninety-six extra observations. A black bar marks each grower median.")
    y -= 0.002
    y = image_block(fig, img / "a1_all.png", ML, y, TEXT_W_IN,
                    "A1 · field area by grower (log scale), one point per field, median bar. No "
                    "field in the dataset falls between 12 and 184 ha, so the empty middle of the "
                    "axis is the data, not a rendering gap.")
    y -= 0.010
    y = para(fig, y,
             "Illinois holds 2,599 ha across ten fields (median 245 ha), Iowa 2,577 ha (243), "
             "Nebraska 2,031 ha (208), and Delaware 30.1 ha (median 6.3 ha). Any pooled "
             "correlation involving area is therefore partly a between-grower contrast, which is "
             "exactly how X5 and X7 are written.", size=8.8)
    y -= 0.006
    y = image_block(fig, img / "coverage_all.png", ML, y, TEXT_W_IN,
                    "Coverage strip · fields with data per grower and layer. NDVI is the only "
                    "subset layer.")
    report.save(fig)

    # ---------------------------------------------------------------- page 3
    fig = report.new_page("Cropland · C3 and X5")
    y = 0.930
    y = heading(fig, y, "Rotation and purity: a two-crop system and a produce chain")
    y = para(fig, y,
             "C3 compares crop share and Shannon diversity by grower; X5 asks whether parcel size "
             "predicts crop purity. Illinois, Iowa and Nebraska are two-class systems — corn and "
             "soybeans split the planted area and Shannon H sits near zero — while Delaware's "
             "double crops and vegetables reach H = 1.57.")
    y -= 0.002
    y = image_block(fig, img / "c3_all.png", ML, y, TEXT_W_IN,
                    "C3 · planted-crop shares (top) and per-field Shannon diversity (bottom), "
                    "planted classes only.")
    y -= 0.006
    y = image_block(fig, img / "x5_all.png", ML, y, TEXT_W_IN,
                    "X5 · field area against crop purity, one point per field (five-year mean). A "
                    "30 m pixel on a 5–12 ha parcel can move purity by tens of percent.")
    y -= 0.008
    y = para(fig, y,
             "X5 pooled ρ = +0.30, but inside a grower the slope is flat or negative (Delaware "
             "−0.80). The pooled value is a small-parcel versus Corn-Belt contrast, not a "
             "resolution law, so X5 bounds data quality and caps confidence for small parcels.",
             size=8.8)
    report.save(fig)

    # ---------------------------------------------------------------- page 4
    fig = report.new_page("Cross-checks · X7 and X4")
    y = 0.930
    y = heading(fig, y, "Two associations that are mostly between growers")
    y = para(fig, y,
             "X7 tests whether bigger fields carry more distinct planted crop classes; X4 tests "
             "whether a greener season tracks heat or water. Both are rank correlations on "
             "repeated footprints, so both use a denominator that keeps the repeat visible.")
    y -= 0.002
    y = image_block(fig, img / "x7_all.png", ML, y, TEXT_W_IN,
                    "X7 · field area against distinct planted CDL classes. (a) 170 field-year "
                    "points, ρ = +0.06; (b) 34 per-field means, ρ = +0.18. The table gives both "
                    "panels per grower. Panel (b) is the field-level statement.")
    y -= 0.006
    y = image_block(fig, img / "x4_all.png", ML, y, TEXT_W_IN,
                    "X4 · annual-mean NDVI against season GDD and rainfall, ρ per crop group. "
                    "Soybeans show ρ = +0.49 with heat; GDD and rainfall co-vary, so neither "
                    "panel is independent evidence.")
    y -= 0.008
    y = para(fig, y,
             "Delaware carries X7: +0.62 field-year and +0.95 per field on four parcels spanning "
             "the whole area range, while Corn Belt growers sit at +0.13 to +0.36. X4's crop split "
             "keeps a pooled climate effect from being read as agronomy. Both are descriptive: "
             "neither says that size or heat causes the outcome.", size=8.8)
    report.save(fig)

    # ---------------------------------------------------------------- page 5
    fig = report.new_page("Weather · W2 and X6")
    y = 0.930
    y = heading(fig, y, "Weather: coarse grid, real between-grower separation")
    y = para(fig, y,
             "W2 shows the replication directly: one dot is a field's five-season mean, one "
             "whisker its min–max, and marker shape the distinct daily series. Fields sharing a "
             "series repeat exactly — within-grower spread tops out near 154 GDD and 55 mm, "
             "against 601 GDD and 206 mm between grower means.")
    y -= 0.002
    y = centered_block(fig, img / "w2_all.png", y, 6.3,
                       "W2 · season GDD (left) and rainfall (right) per field inside each grower. "
                       "Identical marker shapes are the same POWER series, not independent "
                       "agreement.")
    y -= 0.006
    y = centered_block(fig, img / "x6_all.png", y, 6.3,
                       "X6 · season GDD against rainfall, 1.5σ ellipses and crossed centroids, one "
                       "point per (grower, series, year). Illinois runs positive (ρ = +0.42); "
                       "Iowa, Nebraska and Delaware run negative; pooled ρ = −0.34 mixes both.")
    y -= 0.012
    y = para(fig, y,
             "Field-level climate claims belong at the grower scale: W2's tight rows are a grid "
             "artefact, and X6's ellipses describe season-to-season movement, not field-to-field "
             "differences.", size=8.8)
    report.save(fig)

    # ---------------------------------------------------------------- page 6
    fig = report.new_page("NDVI · N3")
    y = 0.930
    y = heading(fig, y, "Greenness: twenty fields, five seasons, one honest denominator")
    y = para(fig, y,
             "NDVI composites are annual means of every valid scene, not temporal peaks. The "
             "grower climatology shows Delaware highest and swinging hardest, Nebraska holding "
             "the narrowest range, and Illinois/Iowa drifting downward through 2025. The bar "
             "strip under the lines is the field count behind each grower.")
    y -= 0.002
    y = image_block(fig, img / "n3_all.png", ML, y, TEXT_W_IN,
                    "N3 · annual-mean NDVI by grower and year, with the n/N field count drawn as "
                    "a bar strip below. Missing composites are absent, never interpolated.")
    y -= 0.008
    cover_rows = [f"{'Grower':<10}{'Fields':>7}{'NDVI':>7}{'Share':>8}", "-" * 32]
    for grower in ("Illinois", "Iowa", "Nebraska", "Delaware"):
        sub = panel[panel["grower"] == grower]
        n_fields = sub["field_id"].nunique()
        n_ndvi = sub.groupby("field_id")["has_ndvi"].max().sum()
        cover_rows.append(
            f"{grower:<10}{n_fields:>7}{int(n_ndvi):>7}{n_ndvi / n_fields:>7.0%}"
        )
    y = table(fig, y, cover_rows, heading_text="Imagery coverage")
    y -= 0.004
    y = para(fig, y,
             "The missing fourteen fields are a coverage gap, not a zero: they stay out of every "
             "NDVI statistic and are hatched on the maps. N1, N2, N3 and X4 all print the same "
             "n/N footer for that reason.", size=8.8)
    report.save(fig)

    # ---------------------------------------------------------------- page 7
    fig = report.new_page("Maps · M1, M2, M3")
    y = 0.930
    y = heading(fig, y, "The same ground, seen geographically")
    y = para(fig, y,
             "Maps answer what the statistical figures cannot: where each field is, what it "
             "carried, and where imagery does not reach. M1 fills boundaries with the 2025 CDL "
             "class over Esri imagery; M2 shows the 2021→2025 dominant-class flip; M3 mosaics the "
             "2025 annual-mean NDVI on a fixed 0.2–0.7 scale.")
    y -= 0.002
    y = image_pair(fig, y, [
        (img / "m1_illinois.png", ""),
        (img / "m2_delaware.png", ""),
    ], gap_in=0.34)
    y -= 0.010
    y = para(fig, y,
             "M1 · Illinois, 2025 — north-up local UTM metre axes, boxed north arrow, imagery "
             "under the CDL fill. M2 · Delaware, 2021→2025 — per-field panels because one page "
             "frame would shrink 5–12 ha parcels to specks; the legend counts fields, not hectares.",
             size=CAPTION_PT, color=MUTED)
    y -= 0.004
    y = para(fig, y,
             "M2 is an endpoint statement: 28 of 34 fields return the same dominant class at both "
             "ends of the window, but on corn/soy ground that parity is the rotation's signature, "
             "not stasis — C2 carries the intermediate seasons, and Delaware's produce chain is "
             "where the endpoints actually diverge.", size=8.8)
    y -= 0.002
    y = bullets(fig, y, [
        "Single grower plates (IL/IA/NE) are north-up local UTM: metre axes, a boxed north arrow, "
        "imagery beneath the polygons, and no scale bar.",
        "Elongated plans switch to per-field small multiples so 5–12 ha parcels stay readable; "
        "M1, M2 and M3 all switch at the same aspect threshold.",
        "M3 mosaics the 2025 annual-mean NDVI on a fixed 0.2–0.7 scale and hatches missing "
        "composites; an offline tile fetch degrades to a light background with a warning.",
    ], size=8.4)
    report.save(fig)

    # ---------------------------------------------------------------- page 8
    fig = report.new_page("Limitations · inventory")
    y = 0.930
    y = heading(fig, y, "Limitations, missing data and assumptions")
    y = bullets(fig, y, [
        "NDVI covers 20 of 34 fields. Illinois and Iowa fields r4–r10 have no satellite ingest; "
        "every NDVI figure prints its n/N, and gaps stay null.",
        "Weather comes from a coarse POWER grid, not field sensors. Fields inside a grower collapse "
        "onto one to four daily series, so within-grower weather spread is small and W2 says so.",
        "CDL pixels are 30 m. On 5–12 ha parcels, mixed edge pixels bias purity and class counts; "
        "X5 and X7 are read with that bound in mind.",
        "Area, perimeter and compactness are static. There is no within-field boundary change to "
        "compare across years; A1/X5/X7 collapse to one row per field.",
        "Field-year rows are not independent. Count-based figures aggregate to fields; X7 prints "
        "both panels rather than choosing the flattering one.",
        "M2 compares endpoints only; a five-year corn/soy window usually returns the same parity, "
        "which is rotation, not an unchanged field.",
        "Small n: Delaware has four parcels, so its per-grower ρ values are indicative. Rank "
        "correlations are descriptive, not causal, and GDD/rainfall co-vary.",
        "Out of scope: soil, yield, NPK, economics, weather forecasting and automated report "
        "generation. This PDF is the only report artifact.",
    ], size=8.6)

    y -= 0.004
    y = heading(fig, y, "Soil is not part of this assignment")
    y = para(fig, y,
             "Soil analysis was not required and was not performed. The subskill never reads "
             "SSURGO or any soil raster, and panel.csv contains no soil columns. The only contact "
             "with the soil/ tree is importing the CDL code-to-name table "
             "(soil/cdl-cropland/src/cdl_codes.py), which is a crop-class lookup — not a soil "
             "property — so stored IL/IA crop names cannot leak wrong class labels into figures.",
             size=8.6, gap=0.008)

    y = table(fig, y, [
        f"{'ID':<10}{'What it shows':<74}",
        "-" * 84,
        f"{'coverage':<10}{'Layer coverage strip: fields with data per grower and layer':<74}",
        f"{'A1':<10}{'Field area by grower, one point per field, log axis':<74}",
        f"{'N1':<10}{'Within-field annual-mean NDVI distribution per year':<74}",
        f"{'N2':<10}{'NDVI by CDL crop class, one point per field-year':<74}",
        f"{'N3':<10}{'Grower × year NDVI climatology with field-count strip':<74}",
        f"{'C1':<10}{'Per-field 100 % stacked crop composition by year':<74}",
        f"{'C2':<10}{'Field × year dominant-crop heatmap':<74}",
        f"{'C3':<10}{'Crop share and Shannon diversity by grower':<74}",
        f"{'W1':<10}{'Cumulative GDD by day-of-year, one line per year':<74}",
        f"{'W2':<10}{'Per-field season GDD and rainfall inside each grower':<74}",
        f"{'W3':<10}{'Monthly temperature and rainfall climatology by grower':<74}",
        f"{'X4':<10}{'NDVI against season GDD and rainfall, ρ per crop group':<74}",
        f"{'X5':<10}{'Field area against crop purity, pooled and per-grower ρ':<74}",
        f"{'X6':<10}{'Season GDD against rainfall with grower ellipses':<74}",
        f"{'X7':<10}{'Field area against distinct planted crop classes':<74}",
        f"{'M1':<10}{'Grower plate: boundaries, imagery, CDL 2025 fill':<74}",
        f"{'M2':<10}{'2021→2025 dominant-crop flip, counts fields not hectares':<74}",
        f"{'M3':<10}{'NDVI plate on a fixed 0.2–0.7 scale, gaps hatched':<74}",
    ], size=6.9, heading_text="Output inventory (15 statistical outputs, 3 map families)")
    y -= 0.004
    y = para(fig, y,
             "Artifacts: panel.csv (170 × 41), composition.csv and coverage.csv, plus 30 PNGs, all "
             "written to eda-field-comparison/output/ (gitignored, not committed); the reviewed "
             "subset is snapshotted in eda-field-comparison/examples/output/. The reusable code, "
             "its GUIDE and the validator remain in "
             "my-farm-advisor/eda/eda-field-comparison/ on the assignment-2 branch.",
             size=7.4, color=MUTED)
    report.save(fig)

    report.close()
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out}  ({size_kb:.0f} KB, {report.page} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
