"""Shared visual language for the field EDA subskill.

One place for grower colours, crop colours, and the CDL raster palette so the
statistical figures and the maps cannot drift apart. Numbers are plain
constants; nothing here reads data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Grower identity colours. Ordered to stay distinguishable in print and in
# greyscale: dark green (IL), blue (IA), amber (NE), magenta (DE).
GROWER_COLORS: dict[str, str] = {
    "Illinois": "#1b7f3b",
    "Iowa": "#2b6cb0",
    "Nebraska": "#b7791f",
    "Delaware": "#b83280",
}

GROWER_ORDER: tuple[str, ...] = ("Illinois", "Iowa", "Nebraska", "Delaware")

# Crop-class colours used by the CDL figures and the 2025 map fill. Values are
# the USDA NASS CDL palette for the classes this dataset actually carries, so
# a reader who knows CDL recognises the colour before reading the legend.
CDL_RGB: dict[int, tuple[int, int, int]] = {
    1: (255, 211, 0),      # Corn
    4: (255, 158, 61),     # Sorghum
    5: (38, 115, 0),       # Soybeans
    12: (255, 214, 0),     # Sweet Corn
    21: (226, 143, 61),    # Barley
    24: (170, 140, 80),    # Winter Wheat
    26: (112, 102, 0),     # Dbl Crop WinWht/Soybeans
    27: (170, 175, 102),   # Rye
    28: (150, 100, 150),   # Oats
    36: (199, 215, 158),   # Alfalfa
    37: (112, 163, 186),   # Other Hay/Non Alfalfa
    42: (163, 0, 0),       # Dry Beans
    48: (255, 102, 102),   # Watermelons
    61: (191, 191, 191),   # Fallow/Idle Cropland
    63: (0, 102, 0),       # Forest
    70: (0, 204, 204),     # Christmas Trees
    82: (153, 153, 153),   # Developed
    121: (153, 153, 153),  # Developed/Open Space
    122: (153, 153, 153),  # Developed/Low Intensity
    176: (226, 199, 127),  # Grass/Pasture
    190: (0, 255, 190),    # Woody Wetlands
    195: (102, 204, 102),  # Herbaceous Wetlands
    # Extended entries: the published CDL palette gives these double crops a
    # base-class hue (barley/soy shares soy green, lettuce/durum shares
    # watermelon red). They are nudged here so categorical figures can tell
    # them apart; the maps still use the source raster's own class values.
    229: (255, 150, 180),  # Dbl Crop Lettuce/Durum Wht
    254: (128, 96, 48),    # Dbl Crop Barley/Soybeans
}

# Stable text-colour choice for a class swatch (dark classes need white text).
def text_color_for(code: int) -> str:
    r, g, b = CDL_RGB.get(code, (204, 204, 204))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 140 else "#111827"


def apply_style() -> None:
    """Apply the subskill's matplotlib defaults (call inside a style context)."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
