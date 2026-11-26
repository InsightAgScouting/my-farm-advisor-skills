"""USDA NASS Cropland Data Layer (CDL) class legend.

Single source of truth for CDL class codes and names inside this skill.
``cdl_reporting.CDL_CODES`` and :class:`CDLCroplandSkill.CROP_CODES` both read
from here so the reporting tables and the downloader cannot drift apart.

Source: USDA NASS Cropland Data Layer class legend (CropScape / CroplandCROS).
Class 254 ("Dbl Crop Barley/Soybeans") is not yet in the older legend excerpt
this skill shipped with. It is part of the current NASS legend, it is a valid
class in third-party CDL readers, and the published CDL color table renders it
with the soybean colour (36, 110, 0), consistent with a barley/soybean
double-crop class.

Codes are stable across years; older rasters simply do not contain newer
classes. ``No Data`` covers CDL background (0) plus the legacy "clouds" class.
"""

from __future__ import annotations

CDL_CLASS_NAMES: dict[int, str] = {
    0: "No Data",
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    6: "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rape Seed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    38: "Camelina",
    39: "Buckwheat",
    41: "Sugarbeets",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    45: "Sugarcane",
    46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    51: "Chick Peas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes",
    55: "Caneberries",
    56: "Hops",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    60: "Switchgrass",
    61: "Fallow/Idle Cropland",
    62: "Pasture/Grass",
    63: "Forest",
    64: "Shrubland",
    65: "Barren",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    70: "Christmas Trees",
    71: "Other Tree Crops",
    72: "Citrus",
    74: "Pecans",
    75: "Almonds",
    76: "Walnuts",
    77: "Pears",
    81: "Clouds/No Data",
    82: "Developed",
    83: "Water",
    87: "Wetlands",
    88: "Nonag/Undefined",
    92: "Aquaculture",
    111: "Open Water",
    112: "Perennial Ice/Snow",
    121: "Developed/Open Space",
    122: "Developed/Low Intensity",
    123: "Developed/Med Intensity",
    124: "Developed/High Intensity",
    131: "Barren",
    141: "Deciduous Forest",
    142: "Evergreen Forest",
    143: "Mixed Forest",
    152: "Shrubland",
    176: "Grass/Pasture",
    190: "Woody Wetlands",
    195: "Herbaceous Wetlands",
    204: "Pistachios",
    205: "Triticale",
    206: "Carrots",
    207: "Asparagus",
    208: "Garlic",
    209: "Cantaloupes",
    210: "Prunes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    215: "Avocados",
    216: "Peppers",
    217: "Pomegranates",
    218: "Nectarines",
    219: "Greens",
    220: "Plums",
    221: "Strawberries",
    222: "Squash",
    223: "Apricots",
    224: "Vetch",
    225: "Dbl Crop WinWht/Corn",
    226: "Dbl Crop Oats/Corn",
    227: "Lettuce",
    228: "Pumpkin",
    229: "Dbl Crop Lettuce/Durum Wht",
    230: "Dbl Crop Lettuce/Cantaloupe",
    231: "Dbl Crop Lettuce/Cotton",
    232: "Dbl Crop Lettuce/Barley",
    233: "Dbl Crop Durum Wht/Sorghum",
    234: "Dbl Crop Barley/Sorghum",
    235: "Dbl Crop WinWht/Sorghum",
    236: "Dbl Crop Barley/Corn",
    237: "Dbl Crop WinWht/Cotton",
    238: "Dbl Crop Soybeans/Cotton",
    239: "Dbl Crop Soybeans/Oats",
    240: "Dbl Crop Corn/Soybeans",
    241: "Blueberries",
    242: "Cabbage",
    243: "Cauliflower",
    244: "Celery",
    245: "Radishes",
    246: "Turnips",
    247: "Eggplants",
    248: "Gourds",
    249: "Cranberries",
    250: "Christmas Trees",
    254: "Dbl Crop Barley/Soybeans",
}

# Back-compat alias: the reporting module historically exported CDL_CODES.
CDL_CODES = CDL_CLASS_NAMES

# Classes that are land cover rather than a planted crop. They stay in the
# per-field composition tables (useful context, especially for small produce
# parcels ringed by development) but are never treated as a rotation year.
ROTATION_EXCLUDED_CODES: frozenset[int] = frozenset(
    {
        0,  # No Data
        61,  # Fallow/Idle Cropland
        62,  # Pasture/Grass
        63,  # Forest
        64,  # Shrubland
        65,  # Barren
        81,  # Clouds/No Data
        82,  # Developed
        83,  # Water
        87,  # Wetlands
        88,  # Nonag/Undefined
        92,  # Aquaculture
        111,  # Open Water
        112,  # Perennial Ice/Snow
        121,  # Developed/Open Space
        122,  # Developed/Low Intensity
        123,  # Developed/Med Intensity
        124,  # Developed/High Intensity
        131,  # Barren
        141,  # Deciduous Forest
        142,  # Evergreen Forest
        143,  # Mixed Forest
        152,  # Shrubland
        176,  # Grass/Pasture
        190,  # Woody Wetlands
        195,  # Herbaceous Wetlands
    }
)
