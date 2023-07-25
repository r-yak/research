import enum

import webcolors


class Color(enum.Enum):
    # WHITE = webcolors.hex_to_rgb('#ffffff')
    YELLOW = webcolors.hex_to_rgb('#ffeb3b')
    ORANGE = webcolors.hex_to_rgb('#ff9800')
    PINK = webcolors.hex_to_rgb('#ff65d5')
    RED = webcolors.hex_to_rgb('#ba000d')
    BROWN = webcolors.hex_to_rgb('#5C4033')
    MAROON = webcolors.hex_to_rgb('#800000')
    LIME = webcolors.hex_to_rgb('#7fe325')
    GREEN = webcolors.hex_to_rgb('#008000')
    TEAL = webcolors.hex_to_rgb('#008080')
    BLUE = webcolors.hex_to_rgb('#4269ff')
    NAVY = webcolors.hex_to_rgb('#000080')
    WINE = webcolors.hex_to_rgb('#b90076')
    PURPLE = webcolors.hex_to_rgb('#9b00b5')
    # GRAY = webcolors.hex_to_rgb('#9e9e9e')
    # BLACK = webcolors.hex_to_rgb('#212121')
    NONE = webcolors.hex_to_rgb('#FF00FF')


class Shape(enum.IntEnum):
    TRIANGLE = 3
    QUADRILATERAL = 4
    PENTAGON = 5
    OVAL_LIKE = 6  # 타원형 혹은 장방형
    CIRCLE = 10
    NONE = enum.auto()