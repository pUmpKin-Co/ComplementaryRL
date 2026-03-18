from typing import Dict, List, Optional

import numpy as np
from minihack.level_generator import TRAP_NAMES


def _decode_screen_description(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00").strip()
    return str(value).strip()


def collect_symbol_descriptions_from_obs(
    obs: Dict,
    *,
    prefer_crop: bool = True,
    include_blank: bool = False,
) -> Dict[str, List[str]]:
    if not isinstance(obs, dict):
        raise TypeError(
            "collect_symbol_descriptions_from_obs expects the *raw MiniHack observation dict* "
            "(with keys like 'chars'/'chars_crop' and 'screen_descriptions'). "
            "You passed a non-dict (often this is the text observation string returned by a wrapper)."
        )
    desc_grid = obs.get("screen_descriptions")
    if desc_grid is None:
        return {}

    candidates = []
    if prefer_crop:
        candidates.extend(["chars_crop", "chars"])
    else:
        candidates.extend(["chars", "chars_crop"])
    candidates.append("chars")  # ensure full-screen fallback is always tried
    candidates.append("chars_crop")

    descs = np.asarray(desc_grid)
    desc_hw = descs.shape[:2] if descs.ndim >= 2 else descs.shape
    char_grid = None
    for k in candidates:
        grid = obs.get(k)
        if grid is None:
            continue
        arr = np.asarray(grid)
        if arr.shape == desc_hw:
            char_grid = grid
            break

    if char_grid is None or desc_grid is None:
        return {}

    chars = np.asarray(char_grid)
    if chars.shape != desc_hw:
        # No compatible char grid available.
        return {}

    symbol_to_descs: Dict[str, set[str]] = {}
    for r in range(chars.shape[0]):
        for c in range(chars.shape[1]):
            chv = int(chars[r, c])
            if chv in (0, 32) and not include_blank:
                continue
            symbol = chr(chv) if 0 <= chv <= 255 else str(chv)
            d = _decode_screen_description(descs[r, c])
            if not d and not include_blank:
                continue
            symbol_to_descs.setdefault(symbol, set()).add(d)

    # Convert to stable lists for downstream JSON/logging.
    return {k: sorted(v) for k, v in symbol_to_descs.items()}


def minihack_walls() -> List[str]:
    """Get list of valid wall types for MiniHack."""
    Walls = [
        "|",  # vertical wall
        "-",  # horizontal wall
        "#",  # corridor/wall
        " ",  # solid wall
    ]
    return Walls


def minihack_terrains() -> List[str]:
    """Get list of valid terrain types for MiniHack."""
    Terrains = [
        "#",  # corridor
        ".",  # room floor (Unlit, unless lit with REGION-command)
        "+",  # door (State is defined with DOOR -command)
        "A",  # air
        "B",  # crosswall / boundary symbol hack (See REGION)
        "C",  # cloud
        "S",  # secret door
        "H",  # secret corridor
        "{",  # fountain
        "\\",  # throne
        "K",  # sink
        "}",  # moat
        "P",  # pool of water
        "L",  # lava pool
        "I",  # ice
        "W",  # water
        "T",  # tree
        "F",  # iron bars
        "`",  # boulder
    ]
    return Terrains


def get_terrain_description(terrain: str) -> str:
    """Get a human-readable description of a terrain type."""
    descriptions = {
        "#": "corridor",
        ".": "room floor",
        "+": "door",
        "A": "air",
        "B": "boundary",
        "C": "cloud",
        "S": "secret door",
        "H": "secret corridor",
        "{": "fountain",
        "\\": "throne",
        "K": "sink",
        "}": "water/lava",
        "P": "pool of water",
        "L": "lava pool",
        "I": "ice",
        "W": "water",
        "T": "tree",
        "F": "iron bars",
        "`": "boulder",
    }
    return descriptions.get(terrain, terrain)


def get_wall_description(wall: str) -> str:
    """Get a human-readable description of a wall type."""
    descriptions = {
        "|": "vertical wall",
        "-": "horizontal wall",
        "#": "corridor wall",
        " ": "solid wall",
    }
    return descriptions.get(wall, wall)


def get_common_monsters() -> List[str]:
    """Get list of common monster names for MiniHack."""
    return [
        "kobold",
        "orc",
        "goblin",
        "gnome",
        "dwarf",
        "human",
        "giant ant",
        "giant beetle",
        "giant spider",
        "grid bug",
        "xan",
        "skeleton",
        "ghost",
        "vampire",
        "lich",
        "minotaur",
        "hell hound",
        "red mold",
    ]


def get_common_objects() -> List[tuple]:
    """
    Get list of common objects for MiniHack DES files.

    Returns tuples of (symbol, name) where symbol is the object class:
    - '%' = food
    - ')' = weapon
    - '!' = potion
    - '?' = scroll
    - '/' = wand
    - '$' = gold
    """
    objects = [
        ("%", "food ration"),
        (")", "weapon"),
        (")", "short sword"),
        ("$", "gold"),
    ]
    return objects


def get_common_traps() -> List[str]:
    """Get list of common trap names for MiniHack."""
    preferred_traps = [
        "teleport",
        "level teleport",
        "pit",
        "trap door",
        "land mine",
    ]
    return [trap for trap in preferred_traps if trap in TRAP_NAMES]


def render_glyphs(
    grid: np.ndarray,
    *,
    crop_to_bounds: bool = False,
    return_bounds: bool = False,
) -> str | tuple[str, tuple[int, int, int, int] | None]:
    """
    Render glyph or ASCII array to text representation.

    Args:
        grid: 2D array of glyph IDs or ASCII codes from NetHack

    Returns:
        String representation of the dungeon view
    """
    if grid is None:
        return "" if not return_bounds else ("", None)

    array = np.asarray(grid)
    if array.ndim != 2:
        raise ValueError("render_glyphs expects a 2D array.")

    if array.size == 0:
        return "" if not return_bounds else ("", None)

    use_ascii = array.dtype == np.uint8 or (np.issubdtype(array.dtype, np.integer) and array.max() <= 255)
    rows, cols = array.shape
    max_rows = min(rows, 21)
    max_cols = min(cols, 79)

    char_array = np.full((max_rows, max_cols), " ", dtype="<U1")

    for r in range(max_rows):
        for c in range(max_cols):
            value = int(array[r, c])

            if use_ascii:
                if value not in (0, 32):
                    char_array[r, c] = chr(value)
            else:
                char_array[r, c] = _approximate_glyph_char(value)

    non_blank_mask = char_array != " "
    if not np.any(non_blank_mask):
        result = ""
        bounds: tuple[int, int, int, int] | None = None
    else:
        row_indices = np.where(np.any(non_blank_mask, axis=1))[0]
        col_indices = np.where(np.any(non_blank_mask, axis=0))[0]
        top, bottom = row_indices[0], row_indices[-1] + 1
        left, right = col_indices[0], col_indices[-1] + 1

        bounds = (top, bottom, left, right)

        if crop_to_bounds:
            char_array = char_array[top:bottom, left:right]

        result = "\n".join("".join(row) for row in char_array)

    if return_bounds:
        return result, bounds
    return result


def _approximate_glyph_char(glyph_id: int) -> str:
    """
    Approximate glyph ID to a representative ASCII character.

    This is a fallback function used when glyph IDs are provided instead of
    ASCII characters. For better accuracy, use ASCII characters directly in des files.
    """
    if glyph_id < 0:
        return " "
    # Monsters and players (typically < 600)
    if glyph_id < 600:
        return "@"
    # Floor and basic terrain (typically 600-1000)
    if glyph_id < 1000:
        return "."
    # Walls and corridors (typically 1000-2000)
    if glyph_id < 2000:
        return "#"
    # Objects and items (typically 2000-4000)
    if glyph_id < 4000:
        return "$"
    # Doors and gates (typically 4000-5000)
    if glyph_id < 5000:
        return "+"
    # Traps and other features (typically 5000+)
    return "^"
