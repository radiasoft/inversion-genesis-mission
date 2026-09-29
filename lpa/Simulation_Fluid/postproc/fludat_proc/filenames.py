"""Parse the physical parameters encoded in simulation export filenames."""

from __future__ import annotations

import re
from pathlib import Path

PRESSURE_FILE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)_bar(?:\.[^.]+)?$", re.IGNORECASE)
GRID_SIZE_STEM = re.compile(r"\d+(?:_\d+)?$")


def parse_pressure_filename(filename: str | Path) -> float:
    """Return the backing pressure in bar from a name such as ``5_bar.txt``."""
    name = Path(filename).name
    match = PRESSURE_FILE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Could not parse backing pressure from {name!r}; expected <pressure>_bar[.ext]"
        )
    return float(match.group(1))


def parse_grid_size_filename(filename: str | Path) -> float:
    """Return the grid size in mm from a name such as ``0_075.txt`` (0.075 mm)."""
    path = Path(filename)
    if GRID_SIZE_STEM.fullmatch(path.stem) is None:
        raise ValueError(
            f"Could not parse a maximum grid size from {path.name!r}; "
            "expected a name such as '0_075.txt'"
        )
    return float(path.stem.replace("_", "."))


def is_grid_size_filename(path: Path) -> bool:
    """Return whether ``path`` is a resolution-tagged export such as ``0_1.cgns``."""
    return path.is_file() and GRID_SIZE_STEM.fullmatch(path.stem) is not None
