"""Read and prepare the sectioned ``(z, density)`` lineout files exported by ANSYS."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SECTION_HEADER = re.compile(r'^\(\(xy/key/label\s+"([^"]+)"\)\s*$')
DATA_LINE = re.compile(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s*$")
X_POSITION_PATTERN = re.compile(r"(?:l-)?x-(\d+)(?:-(\d))?(?:-y-0)?$")


def parse_lineout_file(path: Path) -> dict[str, np.ndarray]:
    """Parse a raw lineout file into labeled ``(z, density)`` arrays."""
    sections: dict[str, list[tuple[float, float]]] = {}
    current_label: str | None = None

    with Path(path).open() as file:
        for line in file:
            line = line.rstrip("\n")

            if match := SECTION_HEADER.match(line):
                current_label = match.group(1)
                if current_label in sections:
                    raise ValueError(
                        f"Duplicate lineout section {current_label!r} in {path}"
                    )
                sections[current_label] = []
                continue

            if line.strip() == ")":
                current_label = None
                continue

            if current_label is None:
                continue

            if match := DATA_LINE.match(line):
                z, density = map(float, match.groups())
                sections[current_label].append((z, density))

    if not sections:
        raise ValueError(f"No lineout sections found in {path}")

    return {
        label: np.array(points, dtype=np.float64) for label, points in sections.items()
    }


def parse_x_position(label: str) -> float:
    """Return the transverse position in mm from a section label such as ``x-1-5``."""
    match = X_POSITION_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"Could not parse x position from {label!r}")

    integer = int(match.group(1))
    if match.group(2) is not None:
        return integer + int(match.group(2)) / 10.0
    return float(integer)


def deduplicate_and_sort_profile(profile: np.ndarray) -> np.ndarray:
    """Sort a ``(z, density)`` profile by z, keeping the last density for repeated z."""
    if profile.ndim != 2 or profile.shape[1] != 2:
        raise ValueError("A density profile must be a two-column (z, density) array")
    if profile.shape[0] < 2:
        raise ValueError("A density profile must contain at least two samples")

    sorted_profile = profile[np.argsort(profile[:, 0], kind="stable")]
    z = sorted_profile[:, 0]
    _, reverse_indices = np.unique(z[::-1], return_index=True)
    indices = np.sort(sorted_profile.shape[0] - 1 - reverse_indices)
    unique_profile = sorted_profile[indices]
    if unique_profile.shape[0] < 2:
        raise ValueError("A density profile must contain two distinct z positions")
    return unique_profile


def mirror_across_z0(profile: np.ndarray) -> np.ndarray:
    """Return the ``z >= 0`` part of a profile mirrored across ``z = 0`` and sorted."""
    profile = deduplicate_and_sort_profile(profile)
    positive = profile[profile[:, 0] >= 0]
    strictly_positive = positive[positive[:, 0] > 0]
    negative = np.column_stack((-strictly_positive[:, 0], strictly_positive[:, 1]))[
        ::-1
    ]
    return np.concatenate((negative, positive))
