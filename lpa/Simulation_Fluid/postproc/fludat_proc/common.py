"""Small helpers shared by the command-line tools."""

from __future__ import annotations

import numpy as np

MM_PER_M = 1000.0


def validate_bounds(
    bounds: tuple[float, float] | None,
    *,
    axis_name: str,
    extent: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Return finite, increasing ``(lower, upper)`` bounds, optionally inside ``extent``.

    ``None`` passes through unchanged so callers can treat it as "use the default".
    """
    if bounds is None:
        return None
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite((lower, upper)).all() or lower >= upper:
        raise ValueError(f"{axis_name} bounds must be finite and increasing")
    if extent is not None and (lower < extent[0] or upper > extent[1]):
        raise ValueError(
            f"{axis_name} bounds [{lower}, {upper}] are outside the valid range "
            f"[{extent[0]}, {extent[1]}]"
        )
    return lower, upper


def mm_bounds_to_m(bounds: list[float] | None) -> tuple[float, float] | None:
    """Convert an optional CLI ``[MIN_MM, MAX_MM]`` pair to metres."""
    if bounds is None:
        return None
    lower, upper = bounds
    return lower / MM_PER_M, upper / MM_PER_M
