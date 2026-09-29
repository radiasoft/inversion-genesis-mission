"""Utility functions and classes for FBPIC simulations."""

from typing import TYPE_CHECKING

from .input_params import InputParameters

if TYPE_CHECKING:
    from .laser import (
        HighOrderLasyLaser,
        polar_fields,
        set_polar_fields,
        transverse_fluence,
    )


_LAZY_LASER_EXPORTS = frozenset(
    {
        "HighOrderLasyLaser",
        "polar_fields",
        "set_polar_fields",
        "transverse_fluence",
    }
)


def __getattr__(name: str):
    """Load LASY-dependent helpers only when they are explicitly requested."""
    if name in _LAZY_LASER_EXPORTS:
        from . import laser

        return getattr(laser, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HighOrderLasyLaser",
    "InputParameters",
    "polar_fields",
    "set_polar_fields",
    "transverse_fluence",
]
