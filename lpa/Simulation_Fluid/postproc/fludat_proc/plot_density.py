#!/usr/bin/env python3
"""Plot a density cube as an ``(x, z)`` map or a lineout along ``z``.

Examples (conda env inv-fbpic):

    python -m fludat_proc.plot_density data/density_lineouts/400_um.h5
    python -m fludat_proc.plot_density data/density_lineouts/400_um.h5 --plot lineout
    python -m fludat_proc.plot_density data/density_lineouts/400_um.h5 \\
        --plot xz_map --pressure 12.5 -o density_map.png
"""

from __future__ import annotations

if __package__ in (
    None,
    "",
):  # run as a plain script, e.g. `python fludat_proc/plot_density.py`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_proc"

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Widget

from .common import MM_PER_M, mm_bounds_to_m, validate_bounds
from .interpolate_density import (
    INTERPOLATION_METHODS,
    DensityInterpolation,
    InterpolationMethod,
    build_density_callable,
    build_density_interpolation,
)

PlotMode = Literal["xz_map", "lineout", "callable_lineout"]
PLOT_MODES: tuple[PlotMode, ...] = ("xz_map", "lineout", "callable_lineout")
DensityFn = Callable[[np.ndarray, float, float], np.ndarray]


@dataclass
class DensityPlot:
    """A figure plus the interactive widgets that must stay referenced to keep working."""

    figure: Figure
    widgets: tuple[Widget, ...] = field(default_factory=tuple)


def _density_label(field: DensityInterpolation) -> str:
    return f"density [{field.density_units}]"


def _resolve_bounds(
    bounds: tuple[float, float] | None,
    extent: tuple[float, float],
    *,
    axis_name: str,
) -> tuple[float, float]:
    return validate_bounds(bounds, axis_name=axis_name, extent=extent) or extent


def _xz_title(field: DensityInterpolation, pressure: float) -> str:
    return f"{field.geometry} @ {pressure:.1f} bar ({field.method})"


def _lineout_title(
    field: DensityInterpolation, x_position: float, pressure: float, *, label: str = ""
) -> str:
    return (
        f"{field.geometry} {label}@ x = {x_position:.2f} mm, "
        f"{pressure:.1f} bar ({field.method})"
    )


def _figure_with_sliders(
    ax: Axes | None, slider_count: int
) -> tuple[Figure, Axes, list[Axes]]:
    """Return a figure, the main axes, and ``slider_count`` axes stacked underneath."""
    if ax is not None:
        return ax.figure, ax, []
    if slider_count == 0:
        fig, plot_ax = plt.subplots()
        return fig, plot_ax, []
    fig = plt.figure()
    grid = gridspec.GridSpec(
        1 + slider_count, 1, height_ratios=[20] + [1] * slider_count, hspace=0.45
    )
    return (
        fig,
        fig.add_subplot(grid[0]),
        [fig.add_subplot(grid[i + 1]) for i in range(slider_count)],
    )


def plot_density_interpolation(
    field: DensityInterpolation,
    *,
    pressure: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
    x_points: int = 400,
    z_points: int = 400,
    ax: Axes | None = None,
    title: str | None = None,
    interactive: bool = True,
) -> DensityPlot:
    """Plot density in the ``(x, z)`` plane, with a backing-pressure slider if interactive.

    Bounds are in the cube's units: ``x`` in mm, ``z`` in m.
    """
    x_values = np.linspace(
        *_resolve_bounds(x_bounds, field.x_extent, axis_name="x"), x_points
    )
    z_values = np.linspace(
        *_resolve_bounds(z_bounds, field.z_extent, axis_name="z"), z_points
    )
    if pressure is None:
        pressure = float(field.pressure[0])

    interactive = interactive and ax is None
    fig, ax, slider_axes = _figure_with_sliders(ax, 1 if interactive else 0)

    if interactive:
        xz_stack = field.xz_grids_at_pressures(x_values, z_values)
        density_grid = field.interpolate_xz_from_pressure_stack(xz_stack, pressure)
    else:
        density_grid = field.interpolate_xz_grid(x_values, z_values, pressure)

    mesh = ax.pcolormesh(
        z_values * MM_PER_M, x_values, density_grid, shading="auto", cmap="viridis"
    )
    colorbar = fig.colorbar(mesh, ax=ax, label=_density_label(field))
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    auto_title = title is None
    ax.set_title(_xz_title(field, pressure) if auto_title else title)

    if not interactive:
        fig.tight_layout()
        return DensityPlot(fig)

    slider = Slider(
        slider_axes[0],
        "backing pressure [bar]",
        *field.pressure_extent,
        valinit=pressure,
    )

    def on_pressure_change(pressure_value: float) -> None:
        density = field.interpolate_xz_from_pressure_stack(xz_stack, pressure_value)
        mesh.set_array(density)
        mesh.set_clim(vmin=density.min(), vmax=density.max())
        colorbar.update_normal(mesh)
        if auto_title:
            ax.set_title(_xz_title(field, pressure_value))
        fig.canvas.draw_idle()

    slider.on_changed(on_pressure_change)
    return DensityPlot(fig, (slider,))


def plot_density_lineout(
    field: DensityInterpolation,
    *,
    pressure: float | None = None,
    x_position: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
    density_fn: DensityFn | None = None,
    title_label: str = "",
    ax: Axes | None = None,
    title: str | None = None,
    interactive: bool = True,
) -> DensityPlot:
    """Plot density along ``z`` at fixed ``x`` and pressure, with sliders if interactive.

    ``density_fn(z_values, x, pressure)`` overrides how the profile is evaluated.
    """
    resolved_x_bounds = _resolve_bounds(x_bounds, field.x_extent, axis_name="x")
    resolved_z_bounds = _resolve_bounds(z_bounds, field.z_extent, axis_name="z")
    if pressure is None:
        pressure = float(field.pressure[0])
    if x_position is None:
        x_position = resolved_x_bounds[0]
    elif not resolved_x_bounds[0] <= x_position <= resolved_x_bounds[1]:
        raise ValueError(
            f"x position {x_position} is outside the plot bounds {list(resolved_x_bounds)}"
        )

    evaluate = density_fn or field.interpolate_along_z
    z_values = np.linspace(*resolved_z_bounds, field.z.size)

    interactive = interactive and ax is None
    fig, plot_ax, slider_axes = _figure_with_sliders(ax, 2 if interactive else 0)
    (line,) = plot_ax.plot(
        z_values * MM_PER_M, evaluate(z_values, x_position, pressure)
    )
    plot_ax.set_xlabel("z [mm]")
    plot_ax.set_ylabel(_density_label(field))
    plot_ax.grid()
    auto_title = title is None
    plot_ax.set_title(
        _lineout_title(field, x_position, pressure, label=title_label)
        if auto_title
        else title
    )

    if not interactive:
        fig.tight_layout()
        return DensityPlot(fig)

    pressure_slider = Slider(
        slider_axes[0],
        "backing pressure [bar]",
        *field.pressure_extent,
        valinit=pressure,
    )
    x_slider = Slider(slider_axes[1], "x [mm]", *resolved_x_bounds, valinit=x_position)

    def refresh(_: float) -> None:
        line.set_ydata(evaluate(z_values, x_slider.val, pressure_slider.val))
        plot_ax.relim()
        plot_ax.autoscale_view()
        if auto_title:
            plot_ax.set_title(
                _lineout_title(
                    field, x_slider.val, pressure_slider.val, label=title_label
                )
            )
        fig.canvas.draw_idle()

    pressure_slider.on_changed(refresh)
    x_slider.on_changed(refresh)
    return DensityPlot(fig, (pressure_slider, x_slider))


def plot_density_callable_lineout(
    hdf5_path: Path,
    *,
    field: DensityInterpolation,
    method: InterpolationMethod = "linear",
    **lineout_kwargs,
) -> DensityPlot:
    """Like :func:`plot_density_lineout`, but evaluated through ``build_density_callable``.

    This exercises the exact code path FBPIC uses, so it is a sanity check of that
    integration rather than a faster or different plot.
    """

    def density_fn(
        z_values: np.ndarray, x_value: float, pressure_value: float
    ) -> np.ndarray:
        density = build_density_callable(
            hdf5_path, pressure_value, x_value, method=method, field=field
        )
        return density(z_values, np.zeros_like(z_values))

    return plot_density_lineout(
        field, density_fn=density_fn, title_label="callable ", **lineout_kwargs
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a density cube interactively or save a static figure."
    )
    parser.add_argument("hdf5_path", type=Path, help="HDF5 density cube for one nozzle")
    parser.add_argument(
        "--plot",
        choices=PLOT_MODES,
        default="xz_map",
        help="xz_map: density in the (x, z) plane vs pressure; lineout: density along z "
        "vs pressure and x; callable_lineout: lineout via build_density_callable",
    )
    parser.add_argument(
        "--method",
        choices=INTERPOLATION_METHODS,
        default="linear",
        help="Interpolation method in the (x, pressure) plane",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help="Backing pressure [bar] (default: lowest in dataset)",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="x position [mm] for lineouts (default: lowest in dataset)",
    )
    parser.add_argument(
        "--x-bounds",
        nargs=2,
        type=float,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="x plot bounds [mm] (default: full dataset extent)",
    )
    parser.add_argument(
        "--z-bounds",
        nargs=2,
        type=float,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="z plot bounds [mm] (default: full dataset extent)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save the figure instead of showing it",
    )
    parser.add_argument("-t", "--title", default=None, help="Figure title")
    args = parser.parse_args()

    field = build_density_interpolation(args.hdf5_path, method=args.method)
    common = dict(
        pressure=args.pressure,
        x_bounds=tuple(args.x_bounds) if args.x_bounds is not None else None,
        z_bounds=mm_bounds_to_m(args.z_bounds),
        title=args.title,
        interactive=args.output is None,
    )
    if args.plot == "xz_map":
        plot = plot_density_interpolation(field, **common)
    elif args.plot == "lineout":
        plot = plot_density_lineout(field, x_position=args.x, **common)
    else:
        plot = plot_density_callable_lineout(
            args.hdf5_path, field=field, method=args.method, x_position=args.x, **common
        )

    if args.output:
        plot.figure.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
