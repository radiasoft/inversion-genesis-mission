#!/usr/bin/env python3
"""Convert unstructured CGNS density fields to a regular nozzle density cube.

Directory input combines one ``<pressure>_bar.cgns`` file per backing pressure into
the pressure axis; single-file input needs ``--pressure``. Only the ``x >= 0``,
``z >= 0`` quadrant of the source is used: the output z grid is symmetric and every
point is sampled at ``|z|``, so the cube is exactly mirrored across ``z = 0``.

Example (conda env inv-fbpic):

    python -m fludat_proc.cgns_to_density_hdf5 data/density_field_raw/htu_fields_7_0 \\
        --output data/density_field/htu_dens_7_0.h5 --density-units kg/m^3
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
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata

from .cgns_io import (
    DEFAULT_DENSITY_NAME,
    DEFAULT_FLOW_SOLUTION,
    DEFAULT_X_COORDINATE,
    DEFAULT_Z_COORDINATE,
    DEFAULT_ZONE_PATH,
    MINIMUM_POINT_COUNT,
    CgnsDataError,
    load_cgns_density_points,
)
from .common import MM_PER_M, mm_bounds_to_m, validate_bounds
from .density_cube import DensityCube, write_density_cube
from .filenames import parse_pressure_filename

INTERPOLATION_METHODS = ("linear", "nearest", "cubic")
OUTSIDE_FILL_MODES = ("zero", "nearest", "raise")

PointField = tuple[np.ndarray, np.ndarray, np.ndarray]
"""``(x, z, density)`` point samples in source units."""


def collect_cgns_inputs(
    input_path: str | Path,
    *,
    pressure_bar: float | None,
) -> list[tuple[float, Path]]:
    """Return pressure-sorted ``(pressure_bar, path)`` CGNS sources for one nozzle."""
    input_path = Path(input_path).resolve()
    if input_path.is_file():
        if pressure_bar is None:
            raise ValueError("pressure_bar is required when input is one CGNS file")
        if not np.isfinite(pressure_bar):
            raise ValueError("pressure_bar must be finite")
        return [(float(pressure_bar), input_path)]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    if pressure_bar is not None:
        raise ValueError("pressure_bar is only valid when input is one CGNS file")

    sources = sorted(
        (parse_pressure_filename(path.name), path)
        for path in input_path.glob("*.cgns")
        if path.is_file()
    )
    if not sources:
        raise ValueError(f"No pressure-tagged .cgns files found in {input_path}")
    pressures = [pressure for pressure, _ in sources]
    if len(set(pressures)) != len(pressures):
        raise ValueError(f"Duplicate backing pressure file in {input_path}")
    return sources


def select_positive_quadrant(
    x: np.ndarray, z: np.ndarray, density: np.ndarray
) -> PointField:
    """Keep the ``x >= 0``, ``z >= 0`` samples that define the symmetric output."""
    mask = (x >= 0.0) & (z >= 0.0)
    if np.count_nonzero(mask) < MINIMUM_POINT_COUNT:
        raise CgnsDataError(
            f"CGNS field has fewer than {MINIMUM_POINT_COUNT} samples at x >= 0, z >= 0"
        )
    return x[mask], z[mask], density[mask]


def symmetric_z_grid(z_max: float, count: int) -> np.ndarray:
    """Return ``count`` evenly spaced points on ``[-z_max, z_max]`` that are exactly mirrored.

    Odd counts include ``z = 0``; even counts straddle it.
    """
    if count % 2:
        positive = np.linspace(0.0, z_max, count // 2 + 1, dtype=np.float64)
        return np.concatenate((-positive[:0:-1], positive))
    positive = np.linspace(z_max / (count - 1), z_max, count // 2, dtype=np.float64)
    return np.concatenate((-positive[::-1], positive))


def shared_regular_grid(
    fields: list[PointField],
    *,
    z_count: int,
    x_count: int,
    z_bounds: tuple[float, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return regular ``(z, x)`` output grids in metres covering every field.

    The default z bounds are ``[-z_max, z_max]`` of the smallest field so the grid is
    symmetric; the default x bounds are the common x extent of all fields.
    """
    if z_count < 2 or x_count < 2:
        raise ValueError("z_count and x_count must both be at least 2")
    if not fields:
        raise ValueError("At least one CGNS field is required")

    common_z_max = min(float(z.max()) for _, z, _ in fields)
    common_x_min = max(float(x.min()) for x, _, _ in fields)
    common_x_max = min(float(x.max()) for x, _, _ in fields)
    z_bounds = validate_bounds(z_bounds, axis_name="z") or (-common_z_max, common_z_max)
    x_bounds = validate_bounds(x_bounds, axis_name="x") or (common_x_min, common_x_max)
    if not z_bounds[0] < z_bounds[1] or not x_bounds[0] < x_bounds[1]:
        raise ValueError("CGNS fields do not have a nonzero common (z, x) domain")

    if np.isclose(z_bounds[0], -z_bounds[1]):
        z_grid = symmetric_z_grid(z_bounds[1], z_count)
    else:
        z_grid = np.linspace(*z_bounds, z_count, dtype=np.float64)
    return z_grid, np.linspace(*x_bounds, x_count, dtype=np.float64)


def grid_density(
    field: PointField,
    z_grid: np.ndarray,
    x_grid: np.ndarray,
    *,
    method: str = "linear",
    outside_fill: str = "zero",
) -> np.ndarray:
    """Interpolate ``x >= 0, z >= 0`` samples onto a ``(z, x)`` grid, sampling at ``|z|``."""
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"Unknown interpolation method {method!r}")
    if outside_fill not in OUTSIDE_FILL_MODES:
        raise ValueError(f"Unknown outside-fill mode {outside_fill!r}")

    x, z, density = field
    points = np.column_stack((z, x))
    z_mesh, x_mesh = np.meshgrid(np.abs(z_grid), x_grid, indexing="ij")
    gridded = np.asarray(griddata(points, density, (z_mesh, x_mesh), method=method))

    missing = np.isnan(gridded)
    if np.any(missing):
        if outside_fill == "raise":
            raise ValueError(
                f"Interpolation left {missing.sum()} grid points outside the CGNS domain"
            )
        if outside_fill == "zero":
            gridded[missing] = 0.0
        else:
            gridded[missing] = griddata(
                points, density, (z_mesh[missing], x_mesh[missing]), method="nearest"
            )
    return gridded


def convert_cgns_to_density_hdf5(
    input_path: str | Path,
    output_path: str | Path,
    *,
    density_scale: float,
    density_units: str,
    pressure_bar: float | None = None,
    z_count: int = 1000,
    x_count: int = 500,
    z_bounds: tuple[float, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    method: str = "linear",
    outside_fill: str = "zero",
    nozzle: str | None = None,
    zone_path: str = DEFAULT_ZONE_PATH,
    flow_solution: str = DEFAULT_FLOW_SOLUTION,
    x_coordinate: str = DEFAULT_X_COORDINATE,
    z_coordinate: str = DEFAULT_Z_COORDINATE,
    density_name: str = DEFAULT_DENSITY_NAME,
) -> Path:
    """Convert one CGNS field or a pressure-tagged CGNS directory to a density cube.

    Bounds are in source metres. ``x`` is stored in mm in the output.
    """
    if not np.isfinite(density_scale):
        raise ValueError("density_scale must be finite")

    input_path = Path(input_path).resolve()
    sources = collect_cgns_inputs(input_path, pressure_bar=pressure_bar)
    fields = [
        select_positive_quadrant(
            *load_cgns_density_points(
                source_path,
                zone_path=zone_path,
                flow_solution=flow_solution,
                x_coordinate=x_coordinate,
                z_coordinate=z_coordinate,
                density_name=density_name,
            )
        )
        for _, source_path in sources
    ]
    z_grid, x_grid_m = shared_regular_grid(
        fields, z_count=z_count, x_count=x_count, z_bounds=z_bounds, x_bounds=x_bounds
    )
    density = np.stack(
        [
            density_scale
            * grid_density(
                field, z_grid, x_grid_m, method=method, outside_fill=outside_fill
            )
            for field in fields
        ],
        axis=-1,
    )

    cube = DensityCube(
        nozzle=nozzle or (input_path.name if input_path.is_dir() else input_path.stem),
        z=z_grid,
        x=x_grid_m * MM_PER_M,
        pressure=np.array([pressure for pressure, _ in sources], dtype=np.float64),
        density=density,
        density_units=density_units,
    )
    return write_density_cube(
        output_path,
        cube,
        density_scale=density_scale,
        attributes={
            "source_format": "CGNS",
            "source_files": np.asarray(
                [str(path) for _, path in sources],
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
            "source_coordinate_mapping": f"z_m={z_coordinate};x_mm={x_coordinate}*1000",
            "source_selection": f"{x_coordinate} >= 0 and {z_coordinate} >= 0",
            "source_z_symmetry": f"sampled at |{z_coordinate}| on a symmetric z grid",
            "interpolation_method": method,
            "outside_fill": outside_fill,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert unstructured CGNS density fields to a regular HDF5 density cube."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Directory of <pressure>_bar.cgns files, or one CGNS file with --pressure",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output HDF5 file"
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help="Backing pressure [bar] of a single-file input",
    )
    parser.add_argument(
        "--density-scale",
        type=float,
        default=1.0,
        help="Factor applied to CGNS density values before writing (default: 1)",
    )
    parser.add_argument(
        "--density-units",
        required=True,
        help="Units of the scaled density dataset, for example kg/m^3 or cm^-3",
    )
    parser.add_argument(
        "--nozzle",
        default=None,
        help="Nozzle name stored in the output (default: input name)",
    )
    parser.add_argument(
        "--z-points", type=int, default=1000, help="Output z grid points"
    )
    parser.add_argument(
        "--x-points", type=int, default=500, help="Output x grid points"
    )
    parser.add_argument(
        "--z-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="Output z bounds [mm] (default: symmetric over the common CGNS extent)",
    )
    parser.add_argument(
        "--x-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="Output x bounds [mm] (default: common CGNS extent at x >= 0)",
    )
    parser.add_argument(
        "--interpolation",
        choices=INTERPOLATION_METHODS,
        default="linear",
        help="Unstructured-to-grid interpolation (default: linear)",
    )
    parser.add_argument(
        "--outside-fill",
        choices=OUTSIDE_FILL_MODES,
        default="zero",
        help="Value for grid points outside the CGNS domain (default: zero)",
    )
    args = parser.parse_args()

    output = convert_cgns_to_density_hdf5(
        args.input,
        args.output,
        pressure_bar=args.pressure,
        density_scale=args.density_scale,
        density_units=args.density_units,
        nozzle=args.nozzle,
        z_count=args.z_points,
        x_count=args.x_points,
        z_bounds=mm_bounds_to_m(args.z_bounds),
        x_bounds=mm_bounds_to_m(args.x_bounds),
        method=args.interpolation,
        outside_fill=args.outside_fill,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
