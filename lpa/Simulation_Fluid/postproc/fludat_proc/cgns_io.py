"""Read point coordinates and scalar solution fields from Fluent CGNS (HDF5) exports.

Fluent writes a 2D axisymmetric field with ``CoordinateX`` along the nozzle axis
and ``CoordinateY`` across it. Throughout this package that pair is renamed to the
FBPIC convention: ``CoordinateY`` is the propagation axis ``z`` and ``CoordinateX``
the transverse position ``x``. Both are in metres.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .common import validate_bounds

DEFAULT_ZONE_PATH = "Base/Zone"
DEFAULT_FLOW_SOLUTION = "FlowSolution.N:1"
DEFAULT_X_COORDINATE = "CoordinateX"
DEFAULT_Z_COORDINATE = "CoordinateY"
DEFAULT_DENSITY_NAME = "Density"
COORDINATE_FIELD_NAMES = {"CoordinateX", "CoordinateY", "CoordinateZ"}
MINIMUM_POINT_COUNT = 3


class CgnsDataError(ValueError):
    """Raised when required pointwise data is absent or invalid in a CGNS file."""


@dataclass(frozen=True)
class CgnsScalarFieldData:
    """Point coordinates and scalar solution fields loaded from one CGNS zone."""

    x: np.ndarray
    z: np.ndarray
    fields: dict[str, np.ndarray]


def read_cgns_data(node: h5py.Group, description: str) -> np.ndarray:
    """Read the data child used by the CGNS HDF5 mapping (``" data"`` or ``"data"``)."""
    for name in (" data", "data"):
        if name in node:
            return np.asarray(node[name], dtype=np.float64)
    raise CgnsDataError(f"{description} does not contain a CGNS data array")


def validate_point_array(
    values: np.ndarray,
    *,
    name: str,
    point_count: int | None = None,
) -> np.ndarray:
    """Validate one finite, one-dimensional point array."""
    if values.ndim != 1:
        raise CgnsDataError(f"{name} must be one-dimensional")
    if point_count is not None and values.size != point_count:
        raise CgnsDataError(
            f"{name} has {values.size} values; expected {point_count} point values"
        )
    if not np.all(np.isfinite(values)):
        raise CgnsDataError(f"{name} must contain only finite values")
    return values


@contextmanager
def _open_zone(cgns_path: Path, zone_path: str) -> Iterator[h5py.Group]:
    """Open one CGNS zone, translating h5py errors into ``CgnsDataError``."""
    if not cgns_path.is_file():
        raise FileNotFoundError(cgns_path)
    try:
        with h5py.File(cgns_path, "r") as cgns_file:
            yield cgns_file[zone_path]
    except KeyError as exc:
        raise CgnsDataError(
            f"{cgns_path} is missing required CGNS node {exc.args[0]!r}"
        ) from exc
    except OSError as exc:
        raise CgnsDataError(f"Could not read CGNS file {cgns_path}: {exc}") from exc


def _read_coordinates(
    zone: h5py.Group,
    cgns_path: Path,
    x_coordinate: str,
    z_coordinate: str,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = zone["GridCoordinates"]
    x = validate_point_array(
        read_cgns_data(
            coordinates[x_coordinate], f"{cgns_path} coordinate {x_coordinate!r}"
        ),
        name=f"{cgns_path} coordinate {x_coordinate!r}",
    )
    z = validate_point_array(
        read_cgns_data(
            coordinates[z_coordinate], f"{cgns_path} coordinate {z_coordinate!r}"
        ),
        name=f"{cgns_path} coordinate {z_coordinate!r}",
        point_count=x.size,
    )
    if x.size < MINIMUM_POINT_COUNT:
        raise CgnsDataError(
            f"{cgns_path} must contain at least {MINIMUM_POINT_COUNT} points"
        )
    return x, z


def _read_field(
    zone: h5py.Group,
    cgns_path: Path,
    flow_solution: str,
    field_name: str,
    point_count: int,
) -> np.ndarray:
    description = f"{cgns_path} field {flow_solution}/{field_name}"
    return validate_point_array(
        read_cgns_data(zone[f"{flow_solution}/{field_name}"], description),
        name=description,
        point_count=point_count,
    )


def load_cgns_density_points(
    cgns_path: str | Path,
    *,
    zone_path: str = DEFAULT_ZONE_PATH,
    flow_solution: str = DEFAULT_FLOW_SOLUTION,
    x_coordinate: str = DEFAULT_X_COORDINATE,
    z_coordinate: str = DEFAULT_Z_COORDINATE,
    density_name: str = DEFAULT_DENSITY_NAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, z, density)`` point samples in source units from one CGNS zone."""
    cgns_path = Path(cgns_path).resolve()
    with _open_zone(cgns_path, zone_path) as zone:
        x, z = _read_coordinates(zone, cgns_path, x_coordinate, z_coordinate)
        density = _read_field(zone, cgns_path, flow_solution, density_name, x.size)
    return x, z, density


def load_cgns_scalar_fields(
    cgns_path: str | Path,
    *,
    zone_path: str = DEFAULT_ZONE_PATH,
    flow_solution: str = DEFAULT_FLOW_SOLUTION,
    x_coordinate: str = DEFAULT_X_COORDINATE,
    z_coordinate: str = DEFAULT_Z_COORDINATE,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
) -> CgnsScalarFieldData:
    """Load every valid point-aligned scalar field, optionally clipped in source metres.

    Solution children that are coordinates, non-groups, or whose data does not align
    with the point coordinates are skipped rather than rejected.
    """
    cgns_path = Path(cgns_path).resolve()
    x_bounds = validate_bounds(x_bounds, axis_name="x")
    z_bounds = validate_bounds(z_bounds, axis_name="z")
    with _open_zone(cgns_path, zone_path) as zone:
        x, z = _read_coordinates(zone, cgns_path, x_coordinate, z_coordinate)
        fields: dict[str, np.ndarray] = {}
        for field_name, node in zone[flow_solution].items():
            if field_name in COORDINATE_FIELD_NAMES or not isinstance(node, h5py.Group):
                continue
            try:
                fields[field_name] = _read_field(
                    zone, cgns_path, flow_solution, field_name, x.size
                )
            except CgnsDataError:
                continue

    if not fields:
        raise CgnsDataError(f"{cgns_path} has no valid point-aligned scalar fields")

    mask = np.ones(x.size, dtype=bool)
    if x_bounds is not None:
        mask &= (x >= x_bounds[0]) & (x <= x_bounds[1])
    if z_bounds is not None:
        mask &= (z >= z_bounds[0]) & (z <= z_bounds[1])
    if np.count_nonzero(mask) < MINIMUM_POINT_COUNT:
        raise CgnsDataError(
            f"Requested x/z bounds contain fewer than {MINIMUM_POINT_COUNT} CGNS points"
        )
    if not np.all(mask):
        x = x[mask]
        z = z[mask]
        fields = {name: values[mask] for name, values in fields.items()}
    return CgnsScalarFieldData(x=x, z=z, fields=fields)
