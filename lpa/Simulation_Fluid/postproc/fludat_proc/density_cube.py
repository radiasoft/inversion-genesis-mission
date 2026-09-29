"""The HDF5 density-cube format shared by every tool in this package.

One file holds one nozzle: a ``density`` dataset indexed as ``(z_m, x_mm, pressure_bar)``
with each axis linked to its coordinate dataset through HDF5 dimension scales.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

COORDINATE_NAMES = ("z_m", "x_mm", "pressure_bar")
COORDINATE_UNITS = ("m", "mm", "bar")
REQUIRED_DATASETS = (*COORDINATE_NAMES, "density")
AXIS_ORDER_ATTRIBUTE = ",".join(COORDINATE_NAMES)


@dataclass(frozen=True)
class DensityCube:
    """One nozzle's density on a regular ``(z, x, pressure)`` grid."""

    nozzle: str
    z: np.ndarray
    x: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    density_units: str

    def __post_init__(self) -> None:
        for name, values in (
            ("z_m", self.z),
            ("x_mm", self.x),
            ("pressure_bar", self.pressure),
        ):
            _validate_coordinates(name, values)
        expected_shape = (self.z.size, self.x.size, self.pressure.size)
        if self.density.shape != expected_shape:
            raise ValueError(
                f"density has shape {self.density.shape}; expected {expected_shape} "
                f"for ({AXIS_ORDER_ATTRIBUTE})"
            )
        if not np.all(np.isfinite(self.density)):
            raise ValueError("density must contain finite values")
        if not self.density_units:
            raise ValueError("density_units must not be empty")

    @property
    def x_extent(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[-1])

    @property
    def z_extent(self) -> tuple[float, float]:
        return float(self.z[0]), float(self.z[-1])

    @property
    def pressure_extent(self) -> tuple[float, float]:
        return float(self.pressure[0]), float(self.pressure[-1])


def _validate_coordinates(name: str, values: np.ndarray) -> None:
    if values.ndim != 1:
        raise ValueError(f"dataset {name!r} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"dataset {name!r} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"dataset {name!r} must contain finite values")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"dataset {name!r} must be strictly increasing")


def _as_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _validate_dimension_scales(hdf5_file: h5py.File, path: Path) -> None:
    """Ensure each density axis is explicitly linked to its coordinate dataset."""
    density_dataset = hdf5_file["density"]
    for axis, name in enumerate(COORDINATE_NAMES):
        dimension = density_dataset.dims[axis]
        if dimension.label != name:
            raise ValueError(
                f"{path} density axis {axis} has label {dimension.label!r}; expected {name!r}"
            )
        scales = list(dimension.values())
        if len(scales) != 1 or scales[0].name != hdf5_file[name].name:
            attached_names = [scale.name for scale in scales]
            raise ValueError(
                f"{path} density axis {axis} must be linked only to {name!r}; "
                f"found {attached_names}"
            )


def write_density_cube(
    output_path: str | Path,
    cube: DensityCube,
    *,
    density_scale: float,
    attributes: Mapping[str, object] | None = None,
) -> Path:
    """Write a density cube to HDF5, returning the resolved output path.

    ``attributes`` are extra root attributes recording how the cube was produced.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hdf5_file:
        hdf5_file.attrs["nozzle"] = cube.nozzle
        hdf5_file.attrs["density_scale"] = density_scale
        hdf5_file.attrs["density_axis_order"] = AXIS_ORDER_ATTRIBUTE
        for key, value in (attributes or {}).items():
            hdf5_file.attrs[key] = value

        density_dataset = hdf5_file.create_dataset("density", data=cube.density)
        density_dataset.attrs["units"] = cube.density_units
        coordinates = (cube.z, cube.x, cube.pressure)
        for axis, (name, units, values) in enumerate(
            zip(COORDINATE_NAMES, COORDINATE_UNITS, coordinates, strict=True)
        ):
            coordinate_dataset = hdf5_file.create_dataset(name, data=values)
            coordinate_dataset.attrs["units"] = units
            coordinate_dataset.make_scale(name)
            density_dataset.dims[axis].label = name
            density_dataset.dims[axis].attach_scale(coordinate_dataset)

    return output_path


def load_density_cube(path: str | Path) -> DensityCube:
    """Load and validate a nozzle density cube from HDF5."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        with h5py.File(path, "r") as hdf5_file:
            missing = [name for name in REQUIRED_DATASETS if name not in hdf5_file]
            if missing:
                raise ValueError(f"{path} is missing required dataset(s): {missing}")

            axis_order = hdf5_file.attrs.get("density_axis_order")
            if axis_order is not None and _as_text(axis_order) != AXIS_ORDER_ATTRIBUTE:
                raise ValueError(
                    f"{path} has density_axis_order={_as_text(axis_order)!r}; "
                    f"expected {AXIS_ORDER_ATTRIBUTE!r}"
                )
            _validate_dimension_scales(hdf5_file, path)

            density_dataset = hdf5_file["density"]
            cube = DensityCube(
                nozzle=_as_text(hdf5_file.attrs.get("nozzle", path.stem)),
                z=np.asarray(hdf5_file["z_m"], dtype=np.float64),
                x=np.asarray(hdf5_file["x_mm"], dtype=np.float64),
                pressure=np.asarray(hdf5_file["pressure_bar"], dtype=np.float64),
                density=np.asarray(density_dataset, dtype=np.float64),
                density_units=_as_text(density_dataset.attrs.get("units", "unknown")),
            )
    except OSError as exc:
        raise ValueError(f"Could not read HDF5 file {path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return cube
