"""Interpolate a nozzle density cube in ``(z, x, backing pressure)``.

Along ``z`` the tabulated grid is interpolated linearly. In the ``(x, pressure)``
plane a :class:`scipy.interpolate.RegularGridInterpolator` with the requested
method is used. Queries outside the tabulated ``z``, ``x``, or pressure range
evaluate to :data:`FILL_VALUE` (zero density) rather than raising, so an FBPIC
window that extends beyond the simulated gas jet simply sees vacuum there.
For plotting, see :mod:`fludat_proc.plot_density`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from .density_cube import DensityCube, load_density_cube

InterpolationMethod = Literal["linear", "cubic", "quintic", "pchip"]
INTERPOLATION_METHODS: tuple[InterpolationMethod, ...] = (
    "linear",
    "cubic",
    "quintic",
    "pchip",
)
FILL_VALUE = 0.0
"""Density returned for queries outside the tabulated ranges."""


@dataclass(frozen=True)
class DensityInterpolation:
    """A density cube together with the ``(x, pressure)`` interpolation method."""

    cube: DensityCube
    method: InterpolationMethod = "linear"

    @property
    def geometry(self) -> str:
        return self.cube.nozzle

    @property
    def density_units(self) -> str:
        return self.cube.density_units

    @property
    def z(self) -> np.ndarray:
        return self.cube.z

    @property
    def x(self) -> np.ndarray:
        return self.cube.x

    @property
    def pressure(self) -> np.ndarray:
        return self.cube.pressure

    @property
    def x_extent(self) -> tuple[float, float]:
        return self.cube.x_extent

    @property
    def z_extent(self) -> tuple[float, float]:
        return self.cube.z_extent

    @property
    def pressure_extent(self) -> tuple[float, float]:
        return self.cube.pressure_extent

    def density_at_z(self, z_value: float) -> np.ndarray:
        """Return the ``(n_x, n_pressure)`` density slice linearly interpolated at ``z``.

        Outside the tabulated z range the slice is :data:`FILL_VALUE` everywhere.
        """
        z_min, z_max = self.z_extent
        if not z_min <= z_value <= z_max:
            return np.full(self.cube.density.shape[1:], FILL_VALUE)
        upper = int(np.searchsorted(self.z, z_value, side="left"))
        if upper == 0:
            return self.cube.density[0].copy()
        lower = upper - 1
        weight = (z_value - self.z[lower]) / (self.z[upper] - self.z[lower])
        return (1.0 - weight) * self.cube.density[lower] + weight * self.cube.density[
            upper
        ]

    def _interpolator_at_z(self, z_value: float) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.x, self.pressure),
            self.density_at_z(z_value),
            method=self.method,
            bounds_error=False,
            fill_value=FILL_VALUE,
        )

    def interpolate(
        self, z_value: float, x_value: float, pressure_value: float
    ) -> float:
        """Interpolate density at one ``(z, x, pressure)`` point; zero outside the cube."""
        return float(self._interpolator_at_z(z_value)((x_value, pressure_value)))

    def interpolate_along_z(
        self,
        z_values: np.ndarray,
        x_value: float,
        pressure_value: float,
    ) -> np.ndarray:
        """Interpolate density along ``z`` at fixed ``x`` and backing pressure.

        Points outside the tabulated ranges evaluate to :data:`FILL_VALUE`.
        """
        z_array = np.asarray(z_values, dtype=np.float64)
        result = np.empty_like(z_array)
        for index, z_value in enumerate(z_array.flat):
            result.flat[index] = self.interpolate(
                float(z_value), x_value, pressure_value
            )
        return result

    def interpolate_xz_grid(
        self,
        x_values: np.ndarray,
        z_values: np.ndarray,
        pressure_value: float,
    ) -> np.ndarray:
        """Interpolate density on an ``(n_x, n_z)`` grid at fixed backing pressure."""
        x_values = np.asarray(x_values, dtype=np.float64)
        query = np.column_stack((x_values, np.full(x_values.size, pressure_value)))
        density_grid = np.empty((x_values.size, len(z_values)), dtype=np.float64)
        for iz, z_value in enumerate(z_values):
            density_grid[:, iz] = self._interpolator_at_z(float(z_value))(query)
        return density_grid

    def xz_grids_at_pressures(
        self, x_values: np.ndarray, z_values: np.ndarray
    ) -> np.ndarray:
        """Return an ``(n_pressure, n_x, n_z)`` stack, one grid per tabulated pressure."""
        return np.stack(
            [
                self.interpolate_xz_grid(x_values, z_values, float(pressure_value))
                for pressure_value in self.pressure
            ]
        )

    def interpolate_xz_from_pressure_stack(
        self,
        xz_stack: np.ndarray,
        pressure_value: float,
    ) -> np.ndarray:
        """Linearly interpolate a precomputed ``(pressure, x, z)`` stack to one pressure.

        Pressures outside the tabulated range give a :data:`FILL_VALUE` grid.
        """
        interpolator = interp1d(
            self.pressure,
            xz_stack,
            axis=0,
            bounds_error=False,
            fill_value=FILL_VALUE,
            assume_sorted=True,
        )
        return np.asarray(interpolator(pressure_value), dtype=np.float64)


def build_density_interpolation(
    hdf5_path: str | Path,
    *,
    method: InterpolationMethod = "linear",
) -> DensityInterpolation:
    """Load an HDF5 density cube and wrap it in a :class:`DensityInterpolation`."""
    return DensityInterpolation(cube=load_density_cube(hdf5_path), method=method)


def build_density_callable(
    hdf5_path: str | Path,
    backing_pressure: float,
    x_position: float,
    *,
    method: InterpolationMethod = "linear",
    field: DensityInterpolation | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return an FBPIC-compatible ``density(z, r)`` function at fixed ``x`` and pressure.

    ``r`` is required by FBPIC but ignored: the profile depends on ``z`` only. The
    cube is loaded from ``hdf5_path`` unless an already-built ``field`` is supplied.
    ``z`` values beyond the tabulated range return :data:`FILL_VALUE` (vacuum).
    """
    if field is None:
        field = build_density_interpolation(hdf5_path, method=method)
    elif method != field.method:
        raise ValueError(
            f"method={method!r} does not match field.method={field.method!r}"
        )

    def density(z: np.ndarray, r: np.ndarray) -> np.ndarray:
        del r
        return field.interpolate_along_z(z, x_position, backing_pressure)

    return density
