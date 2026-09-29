"""Shared fixtures: a headless Matplotlib backend and small synthetic inputs."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from fludat_proc.density_cube import DensityCube, write_density_cube  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def write_lineout_file(
    path: Path, sections: dict[str, list[tuple[float, float]]]
) -> Path:
    """Write an ANSYS-style sectioned lineout export."""
    lines = ['(title "Density")', '(labels "Position" "Density")', ""]
    for label, points in sections.items():
        lines.append(f'((xy/key/label "{label}")')
        lines.extend(f"{z} {density}" for z, density in points)
        lines.extend([")", ""])
    path.write_text("\n".join(lines))
    return path


def write_cgns_file(
    path: Path,
    *,
    x: np.ndarray,
    z: np.ndarray,
    fields: dict[str, np.ndarray],
) -> Path:
    """Write a minimal Fluent-style CGNS (HDF5) file with point-aligned fields."""
    with h5py.File(path, "w") as cgns_file:
        zone = cgns_file.create_group("Base/Zone")
        coordinates = zone.create_group("GridCoordinates")
        coordinates.create_group("CoordinateX").create_dataset(" data", data=x)
        coordinates.create_group("CoordinateY").create_dataset(" data", data=z)
        solution = zone.create_group("FlowSolution.N:1")
        for name, values in fields.items():
            solution.create_group(name).create_dataset(" data", data=values)
    return path


def write_density_cgns(path: Path, density_scale: float = 1.0) -> Path:
    """A six-point field spanning x in [-1, 1], z in [0, 1] with density x + 2z + 3."""
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]
    )
    x, z = points[:, 0], points[:, 1]
    return write_cgns_file(
        path, x=x, z=z, fields={"Density": density_scale * (x + 2 * z + 3)}
    )


@pytest.fixture
def small_cube() -> DensityCube:
    """A 5 x 3 x 2 cube with density = (1 + z) * (1 + x) * pressure."""
    z = np.linspace(-2.0, 2.0, 5)
    x = np.array([0.0, 1.0, 2.0])
    pressure = np.array([5.0, 10.0])
    density = (1 + z)[:, None, None] * (1 + x)[None, :, None] * pressure[None, None, :]
    return DensityCube(
        nozzle="test",
        z=z,
        x=x,
        pressure=pressure,
        density=density,
        density_units="cm^-3",
    )


@pytest.fixture
def small_cube_path(tmp_path: Path, small_cube: DensityCube) -> Path:
    return write_density_cube(tmp_path / "test.h5", small_cube, density_scale=1.0)
