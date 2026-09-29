from __future__ import annotations

import sys
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from fludat_proc.cgns_io import CgnsDataError, load_cgns_density_points
from fludat_proc.cgns_to_density_hdf5 import (
    collect_cgns_inputs,
    convert_cgns_to_density_hdf5,
    grid_density,
    main,
    select_positive_quadrant,
    shared_regular_grid,
    symmetric_z_grid,
)
from fludat_proc.density_cube import load_density_cube

from .conftest import write_cgns_file, write_density_cgns


def test_load_cgns_density_points_returns_x_z_density(tmp_path):
    path = write_density_cgns(tmp_path / "5_bar.cgns")

    x, z, density = load_cgns_density_points(path)

    assert x.shape == z.shape == density.shape == (6,)
    np.testing.assert_allclose(density, x + 2 * z + 3)
    with pytest.raises(CgnsDataError, match="missing required CGNS node"):
        load_cgns_density_points(path, density_name="Nope")


def test_collect_cgns_inputs(tmp_path):
    for name in ("10_bar.cgns", "5_bar.cgns"):
        write_density_cgns(tmp_path / name)
    (tmp_path / "notes.txt").write_text("ignored")

    assert [p for p, _ in collect_cgns_inputs(tmp_path, pressure_bar=None)] == [
        5.0,
        10.0,
    ]
    assert collect_cgns_inputs(tmp_path / "5_bar.cgns", pressure_bar=7.0)[0][0] == 7.0
    with pytest.raises(ValueError, match="pressure_bar is required"):
        collect_cgns_inputs(tmp_path / "5_bar.cgns", pressure_bar=None)
    with pytest.raises(ValueError, match="only valid"):
        collect_cgns_inputs(tmp_path, pressure_bar=5.0)


def test_select_positive_quadrant_requires_three_points():
    x = np.array([-1.0, 0.0, 1.0, 1.0, 0.5])
    z = np.array([0.0, 0.0, 0.0, -1.0, 1.0])

    sx, sz, sd = select_positive_quadrant(x, z, np.arange(5.0))

    np.testing.assert_array_equal(sx, [0.0, 1.0, 0.5])
    np.testing.assert_array_equal(sz, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(sd, [1.0, 2.0, 4.0])
    with pytest.raises(CgnsDataError, match="fewer than 3"):
        select_positive_quadrant(x[:4], z[:4], np.arange(4.0))


def test_symmetric_z_grid_is_exactly_mirrored_and_evenly_spaced():
    for count in (4, 5, 1000):
        grid = symmetric_z_grid(1.0, count)
        assert grid.size == count
        np.testing.assert_array_equal(grid, -grid[::-1])
        np.testing.assert_allclose(np.diff(grid), 2.0 / (count - 1))
    np.testing.assert_allclose(symmetric_z_grid(1.0, 4), [-1.0, -1 / 3, 1 / 3, 1.0])
    assert 0.0 in symmetric_z_grid(1.0, 5)
    assert 0.0 not in symmetric_z_grid(1.0, 4)


def test_shared_regular_grid_defaults_to_symmetric_common_extent():
    fields = [
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.0]), np.ones(3)),
        (np.array([0.5, 1.0, 3.0]), np.array([0.0, 1.0, 2.0]), np.ones(3)),
    ]

    z_grid, x_grid = shared_regular_grid(fields, z_count=5, x_count=4)

    np.testing.assert_allclose(z_grid, [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(x_grid, [0.5, 1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="finite and increasing"):
        shared_regular_grid(fields, z_count=5, x_count=4, z_bounds=(1.0, -1.0))


def test_grid_density_samples_at_abs_z_and_fills_outside():
    field = (
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),  # density = x + 2z
    )
    z_grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    x_grid = np.array([0.0, 0.5, 1.0, 2.0])

    gridded = grid_density(field, z_grid, x_grid, outside_fill="zero")

    np.testing.assert_allclose(
        gridded[:, :3], np.abs(z_grid)[:, None] * 2 + x_grid[None, :3]
    )
    np.testing.assert_array_equal(gridded[:, 3], 0.0)
    with pytest.raises(ValueError, match="outside the CGNS domain"):
        grid_density(field, z_grid, x_grid, outside_fill="raise")
    nearest = grid_density(field, z_grid, x_grid, outside_fill="nearest")
    assert np.all(np.isfinite(nearest))
    assert np.all(np.isin(nearest[:, 3], field[2]))


def test_convert_directory_writes_symmetric_cube_with_mm_x(tmp_path):
    for pressure in (10.0, 5.0):
        write_density_cgns(tmp_path / f"{pressure:g}_bar.cgns", density_scale=pressure)
    output = tmp_path / "out.h5"

    convert_cgns_to_density_hdf5(
        tmp_path,
        output,
        density_scale=2.0,
        density_units="kg/m^3",
        z_count=5,
        x_count=3,
    )

    cube = load_density_cube(output)
    assert cube.nozzle == tmp_path.name
    np.testing.assert_array_equal(cube.pressure, [5.0, 10.0])
    np.testing.assert_allclose(cube.x, [0.0, 500.0, 1000.0])
    np.testing.assert_allclose(cube.z, [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(cube.density, cube.density[::-1])
    # density = scale * pressure * (x + 2z + 3) at x = 0.5 m, z = 0.5 m, 10 bar.
    assert cube.density[3, 1, 1] == pytest.approx(2.0 * 10.0 * 4.5)
    with h5py.File(output) as file:
        assert file.attrs["source_format"] == "CGNS"
        assert len(file.attrs["source_files"]) == 2


def test_cli_single_file_with_pressure_and_mm_bounds(tmp_path):
    path = write_density_cgns(tmp_path / "field.cgns")
    output = tmp_path / "single.h5"
    argv = [
        "cgns_to_density_hdf5",
        str(path),
        "-o",
        str(output),
        "--pressure",
        "7",
        "--density-units",
        "kg/m^3",
        "--z-points",
        "3",
        "--x-points",
        "2",
        "--x-bounds",
        "0",
        "500",
        "--nozzle",
        "htu",
    ]
    with patch.object(sys, "argv", argv):
        main()

    cube = load_density_cube(output)
    assert cube.nozzle == "htu"
    np.testing.assert_array_equal(cube.pressure, [7.0])
    np.testing.assert_allclose(cube.x, [0.0, 500.0])


def test_convert_rejects_field_without_positive_quadrant(tmp_path):
    write_cgns_file(
        tmp_path / "5_bar.cgns",
        x=np.array([-1.0, -2.0, -3.0]),
        z=np.array([0.0, 1.0, 2.0]),
        fields={"Density": np.ones(3)},
    )
    with pytest.raises(CgnsDataError, match="fewer than 3"):
        convert_cgns_to_density_hdf5(
            tmp_path, tmp_path / "o.h5", density_scale=1.0, density_units="u"
        )
