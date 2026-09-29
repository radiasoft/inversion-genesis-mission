from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from fludat_proc.density_cube import load_density_cube
from fludat_proc.lineouts_to_density_hdf5 import (
    build_density_cube,
    collect_pressure_files,
    main,
)

from .conftest import write_lineout_file


def _write_nozzle(directory, pressures=(5.0, 20.0)):
    directory.mkdir()
    for pressure in pressures:
        write_lineout_file(
            directory / f"{pressure:g}_bar.txt",
            {
                "x-0-0": [(0.0, 1.0 * pressure), (0.001, 0.5 * pressure), (0.002, 0.0)],
                "x-1-0": [
                    (0.002, 0.0),
                    (0.001, 0.25 * pressure),
                    (0.0, 0.5 * pressure),
                ],
            },
        )
    return directory


def test_collect_pressure_files_sorts_and_rejects_duplicates(tmp_path):
    nozzle = _write_nozzle(tmp_path / "400_um", pressures=(20.0, 5.0))

    assert [pressure for pressure, _ in collect_pressure_files(nozzle)] == [5.0, 20.0]

    write_lineout_file(nozzle / "5_bar.dat", {"x-0-0": [(0.0, 1.0), (1.0, 1.0)]})
    with pytest.raises(ValueError, match="Duplicate backing pressure"):
        collect_pressure_files(nozzle)


def test_build_density_cube_mirrors_scales_and_stacks(tmp_path):
    nozzle = _write_nozzle(tmp_path / "400_um")

    cube = build_density_cube(nozzle, density_scale=2.0, density_units="cm^-3")

    assert cube.nozzle == "400_um"
    np.testing.assert_array_equal(cube.z, [-0.002, -0.001, 0.0, 0.001, 0.002])
    np.testing.assert_array_equal(cube.x, [0.0, 1.0])
    np.testing.assert_array_equal(cube.pressure, [5.0, 20.0])
    # x = 0, 5 bar: raw [5, 2.5, 0] scaled by 2 and mirrored.
    np.testing.assert_allclose(cube.density[:, 0, 0], [0.0, 5.0, 10.0, 5.0, 0.0])
    # x = 1, 20 bar: raw [10, 5, 0] scaled by 2.
    np.testing.assert_allclose(cube.density[:, 1, 1], [0.0, 10.0, 20.0, 10.0, 0.0])


def test_build_density_cube_restricts_z_to_the_shared_range(tmp_path):
    nozzle = _write_nozzle(tmp_path / "400_um")
    # 10 bar only extends to z = 0.001, so the cube must not extrapolate beyond it.
    write_lineout_file(
        nozzle / "10_bar.txt",
        {"x-0-0": [(0.0, 4.0), (0.001, 2.0)], "x-1-0": [(0.0, 2.0), (0.001, 1.0)]},
    )

    cube = build_density_cube(nozzle, density_scale=1.0, density_units="cm^-3")

    np.testing.assert_array_equal(cube.z, [-0.001, 0.0, 0.001])
    np.testing.assert_array_equal(cube.pressure, [5.0, 10.0, 20.0])
    np.testing.assert_allclose(cube.density[:, 0, 1], [2.0, 4.0, 2.0])
    np.testing.assert_allclose(cube.density[:, 0, 0], [2.5, 5.0, 2.5])


def test_build_density_cube_rejects_mismatched_x_positions(tmp_path):
    nozzle = _write_nozzle(tmp_path / "400_um")
    write_lineout_file(nozzle / "10_bar.txt", {"x-0-5": [(0.0, 1.0), (0.001, 0.0)]})

    with pytest.raises(ValueError, match="x positions .* do not match"):
        build_density_cube(nozzle, density_scale=1.0, density_units="cm^-3")


def test_cli_writes_loadable_cube(tmp_path):
    nozzle = _write_nozzle(tmp_path / "400_um")
    output = tmp_path / "out" / "400_um.h5"

    argv = [
        "lineouts_to_density_hdf5",
        "1e20",
        str(nozzle),
        "-o",
        str(output),
        "--density-units",
        "m^-3",
    ]
    with patch.object(sys, "argv", argv):
        main()

    cube = load_density_cube(output)
    assert cube.density_units == "m^-3"
    assert cube.density.shape == (5, 2, 2)
    assert cube.density[2, 0, 0] == pytest.approx(5e20)
