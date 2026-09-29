from __future__ import annotations

import h5py
import numpy as np
import pytest

from fludat_proc.density_cube import DensityCube, load_density_cube, write_density_cube


def test_round_trip_preserves_data_units_and_attributes(tmp_path, small_cube):
    path = write_density_cube(
        tmp_path / "cube.h5",
        small_cube,
        density_scale=2.0,
        attributes={"source_format": "test"},
    )

    loaded = load_density_cube(path)

    assert loaded.nozzle == "test"
    assert loaded.density_units == "cm^-3"
    np.testing.assert_array_equal(loaded.density, small_cube.density)
    np.testing.assert_array_equal(loaded.z, small_cube.z)
    with h5py.File(path) as file:
        assert file.attrs["density_scale"] == 2.0
        assert file.attrs["source_format"] == "test"
        assert file["x_mm"].attrs["units"] == "mm"
        assert file["density"].dims[2].label == "pressure_bar"


def test_cube_validates_shape_and_monotonic_coordinates():
    z = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        DensityCube(
            "n", z[::-1], np.array([0.0]), np.array([1.0]), np.zeros((2, 1, 1)), "u"
        )
    with pytest.raises(ValueError, match="shape"):
        DensityCube("n", z, np.array([0.0]), np.array([1.0]), np.zeros((1, 2, 1)), "u")
    with pytest.raises(ValueError, match="finite"):
        DensityCube(
            "n", z, np.array([0.0]), np.array([1.0]), np.full((2, 1, 1), np.nan), "u"
        )


def test_load_rejects_missing_dimension_scales(tmp_path, small_cube):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("z_m", data=small_cube.z)
        file.create_dataset("x_mm", data=small_cube.x)
        file.create_dataset("pressure_bar", data=small_cube.pressure)
        file.create_dataset("density", data=small_cube.density)

    with pytest.raises(ValueError, match="density axis 0"):
        load_density_cube(path)


def test_load_reports_missing_datasets(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("z_m", data=[0.0, 1.0])

    with pytest.raises(ValueError, match="missing required dataset"):
        load_density_cube(path)
    with pytest.raises(FileNotFoundError):
        load_density_cube(tmp_path / "nope.h5")
