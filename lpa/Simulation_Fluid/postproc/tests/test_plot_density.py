from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from fludat_proc.interpolate_density import DensityInterpolation
from fludat_proc.plot_density import (
    main,
    plot_density_callable_lineout,
    plot_density_interpolation,
    plot_density_lineout,
)


def test_xz_map_labels_density_with_cube_units(small_cube):
    plot = plot_density_interpolation(
        DensityInterpolation(small_cube), interactive=False
    )

    axes = plot.figure.axes
    assert axes[1].get_ylabel() == "density [cm^-3]"
    assert axes[0].get_title() == "test @ 5.0 bar (linear)"
    assert plot.widgets == ()


def test_interactive_xz_map_slider_updates_mesh(small_cube):
    plot = plot_density_interpolation(
        DensityInterpolation(small_cube), x_points=3, z_points=3
    )

    (slider,) = plot.widgets
    mesh = plot.figure.axes[0].collections[0]
    before = mesh.get_array().copy()
    slider.set_val(10.0)
    np.testing.assert_allclose(mesh.get_array().ravel(), before.ravel() * 2)
    assert plot.figure.axes[0].get_title() == "test @ 10.0 bar (linear)"


def test_lineout_respects_bounds_and_validates_x(small_cube):
    field = DensityInterpolation(small_cube)

    plot = plot_density_lineout(
        field, x_position=1.0, z_bounds=(-1.0, 1.0), interactive=False
    )
    (line,) = plot.figure.axes[0].lines
    np.testing.assert_allclose(line.get_xdata()[[0, -1]], [-1000.0, 1000.0])
    assert plot.figure.axes[0].get_ylabel() == "density [cm^-3]"

    with pytest.raises(ValueError, match="outside the plot bounds"):
        plot_density_lineout(field, x_position=5.0, interactive=False)
    with pytest.raises(ValueError, match="outside the valid range"):
        plot_density_lineout(field, z_bounds=(-5.0, 1.0), interactive=False)


def test_interactive_lineout_sliders_refresh_line(small_cube):
    plot = plot_density_lineout(DensityInterpolation(small_cube))

    pressure_slider, x_slider = plot.widgets
    (line,) = plot.figure.axes[0].lines
    before = line.get_ydata().copy()
    pressure_slider.set_val(10.0)
    np.testing.assert_allclose(line.get_ydata(), before * 2)
    x_slider.set_val(1.0)
    np.testing.assert_allclose(line.get_ydata(), before * 4)


def test_callable_lineout_matches_direct_lineout(small_cube_path, small_cube):
    field = DensityInterpolation(small_cube)
    direct = plot_density_lineout(
        field, x_position=1.0, pressure=7.5, interactive=False
    )
    via_callable = plot_density_callable_lineout(
        small_cube_path, field=field, x_position=1.0, pressure=7.5, interactive=False
    )

    np.testing.assert_allclose(
        via_callable.figure.axes[0].lines[0].get_ydata(),
        direct.figure.axes[0].lines[0].get_ydata(),
    )
    assert "callable" in via_callable.figure.axes[0].get_title()


def test_cli_saves_figure(tmp_path, small_cube_path):
    output = tmp_path / "map.png"
    argv = [
        "plot_density",
        str(small_cube_path),
        "-o",
        str(output),
        "--z-bounds",
        "-1000",
        "1000",
    ]
    with patch.object(sys, "argv", argv):
        main()
    assert output.stat().st_size > 0
