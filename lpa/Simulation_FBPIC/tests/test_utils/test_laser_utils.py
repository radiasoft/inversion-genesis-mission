"""
Tests for ``inversion_fbpic.utils.laser.HighOrderLasyLaser`` helpers.

Run from Simulation_FBPIC::

    pytest tests/test_utils/test_laser_utils.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

PHYSICAL: dict = {
    "laser_wavelength_m": 0.8e-6,
    "laser_energy_J": 1.0,
    "laser_pulse_duration_fwhm_s": 30e-15,
    "laser_spot_size_m": 20e-6,
    "laser_super_gaussian_order": 2.0,
    "laser_focal_position_m": 1.0e-3,
}
HYPER: dict = {
    "n_azimuthal_modes": 1,
    "num_points": (48, 96),
    "centering_angles": 8,
}


def _build():
    from inversion_fbpic.utils.laser import ZERNIKE_OSA_INDICES, HighOrderLasyLaser

    physical = {**PHYSICAL, **{f"zernike_{n}": 0.0 for n in ZERNIKE_OSA_INDICES}}
    return HighOrderLasyLaser(physical, HYPER)


@pytest.mark.integration
class TestFocusA0:
    def test_focus_a0_matches_analytic_gaussian(self) -> None:
        from inversion_fbpic.utils.laser import peak_a0
        from inversion_fbpic.utils.simulation_setup_tools import (
            calculate_laser_a0_from_energy,
        )

        laser = _build()
        focus_a0 = laser.compute_focus_a0()
        start_plane_a0 = peak_a0(laser.laser)

        # The start plane is 1 mm before focus, so the peak field is lower there.
        assert focus_a0 > start_plane_a0
        assert focus_a0 == pytest.approx(
            calculate_laser_a0_from_energy(1.0, 0.8, 20e-6, 30e-15), rel=0.05
        )

    def test_focus_laser_leaves_start_plane_untouched(self) -> None:
        laser = _build()
        before = laser.laser.grid.get_temporal_field().copy()
        focus = laser.focus_laser()
        assert focus is not laser.laser
        np.testing.assert_array_equal(laser.laser.grid.get_temporal_field(), before)
        assert not np.allclose(focus.grid.get_temporal_field(), before)
