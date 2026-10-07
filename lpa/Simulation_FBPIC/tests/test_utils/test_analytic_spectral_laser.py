"""Tests for scalar-defined synthetic spectral laser pulses."""

from __future__ import annotations

import numpy as np
import pytest

from inversion_fbpic.utils.laser import (
    AnalyticSpectralLongitudinalProfile,
    HighOrderLasyLaser,
)


def test_spectral_profile_is_normalized_and_zero_padded() -> None:
    profile = AnalyticSpectralLongitudinalProfile(
        wavelength=800e-9,
        bandwidth_fwhm=1.0e14,
        time_half_width=100e-15,
        npoints=1024,
    )

    intensity = np.abs(profile.evaluate(profile.time_axis)) ** 2

    assert intensity.max() == pytest.approx(1.0)
    assert profile.evaluate(np.array([200e-15]))[0] == 0.0j


def test_gdd_broadens_the_synthesized_temporal_pulse() -> None:
    transform_limited = AnalyticSpectralLongitudinalProfile(
        wavelength=800e-9,
        bandwidth_fwhm=1.0e14,
        time_half_width=200e-15,
        npoints=4096,
    )
    chirped = AnalyticSpectralLongitudinalProfile(
        wavelength=800e-9,
        bandwidth_fwhm=1.0e14,
        time_half_width=200e-15,
        npoints=4096,
        gdd=2.0e-27,
    )

    time_squared = transform_limited.time_axis**2
    transform_limited_width = np.sqrt(
        np.average(time_squared, weights=np.abs(transform_limited.temporal_field) ** 2)
    )
    chirped_width = np.sqrt(
        np.average(time_squared, weights=np.abs(chirped.temporal_field) ** 2)
    )

    assert chirped_width > transform_limited_width


def test_positive_tod_uses_standard_spectral_phase_convention() -> None:
    tod = 1.0e-42
    profile = AnalyticSpectralLongitudinalProfile(
        wavelength=800e-9,
        bandwidth_fwhm=1.0e14,
        time_half_width=200e-15,
        npoints=1024,
        tod=tod,
    )
    frequency_bin = 10
    dt = float(profile.time_axis[1] - profile.time_axis[0])
    angular_frequency_offset = 2.0 * np.pi * np.fft.fftfreq(
        len(profile.time_axis), dt
    )[frequency_bin]
    expected_phase = tod * (-angular_frequency_offset) ** 3 / 6.0

    assert np.angle(profile.spectral_field[frequency_bin]) == pytest.approx(
        expected_phase
    )


def test_auto_bandwidth_matches_transform_limited_gaussian() -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.physical_parameters = {
        "laser_spectral_bandwidth_rad_s": "auto",
        "laser_pulse_duration_fwhm_s": 30e-15,
    }

    bandwidth = laser._resolve_spectral_bandwidth()

    assert bandwidth == pytest.approx(4.0 * np.log(2.0) / 30e-15)


def test_absolute_and_relative_spectral_phase_controls_are_mutually_exclusive() -> None:
    with pytest.raises(
        ValueError,
        match="either laser_gdd_s2 or laser_gdd_relative",
    ):
        HighOrderLasyLaser._validate_phase_parameter_sources(
            {"laser_gdd_s2": 1e-28, "laser_gdd_relative": 1.0}
        )


def test_zero_bandwidth_rejects_nonzero_spectral_phase() -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.physical_parameters = {
        "laser_spectral_bandwidth_rad_s": 0.0,
        "laser_gdd_s2": 1e-28,
        "laser_tod_s3": 0.0,
        "laser_fod_s4": 0.0,
    }

    with pytest.raises(
        ValueError,
        match="Nonzero GDD, TOD, or FOD requires a nonzero",
    ):
        laser._validate_spectral_phase_bandwidth()


def test_peak_delay_expands_temporal_grid() -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.hyperparameters = {"peak_delay_from_file_start_s": 200e-15}

    time_half_width = laser._time_half_width_for_peak_delay(90e-15)

    assert time_half_width == pytest.approx(145e-15)


def test_peak_delay_warns_when_it_crops_a_material_leading_edge() -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.hyperparameters = {"peak_delay_from_file_start_s": 0.0}
    laser._LEADING_EDGE_CROP_INTENSITY_TOLERANCE = 1e-3
    time = np.array([0.0, 1.0, 2.0])
    intensity = np.array([0.0, 1.0, 0.0])

    class Grid:
        @staticmethod
        def get_temporal_field() -> np.ndarray:
            return np.ones((1, 1, 3), dtype=complex)

        @staticmethod
        def set_temporal_field(field: np.ndarray) -> None:
            pass

    class Laser:
        grid = Grid()

    laser.laser = Laser()
    laser._on_axis_intensity = lambda: (time, intensity)

    with pytest.warns(UserWarning, match="crops the leading edge"):
        laser._set_peak_delay_from_file_start()


@pytest.mark.parametrize(
    ("duration", "relative_gdd", "relative_tod", "expected_gdd", "expected_tod"),
    [
        (30e-15, 1.0, -1.0, 5e-28, -1e-41),
        (50e-15, -0.5, 0.5, -0.5 * 5e-28 * (50 / 30) ** 2, 0.5e-41 * (50 / 30) ** 3),
    ],
)
def test_relative_spectral_phase_scales_with_pulse_duration(
    duration: float,
    relative_gdd: float,
    relative_tod: float,
    expected_gdd: float,
    expected_tod: float,
) -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.physical_parameters = {
        "laser_pulse_duration_fwhm_s": duration,
        "laser_gdd_relative": relative_gdd,
        "laser_tod_relative": relative_tod,
        "laser_gdd_s2": 0.0,
        "laser_tod_s3": 0.0,
    }

    laser._resolve_relative_spectral_phase()

    assert laser.physical_parameters["laser_gdd_s2"] == pytest.approx(expected_gdd)
    assert laser.physical_parameters["laser_tod_s3"] == pytest.approx(expected_tod)