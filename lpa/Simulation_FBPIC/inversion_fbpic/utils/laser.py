"""
laser.py

Module containing useful utilities for modeling laser properties.  Contains the following
- Class for loading HTU longitudinal laser profile from FROG data through LASY.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, TypeAlias, Union
import warnings

import numpy as np
from lasy.laser import Laser
from lasy.optical_elements.zernike_aberrations import ZernikeAberrations
from lasy.profiles import Profile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.longitudinal.longitudinal_profile_from_data import (
    LongitudinalProfileFromData,
)
from lasy.profiles.transverse import SuperGaussianTransverseProfile
from scipy.constants import c, e, epsilon_0, m_e
from scipy.interpolate import RegularGridInterpolator


Array: TypeAlias = np.ndarray

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ZERNIKE_OSA_INDICES = {
    "astigmatism_2": 3,
    "astigmatism_4": 5,
    "coma_y": 7,
    "coma_x": 8,
    "trefoil_y": 6,
    "trefoil_x": 9,
    "spherical_3": 12,
    "astigmatism_6": 13,
    "coma_5_y": 17,
    "coma_5_x": 18,
    "secondary_trefoil_y": 16,
    "secondary_trefoil_x": 19,
}


def polar_fields(laser: Laser, angles: Array) -> Array:
    """Reconstruct a modal LASY field on a polar-angle grid."""
    angular_phase = np.exp(-1j * np.outer(angles, laser.grid.azimuthal_modes))
    return np.einsum(
        "am,mrt->art",
        angular_phase,
        laser.grid.get_temporal_field(),
        optimize=True,
    )


def set_polar_fields(laser: Laser, field: Array) -> None:
    """Decompose fields sampled on a ``[0, 2*pi)`` angle grid into modes."""
    n_modes = laser.grid.n_azimuthal_modes
    minimum_angles = 2 * n_modes - 1
    if field.shape[0] < minimum_angles:
        raise ValueError(
            f"Need at least {minimum_angles} angular samples for {n_modes} modes."
        )
    mode_field = np.fft.ifft(field, axis=0)
    retained_modes = mode_field[:n_modes]
    if n_modes > 1:
        retained_modes = np.concatenate((retained_modes, mode_field[-n_modes + 1 :]))
    laser.grid.set_temporal_field(retained_modes)


def transverse_fluence(
    laser: Laser,
    n_angles: int = 361,
) -> tuple[Array, Array, Array, Array]:
    """Calculate fluence and peak-time fields on an FFT-aligned polar grid."""
    radius, time = laser.grid.axes
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    field = polar_fields(laser, angles)
    intensity = 0.5 * epsilon_0 * c * np.abs(field) ** 2
    fluence = np.trapezoid(intensity, x=time, axis=-1)
    peak_time_index = int(np.argmax(intensity[:, 0, :].mean(axis=0)))
    return radius, angles, fluence, field[:, :, peak_time_index]


def peak_a0(laser: Laser, n_angles: int = 361) -> float:
    """Return the peak normalized vector potential of a LASY envelope.

    The full transverse field is reconstructed from the azimuthal modes on
    ``n_angles`` polar angles, and the envelope maximum is converted with
    ``a0 = e |E| / (m_e c omega0)``.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    field = polar_fields(laser, angles)
    return float(e * np.abs(field).max() / (m_e * c * laser.profile.omega0))


class _ZernikeSuperGaussianProfile(Profile):
    """LASY profile with Zernike phase clamped outside its physical pupil.

    LASY evaluates Zernike polynomials beyond the pupil radius. This profile
    instead evaluates the phase at the nearest pupil boundary point for
    ``rho > 1``, preventing high-order terms from growing through low-intensity
    beam wings.
    """

    def __init__(
        self,
        parameters: Mapping[str, Any],
        pupil_radius: float,
        longitudinal_profile: Any,
    ) -> None:
        super().__init__(parameters["wavelength"], parameters["polarization"])
        self.laser_energy = parameters["energy"]
        zernike_amplitudes = {
            ZERNIKE_OSA_INDICES[name]: amplitude
            for name, amplitude in parameters["zernike_coefficients"].items()
            if name in ZERNIKE_OSA_INDICES and amplitude != 0.0
        }
        self.zernike_aberrations = ZernikeAberrations(
            pupil_coords=(0.0, 0.0, pupil_radius),
            zernike_amplitudes=zernike_amplitudes,
        )
        self.pupil_radius = pupil_radius
        self.longitudinal_profile = longitudinal_profile
        self.transverse_profile = SuperGaussianTransverseProfile(
            w0=parameters["spot_size"],
            n_order=parameters["super_gaussian_order"],
        )

    def evaluate(self, x: Array, y: Array, t: Array) -> Array:
        omega = np.full_like(x, self.omega0)
        radius = np.hypot(x, y)
        scale = np.minimum(1.0, self.pupil_radius / np.maximum(radius, 1e-30))
        return (
            self.longitudinal_profile.evaluate(t)
            * self.transverse_profile.evaluate(x, y)
            * self.zernike_aberrations.amplitude_multiplier(
                x * scale,
                y * scale,
                omega,
            )
        )


class AnalyticSpectralLongitudinalProfile:
    """Complex temporal envelope synthesized from a scalar Gaussian spectrum.

    The spectral intensity has a Gaussian FWHM of ``bandwidth_fwhm`` in angular
    frequency. ``gdd``, ``tod``, and ``fod`` are the second-, third-, and
    fourth-order spectral-phase coefficients about the central frequency.
    """

    def __init__(
        self,
        wavelength: float,
        bandwidth_fwhm: float,
        time_half_width: float,
        npoints: int,
        cep_phase: float = 0.0,
        gdd: float = 0.0,
        tod: float = 0.0,
        fod: float = 0.0,
    ) -> None:
        if bandwidth_fwhm <= 0.0:
            raise ValueError("bandwidth_fwhm must be positive.")
        if time_half_width <= 0.0:
            raise ValueError("time_half_width must be positive.")
        if npoints < 2:
            raise ValueError("npoints must be at least 2.")
        self.wavelength = wavelength
        self.bandwidth_fwhm = bandwidth_fwhm
        self.time_axis = np.linspace(
            -time_half_width,
            time_half_width,
            npoints,
            endpoint=False,
        )
        dt = float(self.time_axis[1] - self.time_axis[0])
        angular_frequency_offset = 2.0 * np.pi * np.fft.fftfreq(npoints, d=dt)
        physical_angular_frequency_offset = -angular_frequency_offset
        spectral_amplitude = np.exp(
            -2.0 * np.log(2.0) * (angular_frequency_offset / bandwidth_fwhm) ** 2
        )
        spectral_phase = (
            cep_phase
            + 0.5 * gdd * physical_angular_frequency_offset**2
            + tod * physical_angular_frequency_offset**3 / 6.0
            + fod * physical_angular_frequency_offset**4 / 24.0
        )
        self.spectral_field = spectral_amplitude * np.exp(1j * spectral_phase)
        self.temporal_field = np.fft.fftshift(np.fft.ifft(self.spectral_field))
        self.temporal_field /= np.max(np.abs(self.temporal_field))

    def evaluate(self, t: Array) -> Array:
        """Return the complex envelope, zero-padded outside the synthesis grid."""
        real = np.interp(
            t, self.time_axis, self.temporal_field.real, left=0.0, right=0.0
        )
        imag = np.interp(
            t, self.time_axis, self.temporal_field.imag, left=0.0, right=0.0
        )
        return real + 1j * imag


class HighOrderLasyLaser:
    """Prepare an aberrated LASY laser at an FBPIC simulation start plane.

    ``physical_parameters`` holds required physical laser values;
    ``hyperparameters`` holds numerical-grid and fixed representation values.
    Construction automatically back-propagates to the simulation start and
    centers the field when requested.
    """

    PHYSICAL_PARAMETER_KEYS = frozenset(
        {
            "laser_wavelength_m",
            "laser_energy_J",
            "laser_pulse_duration_fwhm_s",
            "laser_spot_size_m",
            "laser_super_gaussian_order",
            "laser_focal_position_m",
            *(f"zernike_{name}" for name in ZERNIKE_OSA_INDICES),
        }
    )
    _PHYSICAL_PARAMETER_DEFAULTS: dict[str, float] = {
        "laser_spectral_bandwidth_rad_s": 0.0,
        "laser_cep_phase_rad": 0.0,
        "laser_gdd_s2": 0.0,
        "laser_tod_s3": 0.0,
        "laser_fod_s4": 0.0,
        "laser_gdd_relative": None,
        "laser_tod_relative": None,
    }
    OPTIONAL_PHYSICAL_PARAMETER_KEYS = frozenset(_PHYSICAL_PARAMETER_DEFAULTS)
    _REFERENCE_PULSE_DURATION_FWHM_S = 30e-15
    _REFERENCE_GDD_S2 = 5e-28
    _REFERENCE_TOD_S3 = 1e-41
    _LEADING_EDGE_CROP_INTENSITY_TOLERANCE = 1e-3
    _HYPERPARAMETER_DEFAULTS: dict[str, Any] = {
        "polarization": (1, 0),
        "n_azimuthal_modes": 5,
        "num_points": (600, 900),
        "hi_range": 8.0,
        "center_and_remove_tilt": True,
        "centering_angles": 72,
        "spectral_time_window_factor": 6.0,
        "peak_delay_from_file_start_s": None,
        "maximum_pulse_duration_fwhm_s": None,
    }

    def __init__(
        self,
        physical_parameters: Mapping[str, Any],
        hyperparameters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._validate_phase_parameter_sources(physical_parameters)
        self.physical_parameters = {
            **self._PHYSICAL_PARAMETER_DEFAULTS,
            **physical_parameters,
        }
        self.physical_parameters["laser_spectral_bandwidth_rad_s"] = (
            self._resolve_spectral_bandwidth()
        )
        self._resolve_relative_spectral_phase()
        self.hyperparameters = {
            **self._HYPERPARAMETER_DEFAULTS,
            **(hyperparameters or {}),
        }
        self._validate_parameters()
        self.laser = self._build_laser_at_focus()
        self.laser.propagate(
            distance=-self.physical_parameters["laser_focal_position_m"],
        )
        if self.hyperparameters["center_and_remove_tilt"]:
            self.remove_start_plane_offset_and_tilt()
        self.laser.normalize(
            self.physical_parameters["laser_energy_J"],
            kind="energy",
        )
        self._set_peak_delay_from_file_start()
        self._validate_pulse_duration()
        self._validate_start_plane_grid()

    @staticmethod
    def _validate_phase_parameter_sources(
        physical_parameters: Mapping[str, Any],
    ) -> None:
        """Reject ambiguous absolute and duration-relative phase controls."""
        for absolute_name, relative_name in (
            ("laser_gdd_s2", "laser_gdd_relative"),
            ("laser_tod_s3", "laser_tod_relative"),
        ):
            if (
                absolute_name in physical_parameters
                and physical_parameters.get(relative_name) is not None
            ):
                raise ValueError(
                    f"Specify either {absolute_name} or {relative_name}, not both."
                )

    def _resolve_spectral_bandwidth(self) -> float:
        """Resolve a numeric bandwidth or derive a transform-limited Gaussian value."""
        bandwidth = self.physical_parameters["laser_spectral_bandwidth_rad_s"]
        if bandwidth == "auto":
            duration = self.physical_parameters["laser_pulse_duration_fwhm_s"]
            if duration <= 0.0:
                raise ValueError("laser_pulse_duration_fwhm_s must be positive.")
            return float(4.0 * np.log(2.0) / duration)
        if isinstance(bandwidth, str):
            raise ValueError(
                "laser_spectral_bandwidth_rad_s must be a non-negative number or 'auto'."
            )
        return float(bandwidth)

    def _resolve_relative_spectral_phase(self) -> None:
        """Convert duration-normalized GDD/TOD controls to SI coefficients.

        At a 30 fs FWHM reference pulse, relative values of ``+1`` correspond
        to ``+5e-28 s^2`` GDD and ``+1e-41 s^3`` TOD. Scaling the coefficients
        with duration squared/cubed keeps their phase contribution comparable
        when an auto bandwidth is derived from pulse duration.
        """
        duration = self.physical_parameters["laser_pulse_duration_fwhm_s"]
        relative_gdd = self.physical_parameters["laser_gdd_relative"]
        relative_tod = self.physical_parameters["laser_tod_relative"]
        for name, value in (
            ("laser_gdd_relative", relative_gdd),
            ("laser_tod_relative", relative_tod),
        ):
            if value is not None and not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the interval [-1, 1].")
        duration_ratio = duration / self._REFERENCE_PULSE_DURATION_FWHM_S
        if relative_gdd is not None:
            self.physical_parameters["laser_gdd_s2"] = float(
                relative_gdd * self._REFERENCE_GDD_S2 * duration_ratio**2
            )
        if relative_tod is not None:
            self.physical_parameters["laser_tod_s3"] = float(
                relative_tod * self._REFERENCE_TOD_S3 * duration_ratio**3
            )

    def _validate_parameters(self) -> None:
        missing_physical = self.PHYSICAL_PARAMETER_KEYS - set(self.physical_parameters)
        accepted_physical = (
            self.PHYSICAL_PARAMETER_KEYS | self.OPTIONAL_PHYSICAL_PARAMETER_KEYS
        )
        unknown_physical = set(self.physical_parameters) - accepted_physical
        unknown_hyperparameters = set(self.hyperparameters) - set(
            self._HYPERPARAMETER_DEFAULTS
        )
        if missing_physical:
            raise ValueError(
                f"Missing physical laser parameters: {sorted(missing_physical)}"
            )
        if unknown_physical:
            raise ValueError(
                f"Unknown physical laser parameters: {sorted(unknown_physical)}"
            )
        if unknown_hyperparameters:
            raise ValueError(
                "Unknown laser hyperparameters: " f"{sorted(unknown_hyperparameters)}"
            )
        minimum_angles = 2 * self.hyperparameters["n_azimuthal_modes"] - 1
        if self.hyperparameters["centering_angles"] < minimum_angles:
            raise ValueError(
                "centering_angles must be at least "
                f"{minimum_angles} for the configured azimuthal modes."
            )
        if self.physical_parameters["laser_spectral_bandwidth_rad_s"] < 0.0:
            raise ValueError("laser_spectral_bandwidth_rad_s must be non-negative.")
        self._validate_spectral_phase_bandwidth()
        if self.hyperparameters["spectral_time_window_factor"] <= 0.0:
            raise ValueError("spectral_time_window_factor must be positive.")
        peak_delay = self.hyperparameters["peak_delay_from_file_start_s"]
        if peak_delay is not None and peak_delay < 0.0:
            raise ValueError("peak_delay_from_file_start_s must be non-negative.")
        maximum_duration = self.hyperparameters["maximum_pulse_duration_fwhm_s"]
        if maximum_duration is not None and maximum_duration <= 0.0:
            raise ValueError(
                "maximum_pulse_duration_fwhm_s must be positive when specified."
            )

    def _validate_spectral_phase_bandwidth(self) -> None:
        """Reject phase terms that the zero-bandwidth profile cannot represent."""
        if self.physical_parameters["laser_spectral_bandwidth_rad_s"] == 0.0 and any(
            self.physical_parameters[name] != 0.0
            for name in ("laser_gdd_s2", "laser_tod_s3", "laser_fod_s4")
        ):
            raise ValueError(
                "Nonzero GDD, TOD, or FOD requires a nonzero laser_spectral_bandwidth_rad_s."
            )

    def _lasy_parameters(self) -> dict[str, Any]:
        """Translate flat public inputs to the parameter names LASY expects."""
        return {
            "wavelength": self.physical_parameters["laser_wavelength_m"],
            "energy": self.physical_parameters["laser_energy_J"],
            "pulse_duration_fwhm": self.physical_parameters[
                "laser_pulse_duration_fwhm_s"
            ],
            "spectral_bandwidth": self.physical_parameters[
                "laser_spectral_bandwidth_rad_s"
            ],
            "cep_phase": self.physical_parameters["laser_cep_phase_rad"],
            "gdd": self.physical_parameters["laser_gdd_s2"],
            "tod": self.physical_parameters["laser_tod_s3"],
            "fod": self.physical_parameters["laser_fod_s4"],
            "spot_size": self.physical_parameters["laser_spot_size_m"],
            "super_gaussian_order": self.physical_parameters[
                "laser_super_gaussian_order"
            ],
            "focal_position": self.physical_parameters["laser_focal_position_m"],
            "zernike_coefficients": {
                name: self.physical_parameters[f"zernike_{name}"]
                for name in ZERNIKE_OSA_INDICES
            },
        }

    def _build_laser_at_focus(self) -> Laser:
        parameters = self._lasy_parameters() | {
            "polarization": self.hyperparameters["polarization"]
        }
        intrinsic_time_half_width = self._time_half_width(parameters)
        time_half_width = self._time_half_width_for_peak_delay(
            intrinsic_time_half_width
        )
        longitudinal_profile = self._build_longitudinal_profile(
            parameters,
            intrinsic_time_half_width,
        )
        pupil_radius = self._reference_focus_pupil_radius(
            parameters,
            intrinsic_time_half_width,
        )
        return Laser(
            dim="rt",
            lo=(0.0, -time_half_width),
            hi=(
                self.hyperparameters["hi_range"] * parameters["spot_size"],
                time_half_width,
            ),
            npoints=self.hyperparameters["num_points"],
            profile=_ZernikeSuperGaussianProfile(
                parameters,
                pupil_radius,
                longitudinal_profile,
            ),
            n_azimuthal_modes=self.hyperparameters["n_azimuthal_modes"],
        )

    def _time_half_width(self, parameters: Mapping[str, Any]) -> float:
        """Return a temporal extent that retains the transform-limited pulse and dispersion delay."""
        bandwidth = parameters["spectral_bandwidth"]
        if bandwidth == 0.0:
            return 3.0 * parameters["pulse_duration_fwhm"]
        spectral_extent = 3.0 * bandwidth
        dispersion_delay = abs(
            parameters["gdd"] * spectral_extent
            + parameters["tod"] * spectral_extent**2 / 2.0
            + parameters["fod"] * spectral_extent**3 / 6.0
        )
        return (
            self.hyperparameters["spectral_time_window_factor"]
            * 2.0
            * np.log(2.0)
            / bandwidth
            + dispersion_delay
        )

    def _time_half_width_for_peak_delay(
        self, intrinsic_time_half_width: float
    ) -> float:
        """Expand the grid to retain the requested post-peak temporal support."""
        peak_delay = self.hyperparameters["peak_delay_from_file_start_s"]
        if peak_delay is None:
            return intrinsic_time_half_width
        return max(
            intrinsic_time_half_width,
            (peak_delay + intrinsic_time_half_width) / 2.0,
        )

    def _build_longitudinal_profile(
        self,
        parameters: Mapping[str, Any],
        time_half_width: float,
    ) -> Any:
        """Build either the legacy transform-limited pulse or analytic spectral pulse."""
        bandwidth = parameters["spectral_bandwidth"]
        if bandwidth == 0.0:
            return GaussianLongitudinalProfile(
                wavelength=parameters["wavelength"],
                tau=parameters["pulse_duration_fwhm"] / np.sqrt(2.0 * np.log(2.0)),
                t_peak=0.0,
                cep_phase=parameters["cep_phase"],
            )
        return AnalyticSpectralLongitudinalProfile(
            wavelength=parameters["wavelength"],
            bandwidth_fwhm=bandwidth,
            time_half_width=time_half_width,
            npoints=self.hyperparameters["num_points"][1],
            cep_phase=parameters["cep_phase"],
            gdd=parameters["gdd"],
            tod=parameters["tod"],
            fod=parameters["fod"],
        )

    def _on_axis_intensity(self) -> tuple[Array, Array]:
        """Return the start-plane on-axis intensity on the LASY time grid."""
        _, time = self.laser.grid.axes
        field = self.laser.grid.get_temporal_field()[0, 0]
        return time, np.abs(field) ** 2

    @staticmethod
    def _fwhm(time: Array, intensity: Array) -> float:
        """Return the FWHM of the peak containing the global intensity maximum."""
        peak_index = int(np.argmax(intensity))
        threshold = intensity[peak_index] / 2.0
        below_threshold = intensity < threshold
        left_indices = np.flatnonzero(below_threshold[: peak_index + 1])
        right_indices = np.flatnonzero(below_threshold[peak_index:])
        if left_indices.size == 0 or right_indices.size == 0:
            raise ValueError("Laser temporal grid does not contain the pulse FWHM.")
        left_upper = int(left_indices[-1])
        right_upper = peak_index + int(right_indices[0])
        left_crossing = np.interp(
            threshold,
            intensity[left_upper : left_upper + 2],
            time[left_upper : left_upper + 2],
        )
        right_crossing = np.interp(
            threshold,
            intensity[right_upper - 1 : right_upper + 1][::-1],
            time[right_upper - 1 : right_upper + 1][::-1],
        )
        return float(right_crossing - left_crossing)

    def _set_peak_delay_from_file_start(self) -> None:
        """Place the on-axis peak at a fixed delay from FBPIC's file-time origin."""
        target_delay = self.hyperparameters["peak_delay_from_file_start_s"]
        if target_delay is None:
            return
        time, intensity = self._on_axis_intensity()
        current_delay = float(time[int(np.argmax(intensity))] - time[0])
        shift = target_delay - current_delay
        if shift < 0.0:
            leading_edge_intensity = float(np.interp(time[0] - shift, time, intensity))
            relative_intensity = leading_edge_intensity / float(np.max(intensity))
            if relative_intensity > self._LEADING_EDGE_CROP_INTENSITY_TOLERANCE:
                warnings.warn(
                    "Laser peak delay crops the leading edge at relative on-axis "
                    f"intensity {relative_intensity:.3e}, exceeding "
                    f"{self._LEADING_EDGE_CROP_INTENSITY_TOLERANCE:.1e}.",
                    UserWarning,
                    stacklevel=2,
                )
        field = self.laser.grid.get_temporal_field()
        shifted_field = np.empty_like(field)
        for mode_index in range(field.shape[0]):
            for radius_index in range(field.shape[1]):
                source = field[mode_index, radius_index]
                shifted_field[mode_index, radius_index] = np.interp(
                    time - shift,
                    time,
                    source.real,
                    left=0.0,
                    right=0.0,
                ) + 1j * np.interp(
                    time - shift,
                    time,
                    source.imag,
                    left=0.0,
                    right=0.0,
                )
        self.laser.grid.set_temporal_field(shifted_field)

    def _validate_pulse_duration(self) -> None:
        """Reject spectral-phase settings that exceed the configured FWHM limit."""
        maximum_duration = self.hyperparameters["maximum_pulse_duration_fwhm_s"]
        if maximum_duration is None:
            return
        time, intensity = self._on_axis_intensity()
        duration = self._fwhm(time, intensity)
        if duration > maximum_duration:
            raise ValueError(
                "Laser pulse duration exceeds maximum_pulse_duration_fwhm_s: "
                f"{duration:.3e} s > {maximum_duration:.3e} s."
            )

    def _reference_focus_pupil_radius(
        self,
        parameters: Mapping[str, Any],
        time_half_width: float,
    ) -> float:
        """Measure the centered mean $1/e^2$ radius at the Zernike plane.

        Zernike phase is defined at focus. The reference pulse has the same
        amplitude and numerical grid as the requested pulse but no Zernike
        phase, so its fluence centroid is at the origin and its horizontal and
        vertical radii are equal. This keeps the physical pupil independent of
        the numerical radial extent.
        """
        reference_parameters = {**parameters, "zernike_coefficients": {}}
        reference_laser = Laser(
            dim="rt",
            lo=(0.0, -time_half_width),
            hi=(
                self.hyperparameters["hi_range"] * parameters["spot_size"],
                time_half_width,
            ),
            npoints=self.hyperparameters["num_points"],
            profile=_ZernikeSuperGaussianProfile(
                reference_parameters,
                1.0,
                self._build_longitudinal_profile(
                    reference_parameters,
                    time_half_width,
                ),
            ),
            n_azimuthal_modes=self.hyperparameters["n_azimuthal_modes"],
        )
        radius, time = reference_laser.grid.axes
        fluence = np.trapezoid(
            0.5
            * epsilon_0
            * c
            * np.abs(reference_laser.grid.get_temporal_field()[0]) ** 2,
            x=time,
            axis=-1,
        )
        threshold = fluence[0] / np.e**2
        crossing_indices = np.flatnonzero(fluence <= threshold)
        if crossing_indices.size == 0:
            raise ValueError(
                "Reference focus grid does not contain the $1/e^2$ beam edge."
            )
        upper_index = int(crossing_indices[0])
        if upper_index == 0:
            raise ValueError("Reference focus beam is unresolved at the radial axis.")
        lower_index = upper_index - 1
        return float(
            radius[lower_index]
            + (threshold - fluence[lower_index])
            * (radius[upper_index] - radius[lower_index])
            / (fluence[upper_index] - fluence[lower_index])
        )

    def _polar_fields(self, angles: Array) -> Array:
        return polar_fields(self.laser, angles)

    def _set_polar_fields(self, field: Array) -> None:
        set_polar_fields(self.laser, field)

    @staticmethod
    def _one_over_e_squared_radius(
        coordinates: Array,
        fluence: Array,
    ) -> float:
        """Return the half-width at $1/e^2$ of a transverse fluence slice."""
        peak_index = int(np.argmax(fluence))
        threshold = fluence[peak_index] / np.e**2
        left_candidates = np.flatnonzero(fluence[: peak_index + 1] <= threshold)
        right_candidates = np.flatnonzero(fluence[peak_index:] <= threshold)
        if left_candidates.size == 0 or right_candidates.size == 0:
            raise ValueError("Laser grid does not contain the $1/e^2$ beam edge.")
        left_index = int(left_candidates[-1])
        right_index = peak_index + int(right_candidates[0])
        left_edge = coordinates[left_index] + (
            (threshold - fluence[left_index])
            * (coordinates[left_index + 1] - coordinates[left_index])
            / (fluence[left_index + 1] - fluence[left_index])
        )
        right_edge = coordinates[right_index - 1] + (
            (threshold - fluence[right_index - 1])
            * (coordinates[right_index] - coordinates[right_index - 1])
            / (fluence[right_index] - fluence[right_index - 1])
        )
        return float((right_edge - left_edge) / 2.0)

    def _validate_start_plane_grid(self) -> None:
        """Warn when the radial grid is under five times the beam radius."""
        radius, time = self.laser.grid.axes
        angles = np.linspace(0.0, 2.0 * np.pi, 361, endpoint=False)
        field = self._polar_fields(angles)
        fluence = np.trapezoid(
            0.5 * epsilon_0 * c * np.abs(field) ** 2,
            x=time,
            axis=-1,
        )
        radius_grid = radius[np.newaxis, :]
        weights = fluence * radius_grid
        centroid_x = float(
            (weights * radius_grid * np.cos(angles[:, np.newaxis])).sum()
            / weights.sum()
        )
        centroid_y = float(
            (weights * radius_grid * np.sin(angles[:, np.newaxis])).sum()
            / weights.sum()
        )
        periodic_fluence = np.concatenate((fluence, fluence[:1]), axis=0)
        interpolator = RegularGridInterpolator(
            (np.append(angles, 2.0 * np.pi), radius),
            periodic_fluence,
            bounds_error=False,
            fill_value=0.0,
        )
        coordinates = np.linspace(-radius[-1], radius[-1], 4 * len(radius) - 3)

        def line_fluence(x_coordinates: Array, y_coordinates: Array) -> Array:
            points = np.column_stack(
                (
                    np.mod(np.arctan2(y_coordinates, x_coordinates), 2.0 * np.pi),
                    np.hypot(x_coordinates, y_coordinates),
                )
            )
            return interpolator(points)

        x_radius = self._one_over_e_squared_radius(
            coordinates,
            line_fluence(
                coordinates + centroid_x, np.full_like(coordinates, centroid_y)
            ),
        )
        y_radius = self._one_over_e_squared_radius(
            coordinates,
            line_fluence(
                np.full_like(coordinates, centroid_x), coordinates + centroid_y
            ),
        )
        mean_radius = (x_radius + y_radius) / 2.0
        if radius[-1] < 5.0 * mean_radius:
            recommended_hi_range = (
                5.0 * mean_radius / self.physical_parameters["laser_spot_size_m"]
            )
            warnings.warn(
                "Laser radial grid may clip the start-plane field: "
                f"r_max={radius[-1]:.3e} m, mean 1/e^2 radius={mean_radius:.3e} m. "
                f"Use hi_range >= {recommended_hi_range:.2f} for a five-radius margin.",
                stacklevel=2,
            )

    def remove_start_plane_offset_and_tilt(self) -> tuple[float, float, float, float]:
        """Center fluence and remove the mean transverse phase gradient.

        Returns:
            Removed ``(centroid_x, centroid_y, k_x, k_y)`` values in meters
            and radians per meter.
        """
        radius, time = self.laser.grid.axes
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.hyperparameters["centering_angles"],
            endpoint=False,
        )
        field = self._polar_fields(angles)
        intensity = 0.5 * epsilon_0 * c * np.abs(field) ** 2
        fluence = np.trapezoid(intensity, x=time, axis=-1)
        radius_grid = radius[np.newaxis, :]
        x_grid = radius_grid * np.cos(angles[:, np.newaxis])
        y_grid = radius_grid * np.sin(angles[:, np.newaxis])
        weights = fluence * radius_grid
        centroid_x = float((weights * x_grid).sum() / weights.sum())
        centroid_y = float((weights * y_grid).sum() / weights.sum())
        peak_field = field[:, :, int(np.argmax(intensity[:, 0, :].mean(axis=0)))]
        field_weight = np.abs(peak_field) ** 2 * radius_grid
        radial_derivative = np.gradient(peak_field, radius, axis=1)
        angular_derivative = (
            np.roll(peak_field, -1, axis=0) - np.roll(peak_field, 1, axis=0)
        ) / (2.0 * (angles[1] - angles[0]))
        angular_over_radius = np.divide(
            angular_derivative,
            radius_grid,
            out=np.zeros_like(angular_derivative),
            where=radius_grid != 0.0,
        )
        field_dx = (
            np.cos(angles[:, np.newaxis]) * radial_derivative
            - np.sin(angles[:, np.newaxis]) * angular_over_radius
        )
        field_dy = (
            np.sin(angles[:, np.newaxis]) * radial_derivative
            + np.cos(angles[:, np.newaxis]) * angular_over_radius
        )
        normalization = float(field_weight.sum())
        k_x = float(
            (radius_grid * np.imag(np.conj(peak_field) * field_dx)).sum()
            / normalization
        )
        k_y = float(
            (radius_grid * np.imag(np.conj(peak_field) * field_dy)).sum()
            / normalization
        )
        source_x = x_grid + centroid_x
        source_y = y_grid + centroid_y
        interpolator = RegularGridInterpolator(
            (np.append(angles, 2.0 * np.pi), radius),
            np.concatenate((field, field[:1]), axis=0),
            bounds_error=False,
            fill_value=0.0,
        )
        translated_field = interpolator(
            np.column_stack(
                (
                    np.mod(np.arctan2(source_y, source_x), 2.0 * np.pi).ravel(),
                    np.hypot(source_x, source_y).ravel(),
                )
            )
        ).reshape(field.shape)
        self._set_polar_fields(
            translated_field
            * np.exp(-1j * (k_x * x_grid + k_y * y_grid))[:, :, np.newaxis]
        )
        return centroid_x, centroid_y, k_x, k_y

    def save(self, filename: Union[str, Path]) -> Path:
        """Write the prepared field to LASY HDF5 and return the written path."""
        requested_path = Path(filename)
        requested_path.parent.mkdir(parents=True, exist_ok=True)
        self.laser.write_to_file(
            file_prefix=requested_path.stem,
            file_format="h5",
            write_dir=str(requested_path.parent),
        )
        return requested_path.parent / f"{requested_path.stem}_00000.h5"

    def focus_laser(self) -> Laser:
        """Return a copy of the prepared laser propagated forward to focus.

        ``self.laser`` sits at the simulation start plane and is left untouched.
        """
        focus = copy.deepcopy(self.laser)
        focus.propagate(distance=self.physical_parameters["laser_focal_position_m"])
        return focus

    def compute_focus_a0(self, n_angles: int = 361) -> float:
        """Return the peak normalized vector potential of the prepared pulse at focus."""
        return peak_a0(self.focus_laser(), n_angles)


class HTULasyLaser:
    """Utility wrapper around LASY laser profiles for HTU data.

    The class reads data exported from a FROG (frequency-resolved optical gating)
    measurement and prepares it for consumption by
    `LongitudinalProfileFromData`. In addition, several helpers are provided to
    visualize and analyze the temporal laser profile.

    Attributes:
        data_file: Absolute path to the FROG export containing the six-column
            data (wavelength [nm], spectral amplitude, spectral phase, time
            [fs], temporal amplitude, temporal phase).
        domain: Representation domain to expose via LASY. Must be either
            `temporal` or `spectral`.
        lo: Minimum time (seconds) covered by the data after conversion.
        hi: Maximum time (seconds) covered by the data after conversion.
        laser_profile: LASY longitudinal profile instance constructed from the
            data file.
    """

    def __init__(self, data_file: Path, domain: str = "temporal") -> None:
        """Initialize the HTU LASY laser profile.

        Args:
            data_file: Path to the tab-separated FROG export file.
            domain: Representation domain used by LASY, either `temporal` or
                `spectral`.  Note: 'spectral' has issues with plotting

        Raises:
            ValueError: If `domain` is not one of the supported options.
        """
        self.data_file: Path = data_file
        self.domain = domain
        self.lo: float = 0.0
        self.hi: float = 0.0

        data = self.load_frog_as_lasy_data()
        self.laser_profile: LongitudinalProfileFromData = LongitudinalProfileFromData(
            data, self.lo, self.hi
        )

    def get_laser_profile(self) -> LongitudinalProfileFromData:
        """Return the LASY longitudinal profile instance."""
        return self.laser_profile

    def load_frog_as_lasy_data(self) -> Mapping[str, Union[np.ndarray, float, bool]]:
        """Load the FROG export and format it for LASY consumption.

        Returns:
            Mapping with the keys expected by `LongitudinalProfileFromData`.

        Raises:
            ValueError: If fewer than two temporal samples are present when
            `domain` is `spectral` or if the domain selection is invalid.
        """
        raw = np.loadtxt(self.data_file, skiprows=1)

        wavelength_nm = raw[:, 0]
        spectral_amp = raw[:, 1]
        spectral_phase = raw[:, 2]
        time_fs = raw[:, 3]
        temporal_amp = raw[:, 4]
        temporal_phase = raw[:, 5]

        if self.domain == "spectral":
            spectral_mask = wavelength_nm != 0
            wavelength_nm = wavelength_nm[spectral_mask]
            spectral_amp = spectral_amp[spectral_mask]
            spectral_phase = spectral_phase[spectral_mask]
            time_fs = time_fs[spectral_mask]

        wavelength_m = wavelength_nm * 1e-9
        time_s = time_fs * 1e-15

        # Use weighted average for central wavelength in temporal dict
        central_wavelength = np.average(
            wavelength_m, weights=np.maximum(spectral_amp, 1e-30)
        )

        if self.domain == "spectral":
            # Use temporal spacing to set requested dt for the FFT
            if len(time_s) <= 1:
                msg = "Need at least two time samples to infer dt for spectral data."
                raise ValueError(msg)

            dt = float(np.abs(np.mean(np.diff(time_s))))

            data: dict[str, Union[np.ndarray, float, bool]] = {
                "datatype": "spectral",
                "axis_is_wavelength": True,
                "axis": wavelength_m,
                "intensity": spectral_amp,
                "phase": spectral_phase,
                "dt": dt,
            }

        elif self.domain == "temporal":
            data = {
                "datatype": "temporal",
                "axis": time_s,
                "intensity": temporal_amp,
                "phase": temporal_phase,
                "wavelength": central_wavelength,
            }
        else:
            msg = "domain must be 'spectral' or 'temporal'"
            raise ValueError(msg)

        self.lo = float(time_s.min())
        self.hi = float(time_s.max())

        return data

    def get_time_axis(self, num: int = 2000) -> np.ndarray:
        """Return a uniformly spaced temporal axis covering the raw data range.

        Args:
            num: Number of points used to represent the time axis.

        Returns:
            Array containing the time steps in seconds.
        """
        return np.linspace(self.lo, self.hi, num)

    def get_envelope(self, num: int = 2000) -> np.ndarray:
        """Evaluate the complex laser envelope on a sampled grid.

        Args:
            num: Number of points used to evaluate the envelope.

        Returns:
            Complex-valued array representing the laser envelope.
        """
        time_axis = self.get_time_axis(num=num)
        return self.laser_profile.evaluate(time_axis)

    def get_intensity_profile(self, num: int = 2000) -> np.ndarray:
        """Compute the intensity profile from the laser envelope.

        Args:
            num: Number of points used to evaluate the profile.

        Returns:
            Real-valued array containing the intensity at each sampled point.
        """
        envelope = self.get_envelope(num=num)
        return np.abs(envelope) ** 2

    def get_phase_profile(self, num: int = 2000) -> np.ndarray:
        """Compute the phase profile from the laser envelope.

        Args:
            num: Number of points used to evaluate the profile.

        Returns:
            Real-valued array containing the unwrapped phase in radians.
        """
        envelope = self.get_envelope(num=num)
        return np.unwrap(np.angle(envelope))

    def calculate_fwhm_intensity(self, num: int = 2000) -> float:
        """Estimate the full width at half maximum (FWHM) of the intensity.

        Args:
            num: Number of points used to evaluate the profile.

        Returns:
            Estimated FWHM value in femtoseconds.
        """
        t_fs = self.get_time_axis(num=num) * 1e15
        intensity = self.get_intensity_profile(num=num)

        hist, bin_edges = np.histogram(t_fs, bins=num, weights=intensity)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_idx = int(np.argmax(hist))
        peak_value = float(hist[peak_idx])
        half_max = peak_value / 2
        left_idx = peak_idx
        right_idx = peak_idx
        while left_idx > 0 and hist[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
            right_idx += 1
        fwhm = bin_centers[right_idx] - bin_centers[left_idx]

        return float(fwhm)

    def plot_temporal_profile(
        self,
        title: Optional[str] = None,
        num: int = 2000,
        show: bool = True,
    ) -> Figure:
        """Plot the temporal intensity and phase profiles.

        Args:
            title: Optional title for the intensity subplot.
            num: Number of points used to evaluate the profiles.
            show: Whether to call `plt.show()` before returning.

        Returns:
            Matplotlib figure instance containing the generated subplots.
        """
        import matplotlib.pyplot as plt

        t_fs = self.get_time_axis(num=num) * 1e15
        intensity = self.get_intensity_profile(num=num)
        phase = self.get_phase_profile(num=num)

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t_fs, intensity, label="Imported via LASY")
        ax[0].set_ylabel("Normalized Intensity")
        if title is not None:
            ax[0].set_title(title)
        ax[0].legend()

        ax[1].plot(t_fs, phase)
        ax[1].set_ylabel("Phase [rad]")
        ax[1].set_xlabel("Time [fs]")

        plt.tight_layout()
        if show:
            plt.show()

        return fig


def format_jones(polarization: tuple[float, float]) -> str:
    """Return a human-readable label for a real Jones vector."""
    return f"Jones ({polarization[0]:g}, {polarization[1]:g})"


def plot_start_plane(
    laser: Laser,
    *,
    z0: float = 0.0,
    polarization: tuple[float, float] = (1.0, 0.0),
    mode: str = "lineout_and_2d",
    ax: Any = None,
    num: int = 600,
    output_path: Union[str, Path, None] = None,
    show: bool = False,
    label: Optional[str] = None,
    title: str = "lasy",
    a0_annotation: Optional[float] = None,
) -> Figure:
    """Plot a LASY pulse's on-axis envelope and, optionally, a face-on |E| map.

    The on-axis envelope (in a0 units) is mapped from the LASY time axis to
    ``z`` about ``z0`` with the peak at ``z0``.

    Args:
        laser: LASY ``Laser`` whose current grid (e.g. the simulation start
            plane) is plotted.
        z0: (float) [m] Position assigned to the on-axis intensity peak when the
            time axis is mapped to ``z``.
        polarization: (tuple[float, float]) Real Jones vector ``(Ex, Ey)``; used
            for the panel labels and the quiver of ``Re(E) * polarization``.
        mode: (str) ``"lineout"`` shows only the longitudinal envelope.
            ``"lineout_and_2d"`` adds a face-on field-amplitude map at the peak
            time with the polarization quiver, zoomed to where the azimuthally
            averaged fluence exceeds 1e-3 of its peak.
        ax: (matplotlib.axes.Axes|None) If provided, the longitudinal envelope
            is also drawn on this external axes with ``z`` in mm (for combined
            overlay figures).
        num: (int) Number of points for the resampled longitudinal lineout.
        output_path: (str|Path|None) If given, the figure is saved here; parent
            directories are created.
        show: (bool) Whether to call ``plt.show()``; otherwise the figure is
            closed before returning.
        label: (str|None) Label for the external *ax* lineout. Defaults to
            ``"<title> (<polarization>)"``.
        title: (str) Name used in the figure title and the default *ax* label.
        a0_annotation: (float|None) If given, written on the face-on panel as
            the pulse's a0 at focus.

    Returns:
        The created matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    _, time = laser.grid.axes
    e_to_a0 = e / (m_e * c * laser.profile.omega0)
    pol_label = format_jones(polarization)

    # On-axis envelope vs. time, mapped to z about the nominal centroid z0.
    on_axis = np.abs(polar_fields(laser, np.array([0.0]))[0, 0, :]) * e_to_a0
    t_peak = time[int(np.argmax(on_axis))]
    z_of_t = z0 - c * (time - t_peak)
    order = np.argsort(z_of_t)
    z_arr = np.linspace(z_of_t.min(), z_of_t.max(), num)
    envelope = np.interp(z_arr, z_of_t[order], on_axis[order])

    if ax is not None:
        default_label = f"{title} ({pol_label})"
        ax.plot(
            z_arr * 1e3,
            envelope,
            lw=1.5,
            label=label if label is not None else default_label,
        )

    if mode == "lineout":
        fig, ax_z = plt.subplots(1, 1, figsize=(8, 4.5))
    else:
        fig, (ax_z, ax_xy) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_z.plot(z_arr * 1e6, envelope, color="C0", lw=1.5)
    ax_z.set_xlabel("z (um)")
    ax_z.set_ylabel("On-axis envelope amplitude (a\u2080)")
    ax_z.set_title(f"Longitudinal envelope at start plane\n{pol_label}")
    ax_z.grid(True, alpha=0.3)

    if mode == "lineout_and_2d":
        radius, angles, fluence, peak_field = transverse_fluence(laser, 361)
        amplitude = np.abs(peak_field)
        # Explicit polar cell edges: the Cartesian mesh is not monotonic, so
        # pcolormesh cannot infer them from cell centres.
        d_theta = angles[1] - angles[0]
        theta_edges = np.append(angles - d_theta / 2.0, angles[-1] + d_theta / 2.0)
        r_edges = np.concatenate(
            ([0.0], 0.5 * (radius[1:] + radius[:-1]), [radius[-1]])
        )
        theta_grid, r_grid = np.meshgrid(theta_edges, r_edges, indexing="ij")
        x_um = r_grid * np.cos(theta_grid) * 1e6
        y_um = r_grid * np.sin(theta_grid) * 1e6
        im = ax_xy.pcolormesh(x_um, y_um, amplitude, cmap="inferno", shading="flat")
        ax_xy.set_aspect("equal")

        # Zoom to where the azimuthally averaged fluence is above 1e-3 of peak.
        radial_fluence = fluence.mean(axis=0)
        above = np.flatnonzero(radial_fluence > 1e-3 * radial_fluence.max())
        r_view = radius[int(above[-1])] * 1e6 if above.size else radius[-1] * 1e6
        ax_xy.set_xlim(-r_view, r_view)
        ax_xy.set_ylim(-r_view, r_view)

        ax_xy.set_xlabel(r"x ($\mu$m)")
        ax_xy.set_ylabel(r"y ($\mu$m)")
        ax_xy.set_title(f"Face-on |E| at start plane\n{pol_label}")
        cbar = fig.colorbar(im, ax=ax_xy, fraction=0.046, pad=0.04)
        cbar.set_label("|E| (V/m)")

        # Quiver the polarization field: LASY stores a scalar envelope and
        # applies the real Jones vector (px, py) as a fixed spatial
        # scaling, so the local field vector is Re(peak_field) * (px, py).
        # Its direction is always +/-(px, py), but the sign flips across
        # the aberrated wavefront's phase structure, which a single static
        # arrow cannot show.
        px, py = polarization
        theta_grid_c, r_grid_c = np.meshgrid(angles, radius, indexing="ij")
        field_real = peak_field.real
        ex_grid = field_real * px
        ey_grid = field_real * py

        stride_theta = max(1, len(angles) // 24)
        stride_r = max(1, len(radius) // 10)
        in_view = r_grid_c <= (r_view * 1e-6)
        sl = (slice(None, None, stride_theta), slice(None, None, stride_r))
        xs = (r_grid_c * np.cos(theta_grid_c))[sl] * 1e6
        ys = (r_grid_c * np.sin(theta_grid_c))[sl] * 1e6
        us = ex_grid[sl]
        vs = ey_grid[sl]
        view_mask = in_view[sl]
        mag = np.hypot(us, vs)
        mask = view_mask & (mag > 0.05 * np.max(mag))
        if mask.any():
            ax_xy.quiver(
                xs[mask],
                ys[mask],
                us[mask] / mag[mask],
                vs[mask] / mag[mask],
                color="white",
                alpha=0.6,
                scale=25,
                width=0.004,
                headwidth=3,
            )
        if a0_annotation is not None:
            ax_xy.text(
                0.02,
                0.98,
                f"a\u2080 at focus = {a0_annotation:.3g}",
                transform=ax_xy.transAxes,
                color="white",
                va="top",
                fontsize=9,
            )

    fig.suptitle(f"{title}  \u2014  {pol_label}", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
