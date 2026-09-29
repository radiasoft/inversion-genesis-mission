"""
laser.py

Module containing useful utilities for modeling laser properties.  Contains the following
- Class for loading HTU longitudinal laser profile from FROG data through LASY.
"""

from __future__ import annotations

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
from scipy.constants import c, epsilon_0
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


class _ZernikeSuperGaussianProfile(Profile):
    """LASY profile with Zernike phase clamped outside its physical pupil.

    LASY evaluates Zernike polynomials beyond the pupil radius. This profile
    instead evaluates the phase at the nearest pupil boundary point for
    ``rho > 1``, preventing high-order terms from growing through low-intensity
    beam wings.
    """

    def __init__(self, parameters: Mapping[str, Any], pupil_radius: float) -> None:
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
        self.longitudinal_profile = GaussianLongitudinalProfile(
            wavelength=parameters["wavelength"],
            tau=parameters["pulse_duration_fwhm"] / np.sqrt(2.0 * np.log(2.0)),
            t_peak=0.0,
        )
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
    _HYPERPARAMETER_DEFAULTS: dict[str, Any] = {
        "polarization": (1, 0),
        "n_azimuthal_modes": 5,
        "num_points": (600, 900),
        "hi_range": 8.0,
        "center_and_remove_tilt": True,
        "centering_angles": 72,
    }

    def __init__(
        self,
        physical_parameters: Mapping[str, Any],
        hyperparameters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.physical_parameters = dict(physical_parameters)
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
        self._validate_start_plane_grid()

    def _validate_parameters(self) -> None:
        missing_physical = self.PHYSICAL_PARAMETER_KEYS - set(self.physical_parameters)
        unknown_physical = set(self.physical_parameters) - self.PHYSICAL_PARAMETER_KEYS
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

    def _lasy_parameters(self) -> dict[str, Any]:
        """Translate flat public inputs to the parameter names LASY expects."""
        return {
            "wavelength": self.physical_parameters["laser_wavelength_m"],
            "energy": self.physical_parameters["laser_energy_J"],
            "pulse_duration_fwhm": self.physical_parameters[
                "laser_pulse_duration_fwhm_s"
            ],
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
        time_half_width = 3.0 * parameters["pulse_duration_fwhm"]
        pupil_radius = self._reference_focus_pupil_radius(
            parameters,
            time_half_width,
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
            ),
            n_azimuthal_modes=self.hyperparameters["n_azimuthal_modes"],
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
