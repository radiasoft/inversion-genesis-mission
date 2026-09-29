"""Load, propagate, and diagnose an exported LASY laser field.

Run this module with an LASY ``.h5`` field filename to visualize the field at
the simulation entrance and at its nominal focus.
"""

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from lasy.laser import Laser
from lasy.profiles import FromOpenPMDProfile
from lasy.utils.laser_utils import get_full_field
from scipy.interpolate import RegularGridInterpolator

from inversion_fbpic.utils.laser import transverse_fluence


Array: TypeAlias = np.ndarray

N_AZIMUTHAL_MODES = 5
N_PROPAGATION_SAMPLES = 5
N_TRANSVERSE_DIAGNOSTIC_ANGLES = 361
N_EVOLUTION_METRIC_ANGLES = 73
ZERNIKE_MAX_RADIAL_ORDER = 5
DEFAULT_FOCAL_POSITION_M = 3.5e-3


def zernike_radial_polynomial(
    radial_order: int,
    azimuthal_order: int,
    normalized_radius: Array,
) -> Array:
    """Evaluate the radial component of an OSA/ANSI Zernike polynomial."""
    absolute_azimuthal_order = abs(azimuthal_order)
    radial_polynomial = np.zeros_like(normalized_radius, dtype=float)
    for summation_index in range((radial_order - absolute_azimuthal_order) // 2 + 1):
        coefficient = (-1) ** summation_index * math.factorial(
            radial_order - summation_index
        )
        coefficient /= math.factorial(summation_index)
        coefficient /= math.factorial(
            (radial_order + absolute_azimuthal_order) // 2 - summation_index
        )
        coefficient /= math.factorial(
            (radial_order - absolute_azimuthal_order) // 2 - summation_index
        )
        radial_polynomial += coefficient * normalized_radius ** (
            radial_order - 2 * summation_index
        )
    return radial_polynomial


def osa_zernike_terms(max_radial_order: int) -> list[tuple[int, int, int, str]]:
    """Return OSA/ANSI indexed Zernike terms through ``max_radial_order``."""
    terms: list[tuple[int, int, int, str]] = []
    for radial_order in range(1, max_radial_order + 1):
        for azimuthal_order in range(-radial_order, radial_order + 1, 2):
            osa_index = (radial_order * (radial_order + 2) + azimuthal_order) // 2
            label = f"Z{osa_index}\n({radial_order}, {azimuthal_order})"
            terms.append((osa_index, radial_order, azimuthal_order, label))
    return terms


def mean_hwhm_radius(radius: Array, polar_intensity: Array) -> float:
    """Calculate the mean angular half-width at half maximum in meters."""
    half_widths: list[float] = []
    for intensity_profile in polar_intensity:
        peak_index = int(np.argmax(intensity_profile))
        half_maximum = intensity_profile[peak_index] / 2.0
        crossing_indices = np.flatnonzero(
            intensity_profile[peak_index:] <= half_maximum
        )
        if crossing_indices.size == 0:
            continue
        upper_index = peak_index + int(crossing_indices[0])
        if upper_index == peak_index:
            half_widths.append(float(radius[upper_index]))
            continue
        lower_index = upper_index - 1
        lower_intensity = intensity_profile[lower_index]
        upper_intensity = intensity_profile[upper_index]
        crossing_radius = radius[lower_index] + (
            (half_maximum - lower_intensity)
            * (radius[upper_index] - radius[lower_index])
            / (upper_intensity - lower_intensity)
        )
        half_widths.append(float(crossing_radius))
    if not half_widths:
        raise ValueError("Could not find a half-maximum radius for the laser field.")
    return float(np.mean(half_widths))


def zernike_phase_coefficients(
    laser: Laser,
    max_radial_order: int = ZERNIKE_MAX_RADIAL_ORDER,
) -> tuple[Array, list[str]]:
    """Fit peak-time phase to OSA/ANSI Zernike terms in radians.

    The phase fit is performed on the circular pupil whose radius is the mean
    HWHM over the angular peak-intensity profiles, and is weighted by
    peak-time intensity. Piston is omitted because it does not affect laser
    propagation. These diagnostic coefficients use the local beam profile and
    are not directly comparable to the injected ``zernike_*`` amplitudes when
    their physical pupil definitions differ.
    """
    radius, angles, _, polar_peak_field = transverse_fluence(laser)
    intensity = np.abs(polar_peak_field) ** 2
    pupil_radius = mean_hwhm_radius(radius, intensity)
    normalized_radius = radius[np.newaxis, :] / pupil_radius
    polar_angle = angles[:, np.newaxis]
    core_pupil = normalized_radius <= 1.0
    phase = np.unwrap(np.angle(polar_peak_field), axis=1)

    basis_columns: list[Array] = []
    labels: list[str] = []
    for _, radial_order, azimuthal_order, label in osa_zernike_terms(max_radial_order):
        radial_component = zernike_radial_polynomial(
            radial_order,
            azimuthal_order,
            np.broadcast_to(normalized_radius, polar_peak_field.shape),
        )
        normalization = np.sqrt(radial_order + 1)
        if azimuthal_order != 0:
            normalization *= np.sqrt(2.0)
        angular_component = (
            np.cos(azimuthal_order * polar_angle)
            if azimuthal_order >= 0
            else np.sin(abs(azimuthal_order) * polar_angle)
        )
        basis_columns.append(normalization * radial_component * angular_component)
        labels.append(label)

    basis = np.column_stack(
        [column[:, core_pupil[0]].ravel() for column in basis_columns]
    )
    weights = np.sqrt(intensity[:, core_pupil[0]] * radius[core_pupil[0]]).ravel()
    coefficients, _, _, _ = np.linalg.lstsq(
        basis * weights[:, np.newaxis],
        phase[:, core_pupil[0]].ravel() * weights,
        rcond=None,
    )
    return coefficients, labels


def plot_zernike_decomposition(laser: Laser, title: str) -> None:
    """Plot OSA/ANSI Zernike coefficients fitted to the peak-time phase."""
    coefficients, labels = zernike_phase_coefficients(laser)
    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    axis.bar(np.arange(len(coefficients)), coefficients, color="tab:blue")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(coefficients)), labels, fontsize=8)
    axis.set_ylabel("Phase coefficient (rad RMS)")
    axis.set_title(f"{title}: OSA/ANSI Zernike phase decomposition")
    axis.grid(axis="y")


def load_saved_laser(file_path: Path) -> Laser:
    """Load an exported LASY field into a laser with the original ``rt`` grid."""
    profile = FromOpenPMDProfile(str(file_path), envelope_name="laserEnvelope")
    radius, time = profile.axes["r"], profile.axes["t"]
    return Laser(
        dim="rt",
        lo=(float(radius[0]), float(time[0])),
        hi=(float(radius[-1]), float(time[-1])),
        npoints=(len(radius), len(time)),
        profile=profile,
        n_azimuthal_modes=N_AZIMUTHAL_MODES,
    )


def plot_field(laser: Laser, title: str) -> None:
    """Plot the real electric field in the radial-time plane."""
    field_rt, extent = get_full_field(laser)
    time_min, time_max, radius_min, radius_max = extent
    field_limit = np.abs(field_rt).max()
    plt.figure()
    plt.imshow(
        field_rt,
        origin="lower",
        aspect="auto",
        cmap="bwr",
        vmin=-field_limit,
        vmax=field_limit,
        extent=(
            time_min * 1e15,
            time_max * 1e15,
            radius_min * 1e6,
            radius_max * 1e6,
        ),
    )
    plt.colorbar(label="E_x (V/m)")
    plt.xlabel("Time (fs)")
    plt.ylabel("Radius (um)")
    plt.title(title)
    plt.tight_layout()


def polar_plane(
    radius: Array, angles: Array, polar_values: Array
) -> tuple[Array, Array]:
    """Interpolate a polar quantity onto a Cartesian display plane."""
    x_axis = np.concatenate((-radius[:0:-1], radius))
    x_coordinates, y_coordinates = np.meshgrid(x_axis, x_axis)
    polar_angle = np.mod(np.arctan2(y_coordinates, x_coordinates), 2.0 * np.pi)
    interpolator = RegularGridInterpolator(
        (np.append(angles, 2.0 * np.pi), radius),
        np.concatenate((polar_values, polar_values[:1]), axis=0),
        bounds_error=False,
        fill_value=0.0,
    )
    plane = interpolator(
        np.column_stack(
            (polar_angle.ravel(), np.hypot(x_coordinates, y_coordinates).ravel())
        )
    ).reshape(x_coordinates.shape)
    return x_axis, plane


def plot_transverse_diagnostics(laser: Laser, title: str) -> None:
    """Plot transverse fluence, projections, and peak-time phase."""
    radius, angles, polar_fluence, polar_peak_field = transverse_fluence(laser)
    x_axis, fluence_2d = polar_plane(radius, angles, polar_fluence)
    _, peak_field_2d = polar_plane(radius, angles, polar_peak_field)
    phase_2d = np.angle(peak_field_2d)
    horizontal_projection = fluence_2d.sum(axis=0)
    vertical_projection = fluence_2d.sum(axis=1)

    figure = plt.figure(figsize=(11, 4.5), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.2, 4.5, 0.25, 4.5),
        height_ratios=(4.5, 1.2),
    )
    axis_vertical = figure.add_subplot(grid[0, 0])
    axis_intensity = figure.add_subplot(grid[0, 1])
    axis_colorbar = figure.add_subplot(grid[0, 2])
    axis_phase = figure.add_subplot(grid[0, 3])
    axis_horizontal = figure.add_subplot(grid[1, 1], sharex=axis_intensity)

    x_um = x_axis * 1e6
    image = axis_intensity.imshow(
        fluence_2d,
        origin="lower",
        extent=(x_um.min(), x_um.max(), x_um.min(), x_um.max()),
        cmap="inferno",
    )
    zernike_pupil = plt.Circle(
        (0.0, 0.0),
        mean_hwhm_radius(radius, np.abs(polar_peak_field) ** 2) * 1e6,
        fill=False,
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
    )
    axis_intensity.add_patch(zernike_pupil)
    figure.colorbar(image, cax=axis_colorbar, label="Fluence (J/m$^2$)")
    axis_intensity.set_title(f"{title}: time-integrated field intensity")
    axis_intensity.set_ylabel("Vertical position (um)")
    axis_intensity.tick_params(labelbottom=False)

    axis_vertical.plot(vertical_projection, x_um)
    axis_vertical.set_xlabel("Integrated fluence (J/m)")
    axis_vertical.set_ylabel("Vertical position (um)")
    axis_vertical.set_ylim(x_um.min(), x_um.max())

    axis_horizontal.plot(x_um, horizontal_projection)
    axis_horizontal.set_xlabel("Horizontal position (um)")
    axis_horizontal.set_ylabel("Integrated fluence (J/m)")
    axis_horizontal.set_xlim(x_um.min(), x_um.max())

    phase_image = axis_phase.imshow(
        phase_2d,
        origin="lower",
        extent=(x_um.min(), x_um.max(), x_um.min(), x_um.max()),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    figure.colorbar(phase_image, ax=axis_phase, label="Phase (rad)")
    axis_phase.set_title(f"{title}: phase")
    axis_phase.set_xlabel("Horizontal position (um)")
    axis_phase.set_ylabel("Vertical position (um)")


def one_over_e_squared_radius(coordinates: Array, projection: Array) -> float:
    """Calculate a $1/e^2$ half-width from a full signed projection."""
    peak_index = int(np.argmax(projection))
    threshold = projection[peak_index] / np.e**2

    left_indices = np.flatnonzero(projection[: peak_index + 1] <= threshold)
    if left_indices.size == 0:
        left_crossing = float(coordinates[0])
    else:
        left_upper = int(left_indices[-1])
        left_lower = min(left_upper + 1, peak_index)
        left_crossing = float(
            np.interp(
                threshold,
                projection[left_upper : left_lower + 1],
                coordinates[left_upper : left_lower + 1],
            )
        )

    right_indices = np.flatnonzero(projection[peak_index:] <= threshold)
    if right_indices.size == 0:
        right_crossing = float(coordinates[-1])
    else:
        right_lower = peak_index + int(right_indices[0])
        right_upper = max(right_lower - 1, peak_index)
        right_crossing = float(
            np.interp(
                threshold,
                projection[right_upper : right_lower + 1][::-1],
                coordinates[right_upper : right_lower + 1][::-1],
            )
        )

    return 0.5 * (right_crossing - left_crossing)


def laser_metrics(laser: Laser) -> tuple[float, float, float]:
    """Calculate transverse $1/e^2$ radii and peak fluence."""
    radius, angles, polar_fluence, _ = transverse_fluence(
        laser, n_angles=N_EVOLUTION_METRIC_ANGLES
    )
    coordinates, fluence_2d = polar_plane(radius, angles, polar_fluence)
    x_projection = fluence_2d.sum(axis=0)
    y_projection = fluence_2d.sum(axis=1)
    x_spot_size = one_over_e_squared_radius(coordinates, x_projection)
    y_spot_size = one_over_e_squared_radius(coordinates, y_projection)
    return x_spot_size, y_spot_size, float(fluence_2d.max())


def plot_vacuum_evolution(laser_at_start: Laser, focal_position: float) -> None:
    """Plot x/y spot sizes and peak fluence around the nominal focus."""
    z_relative_to_focus = np.linspace(
        -focal_position,
        focal_position,
        N_PROPAGATION_SAMPLES,
    )
    x_spot_sizes: list[float] = []
    y_spot_sizes: list[float] = []
    peak_fluences: list[float] = []
    for z_position in z_relative_to_focus:
        laser = deepcopy(laser_at_start)
        laser.propagate(
            distance=float(z_position + focal_position), show_progress=False
        )
        x_spot_size, y_spot_size, peak_fluence = laser_metrics(laser)
        x_spot_sizes.append(x_spot_size)
        y_spot_sizes.append(y_spot_size)
        peak_fluences.append(peak_fluence)

    figure, (axis_spot, axis_fluence) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6), constrained_layout=True
    )
    axis_spot.plot(
        z_relative_to_focus * 1e3,
        np.asarray(x_spot_sizes) * 1e6,
        marker="o",
        label="x projection",
    )
    axis_spot.plot(
        z_relative_to_focus * 1e3,
        np.asarray(y_spot_sizes) * 1e6,
        marker="s",
        label="y projection",
    )
    axis_spot.set_ylabel("1/e^2 radius (um)")
    axis_spot.set_title("Vacuum laser evolution")
    axis_spot.legend()
    axis_spot.grid()
    axis_fluence.plot(
        z_relative_to_focus * 1e3,
        peak_fluences,
        marker="o",
        color="tab:red",
    )
    axis_fluence.set_xlabel("Position relative to focus (mm)")
    axis_fluence.set_ylabel("Peak fluence (J/m$^2$)")
    axis_fluence.grid()


def run_propagation_diagnostics(
    file_path: Path,
    focal_position: float = DEFAULT_FOCAL_POSITION_M,
) -> None:
    """Load a saved laser, plot its evolution, and show its two end planes."""
    saved_laser = load_saved_laser(file_path)
    plot_vacuum_evolution(saved_laser, focal_position)
    plot_field(saved_laser, "Laser at simulation entrance")
    plot_transverse_diagnostics(saved_laser, "Laser at simulation entrance")
    plot_zernike_decomposition(saved_laser, "Laser at simulation entrance")
    saved_laser.propagate(distance=focal_position, show_progress=False)
    plot_field(saved_laser, "Laser at nominal focus")
    plot_transverse_diagnostics(saved_laser, "Laser at nominal focus")
    plot_zernike_decomposition(saved_laser, "Laser at nominal focus")
    plt.show()


def main() -> None:
    """Run diagnostics for a LASY field file specified on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--file-path",
        "--file_path",
        dest="file_path",
        type=Path,
        required=True,
        help="Path to the LASY .h5 field file.",
    )
    parser.add_argument(
        "--focal-position",
        type=float,
        default=DEFAULT_FOCAL_POSITION_M,
        help=(
            "Nominal focus position in meters "
            f"(default: {DEFAULT_FOCAL_POSITION_M:.1e})."
        ),
    )
    arguments = parser.parse_args()
    run_propagation_diagnostics(arguments.file_path, arguments.focal_position)


if __name__ == "__main__":
    main()
