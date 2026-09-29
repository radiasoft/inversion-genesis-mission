"""Particle-distribution loading, selection, and descriptor calculations."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

ParticleArray: TypeAlias = npt.NDArray[np.float64]
WeightArray: TypeAlias = npt.NDArray[np.float64]

COORD_NAMES = ("x", "ux", "y", "uy", "z", "uz")
ELEMENTARY_CHARGE_C = 1.602176634e-19
OFF = 0
MOMENTS = 1
SPLINE = 2
LONGITUDINAL_PROFILE_BINS = 4


def load_openpmd_particles(
    h5_path: str | Path, species: str, iteration: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load a particle species from an openPMD HDF5 diagnostic.

    Args:
        h5_path: Path to an ``data########.h5`` diagnostic file.
        species: openPMD particle-species name.
        iteration: Iteration to load. When omitted, infer it from the filename.

    Returns:
        A ``(N, 6)`` array ordered as ``x, ux, y, uy, z, uz`` and its macro-weights.
    """
    from openpmd_viewer import OpenPMDTimeSeries

    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise ValueError(f"openPMD HDF5 file does not exist: {h5_path}")
    if iteration is None:
        match = re.fullmatch(r"data(\d+)\.h5", h5_path.name)
        if match is None:
            raise ValueError(
                "Cannot infer openPMD iteration from filename; use data########.h5"
            )
        iteration = int(match.group(1))

    time_series = OpenPMDTimeSeries(str(h5_path.parent))
    x, ux, y, uy, z, uz, weights = time_series.get_particle(
        var_list=["x", "ux", "y", "uy", "z", "uz", "w"],
        iteration=iteration,
        species=species,
    )
    return np.column_stack([x, ux, y, uy, z, uz]).astype(np.float64), np.asarray(
        weights, dtype=np.float64
    )


def select_by_uz(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    uz_min: float | None = None,
) -> tuple[ParticleArray, WeightArray | None]:
    """Select particles with normalized longitudinal momentum at least ``uz_min``."""
    particles = np.asarray(particles, dtype=np.float64)
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    if uz_min is None:
        return particles, weights
    selected = particles[:, 5] >= uz_min
    particles = particles[selected]
    weights = None if weights is None else np.asarray(weights)[selected]
    if len(particles) < 2:
        raise ValueError(f"fewer than two particles remain after uz >= {uz_min:g}")
    return particles, weights


def crop_central_particles(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    central_fraction: float = 0.95,
) -> tuple[ParticleArray, WeightArray]:
    """Retain the central fraction about the weighted mean on all six axes."""
    if not 0 < central_fraction <= 1:
        raise ValueError("central_fraction must be in (0, 1]")
    particles, weights = _validate_particles(particles, weights)
    mean = np.average(particles, axis=0, weights=weights)
    distances = np.abs(particles - mean)
    limits = np.quantile(distances, central_fraction, axis=0)
    retained = np.all(distances <= limits, axis=1)
    if retained.sum() < 2:
        raise ValueError("fewer than two particles remain after central crop")
    return particles[retained], weights[retained]


def _validate_particles(
    particles: npt.ArrayLike, weights: npt.ArrayLike | None
) -> tuple[ParticleArray, WeightArray]:
    """Validate a 6D particle distribution and return finite positive weights."""
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] != 6:
        raise ValueError(f"particles must have shape (N, 6), got {particles.shape}")
    finite = np.all(np.isfinite(particles), axis=1)
    if weights is None:
        weights = np.ones(len(particles), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(particles):
            raise ValueError("weights and particles have different lengths")
        finite &= np.isfinite(weights)
    particles, weights = particles[finite], np.abs(weights[finite])
    if len(particles) < 2 or not np.any(weights > 0):
        raise ValueError("fewer than two usable particles remain")
    return particles, weights


def weighted_mean_cov(
    particles: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> tuple[ParticleArray, ParticleArray]:
    """Calculate the weighted population mean and symmetric 6D covariance."""
    x, w = _validate_particles(particles, weights)
    mean = np.average(x, axis=0, weights=w)
    delta = x - mean
    covariance = (delta * w[:, None]).T @ delta / w.sum()
    return mean, 0.5 * (covariance + covariance.T)


def weighted_skew_kurtosis(
    particles: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> tuple[ParticleArray, ParticleArray]:
    """Calculate coordinate-wise weighted skewness and excess kurtosis."""
    x, w = _validate_particles(particles, weights)
    mean = np.average(x, axis=0, weights=w)
    delta = x - mean
    sigma = np.sqrt(np.maximum(np.average(delta**2, axis=0, weights=w), 1e-300))
    skew = np.average((delta / sigma) ** 3, axis=0, weights=w)
    kurtosis = np.average((delta / sigma) ** 4, axis=0, weights=w) - 3.0
    return skew, kurtosis


def longitudinal_shape_features(
    particles: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> tuple[ParticleArray, ParticleArray, ParticleArray, float]:
    """Calculate mixed longitudinal moments and a weighted quadratic ``uz(z)`` fit.

    Returns the two mixed central moments and quadratic curvature, the physical
    fit coefficients ``[quadratic, linear, constant]``, ``[mean_z, mean_uz]``,
    and the weighted RMS residual about the fit.
    """
    x, w = _validate_particles(particles, weights)
    z, uz = x[:, 4], x[:, 5]
    z_mean, uz_mean = np.average(z, weights=w), np.average(uz, weights=w)
    z_delta, uz_delta = z - z_mean, uz - uz_mean
    mixed_z2_uz = np.average(z_delta**2 * uz_delta, weights=w)
    mixed_z3_uz = np.average(z_delta**3 * uz_delta, weights=w)
    z_rms = np.sqrt(np.average(z_delta**2, weights=w))
    if z_rms == 0:
        raise ValueError("longitudinal coordinate has zero weighted RMS size")
    z_normalized = z_delta / z_rms
    design = np.column_stack([z_normalized**2, z_normalized, np.ones(len(z))])
    normalized, *_ = np.linalg.lstsq(design * np.sqrt(w)[:, None], uz * np.sqrt(w), rcond=None)
    coefficients = normalized.copy()
    coefficients[0] /= z_rms**2
    coefficients[1] /= z_rms
    fitted_uz = np.polynomial.polynomial.polyval(z_delta, coefficients[::-1])
    residual_rms = np.sqrt(np.average((uz - fitted_uz) ** 2, weights=w))
    return np.asarray([mixed_z2_uz, mixed_z3_uz, coefficients[0]]), coefficients, np.asarray([z_mean, uz_mean]), residual_rms


def longitudinal_slice_profile(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    bins: int = LONGITUDINAL_PROFILE_BINS,
) -> dict[str, ParticleArray]:
    """Calculate a fixed-bin, charge-weighted longitudinal beam profile.

    The profile is expressed in centered ``z - mean(z)`` coordinates. Each
    retained bin stores its mean ``uz`` and RMS ``uz`` spread. It describes the
    longitudinal distribution statistically without imposing a global
    polynomial trajectory.
    """
    if bins < 4:
        raise ValueError("bins must be at least 4")
    x, w = _validate_particles(particles, weights)
    z_mean = np.average(x[:, 4], weights=w)
    z_centered = x[:, 4] - z_mean
    edges = np.quantile(z_centered, np.linspace(0, 1, bins + 1))
    edges[0] -= np.finfo(np.float64).eps
    edges[-1] += np.finfo(np.float64).eps
    bin_index = np.clip(np.digitize(z_centered, edges) - 1, 0, bins - 1)
    centers, mean_uz, rms_uz = [], [], []
    for index in range(bins):
        selected = bin_index == index
        if not np.any(selected):
            raise ValueError(
                f"longitudinal profile bin {index} is empty; cannot create a fixed schema"
            )
        bin_weights = w[selected]
        bin_uz = x[selected, 5]
        bin_center = np.average(z_centered[selected], weights=bin_weights)
        bin_mean_uz = np.average(bin_uz, weights=bin_weights)
        centers.append(bin_center)
        mean_uz.append(bin_mean_uz)
        rms_uz.append(np.sqrt(np.average((bin_uz - bin_mean_uz) ** 2, weights=bin_weights)))
    return {
        "z_centered": np.asarray(centers, dtype=np.float64),
        "mean_uz": np.asarray(mean_uz, dtype=np.float64),
        "rms_uz": np.maximum(np.asarray(rms_uz, dtype=np.float64), 1e-12),
    }


def longitudinal_profile_density(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None,
    z_grid: ParticleArray,
    uz_grid: ParticleArray,
    *,
    bins: int = LONGITUDINAL_PROFILE_BINS,
) -> ParticleArray:
    """Evaluate a spline-interpolated conditional ``uz`` density from slices."""
    x, w = _validate_particles(particles, weights)
    profile = longitudinal_slice_profile(x, w, bins=bins)
    z_mean = np.average(x[:, 4], weights=w)
    z_values = z_grid - z_mean
    knots = profile["z_centered"]
    mean_spline = CubicSpline(knots, profile["mean_uz"], bc_type="natural", extrapolate=False)
    log_spread_spline = CubicSpline(
        knots, np.log(profile["rms_uz"]), bc_type="natural", extrapolate=False
    )
    mean_uz = mean_spline(z_values)
    rms_uz = np.exp(log_spread_spline(z_values))
    conditional_density = np.exp(-0.5 * ((uz_grid - mean_uz) / rms_uz) ** 2) / rms_uz
    return np.nan_to_num(conditional_density)


def gram_charlier_projection_density(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None,
    first_index: int,
    second_index: int,
    first_grid: ParticleArray,
    second_grid: ParticleArray,
    order: int = 4,
) -> ParticleArray:
    """Evaluate a bivariate Gram-Charlier A density through an order.

    The projection is whitened before evaluating the expansion. Its correction
    ``order=2`` gives the Gaussian baseline. Higher values add every mixed
    cumulant through that total order, expressed against probabilists' Hermite
    polynomials after whitening the pair. Negative values are clipped because
    a truncated expansion is not guaranteed positive.
    """
    if order not in {2, 3, 4, 5}:
        raise ValueError("Gram-Charlier order must be 2, 3, 4, or 5")
    x, w = _validate_particles(particles, weights)
    projection = x[:, [first_index, second_index]]
    mean = np.average(projection, axis=0, weights=w)
    delta = projection - mean
    covariance = (delta * w[:, None]).T @ delta / w.sum()
    transform = np.linalg.inv(np.linalg.cholesky(covariance))
    whitened = delta @ transform.T
    grid = np.stack([first_grid - mean[0], second_grid - mean[1]], axis=-1)
    standardized_grid = grid @ transform.T
    a, b = standardized_grid[..., 0], standardized_grid[..., 1]

    def central_moment(power_a: int, power_b: int) -> float:
        return float(np.average(whitened[:, 0] ** power_a * whitened[:, 1] ** power_b, weights=w))

    third = {(3, 0): central_moment(3, 0), (2, 1): central_moment(2, 1), (1, 2): central_moment(1, 2), (0, 3): central_moment(0, 3)}
    fourth = {(4, 0): central_moment(4, 0) - 3, (3, 1): central_moment(3, 1), (2, 2): central_moment(2, 2) - 1, (1, 3): central_moment(1, 3), (0, 4): central_moment(0, 4) - 3}
    fifth = {(5, 0): central_moment(5, 0) - 10 * central_moment(3, 0), (4, 1): central_moment(4, 1) - 6 * central_moment(2, 1), (3, 2): central_moment(3, 2) - 3 * central_moment(1, 2) - central_moment(3, 0), (2, 3): central_moment(2, 3) - 3 * central_moment(2, 1) - central_moment(0, 3), (1, 4): central_moment(1, 4) - 6 * central_moment(1, 2), (0, 5): central_moment(0, 5) - 10 * central_moment(0, 3)}
    hermite3 = {(3, 0): a**3 - 3 * a, (2, 1): (a**2 - 1) * b, (1, 2): a * (b**2 - 1), (0, 3): b**3 - 3 * b}
    hermite4 = {(4, 0): a**4 - 6 * a**2 + 3, (3, 1): (a**3 - 3 * a) * b, (2, 2): (a**2 - 1) * (b**2 - 1), (1, 3): a * (b**3 - 3 * b), (0, 4): b**4 - 6 * b**2 + 3}
    hermite5 = {(5, 0): a**5 - 10 * a**3 + 15 * a, (4, 1): (a**4 - 6 * a**2 + 3) * b, (3, 2): (a**3 - 3 * a) * (b**2 - 1), (2, 3): (a**2 - 1) * (b**3 - 3 * b), (1, 4): a * (b**4 - 6 * b**2 + 3), (0, 5): b**5 - 10 * b**3 + 15 * b}
    correction = 1.0
    if order >= 3:
        correction += third[(3, 0)] * hermite3[(3, 0)] / 6 + third[(2, 1)] * hermite3[(2, 1)] / 2 + third[(1, 2)] * hermite3[(1, 2)] / 2 + third[(0, 3)] * hermite3[(0, 3)] / 6
    if order >= 4:
        correction += fourth[(4, 0)] * hermite4[(4, 0)] / 24 + fourth[(3, 1)] * hermite4[(3, 1)] / 6 + fourth[(2, 2)] * hermite4[(2, 2)] / 4 + fourth[(1, 3)] * hermite4[(1, 3)] / 6 + fourth[(0, 4)] * hermite4[(0, 4)] / 24
    if order >= 5:
        correction += fifth[(5, 0)] * hermite5[(5, 0)] / 120 + fifth[(4, 1)] * hermite5[(4, 1)] / 24 + fifth[(3, 2)] * hermite5[(3, 2)] / 12 + fifth[(2, 3)] * hermite5[(2, 3)] / 12 + fifth[(1, 4)] * hermite5[(1, 4)] / 24 + fifth[(0, 5)] * hermite5[(0, 5)] / 120
    gaussian = np.exp(-0.5 * (a**2 + b**2)) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))
    return np.maximum(gaussian * correction, 0.0)


def gram_charlier_marginal_density(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None,
    coordinate_index: int,
    grid: ParticleArray,
    order: int = 4,
) -> ParticleArray:
    """Evaluate a univariate Gram-Charlier A density through the requested order.

    The correction uses the standardized third central moment and excess
    kurtosis. ``order=2`` is Gaussian, ``order=3`` adds skewness, and orders
    4 and 5 add excess kurtosis and the fifth cumulant. Negative values are
    clipped because a truncated Gram-Charlier series is not guaranteed nonnegative.
    """
    if order not in {2, 3, 4, 5}:
        raise ValueError("Gram-Charlier order must be 2, 3, 4, or 5")
    x, w = _validate_particles(particles, weights)
    values = x[:, coordinate_index]
    mean = np.average(values, weights=w)
    sigma = np.sqrt(np.average((values - mean) ** 2, weights=w))
    if sigma == 0:
        raise ValueError("coordinate has zero weighted RMS size")
    standardized = (values - mean) / sigma
    skewness = np.average(standardized**3, weights=w)
    excess_kurtosis = np.average(standardized**4, weights=w) - 3.0
    fifth_cumulant = np.average(standardized**5, weights=w) - 10 * skewness
    standardized_grid = (grid - mean) / sigma
    hermite3 = standardized_grid**3 - 3 * standardized_grid
    hermite4 = standardized_grid**4 - 6 * standardized_grid**2 + 3
    hermite5 = standardized_grid**5 - 10 * standardized_grid**3 + 15 * standardized_grid
    correction = np.ones_like(standardized_grid)
    if order >= 3:
        correction += skewness * hermite3 / 6
    if order >= 4:
        correction += excess_kurtosis * hermite4 / 24
    if order >= 5:
        correction += fifth_cumulant * hermite5 / 120
    gaussian = np.exp(-0.5 * standardized_grid**2) / (sigma * np.sqrt(2 * np.pi))
    return np.maximum(gaussian * correction, 0.0)


def compute_moment_descriptor(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    longitudinal_mode: int = SPLINE,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
    include_total_weight: bool = False,
    include_higher_moments: bool = False,
) -> dict[str, float]:
    """Create a moment descriptor for one selected electron bunch.

    The default spline schema has 33 features: three momentum centroids, 21
    unique 6D covariance entries, eight longitudinal ``uz`` profile features,
    and selected beam charge. ``longitudinal_mode`` may be ``OFF``,
    ``MOMENTS``, or ``SPLINE``. The spline mode emits exactly
    ``longitudinal_bins`` mean and RMS ``uz`` features, or raises when that
    fixed schema cannot be produced.
    """
    if longitudinal_mode not in {OFF, MOMENTS, SPLINE}:
        raise ValueError("longitudinal_mode must be OFF, MOMENTS, or SPLINE")
    x, w = _validate_particles(particles, weights)
    mean, covariance = weighted_mean_cov(x, w)
    features = {
        "mean_ux": float(mean[1]),
        "mean_uy": float(mean[3]),
        "mean_uz": float(mean[5]),
    }
    for row in range(6):
        for column in range(row, 6):
            features[f"cov_{COORD_NAMES[row]}_{COORD_NAMES[column]}"] = float(
                covariance[row, column]
            )
    if longitudinal_mode == MOMENTS:
        longitudinal, _, _, residual_rms = longitudinal_shape_features(x, w)
        features.update(
            {
                "central_moment_z2_uz": float(longitudinal[0]),
                "central_moment_z3_uz": float(longitudinal[1]),
                "quadratic_fit_uz_z2": float(longitudinal[2]),
                "quadratic_fit_uz_residual_rms": float(residual_rms),
            }
        )
    elif longitudinal_mode == SPLINE:
        profile = longitudinal_slice_profile(x, w, bins=longitudinal_bins)
        for index, (mean_uz, rms_uz) in enumerate(
            zip(profile["mean_uz"], profile["rms_uz"])
        ):
            features[f"longitudinal_mean_uz_{index:02d}"] = float(mean_uz)
            features[f"longitudinal_rms_uz_{index:02d}"] = float(rms_uz)
    features["total_beam_charge_c"] = float(ELEMENTARY_CHARGE_C * w.sum())
    if include_total_weight:
        features["log_total_weight"] = float(np.log(w.sum()))
    if include_higher_moments:
        skew, kurtosis = weighted_skew_kurtosis(x, w)
        features.update(
            {f"skew_{name}": float(value) for name, value in zip(COORD_NAMES, skew)}
        )
        features.update(
            {
                f"excess_kurt_{name}": float(value)
                for name, value in zip(COORD_NAMES, kurtosis)
            }
        )
    return features