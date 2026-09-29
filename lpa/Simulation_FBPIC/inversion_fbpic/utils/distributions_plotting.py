"""Shared phase-space plotting for standardized particle distributions."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from inversion_fbpic.utils.distributions import (
    COORD_NAMES,
    LONGITUDINAL_PROFILE_BINS,
    MOMENTS,
    OFF,
    SPLINE,
    compute_moment_descriptor,
    gram_charlier_marginal_density,
    gram_charlier_projection_density,
    longitudinal_shape_features,
    longitudinal_profile_density,
    select_by_uz,
    weighted_mean_cov,
)

PLOTS = (("z", "uz"), ("x", "ux"), ("y", "uy"), ("x", "y"))
DISPLAY_NAMES = ("x", "x'", "y", "y'", "z", "uz")
MICROMETER_COORDINATES = {"x", "y", "z"}


def _weighted_limits(
    values: np.ndarray, weights: np.ndarray, central_fraction: float
) -> np.ndarray:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    quantiles = ((1 - central_fraction) / 2, (1 + central_fraction) / 2)
    return np.interp(np.asarray(quantiles) * cumulative[-1], cumulative, values)


def _analytic_contours(
    axis: Axes,
    particles: np.ndarray,
    weights: np.ndarray,
    x_index: int,
    y_index: int,
    scale: np.ndarray,
    x_limits: np.ndarray,
    y_limits: np.ndarray,
    gram_charlier_order: int | None = None,
    longitudinal_mode: int = SPLINE,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
) -> None:
    """Overlay analytic Gaussian or Gram-Charlier 68% and 95% contours."""
    centers_x = np.linspace(float(x_limits[0]), float(x_limits[1]), 250)
    centers_y = np.linspace(float(y_limits[0]), float(y_limits[1]), 250)
    grid_x, grid_y = np.meshgrid(centers_x, centers_y)
    if gram_charlier_order is not None:
        density = gram_charlier_projection_density(
            particles,
            weights,
            x_index,
            y_index,
            grid_x / scale[0],
            grid_y / scale[1],
            order=gram_charlier_order,
        )
    elif longitudinal_mode == MOMENTS and (x_index, y_index) == (4, 5):
        mean, covariance = weighted_mean_cov(particles, weights)
        _, coefficients, _, residual_rms = longitudinal_shape_features(
            particles, weights
        )
        z_delta = grid_x / scale[0] - mean[4]
        fitted_uz = np.polynomial.polynomial.polyval(z_delta, coefficients[::-1])
        density = np.exp(
            -0.5
            * (
                (z_delta / np.sqrt(covariance[4, 4])) ** 2
                + ((grid_y / scale[1] - fitted_uz) / residual_rms) ** 2
            )
        )
    elif longitudinal_mode == SPLINE and (x_index, y_index) == (4, 5):
        density = longitudinal_profile_density(
            particles,
            weights,
            grid_x / scale[0],
            grid_y / scale[1],
            bins=longitudinal_bins,
        )
    else:
        mean, covariance = weighted_mean_cov(particles, weights)
        projection_mean = mean[[x_index, y_index]] * scale
        projection_covariance = (
            np.diag(scale)
            @ covariance[np.ix_([x_index, y_index], [x_index, y_index])]
            @ np.diag(scale)
        )
        inverse_covariance = np.linalg.inv(projection_covariance)
        offsets = np.stack(
            [grid_x - projection_mean[0], grid_y - projection_mean[1]], axis=-1
        )
        density = np.exp(
            -0.5 * np.einsum(
                "...i,ij,...j->...", offsets, inverse_covariance, offsets
            )
        ) / (2 * np.pi * np.sqrt(np.linalg.det(projection_covariance)))
    positive = density[density > 0]
    if len(positive):
        sorted_density = np.sort(positive)[::-1]
        enclosed_mass = np.cumsum(sorted_density) / sorted_density.sum()
        levels = [
            sorted_density[np.searchsorted(enclosed_mass, probability)]
            for probability in (0.95, 0.68)
        ]
        axis.contour(
            grid_x,
            grid_y,
            density,
            levels=levels,
            colors="white",
            linestyles="--",
            linewidths=[1.1, 1.6],
        )


def _coordinate_scale(name: str) -> float:
    """Return the plotting scale for a phase-space coordinate."""
    return 1e6 if name in MICROMETER_COORDINATES else 1.0


def _coordinate_label(index: int) -> str:
    """Return the display label, including micrometer units when appropriate."""
    name = COORD_NAMES[index]
    label = DISPLAY_NAMES[index]
    return f"{label} (um)" if name in MICROMETER_COORDINATES else label


def _projection(
    axis: Axes,
    values: np.ndarray,
    weights: np.ndarray,
    particles: np.ndarray,
    coordinate_index: int,
    coordinate_scale: float,
    bins: int,
    gram_charlier_order: int | None = None,
) -> None:
    """Plot a weighted 1D projection and its analytic moment-model density."""
    density, edges, _ = axis.hist(values, bins=bins, weights=weights, density=True, color="#365f8b", alpha=0.7)
    density = np.asarray(density)
    edges = np.asarray(edges)
    mean = np.average(values, weights=weights)
    sigma = np.sqrt(np.average((values - mean) ** 2, weights=weights))
    if sigma > 0:
        grid = np.linspace(edges[0], edges[-1], 300)
        if gram_charlier_order is not None:
            density_model = gram_charlier_marginal_density(
                particles,
                weights,
                coordinate_index,
                grid / coordinate_scale,
                order=gram_charlier_order,
            ) / coordinate_scale
        else:
            density_model = np.exp(-0.5 * ((grid - mean) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )
        axis.plot(grid, density_model, "--", color="black", linewidth=1.3)
    axis.set_ylim(0, max(density.max() * 1.1, 1e-300))


def plot_phase_space_moments(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    uz_min: float | None = 30.0,
    central_fraction: float = 0.99,
    bins: int = 150,
    title: str | None = None,
    gram_charlier_order: int | None = None,
    longitudinal_mode: int = SPLINE,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
) -> Figure:
    """Plot raw weighted density against the shared moment-model projections.

    Set ``gram_charlier_order`` to 2, 3, 4, or 5 to override the default plotting
    path with a Gram-Charlier A approximation for every displayed panel.
    """
    if not 0 < central_fraction < 1:
        raise ValueError("central_fraction must be strictly between 0 and 1")
    if gram_charlier_order is not None and gram_charlier_order not in {2, 3, 4, 5}:
        raise ValueError("gram_charlier_order must be 2, 3, 4, or 5")
    if longitudinal_mode not in {OFF, MOMENTS, SPLINE}:
        raise ValueError("longitudinal_mode must be OFF, MOMENTS, or SPLINE")
    particles, weights = select_by_uz(particles, weights, uz_min=uz_min)
    weights = np.ones(len(particles)) if weights is None else np.abs(weights)
    descriptor = compute_moment_descriptor(
        particles,
        weights,
        longitudinal_mode=longitudinal_mode,
        longitudinal_bins=longitudinal_bins,
    )
    print(
        f"Beam descriptor: {len(descriptor)} parameters "
        f"(longitudinal mode={longitudinal_mode})"
    )
    indices = {name: index for index, name in enumerate(COORD_NAMES)}
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, (x_name, y_name) in zip(axes.flat, PLOTS):
        x_index, y_index = indices[x_name], indices[y_name]
        scale = np.array([_coordinate_scale(x_name), _coordinate_scale(y_name)])
        x, y = particles[:, x_index] * scale[0], particles[:, y_index] * scale[1]
        x_limits, y_limits = _weighted_limits(x, weights, central_fraction), _weighted_limits(y, weights, central_fraction)
        axis.hist2d(x, y, bins=bins, range=(x_limits, y_limits), weights=weights, cmap="viridis")
        _analytic_contours(
            axis, particles, weights, x_index, y_index, scale, x_limits, y_limits,
            gram_charlier_order,
            longitudinal_mode,
            longitudinal_bins,
        )
        axis.set(
            xlim=x_limits,
            ylim=y_limits,
            xlabel=_coordinate_label(x_index),
            ylabel=_coordinate_label(y_index),
        )
    figure.suptitle(title or "Raw particle phase space and moment models")
    return figure


def plot_all_phase_space_moments(
    particles: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    uz_min: float | None = 30.0,
    central_fraction: float = 0.99,
    bins: int = 100,
    title: str | None = None,
    gram_charlier_order: int | None = None,
    longitudinal_mode: int = SPLINE,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
) -> Figure:
    """Plot every unique 1D and 2D phase-space projection in a triangular grid.

    Set ``gram_charlier_order`` to 2, 3, 4, or 5 to use a Gram-Charlier A
    approximation for every diagonal and lower-triangle panel. Upper-triangle
    cells are hidden because they mirror the corresponding lower-triangle views.
    """
    if not 0 < central_fraction < 1:
        raise ValueError("central_fraction must be strictly between 0 and 1")
    if gram_charlier_order is not None and gram_charlier_order not in {2, 3, 4, 5}:
        raise ValueError("gram_charlier_order must be 2, 3, 4, or 5")
    if longitudinal_mode not in {OFF, MOMENTS, SPLINE}:
        raise ValueError("longitudinal_mode must be OFF, MOMENTS, or SPLINE")
    particles, weights = select_by_uz(particles, weights, uz_min=uz_min)
    weights = np.ones(len(particles)) if weights is None else np.abs(weights)
    descriptor = compute_moment_descriptor(
        particles,
        weights,
        longitudinal_mode=longitudinal_mode,
        longitudinal_bins=longitudinal_bins,
    )
    print(
        f"Beam descriptor: {len(descriptor)} parameters "
        f"(longitudinal mode={longitudinal_mode})"
    )

    figure, axes = plt.subplots(6, 6, figsize=(16, 16), constrained_layout=True)
    for row, y_name in enumerate(COORD_NAMES):
        for column, x_name in enumerate(COORD_NAMES):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            x_scale = _coordinate_scale(x_name)
            x_values = particles[:, column] * x_scale
            x_limits = _weighted_limits(x_values, weights, central_fraction)
            if row == column:
                _projection(
                    axis,
                    x_values,
                    weights,
                    particles,
                    column,
                    x_scale,
                    bins,
                    gram_charlier_order,
                )
                axis.set_xlim(x_limits)
                axis.set_xlabel(_coordinate_label(column), fontsize=8)
                axis.set_ylabel("density", fontsize=8)
            else:
                y_scale = _coordinate_scale(y_name)
                y_values = particles[:, row] * y_scale
                y_limits = _weighted_limits(y_values, weights, central_fraction)
                axis.hist2d(x_values, y_values, bins=bins, range=(x_limits, y_limits), weights=weights, cmap="viridis")
                _analytic_contours(
                    axis,
                    particles,
                    weights,
                    column,
                    row,
                    np.array([x_scale, y_scale]),
                    x_limits,
                    y_limits,
                    gram_charlier_order,
                    longitudinal_mode,
                    longitudinal_bins,
                )
                axis.set(xlim=x_limits, ylim=y_limits, xlabel=_coordinate_label(column), ylabel=_coordinate_label(row))
            axis.tick_params(labelsize=7)
    figure.suptitle(title or "All raw particle phase-space projections and moment models")
    return figure
