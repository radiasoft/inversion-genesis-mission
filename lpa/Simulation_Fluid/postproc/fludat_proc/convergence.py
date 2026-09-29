#!/usr/bin/env python3
"""Measure density convergence across ANSYS lineout or CGNS grid resolutions.

Each resolution is compared with a reference: by default the finest available grid,
optionally the next-finer grid. The headline error is the integrated absolute
density difference normalised by the reference density magnitude. CGNS field
comparisons additionally report pointwise error distributions.
"""

from __future__ import annotations

if __package__ in (
    None,
    "",
):  # run as a plain script, e.g. `python fludat_proc/plot_density.py`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_proc"

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay

from .ansys_lineouts import deduplicate_and_sort_profile, parse_lineout_file
from .cgns_io import load_cgns_density_points
from .common import mm_bounds_to_m, validate_bounds
from .filenames import is_grid_size_filename, parse_grid_size_filename

REFERENCE_MODES = ("finest", "adjacent")
FIELD_INTERPOLATIONS = ("linear", "cubic")
FIELD_ERROR_MODES = ("local-relative", "peak-normalized", "both")
PLOT_VIEWS = ("summary", "lineouts", "both")

T = TypeVar("T")


@dataclass(frozen=True)
class ResolutionLineouts:
    """Lineouts from one simulation resolution."""

    grid_size_mm: float
    path: Path
    sections: dict[str, np.ndarray]


@dataclass(frozen=True)
class ResolutionField:
    """Pointwise CGNS density field from one simulation resolution."""

    grid_size_mm: float
    path: Path
    x: np.ndarray
    z: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class ErrorDistribution:
    """Summary statistics of a set of (dimensionless) error samples."""

    sample_count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> ErrorDistribution:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.size == 0:
            raise ValueError("At least one error sample is required")
        return cls(
            sample_count=int(samples.size),
            mean=float(np.mean(samples)),
            standard_deviation=float(np.std(samples)),
            minimum=float(np.min(samples)),
            maximum=float(np.max(samples)),
        )


def pool_distributions(distributions: Sequence[ErrorDistribution]) -> ErrorDistribution:
    """Combine distributions of disjoint sample sets into one (exact mean and std)."""
    if not distributions:
        raise ValueError("At least one distribution is required")
    count = sum(distribution.sample_count for distribution in distributions)
    mean = sum(d.sample_count * d.mean for d in distributions) / count
    second_moment = (
        sum(
            d.sample_count * (d.standard_deviation**2 + d.mean**2)
            for d in distributions
        )
        / count
    )
    return ErrorDistribution(
        sample_count=count,
        mean=float(mean),
        standard_deviation=float(np.sqrt(max(second_moment - mean**2, 0.0))),
        minimum=min(d.minimum for d in distributions),
        maximum=max(d.maximum for d in distributions),
    )


@dataclass(frozen=True)
class ConvergenceMetric:
    """Relative integrated error for one lineout (or whole field) and one grid pair."""

    coarse_grid_size_mm: float
    finer_grid_size_mm: float
    lineout_label: str
    relative_error: float
    z_min_m: float
    z_max_m: float
    x_min_m: float | None = None
    x_max_m: float | None = None
    local_relative_errors: ErrorDistribution | None = None
    """CGNS only: ``|coarse - ref| / |ref|`` per grid cell where ``ref != 0``."""
    peak_normalized_errors: ErrorDistribution | None = None
    """CGNS only: ``|coarse - ref| / max|ref|`` per grid cell."""


@dataclass(frozen=True)
class ConvergenceSummary:
    """Aggregate errors for one (coarse, reference) grid pair.

    ``relative_error`` is the distribution behind the CSV ``*_relative_error`` columns:
    for lineout input the per-lineout integrated errors, for CGNS input the pooled
    pointwise local relative cell errors.
    """

    coarse_grid_size_mm: float
    finer_grid_size_mm: float
    relative_error: ErrorDistribution
    peak_normalized: ErrorDistribution | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _resolution_tagged_files(input_dir: Path, suffix: str | None) -> list[Path]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if is_grid_size_filename(path)
        and not path.name.startswith(".")
        and (suffix is None or path.suffix.lower() == suffix)
    )
    if len(paths) < 2:
        raise ValueError(
            f"Expected at least two resolution-tagged files such as '0_1{suffix or '.txt'}' "
            f"in {input_dir}"
        )
    grid_sizes = [parse_grid_size_filename(path) for path in paths]
    if len(set(grid_sizes)) != len(grid_sizes):
        raise ValueError(f"Duplicate maximum grid size file in {input_dir}")
    return paths


def load_resolution_lineouts(input_dir: Path) -> list[ResolutionLineouts]:
    """Load resolution-tagged ANSYS lineout files, sorted from coarsest to finest."""
    resolutions = sorted(
        (
            ResolutionLineouts(
                grid_size_mm=parse_grid_size_filename(path),
                path=path,
                sections={
                    label: deduplicate_and_sort_profile(profile)
                    for label, profile in parse_lineout_file(path).items()
                },
            )
            for path in _resolution_tagged_files(input_dir, None)
        ),
        key=lambda resolution: resolution.grid_size_mm,
    )

    expected_labels = set(resolutions[0].sections)
    for resolution in resolutions[1:]:
        current_labels = set(resolution.sections)
        if current_labels != expected_labels:
            raise ValueError(
                f"Lineout labels in {resolution.path.name} do not match "
                f"{resolutions[0].path.name}; "
                f"missing={sorted(expected_labels - current_labels)}, "
                f"extra={sorted(current_labels - expected_labels)}"
            )
    return resolutions


def load_resolution_cgns_fields(input_dir: Path) -> list[ResolutionField]:
    """Load resolution-tagged CGNS density fields, sorted from coarsest to finest."""
    fields = []
    for path in _resolution_tagged_files(input_dir, ".cgns"):
        x, z, density = load_cgns_density_points(path)
        fields.append(
            ResolutionField(
                grid_size_mm=parse_grid_size_filename(path),
                path=path,
                x=x,
                z=z,
                density=density,
            )
        )
    return sorted(fields, key=lambda field: field.grid_size_mm)


# ---------------------------------------------------------------------------
# Error calculation
# ---------------------------------------------------------------------------


def reference_pairs(
    items: Sequence[T],
    *,
    grid_size: Callable[[T], float],
    reference_mode: str,
) -> list[tuple[T, T]]:
    """Return ``(coarse, reference)`` pairs ordered from coarsest to finest.

    ``finest`` pairs every coarser item with the finest one; ``adjacent`` pairs each
    item with the next finer one.
    """
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"Unknown convergence reference mode {reference_mode!r}")
    if len(items) < 2:
        raise ValueError("At least two resolutions are required for convergence")
    descending = sorted(items, key=grid_size, reverse=True)
    if reference_mode == "finest":
        return [(coarse, descending[-1]) for coarse in descending[:-1]]
    return list(zip(descending, descending[1:], strict=False))


def relative_integrated_difference(
    coarse_profile: np.ndarray,
    finer_profile: np.ndarray,
) -> tuple[float, float, float]:
    """Return the relative integrated error and shared z limits of two profiles."""
    coarse_z, coarse_density = deduplicate_and_sort_profile(coarse_profile).T
    finer_z, finer_density = deduplicate_and_sort_profile(finer_profile).T

    z_min = max(float(coarse_z[0]), float(finer_z[0]))
    z_max = min(float(coarse_z[-1]), float(finer_z[-1]))
    if z_max <= z_min:
        raise ValueError("Density profiles do not have a nonzero shared z range")

    z_values = np.unique(
        np.concatenate(
            (
                coarse_z[(coarse_z >= z_min) & (coarse_z <= z_max)],
                finer_z[(finer_z >= z_min) & (finer_z <= z_max)],
            )
        )
    )
    if z_values.size < 2:
        raise ValueError("Density profiles do not share enough z samples to integrate")

    coarse_values = np.interp(z_values, coarse_z, coarse_density)
    finer_values = np.interp(z_values, finer_z, finer_density)
    difference_integral = np.trapezoid(np.abs(coarse_values - finer_values), z_values)
    reference_integral = np.trapezoid(np.abs(finer_values), z_values)
    if reference_integral == 0:
        raise ValueError("Finer density profile has zero integrated magnitude")
    return float(difference_integral / reference_integral), z_min, z_max


def calculate_convergence(
    resolutions: list[ResolutionLineouts],
    *,
    reference_mode: str = "finest",
) -> list[ConvergenceMetric]:
    """Calculate per-lineout relative errors against the requested reference grid."""
    metrics: list[ConvergenceMetric] = []
    for coarse, finer in reference_pairs(
        resolutions, grid_size=lambda r: r.grid_size_mm, reference_mode=reference_mode
    ):
        for label in sorted(coarse.sections):
            relative_error, z_min, z_max = relative_integrated_difference(
                coarse.sections[label], finer.sections[label]
            )
            metrics.append(
                ConvergenceMetric(
                    coarse_grid_size_mm=coarse.grid_size_mm,
                    finer_grid_size_mm=finer.grid_size_mm,
                    lineout_label=label,
                    relative_error=relative_error,
                    z_min_m=z_min,
                    z_max_m=z_max,
                )
            )
    return metrics


def _clip_field(
    field: ResolutionField,
    *,
    x_min: float | None,
    x_max: float | None,
    z_min: float | None,
    z_max: float | None,
) -> ResolutionField:
    mask = np.ones(field.x.size, dtype=bool)
    if x_min is not None:
        mask &= field.x >= x_min
    if x_max is not None:
        mask &= field.x <= x_max
    if z_min is not None:
        mask &= field.z >= z_min
    if z_max is not None:
        mask &= field.z <= z_max
    if np.count_nonzero(mask) < 3:
        raise ValueError(
            f"Clipping bounds leave fewer than three CGNS samples in {field.path}"
        )
    return ResolutionField(
        grid_size_mm=field.grid_size_mm,
        path=field.path,
        x=field.x[mask],
        z=field.z[mask],
        density=field.density[mask],
    )


def regrid_cgns_fields(
    fields: list[ResolutionField],
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    x_points: int = 200,
    z_points: int = 200,
    interpolation: str = "linear",
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    """Clip every field and interpolate it once onto one shared regular ``(z, x)`` grid.

    Grid points outside a field's convex hull get zero density, matching the
    CGNS-to-HDF5 converter's default outside-domain policy.
    """
    if x_points < 2 or z_points < 2:
        raise ValueError("x_points and z_points must both be at least 2")
    for axis_name, lower, upper in (("x", x_min, x_max), ("z", z_min, z_max)):
        if lower is not None and upper is not None:
            validate_bounds((lower, upper), axis_name=axis_name)
    if interpolation not in FIELD_INTERPOLATIONS:
        raise ValueError(f"Unknown CGNS interpolation method {interpolation!r}")
    clipped = [
        _clip_field(field, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max)
        for field in fields
    ]
    common_x_min = max(float(field.x.min()) for field in clipped)
    common_x_max = min(float(field.x.max()) for field in clipped)
    common_z_min = max(float(field.z.min()) for field in clipped)
    common_z_max = min(float(field.z.max()) for field in clipped)
    if common_x_max <= common_x_min or common_z_max <= common_z_min:
        raise ValueError("CGNS fields do not have a nonzero shared 2D domain")

    x_grid = np.linspace(common_x_min, common_x_max, x_points)
    z_grid = np.linspace(common_z_min, common_z_max, z_points)
    z_mesh, x_mesh = np.meshgrid(z_grid, x_grid, indexing="ij")
    interpolator_class = (
        LinearNDInterpolator
        if interpolation == "linear"
        else CloughTocher2DInterpolator
    )
    grids = {
        field.grid_size_mm: np.asarray(
            interpolator_class(
                Delaunay(np.column_stack((field.z, field.x))),
                field.density,
                fill_value=0.0,
            )(z_mesh, x_mesh)
        )
        for field in clipped
    }
    return z_grid, x_grid, grids


def _integrated_relative_field_error(
    coarse: np.ndarray, finer: np.ndarray, *, x_grid: np.ndarray, z_grid: np.ndarray
) -> float:
    def area_integral(values: np.ndarray) -> float:
        return float(np.trapezoid(np.trapezoid(values, x_grid, axis=1), z_grid))

    reference_integral = area_integral(np.abs(finer))
    if reference_integral == 0:
        raise ValueError("Finer CGNS field has zero integrated magnitude")
    return area_integral(np.abs(coarse - finer)) / reference_integral


def _local_relative_errors(coarse: np.ndarray, finer: np.ndarray) -> ErrorDistribution:
    nonzero = np.abs(finer) > 0.0
    if not np.any(nonzero):
        raise ValueError("Finer CGNS field has no nonzero density samples")
    return ErrorDistribution.from_samples(
        np.abs(coarse[nonzero] - finer[nonzero]) / np.abs(finer[nonzero])
    )


def _peak_normalized_errors(coarse: np.ndarray, finer: np.ndarray) -> ErrorDistribution:
    peak = float(np.max(np.abs(finer)))
    if peak == 0.0:
        raise ValueError("Finer CGNS field has zero peak density")
    return ErrorDistribution.from_samples(np.abs(coarse - finer) / peak)


def calculate_field_convergence(
    fields: list[ResolutionField],
    *,
    reference_mode: str = "finest",
    x_min: float | None = None,
    x_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    x_points: int = 200,
    z_points: int = 200,
    interpolation: str = "linear",
) -> list[ConvergenceMetric]:
    """Calculate whole-field CGNS convergence for each grid pair on one shared grid."""
    pairs = reference_pairs(
        fields, grid_size=lambda f: f.grid_size_mm, reference_mode=reference_mode
    )
    z_grid, x_grid, grids = regrid_cgns_fields(
        fields,
        x_min=x_min,
        x_max=x_max,
        z_min=z_min,
        z_max=z_max,
        x_points=x_points,
        z_points=z_points,
        interpolation=interpolation,
    )
    metrics: list[ConvergenceMetric] = []
    for coarse, finer in pairs:
        coarse_density = grids[coarse.grid_size_mm]
        finer_density = grids[finer.grid_size_mm]
        metrics.append(
            ConvergenceMetric(
                coarse_grid_size_mm=coarse.grid_size_mm,
                finer_grid_size_mm=finer.grid_size_mm,
                lineout_label="field",
                relative_error=_integrated_relative_field_error(
                    coarse_density, finer_density, x_grid=x_grid, z_grid=z_grid
                ),
                z_min_m=float(z_grid[0]),
                z_max_m=float(z_grid[-1]),
                x_min_m=float(x_grid[0]),
                x_max_m=float(x_grid[-1]),
                local_relative_errors=_local_relative_errors(
                    coarse_density, finer_density
                ),
                peak_normalized_errors=_peak_normalized_errors(
                    coarse_density, finer_density
                ),
            )
        )
    return metrics


def summarize_convergence(metrics: list[ConvergenceMetric]) -> list[ConvergenceSummary]:
    """Aggregate metrics per grid pair, ordered from coarsest to finest."""
    grouped: dict[tuple[float, float], list[ConvergenceMetric]] = {}
    for metric in metrics:
        grouped.setdefault(
            (metric.coarse_grid_size_mm, metric.finer_grid_size_mm), []
        ).append(metric)

    summaries: list[ConvergenceSummary] = []
    for (coarse_size, finer_size), group in sorted(grouped.items()):
        local = [metric.local_relative_errors for metric in group]
        peak = [metric.peak_normalized_errors for metric in group]
        summaries.append(
            ConvergenceSummary(
                coarse_grid_size_mm=coarse_size,
                finer_grid_size_mm=finer_size,
                relative_error=(
                    pool_distributions(local)
                    if all(d is not None for d in local)
                    else ErrorDistribution.from_samples(
                        [m.relative_error for m in group]
                    )
                ),
                peak_normalized=(
                    pool_distributions(peak)
                    if all(d is not None for d in peak)
                    else None
                ),
            )
        )
    return summaries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = (
    "coarse_grid_size_mm",
    "finer_grid_size_mm",
    "lineout_label",
    "relative_integrated_absolute_difference",
    "z_min_m",
    "z_max_m",
    "x_min_m",
    "x_max_m",
    "mean_relative_error",
    "standard_deviation",
    "minimum_relative_error",
    "maximum_relative_error",
    "mean_peak_normalized_absolute_error",
    "standard_deviation_peak_normalized_absolute_error",
    "minimum_peak_normalized_absolute_error",
    "maximum_peak_normalized_absolute_error",
)


def write_convergence_csv(
    output_path: Path,
    metrics: list[ConvergenceMetric],
    summaries: list[ConvergenceSummary],
) -> Path:
    """Write per-lineout metrics alongside their grid-pair aggregate statistics."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_by_pair = {
        (s.coarse_grid_size_mm, s.finer_grid_size_mm): s for s in summaries
    }

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for metric in metrics:
            summary = summary_by_pair[
                (metric.coarse_grid_size_mm, metric.finer_grid_size_mm)
            ]
            local = summary.relative_error
            peak = summary.peak_normalized
            writer.writerow(
                {
                    "coarse_grid_size_mm": metric.coarse_grid_size_mm,
                    "finer_grid_size_mm": metric.finer_grid_size_mm,
                    "lineout_label": metric.lineout_label,
                    "relative_integrated_absolute_difference": metric.relative_error,
                    "z_min_m": metric.z_min_m,
                    "z_max_m": metric.z_max_m,
                    "x_min_m": metric.x_min_m,
                    "x_max_m": metric.x_max_m,
                    "mean_relative_error": local.mean,
                    "standard_deviation": local.standard_deviation,
                    "minimum_relative_error": local.minimum,
                    "maximum_relative_error": local.maximum,
                    "mean_peak_normalized_absolute_error": peak.mean if peak else None,
                    "standard_deviation_peak_normalized_absolute_error": (
                        peak.standard_deviation if peak else None
                    ),
                    "minimum_peak_normalized_absolute_error": (
                        peak.minimum if peak else None
                    ),
                    "maximum_peak_normalized_absolute_error": (
                        peak.maximum if peak else None
                    ),
                }
            )
    return output_path


def _positive_plot_values(values: np.ndarray) -> np.ndarray:
    """Replace exact zeroes so they remain visible on logarithmic axes."""
    positive = values[values > 0]
    if positive.size == 0:
        return np.full_like(values, np.finfo(float).tiny)
    return np.maximum(values, positive.min() / 10)


def _style_convergence_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("maximum grid size [mm]")
    axis.set_ylabel("relative density error")
    axis.grid(which="both")


def _set_log_y_limits(axis: plt.Axes, values: np.ndarray) -> None:
    """Set whole-decade log limits from the positive plotted data values."""
    positive_values = values[np.isfinite(values) & (values > 0.0)]
    if positive_values.size == 0:
        return
    y_min = 10.0 ** np.floor(np.log10(positive_values.min()))
    y_max = 10.0 ** np.ceil(np.log10(positive_values.max()))
    if y_min == y_max:
        y_min /= 10.0
        y_max *= 10.0
    axis.set_ylim(y_min, y_max)


def _plot_mean_band(
    axis: plt.Axes,
    grid_sizes: np.ndarray,
    distributions: Sequence[ErrorDistribution],
    *,
    marker: str,
    label: str,
) -> np.ndarray:
    """Plot mean +/- one standard deviation; return the values that set the y limits."""
    means = np.array([d.mean for d in distributions])
    deviations = np.array([d.standard_deviation for d in distributions])
    plot_means = _positive_plot_values(means)
    upper = _positive_plot_values(means + deviations)
    axis.plot(grid_sizes, plot_means, marker=marker, label=label)
    axis.fill_between(
        grid_sizes,
        _positive_plot_values(means - deviations),
        upper,
        alpha=0.25,
        label=f"{label} +/- 1 std",
    )
    return np.concatenate((plot_means, upper))


def plot_convergence(
    metrics: list[ConvergenceMetric],
    summaries: list[ConvergenceSummary],
    *,
    view: str = "both",
    field_error: str = "both",
    title: str | None = None,
) -> plt.Figure:
    """Plot aggregate convergence curves and, for lineout input, one curve per lineout."""
    if view not in PLOT_VIEWS:
        raise ValueError(f"Unknown convergence plot view {view!r}")
    if field_error not in FIELD_ERROR_MODES:
        raise ValueError(f"Unknown CGNS field-error mode {field_error!r}")

    if view == "both":
        figure, (summary_axis, lineout_axis) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 8)
        )
    else:
        figure, axis = plt.subplots(figsize=(8, 5))
        summary_axis = axis if view == "summary" else None
        lineout_axis = axis if view == "lineouts" else None

    if summary_axis is not None:
        ordered = sorted(summaries, key=lambda summary: summary.coarse_grid_size_mm)
        grid_sizes = np.array([summary.coarse_grid_size_mm for summary in ordered])
        is_field_summary = bool(ordered) and all(
            s.peak_normalized is not None for s in ordered
        )
        _style_convergence_axis(summary_axis)
        plot_values: list[np.ndarray] = []
        if not is_field_summary or field_error in {"local-relative", "both"}:
            plot_values.append(
                _plot_mean_band(
                    summary_axis,
                    grid_sizes,
                    [s.relative_error for s in ordered],
                    marker="o",
                    label="mean local relative error" if is_field_summary else "mean",
                )
            )
        if is_field_summary and field_error in {"peak-normalized", "both"}:
            plot_values.append(
                _plot_mean_band(
                    summary_axis,
                    grid_sizes,
                    [s.peak_normalized for s in ordered],  # type: ignore[misc]
                    marker="s",
                    label="mean absolute error / peak reference density",
                )
            )
        if plot_values:
            _set_log_y_limits(summary_axis, np.concatenate(plot_values))
        summary_axis.set_title(
            "Mean field errors"
            if is_field_summary
            else "Mean convergence across lineouts"
        )
        summary_axis.legend()

    if lineout_axis is not None:
        metrics_by_label: dict[str, list[ConvergenceMetric]] = {}
        for metric in metrics:
            metrics_by_label.setdefault(metric.lineout_label, []).append(metric)
        for label, label_metrics in sorted(metrics_by_label.items()):
            ordered_metrics = sorted(label_metrics, key=lambda m: m.coarse_grid_size_mm)
            lineout_axis.plot(
                [m.coarse_grid_size_mm for m in ordered_metrics],
                _positive_plot_values(
                    np.array([m.relative_error for m in ordered_metrics])
                ),
                marker="o",
                label=label,
            )
        _style_convergence_axis(lineout_axis)
        _set_log_y_limits(lineout_axis, np.array([m.relative_error for m in metrics]))
        lineout_axis.set_title("Convergence by lineout")
        lineout_axis.legend(ncol=2, fontsize="small")

    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare density at each grid size with a reference resolution."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory of resolution-tagged ANSYS lineout (.txt) or CGNS files, e.g. 0_1.txt",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "lineout", "cgns"),
        default="auto",
        help="Input format (default: cgns when the directory holds .cgns files, else lineout)",
    )
    parser.add_argument(
        "--view",
        choices=PLOT_VIEWS,
        default="both",
        help="Plot mean-plus-spread, individual lineouts, or both (default: both; "
        "CGNS input always uses summary)",
    )
    parser.add_argument(
        "--reference",
        choices=REFERENCE_MODES,
        default="finest",
        help="finest treats the highest resolution as ground truth; adjacent uses the "
        "next finer grid (default: finest)",
    )
    parser.add_argument(
        "--field-error",
        choices=FIELD_ERROR_MODES,
        default="both",
        help="CGNS summary curves: local-relative divides each cell's error by its "
        "reference density; peak-normalized divides by the peak reference density "
        "(default: both)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save the plot instead of showing it",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="CSV output path (default: next to --output; no CSV for interactive plots)",
    )
    parser.add_argument(
        "--x-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="CGNS comparison x bounds [mm] (default: full common extent)",
    )
    parser.add_argument(
        "--z-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="CGNS comparison z bounds [mm] (default: full common extent)",
    )
    parser.add_argument(
        "--x-points", type=int, default=200, help="CGNS grid points along x"
    )
    parser.add_argument(
        "--z-points", type=int, default=200, help="CGNS grid points along z"
    )
    parser.add_argument(
        "--interpolation",
        choices=FIELD_INTERPOLATIONS,
        default="linear",
        help="CGNS unstructured interpolation: linear (default) or cubic Clough-Tocher",
    )
    parser.add_argument("-t", "--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    input_format = args.input_format
    if input_format == "auto":
        input_format = "cgns" if any(args.input_dir.glob("*.cgns")) else "lineout"

    if input_format == "cgns":
        try:
            x_bounds = validate_bounds(
                mm_bounds_to_m(args.x_bounds), axis_name="x"
            ) or (None, None)
            z_bounds = validate_bounds(
                mm_bounds_to_m(args.z_bounds), axis_name="z"
            ) or (None, None)
        except ValueError as exc:
            parser.error(str(exc))
        metrics = calculate_field_convergence(
            load_resolution_cgns_fields(args.input_dir),
            reference_mode=args.reference,
            x_min=x_bounds[0],
            x_max=x_bounds[1],
            z_min=z_bounds[0],
            z_max=z_bounds[1],
            x_points=args.x_points,
            z_points=args.z_points,
            interpolation=args.interpolation,
        )
    else:
        metrics = calculate_convergence(
            load_resolution_lineouts(args.input_dir), reference_mode=args.reference
        )
    summaries = summarize_convergence(metrics)
    figure = plot_convergence(
        metrics,
        summaries,
        view="summary" if input_format == "cgns" else args.view,
        field_error=args.field_error,
        title=args.title,
    )

    metrics_output = args.metrics_output
    if metrics_output is None and args.output is not None:
        metrics_output = args.output.with_suffix(".csv")
    if metrics_output is not None:
        print(f"Wrote {write_convergence_csv(metrics_output, metrics, summaries)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
