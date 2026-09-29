#!/usr/bin/env python3
"""Interactively inspect density-gradient shocks in an unstructured CGNS field.

The heatmap shows |grad rho| on a log scale. Dragging a segment across it samples
the checked scalar fields along that segment and compares the downstream/upstream
ratios of pressure, density, and temperature with oblique-shock predictions. The
endpoint with the larger Mach number is treated as upstream.

Example (conda env inv-fbpic):

    python -m fludat_proc.shock_evaluator data/generic_field_raw/htu_all_fields.cgns
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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation
from matplotlib.widgets import CheckButtons, TextBox
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from .cgns_io import (
    DEFAULT_FLOW_SOLUTION,
    DEFAULT_X_COORDINATE,
    DEFAULT_Z_COORDINATE,
    CgnsDataError,
    CgnsScalarFieldData,
    load_cgns_scalar_fields,
)
from .common import MM_PER_M, mm_bounds_to_m

DEFAULT_DENSITY_X_GRADIENT = "dp-dX"
DEFAULT_DENSITY_Y_GRADIENT = "dp-dY"
SHOCK_ANGLE_MODES = ("auto", "perp", "manual")
SHOCK_STATE_FIELD_NAMES = (
    "Mach",
    "Axial_Velocity",
    "Radial_Velocity",
    "Pressure",
    "Density",
    "Temperature",
)

Point = tuple[float, float]
"""A heatmap point in source-coordinate ``(z, x)`` metres."""


# ---------------------------------------------------------------------------
# Oblique-shock relations
# ---------------------------------------------------------------------------


def validate_gamma(gamma: float) -> float:
    """Validate the heat-capacity ratio used by the oblique-shock equations."""
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than 1")
    return float(gamma)


def oblique_p2p1(gamma: float, m1: float, beta: float) -> float:
    """Static pressure ratio p2/p1 across an oblique shock.

    ``m1`` is the upstream Mach number and ``beta`` the angle between the shock and
    the upstream flow direction, in radians.
    """
    gamma = validate_gamma(gamma)
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (m1**2 * np.sin(beta) ** 2 - 1.0)


def oblique_rho2rho1(gamma: float, m1: float, beta: float) -> float:
    """Density ratio rho2/rho1 across an oblique shock (arguments as ``oblique_p2p1``)."""
    gamma = validate_gamma(gamma)
    normal_mach_squared = m1**2 * np.sin(beta) ** 2
    return (
        (gamma + 1.0)
        * normal_mach_squared
        / ((gamma - 1.0) * normal_mach_squared + 2.0)
    )


def oblique_T2T1(gamma: float, m1: float, beta: float) -> float:
    """Temperature ratio T2/T1 across an oblique shock (arguments as ``oblique_p2p1``)."""
    return oblique_p2p1(gamma, m1, beta) / oblique_rho2rho1(gamma, m1, beta)


# ---------------------------------------------------------------------------
# Shock evaluation from sampled endpoint states
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShockState:
    """Thermodynamic state sampled at one segment endpoint."""

    pressure: float
    density: float
    temperature: float

    @classmethod
    def from_fields(cls, values: Mapping[str, float]) -> ShockState:
        return cls(values["Pressure"], values["Density"], values["Temperature"])

    def as_array(self) -> np.ndarray:
        return np.array(
            (self.pressure, self.density, self.temperature), dtype=np.float64
        )


@dataclass(frozen=True)
class ShockEvaluation:
    """Observed and oblique-shock-predicted downstream/upstream state ratios.

    The upstream endpoint (state 1) is the one with the larger Mach number; every
    ratio is state 2 (downstream) over state 1 (upstream).
    """

    gamma: float
    mach_upstream: float
    beta_rad: float
    velocity_change: np.ndarray
    pressure_ratio_simulated: float
    density_ratio_simulated: float
    temperature_ratio_simulated: float
    pressure_ratio_predicted: float
    density_ratio_predicted: float
    temperature_ratio_predicted: float
    angle_source: str = "velocity change"
    upstream_endpoint: str = "drag start"


def validate_shock_angle_mode(mode: str) -> str:
    """Validate the method used to determine the oblique-shock angle."""
    if mode not in SHOCK_ANGLE_MODES:
        raise ValueError(f"shock_angle must be one of {', '.join(SHOCK_ANGLE_MODES)}")
    return mode


def select_upstream_endpoint(start_mach: float, end_mach: float) -> tuple[bool, str]:
    """Return whether the drag start is upstream, based on the larger endpoint Mach."""
    if (
        not np.isfinite((start_mach, end_mach)).all()
        or min(start_mach, end_mach) <= 0.0
    ):
        raise ValueError("Endpoint Mach numbers must be finite and positive")
    if np.isclose(start_mach, end_mach):
        raise ValueError("Cannot infer upstream endpoint from equal Mach numbers")
    if start_mach > end_mach:
        return True, "drag start"
    return False, "drag end"


def _as_point(value: tuple[float, float] | np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain two finite components")
    return point


def _to_physical(plot_direction: np.ndarray) -> np.ndarray:
    """Convert a heatmap ``(z, x)`` direction to physical ``(axial, radial)`` order."""
    return np.array((plot_direction[1], plot_direction[0]))


def _beta_from_direction(
    shock_direction: np.ndarray, upstream_velocity: np.ndarray
) -> float:
    """Return the angle between a unit shock direction and the upstream velocity."""
    upstream_speed = float(np.linalg.norm(upstream_velocity))
    if upstream_speed == 0.0:
        raise ValueError("Upstream velocity must be nonzero")
    cosine = float(np.dot(shock_direction, upstream_velocity)) / upstream_speed
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def perpendicular_segment_shock_direction(
    segment_start: Point,
    segment_end: Point,
    upstream_velocity: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return beta and a unit physical ``(axial, radial)`` normal to the selected segment.

    The normal is oriented along the upstream flow.
    """
    start = _as_point(segment_start, "Segment start")
    end = _as_point(segment_end, "Segment end")
    upstream_velocity = _as_point(upstream_velocity, "Upstream velocity")
    if np.allclose(end - start, 0.0):
        raise ValueError("Segment endpoints must be distinct")

    tangent = _to_physical(end - start)
    shock_direction = np.array((-tangent[1], tangent[0]))
    shock_direction /= np.linalg.norm(shock_direction)
    if np.dot(shock_direction, upstream_velocity) < 0.0:
        shock_direction *= -1.0
    return _beta_from_direction(shock_direction, upstream_velocity), shock_direction


def third_point_shock_direction(
    segment_start: Point,
    segment_end: Point,
    direction_point: Point,
    upstream_velocity: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return beta and the unit physical direction from the segment midpoint to a third point."""
    start = _as_point(segment_start, "Segment start")
    end = _as_point(segment_end, "Segment end")
    direction_point = _as_point(direction_point, "Direction point")
    upstream_velocity = _as_point(upstream_velocity, "Upstream velocity")
    plot_direction = direction_point - 0.5 * (start + end)
    if np.allclose(plot_direction, 0.0):
        raise ValueError("Third point must differ from the segment midpoint")

    shock_direction = _to_physical(plot_direction)
    shock_direction /= np.linalg.norm(shock_direction)
    return _beta_from_direction(shock_direction, upstream_velocity), shock_direction


def evaluate_oblique_shock(
    *,
    gamma: float,
    mach_upstream: float,
    upstream_velocity: np.ndarray,
    downstream_velocity: np.ndarray,
    upstream_state: ShockState,
    downstream_state: ShockState,
    beta_rad: float | None = None,
    angle_source: str = "velocity change",
    upstream_endpoint: str = "drag start",
) -> ShockEvaluation:
    """Compare observed downstream/upstream ratios with oblique-shock predictions.

    Velocity components are ordered ``(axial, radial)``, matching the CGNS fields.
    When ``beta_rad`` is omitted the shock angle is taken from the velocity change.
    """
    gamma = validate_gamma(gamma)
    if not np.isfinite(mach_upstream) or mach_upstream <= 0.0:
        raise ValueError("Upstream Mach number must be finite and positive")
    upstream_velocity = _as_point(upstream_velocity, "Upstream velocity")
    downstream_velocity = _as_point(downstream_velocity, "Downstream velocity")
    velocity_change = upstream_velocity - downstream_velocity

    if beta_rad is None:
        change_speed = float(np.linalg.norm(velocity_change))
        if change_speed == 0.0:
            raise ValueError("Upstream and downstream velocities must differ")
        beta_rad = np.pi / 2 - _beta_from_direction(
            velocity_change / change_speed, upstream_velocity
        )
    elif not np.isfinite(beta_rad) or not 0.0 <= beta_rad <= np.pi:
        raise ValueError("Shock angle beta must be finite and between 0 and pi radians")

    upstream_values = upstream_state.as_array()
    downstream_values = downstream_state.as_array()
    if not np.all(np.isfinite(np.concatenate((upstream_values, downstream_values)))):
        raise ValueError("Shock-state values must be finite")
    if np.any(upstream_values == 0.0):
        raise ValueError("Upstream pressure, density, and temperature must be nonzero")
    observed = downstream_values / upstream_values

    return ShockEvaluation(
        gamma=gamma,
        mach_upstream=float(mach_upstream),
        beta_rad=beta_rad,
        velocity_change=velocity_change,
        pressure_ratio_simulated=float(observed[0]),
        density_ratio_simulated=float(observed[1]),
        temperature_ratio_simulated=float(observed[2]),
        pressure_ratio_predicted=oblique_p2p1(gamma, mach_upstream, beta_rad),
        density_ratio_predicted=oblique_rho2rho1(gamma, mach_upstream, beta_rad),
        temperature_ratio_predicted=oblique_T2T1(gamma, mach_upstream, beta_rad),
        angle_source=angle_source,
        upstream_endpoint=upstream_endpoint,
    )


def format_shock_report(evaluation: ShockEvaluation) -> str:
    """Format observed and predicted shock ratios as aligned monospace columns."""
    label_width = 23
    value_width = 18

    def row(label: str, observed: float, predicted: float) -> str:
        if predicted == 0.0:
            percent_error = 0.0 if observed == 0.0 else float("nan")
        else:
            percent_error = abs(observed - predicted) / abs(predicted) * 100.0
        return (
            f"{label:<{label_width}}{observed:>{value_width}.2f}"
            f"{predicted:>{value_width}.2f}{percent_error:>{value_width}.2f}"
        )

    return "\n".join(
        (
            f"gamma = {evaluation.gamma:.3f}  M1 = {evaluation.mach_upstream:.2f}  "
            f"beta = {np.rad2deg(evaluation.beta_rad):.1f} deg",
            f"upstream (1) = {evaluation.upstream_endpoint}  "
            f"angle = {evaluation.angle_source}",
            f"{'ratio (2 = downstream)':<{label_width}}{'simulated':>{value_width}}"
            f"{'predicted':>{value_width}}{'percent error [%]':>{value_width}}",
            row(
                "P2/P1",
                evaluation.pressure_ratio_simulated,
                evaluation.pressure_ratio_predicted,
            ),
            row(
                "rho2/rho1",
                evaluation.density_ratio_simulated,
                evaluation.density_ratio_predicted,
            ),
            row(
                "T2/T1",
                evaluation.temperature_ratio_simulated,
                evaluation.temperature_ratio_predicted,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Field sampling helpers
# ---------------------------------------------------------------------------


def density_gradient_magnitude(
    fields: Mapping[str, np.ndarray],
    *,
    x_gradient_name: str = DEFAULT_DENSITY_X_GRADIENT,
    y_gradient_name: str = DEFAULT_DENSITY_Y_GRADIENT,
) -> np.ndarray:
    """Return |grad rho| from the two exported gradient components."""
    try:
        x_gradient = fields[x_gradient_name]
        y_gradient = fields[y_gradient_name]
    except KeyError as exc:
        raise CgnsDataError(
            "Required density-gradient field is missing: "
            f"{exc.args[0]!r}; expected {x_gradient_name!r} and {y_gradient_name!r}"
        ) from exc
    if x_gradient.shape != y_gradient.shape:
        raise CgnsDataError(
            "Density-gradient component arrays must have matching shapes"
        )
    return np.hypot(x_gradient, y_gradient)


def initial_log_limits(values: np.ndarray) -> tuple[float, float]:
    """Choose positive, finite logarithmic limits robust to isolated outliers."""
    positive_values = values[np.isfinite(values) & (values > 0.0)]
    if positive_values.size == 0:
        raise CgnsDataError("Density-gradient magnitude has no positive finite values")
    lower, upper = np.quantile(positive_values, (0.01, 0.99))
    if lower == upper:
        lower = float(positive_values.min())
        upper = float(positive_values.max())
    if lower == upper:
        lower *= 0.9
        upper *= 1.1
    return float(lower), float(upper)


def parse_log_limits(lower_text: str, upper_text: str) -> tuple[float, float]:
    """Parse and validate explicitly entered positive heatmap limits."""
    try:
        lower = float(lower_text)
        upper = float(upper_text)
    except ValueError as exc:
        raise ValueError("Limits must be numeric values") from exc
    if not np.isfinite((lower, upper)).all() or lower <= 0.0 or lower >= upper:
        raise ValueError("Limits must be finite, positive, and increasing")
    return lower, upper


def sample_segment(
    start: Point, end: Point, *, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(z, x)`` samples along a segment and their distance from its start in mm."""
    if count < 2:
        raise ValueError("line sample count must be at least 2")
    start_point = _as_point(start, "Line start")
    end_point = _as_point(end, "Line end")
    if np.array_equal(start_point, end_point):
        raise ValueError("Line endpoints must be distinct")
    fractions = np.linspace(0.0, 1.0, count)
    points = start_point + fractions[:, np.newaxis] * (end_point - start_point)
    distance_mm = fractions * np.linalg.norm(end_point - start_point) * MM_PER_M
    return points, distance_mm


def normalize_lineout_values(values: np.ndarray) -> np.ndarray:
    """Scale one lineout to unit magnitude without shifting its zero baseline."""
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return values.copy()
    magnitude = float(np.max(np.abs(finite_values)))
    if magnitude == 0.0:
        return values.copy()
    return values / magnitude


# ---------------------------------------------------------------------------
# Interactive controller
# ---------------------------------------------------------------------------


class InteractiveShockEvaluator:
    """Coordinate the heatmap selector, live lineout panel, and shock report."""

    def __init__(
        self,
        data: CgnsScalarFieldData,
        *,
        line_samples: int = 400,
        gamma: float = 1.4,
        shock_angle: str = "auto",
        show_color_limit_boxes: bool = False,
        title: str | None = None,
    ) -> None:
        if line_samples < 2:
            raise ValueError("line_samples must be at least 2")
        self.data = data
        self.line_samples = line_samples
        self.gamma = validate_gamma(gamma)
        self.shock_angle = validate_shock_angle_mode(shock_angle)
        self.show_color_limit_boxes = show_color_limit_boxes
        self.gradient_magnitude = density_gradient_magnitude(data.fields)
        self.limits = initial_log_limits(self.gradient_magnitude)
        self.title = title or "Density-gradient magnitude"
        self.triangulation = Triangulation(data.z, data.x)
        delaunay = Delaunay(np.column_stack((data.z, data.x)))
        self._samplers = {
            name: LinearNDInterpolator(delaunay, values, fill_value=np.nan)
            for name, values in data.fields.items()
        }
        self.segment: tuple[Point, Point] | None = None
        self._drag_start: Point | None = None
        self._third_point: Point | None = None
        self.shock_evaluation: ShockEvaluation | None = None

        self.figure, self.heatmap_axes, self.lineout_axes = self._create_figure()

    def _create_figure(self) -> tuple[Figure, Axes, Axes]:
        figure = plt.figure(figsize=(17, 10))
        heatmap_axes = figure.add_axes((0.06, 0.48, 0.43, 0.43))
        lineout_axes = figure.add_axes((0.56, 0.48, 0.38, 0.43))
        checkbox_axes = figure.add_axes((0.76, 0.08, 0.18, 0.30))

        self.heatmap = heatmap_axes.tripcolor(
            self.triangulation,
            self.gradient_magnitude,
            shading="gouraud",
            cmap="magma",
            norm=colors.LogNorm(*self.limits),
        )
        self.colorbar = figure.colorbar(
            self.heatmap, ax=heatmap_axes, label=r"$|\nabla \rho|$ [source units]"
        )
        heatmap_axes.set_xlabel("z [mm]")
        heatmap_axes.set_ylabel("x [mm]")
        heatmap_axes.set_title(self.title)
        heatmap_axes.set_aspect("equal")
        heatmap_axes.xaxis.set_major_formatter(lambda value, _: f"{value * MM_PER_M:g}")
        heatmap_axes.yaxis.set_major_formatter(lambda value, _: f"{value * MM_PER_M:g}")

        self.lower_limit_box: TextBox | None = None
        self.upper_limit_box: TextBox | None = None
        if self.show_color_limit_boxes:
            self.lower_limit_box = TextBox(
                figure.add_axes((0.07, 0.39, 0.18, 0.04)),
                "lower limit",
                initial=f"{self.limits[0]:.6g}",
            )
            self.upper_limit_box = TextBox(
                figure.add_axes((0.31, 0.39, 0.18, 0.04)),
                "upper limit",
                initial=f"{self.limits[1]:.6g}",
            )
            self.lower_limit_box.on_submit(self._on_limits_submitted)
            self.upper_limit_box.on_submit(self._on_limits_submitted)

        self.status_text = figure.text(
            0.06, 0.44, "Drag across the map to sample a lineout."
        )
        self.segment_artist = Line2D([], [], color="cyan", linewidth=1.5)
        self.shock_direction_artist = Line2D(
            [], [], color="lime", linestyle="--", linewidth=1.5
        )
        self.shock_direction_point_artist = Line2D(
            [], [], color="lime", marker="o", linestyle="None"
        )
        for artist in (
            self.segment_artist,
            self.shock_direction_artist,
            self.shock_direction_point_artist,
        ):
            heatmap_axes.add_line(artist)
        figure.canvas.mpl_connect("button_press_event", self._on_press)
        figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        figure.canvas.mpl_connect("button_release_event", self._on_release)

        field_names = list(self.data.fields)
        self.field_checkboxes = CheckButtons(
            checkbox_axes, field_names, [name == "Density" for name in field_names]
        )
        self.field_checkboxes.on_clicked(self._on_field_selection_changed)
        checkbox_axes.set_title("Lineout fields", fontsize=10)
        for label in self.field_checkboxes.labels:
            label.set_fontsize(8)

        self.lineout_axes = lineout_axes
        self._style_lineout_axes()
        lineout_axes.text(
            0.5,
            0.5,
            "Drag a segment on the heatmap to display checked fields.",
            ha="center",
            va="center",
            transform=lineout_axes.transAxes,
        )
        self.shock_report_text = figure.text(
            0.06,
            0.06,
            "Shock ratios will appear after selecting a segment.",
            family="monospace",
            fontsize=11,
            va="bottom",
        )
        return figure, heatmap_axes, lineout_axes

    def _style_lineout_axes(self) -> None:
        self.lineout_axes.set_xlabel("distance from segment start [mm]")
        self.lineout_axes.set_ylabel("field value [source units]")
        self.lineout_axes.grid()

    def _redraw(self) -> None:
        self.figure.canvas.draw_idle()

    def _on_limits_submitted(self, _: str) -> None:
        assert self.lower_limit_box is not None and self.upper_limit_box is not None
        try:
            self.limits = parse_log_limits(
                self.lower_limit_box.text, self.upper_limit_box.text
            )
        except ValueError as exc:
            self.status_text.set_text(str(exc))
            self._redraw()
            return
        self.heatmap.set_norm(colors.LogNorm(*self.limits))
        self.colorbar.update_normal(self.heatmap)
        self.status_text.set_text("Applied logarithmic limits.")
        self._redraw()

    def _event_point(self, event: MouseEvent) -> Point | None:
        if (
            event.inaxes is not self.heatmap_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return None
        return float(event.xdata), float(event.ydata)

    def _on_press(self, event: MouseEvent) -> None:
        if event.button != 1:
            return
        awaiting_third_point = (
            self.shock_angle == "manual"
            and self.segment is not None
            and self._third_point is None
        )
        if awaiting_third_point:
            third_point = self._event_point(event)
            if third_point is None:
                return
            self._third_point = third_point
            self._update_shock_evaluation()
            self._update_lineout()
            self._redraw()
            return
        self._drag_start = self._event_point(event)

    def _on_motion(self, event: MouseEvent) -> None:
        if self._drag_start is None:
            return
        end = self._event_point(event)
        if end is None:
            return
        self.segment_artist.set_data(
            (self._drag_start[0], end[0]), (self._drag_start[1], end[1])
        )
        self._redraw()

    def _on_release(self, event: MouseEvent) -> None:
        if event.button != 1 or self._drag_start is None:
            return
        end = self._event_point(event)
        start = self._drag_start
        self._drag_start = None
        if end is None or np.array_equal(start, end):
            self.status_text.set_text("Select two distinct points inside the heatmap.")
            self._redraw()
            return
        self.segment = (start, end)
        self._third_point = None
        self.segment_artist.set_data((start[0], end[0]), (start[1], end[1]))
        if self.shock_angle == "manual":
            self._clear_shock_evaluation(
                "Click a third point on the heatmap to select the shock direction."
            )
            self.status_text.set_text(
                "Segment selected; click a third point for shock direction."
            )
        else:
            self._update_shock_evaluation()
        self._update_lineout()
        self._redraw()

    def _on_field_selection_changed(self, _: str) -> None:
        if self.segment is not None:
            self._update_lineout()

    def _selected_field_names(self) -> list[str]:
        return [
            name
            for name, selected in zip(
                self.data.fields, self.field_checkboxes.get_status(), strict=True
            )
            if selected
        ]

    def _sample_state(self, point: Point) -> dict[str, float]:
        """Sample every shock-state field at one ``(z, x)`` point inside the mesh."""
        values = {}
        for field_name in SHOCK_STATE_FIELD_NAMES:
            value = float(np.asarray(self._samplers[field_name]([point])).item())
            if not np.isfinite(value):
                raise ValueError(f"{field_name} is outside the interpolation domain")
            values[field_name] = value
        return values

    @staticmethod
    def _velocity(values: Mapping[str, float]) -> np.ndarray:
        return np.array((values["Axial_Velocity"], values["Radial_Velocity"]))

    def _clear_shock_evaluation(self, message: str) -> None:
        self.shock_evaluation = None
        self.shock_direction_artist.set_data([], [])
        self.shock_direction_point_artist.set_data([], [])
        self.shock_report_text.set_text(message)

    def _update_shock_evaluation(self) -> None:
        """Calculate and display endpoint shock ratios for the selected segment."""
        assert self.segment is not None
        missing_fields = [
            name for name in SHOCK_STATE_FIELD_NAMES if name not in self._samplers
        ]
        if missing_fields:
            self._clear_shock_evaluation(
                "Shock evaluation unavailable: missing " + ", ".join(missing_fields)
            )
            self.status_text.set_text(
                "Lineout updated; shock-state fields are unavailable."
            )
            return

        start, end = self.segment
        try:
            start_values = self._sample_state(start)
            end_values = self._sample_state(end)
            # In every mode the endpoint with the larger Mach number is upstream, so the
            # reported ratios are downstream/upstream regardless of drag direction.
            start_is_upstream, upstream_endpoint = select_upstream_endpoint(
                start_values["Mach"], end_values["Mach"]
            )
            upstream_values, downstream_values = (
                (start_values, end_values)
                if start_is_upstream
                else (end_values, start_values)
            )
            beta_rad: float | None = None

            if self.shock_angle == "perp":
                beta_rad, shock_direction = perpendicular_segment_shock_direction(
                    start, end, self._velocity(upstream_values)
                )
                angle_source = "segment normal"
            elif self.shock_angle == "manual":
                if self._third_point is None:
                    raise ValueError("Select a third point for the shock direction")
                beta_rad, shock_direction = third_point_shock_direction(
                    start, end, self._third_point, self._velocity(upstream_values)
                )
                angle_source = "third point"
            else:
                shock_direction = (
                    self._velocity(upstream_values) - self._velocity(downstream_values)
                ) @ np.array([[0, -1], [1, 0]])
                angle_source = "velocity change"

            self.shock_evaluation = evaluate_oblique_shock(
                gamma=self.gamma,
                mach_upstream=upstream_values["Mach"],
                upstream_velocity=self._velocity(upstream_values),
                downstream_velocity=self._velocity(downstream_values),
                upstream_state=ShockState.from_fields(upstream_values),
                downstream_state=ShockState.from_fields(downstream_values),
                beta_rad=beta_rad,
                angle_source=angle_source,
                upstream_endpoint=upstream_endpoint,
            )
        except ValueError as exc:
            self._clear_shock_evaluation(f"Shock evaluation unavailable: {exc}")
            self.status_text.set_text(
                "Lineout updated; select endpoints inside the field."
            )
            return

        self._draw_shock_direction(
            start,
            end,
            shock_direction,
            direction_point=self._third_point if self.shock_angle == "manual" else None,
        )
        self.shock_report_text.set_text(format_shock_report(self.shock_evaluation))
        self.status_text.set_text(
            f"Lineout updated; beta = {np.rad2deg(self.shock_evaluation.beta_rad):.2f} degrees."
        )

    def _draw_shock_direction(
        self,
        start: Point,
        end: Point,
        physical_direction: np.ndarray,
        *,
        direction_point: Point | None = None,
    ) -> None:
        """Draw the physical ``(axial, radial)`` direction from the segment midpoint."""
        start_point, end_point = np.asarray(start), np.asarray(end)
        midpoint = 0.5 * (start_point + end_point)
        plot_direction = _to_physical(physical_direction)
        plot_direction /= np.linalg.norm(plot_direction)
        line_end = (
            np.asarray(direction_point)
            if direction_point is not None
            else midpoint
            + float(np.linalg.norm(end_point - start_point)) * plot_direction
        )
        self.shock_direction_artist.set_data(
            (midpoint[0], line_end[0]), (midpoint[1], line_end[1])
        )
        if direction_point is None:
            self.shock_direction_point_artist.set_data([], [])
        else:
            self.shock_direction_point_artist.set_data((line_end[0],), (line_end[1],))

    def _update_lineout(self) -> None:
        if self.segment is None:
            return
        points, distance_mm = sample_segment(*self.segment, count=self.line_samples)
        self.lineout_axes.clear()
        self._style_lineout_axes()
        selected_names = self._selected_field_names()
        if not selected_names:
            self.lineout_axes.text(
                0.5,
                0.5,
                "Select one or more fields to plot.",
                ha="center",
                va="center",
                transform=self.lineout_axes.transAxes,
            )
        else:
            normalize = len(selected_names) > 1
            if normalize:
                self.lineout_axes.set_ylabel(
                    "independently magnitude-normalized field value"
                )
                self.lineout_axes.set_title(
                    "Selected fields independently scaled to unit magnitude"
                )
            for field_name in selected_names:
                values = np.asarray(self._samplers[field_name](points))
                if normalize:
                    values = normalize_lineout_values(values)
                self.lineout_axes.plot(distance_mm, values, label=field_name)
            self.lineout_axes.legend()
        self._redraw()

    def show(self) -> None:
        """Run the GUI event loop while retaining this controller's callbacks."""
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively evaluate density-gradient shocks in one CGNS field."
    )
    parser.add_argument("cgns_path", type=Path, help="Input unstructured CGNS field")
    parser.add_argument(
        "--line-samples",
        type=int,
        default=400,
        help="Number of samples per selected segment (default: 400)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.4,
        help="Heat-capacity ratio for oblique-shock predictions (default: 1.4)",
    )
    parser.add_argument(
        "--shock-angle",
        choices=SHOCK_ANGLE_MODES,
        default="auto",
        help="auto: from endpoint velocities; perp: segment normal; manual: third click "
        "from the segment midpoint (default: auto)",
    )
    parser.add_argument(
        "--show-color-limit-boxes",
        action="store_true",
        help="Show editable lower and upper heatmap color-limit text boxes",
    )
    parser.add_argument(
        "--flow-solution",
        default=DEFAULT_FLOW_SOLUTION,
        help=f"CGNS flow-solution group (default: {DEFAULT_FLOW_SOLUTION})",
    )
    parser.add_argument(
        "--x-coordinate",
        default=DEFAULT_X_COORDINATE,
        help=f"CGNS transverse coordinate name (default: {DEFAULT_X_COORDINATE})",
    )
    parser.add_argument(
        "--z-coordinate",
        default=DEFAULT_Z_COORDINATE,
        help=f"CGNS axial coordinate name (default: {DEFAULT_Z_COORDINATE})",
    )
    parser.add_argument(
        "--x-bounds",
        nargs=2,
        type=float,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="Optional transverse x bounds [mm]",
    )
    parser.add_argument(
        "--z-bounds",
        nargs=2,
        type=float,
        metavar=("MIN_MM", "MAX_MM"),
        default=None,
        help="Optional axial z bounds [mm]",
    )
    args = parser.parse_args()

    try:
        data = load_cgns_scalar_fields(
            args.cgns_path,
            flow_solution=args.flow_solution,
            x_coordinate=args.x_coordinate,
            z_coordinate=args.z_coordinate,
            x_bounds=mm_bounds_to_m(args.x_bounds),
            z_bounds=mm_bounds_to_m(args.z_bounds),
        )
        evaluator = InteractiveShockEvaluator(
            data,
            line_samples=args.line_samples,
            gamma=args.gamma,
            shock_angle=args.shock_angle,
            show_color_limit_boxes=args.show_color_limit_boxes,
            title=args.cgns_path.name,
        )
    except (CgnsDataError, ValueError) as exc:
        parser.error(str(exc))
    evaluator.show()


if __name__ == "__main__":
    main()
