from __future__ import annotations

import gc
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from fludat_proc.cgns_io import (
    CgnsDataError,
    CgnsScalarFieldData,
    load_cgns_scalar_fields,
)
from fludat_proc.shock_evaluator import (
    InteractiveShockEvaluator,
    ShockEvaluation,
    ShockState,
    density_gradient_magnitude,
    evaluate_oblique_shock,
    format_shock_report,
    initial_log_limits,
    main,
    normalize_lineout_values,
    oblique_p2p1,
    parse_log_limits,
    perpendicular_segment_shock_direction,
    sample_segment,
    select_upstream_endpoint,
    third_point_shock_direction,
    validate_gamma,
    validate_shock_angle_mode,
)

from .conftest import write_cgns_file


def write_shock_field(path, *, include_gradients=True):
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    z = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    fields = {
        "Density": x + 2.0 * z + 1.0,
        "Temperature": 300.0 + x - z,
        "Mach": 3.0 - z,  # decreases from the drag start (z = 0) to the drag end
        "Axial_Velocity": 100.0 - 50.0 * z,
        "Radial_Velocity": 50.0 * z,
        "Pressure": 1.0 + z,
        "CoordinateX": x,
        "InvalidLength": x[:3],
    }
    if include_gradients:
        fields["dp-dX"] = np.ones_like(x)
        fields["dp-dY"] = np.full_like(x, 2.0)
    return write_cgns_file(path, x=x, z=z, fields=fields)


def square_field_data(mach):
    x = np.array([0.0, 1.0, 0.0, 1.0])
    return CgnsScalarFieldData(
        x=x,
        z=np.array([0.0, 0.0, 1.0, 1.0]),
        fields={
            "Density": 1.0 + x,
            "dp-dX": np.ones(4),
            "dp-dY": np.ones(4),
            "Mach": mach,
            "Axial_Velocity": np.full(4, 100.0),
            "Radial_Velocity": np.zeros(4),
            "Pressure": 1.0 + x,
            "Temperature": 1.0 + x,
        },
    )


def test_load_discovers_point_aligned_fields_and_ignores_metadata(tmp_path):
    data = load_cgns_scalar_fields(write_shock_field(tmp_path / "field.cgns"))

    assert set(data.fields) == {
        "Density",
        "Temperature",
        "Mach",
        "Axial_Velocity",
        "Radial_Velocity",
        "Pressure",
        "dp-dX",
        "dp-dY",
    }
    np.testing.assert_array_equal(data.x, [0.0, 1.0, 0.0, 1.0, 0.5])


def test_load_clips_all_fields_to_requested_bounds(tmp_path):
    path = write_shock_field(tmp_path / "field.cgns")

    data = load_cgns_scalar_fields(path, x_bounds=(0.5, 1.0), z_bounds=(0.0, 1.0))

    np.testing.assert_array_equal(data.x, [1.0, 1.0, 0.5])
    assert data.fields["Density"].size == 3
    with pytest.raises(CgnsDataError, match="fewer than 3"):
        load_cgns_scalar_fields(path, x_bounds=(0.75, 1.0))
    with pytest.raises(ValueError, match="finite and increasing"):
        load_cgns_scalar_fields(path, x_bounds=(1.0, 0.0))


def test_gradient_magnitude_requires_both_components():
    np.testing.assert_array_equal(
        density_gradient_magnitude(
            {"dp-dX": np.array([3.0]), "dp-dY": np.array([4.0])}
        ),
        [5.0],
    )
    with pytest.raises(CgnsDataError, match="dp-dY"):
        density_gradient_magnitude({"dp-dX": np.array([3.0])})


def test_log_limits_require_positive_increasing_values():
    lower, upper = initial_log_limits(np.array([0.0, 1.0, 10.0, np.nan]))
    assert 0.0 < lower < upper
    assert parse_log_limits("1e-3", "2.5") == (1e-3, 2.5)
    for limits in (("0", "2"), ("3", "2"), ("not a number", "2")):
        with pytest.raises(ValueError):
            parse_log_limits(*limits)
    with pytest.raises(CgnsDataError, match="no positive"):
        initial_log_limits(np.array([0.0, np.nan]))


def test_segment_samples_follow_distance_in_millimetres():
    points, distance_mm = sample_segment((0.0, 0.0), (0.003, 0.004), count=3)

    np.testing.assert_allclose(points, [[0.0, 0.0], [0.0015, 0.002], [0.003, 0.004]])
    np.testing.assert_allclose(distance_mm, [0.0, 2.5, 5.0])
    with pytest.raises(ValueError, match="distinct"):
        sample_segment((0.0, 0.0), (0.0, 0.0), count=3)


def test_normalize_lineout_values_preserves_gaps_and_zero_baseline():
    np.testing.assert_allclose(
        normalize_lineout_values(np.array([0.0, 10.0, np.nan, 20.0])),
        [0.0, 0.5, np.nan, 1.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        normalize_lineout_values(np.array([-4.0, 0.0, 2.0, np.nan])),
        [-1.0, 0.0, 0.5, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        normalize_lineout_values(np.array([0.0, 0.0])), [0.0, 0.0]
    )


def test_oblique_relations_validate_gamma():
    assert validate_gamma(1.67) == 1.67
    with pytest.raises(ValueError, match="greater than 1"):
        validate_gamma(1.0)
    with pytest.raises(ValueError, match="greater than 1"):
        oblique_p2p1(1.0, 2.0, np.pi / 2)
    assert validate_shock_angle_mode("auto") == "auto"
    with pytest.raises(ValueError, match="auto, perp, manual"):
        validate_shock_angle_mode("invalid")


def test_oblique_shock_evaluation_calculates_angle_and_downstream_over_upstream_ratios():
    result = evaluate_oblique_shock(
        gamma=1.4,
        mach_upstream=2.0,
        upstream_velocity=np.array([100.0, 0.0]),
        downstream_velocity=np.array([50.0, 50.0]),
        upstream_state=ShockState(pressure=1.0, density=1.0, temperature=1.0),
        downstream_state=ShockState(pressure=2.0, density=1.5, temperature=1.25),
    )

    assert np.rad2deg(result.beta_rad) == pytest.approx(45.0)
    assert result.pressure_ratio_simulated == 2.0
    assert result.density_ratio_simulated == 1.5
    assert result.temperature_ratio_simulated == 1.25
    assert result.pressure_ratio_predicted == pytest.approx(2.166666666666666)
    assert result.density_ratio_predicted == pytest.approx(1.714285714285714)
    assert result.temperature_ratio_predicted == pytest.approx(1.2638888888888888)


def test_oblique_shock_evaluation_rejects_zero_upstream_state():
    with pytest.raises(ValueError, match="Upstream pressure"):
        evaluate_oblique_shock(
            gamma=1.4,
            mach_upstream=2.0,
            upstream_velocity=np.array([100.0, 0.0]),
            downstream_velocity=np.array([50.0, 50.0]),
            upstream_state=ShockState(0.0, 1.0, 1.0),
            downstream_state=ShockState(2.0, 1.5, 1.25),
        )


def test_shock_report_uses_aligned_fixed_point_columns():
    evaluation = ShockEvaluation(
        gamma=1.532,
        mach_upstream=3.3076,
        beta_rad=np.deg2rad(88.083),
        velocity_change=np.array([1.0, 2.0]),
        pressure_ratio_simulated=0.022930661,
        density_ratio_simulated=0.084918933,
        temperature_ratio_simulated=0.270043084,
        pressure_ratio_predicted=0.36077325,
        density_ratio_predicted=0.5306587,
        temperature_ratio_predicted=0.6798593,
    )

    header, *data_rows = format_shock_report(evaluation).splitlines()[2:]

    assert header[:23].strip() == "ratio (2 = downstream)"
    assert header[23:41].strip() == "simulated"
    assert [row[:23].strip() for row in data_rows] == ["P2/P1", "rho2/rho1", "T2/T1"]
    for row in data_rows:
        for column in (row[23:41], row[41:59], row[59:77]):
            assert len(column) == 18
            assert column.strip() == f"{float(column):.2f}"
    assert data_rows[0][23:41].strip() == "0.02"
    assert data_rows[0][41:59].strip() == "0.36"
    assert float(data_rows[0][59:77]) == pytest.approx(
        abs(0.022930661 - 0.36077325) / 0.36077325 * 100.0, abs=0.005
    )


def test_perpendicular_mode_selects_higher_mach_endpoint_as_upstream():
    assert select_upstream_endpoint(1.0, 3.0) == (False, "drag end")
    assert select_upstream_endpoint(3.0, 1.0) == (True, "drag start")
    with pytest.raises(ValueError, match="equal Mach"):
        select_upstream_endpoint(2.0, 2.0)

    beta_rad, direction = perpendicular_segment_shock_direction(
        (0.0, 0.0), (0.0, 1.0), np.array([100.0, 0.0])
    )
    assert np.rad2deg(beta_rad) == pytest.approx(90.0)
    np.testing.assert_allclose(direction, [0.0, 1.0])


def test_third_point_shock_direction_uses_midpoint_ray():
    beta_rad, direction = third_point_shock_direction(
        (0.0, 0.0), (0.0, 2.0), (1.0, 1.0), np.array([100.0, 0.0])
    )

    assert np.rad2deg(beta_rad) == pytest.approx(90.0)
    np.testing.assert_allclose(direction, [0.0, 1.0])
    with pytest.raises(ValueError, match="differ from the segment midpoint"):
        third_point_shock_direction(
            (0.0, 0.0), (0.0, 2.0), (0.0, 1.0), np.array([100.0, 0.0])
        )


def test_controller_updates_checked_lineouts_from_selected_segment(tmp_path):
    viewer = InteractiveShockEvaluator(
        load_cgns_scalar_fields(write_shock_field(tmp_path / "field.cgns")),
        line_samples=5,
        show_color_limit_boxes=True,
    )

    viewer.segment = ((0.0, 0.0), (1.0, 1.0))
    viewer._update_lineout()
    assert [line.get_label() for line in viewer.lineout_axes.lines] == ["Density"]
    np.testing.assert_allclose(
        viewer.lineout_axes.lines[0].get_ydata(), np.linspace(1.0, 4.0, 5)
    )

    viewer.field_checkboxes.set_active(list(viewer.data.fields).index("Temperature"))
    assert len(viewer.lineout_axes.lines) == 2
    assert (
        viewer.lineout_axes.get_ylabel()
        == "independently magnitude-normalized field value"
    )
    for line in viewer.lineout_axes.lines:
        finite = line.get_ydata()[np.isfinite(line.get_ydata())]
        assert 0.0 <= finite.min() and finite.max() <= 1.0

    viewer.lower_limit_box.set_val("1")
    viewer.upper_limit_box.set_val("10")
    assert viewer.limits == (1.0, 10.0)


def test_color_limit_boxes_are_hidden_unless_requested(tmp_path):
    data = load_cgns_scalar_fields(write_shock_field(tmp_path / "field.cgns"))

    assert InteractiveShockEvaluator(data).lower_limit_box is None
    assert (
        InteractiveShockEvaluator(data, show_color_limit_boxes=True).upper_limit_box
        is not None
    )


def test_controller_reports_shock_state_after_segment_selection(tmp_path):
    viewer = InteractiveShockEvaluator(
        load_cgns_scalar_fields(write_shock_field(tmp_path / "field.cgns")),
        line_samples=5,
        gamma=1.67,
    )
    viewer.segment = ((0.0, 0.0), (1.0, 1.0))

    viewer._update_shock_evaluation()

    assert viewer.shock_evaluation is not None
    assert viewer.shock_evaluation.gamma == 1.67
    assert np.rad2deg(viewer.shock_evaluation.beta_rad) == pytest.approx(45.0)
    assert len(viewer.shock_direction_artist.get_xdata()) == 2
    assert viewer.shock_evaluation.upstream_endpoint == "drag start"
    assert viewer.shock_evaluation.pressure_ratio_simulated == pytest.approx(2.0)
    assert "upstream (1) = drag start" in viewer.shock_report_text.get_text()


def test_controller_reports_missing_state_fields(tmp_path):
    path = write_cgns_file(
        tmp_path / "field.cgns",
        x=np.array([0.0, 1.0, 0.0, 1.0]),
        z=np.array([0.0, 0.0, 1.0, 1.0]),
        fields={"Density": np.ones(4), "dp-dX": np.ones(4), "dp-dY": np.ones(4)},
    )
    viewer = InteractiveShockEvaluator(load_cgns_scalar_fields(path))
    viewer.segment = ((0.0, 0.0), (1.0, 1.0))

    viewer._update_shock_evaluation()

    assert viewer.shock_evaluation is None
    assert "missing Mach" in viewer.shock_report_text.get_text()


@pytest.mark.parametrize("shock_angle", ["auto", "perp"])
def test_controller_reports_downstream_over_upstream_regardless_of_drag_direction(
    shock_angle,
):
    x = np.array([0.0, 1.0, 0.0, 1.0])
    data = square_field_data(mach=1.0 + 2.0 * x)
    # Mach and pressure both grow with x, so x = 1 is upstream whichever way we drag.
    data.fields["Axial_Velocity"] = 100.0 + 100.0 * x
    data.fields["Radial_Velocity"] = 20.0 * (1.0 - x)
    results = {}
    for direction, segment in (
        ("forward", ((0.0, 0.0), (0.0, 1.0))),
        ("reverse", ((0.0, 1.0), (0.0, 0.0))),
    ):
        viewer = InteractiveShockEvaluator(data, shock_angle=shock_angle)
        viewer.segment = segment
        viewer._update_shock_evaluation()
        assert viewer.shock_evaluation is not None, viewer.shock_report_text.get_text()
        results[direction] = viewer.shock_evaluation

    assert results["forward"].upstream_endpoint == "drag end"
    assert results["reverse"].upstream_endpoint == "drag start"
    for evaluation in results.values():
        assert evaluation.mach_upstream == 3.0
        assert evaluation.pressure_ratio_simulated == pytest.approx(0.5)
        assert evaluation.density_ratio_simulated == pytest.approx(0.5)
    assert results["forward"].beta_rad == pytest.approx(results["reverse"].beta_rad)
    assert results["forward"].pressure_ratio_predicted == pytest.approx(
        results["reverse"].pressure_ratio_predicted
    )
    if shock_angle == "perp":
        assert np.rad2deg(results["forward"].beta_rad) == pytest.approx(90.0, abs=15.0)
        assert results["forward"].angle_source == "segment normal"


def test_controller_third_point_mode_waits_then_evaluates_direction():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    viewer = InteractiveShockEvaluator(
        square_field_data(mach=3.0 - x), shock_angle="manual"
    )
    viewer.segment = ((0.0, 0.0), (0.0, 1.0))
    viewer._third_point = None

    viewer._update_shock_evaluation()
    assert viewer.shock_evaluation is None
    assert "third point" in viewer.shock_report_text.get_text()

    viewer._third_point = (1.0, 0.5)
    viewer._update_shock_evaluation()
    assert viewer.shock_evaluation is not None
    assert viewer.shock_evaluation.angle_source == "third point"
    assert np.rad2deg(viewer.shock_evaluation.beta_rad) == pytest.approx(90.0)
    np.testing.assert_allclose(viewer.shock_direction_artist.get_xdata(), [0.0, 1.0])
    np.testing.assert_allclose(viewer.shock_direction_point_artist.get_ydata(), [0.5])


def test_manual_third_point_click_refreshes_lineout_window():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    viewer = InteractiveShockEvaluator(
        square_field_data(mach=3.0 - x), shock_angle="manual"
    )
    viewer.segment = ((0.0, 0.0), (0.0, 1.0))
    viewer.figure.canvas.draw()
    pixel = viewer.heatmap_axes.transData.transform((1.0, 0.5))
    event = MouseEvent("button_press_event", viewer.figure.canvas, *pixel, button=1)

    with patch.object(viewer, "_update_lineout") as update_lineout:
        viewer._on_press(event)

    update_lineout.assert_called_once_with()
    assert viewer.shock_evaluation is not None


def test_main_retains_drag_callbacks_through_show(tmp_path):
    path = write_shock_field(tmp_path / "field.cgns")

    def exercise_drag():
        gc.collect()
        assert len(plt.get_fignums()) == 1
        figure = plt.figure(plt.get_fignums()[0])
        heatmap_axes = figure.axes[0]
        canvas = figure.canvas
        canvas.draw()
        start = heatmap_axes.transData.transform((0.0, 0.0))
        end = heatmap_axes.transData.transform((1.0, 1.0))
        for event_name, position in (
            ("button_press_event", start),
            ("motion_notify_event", end),
            ("button_release_event", end),
        ):
            canvas.callbacks.process(
                event_name, MouseEvent(event_name, canvas, *position, button=1)
            )
        assert len(figure.axes[1].lines) == 1

    with (
        patch.object(
            sys, "argv", ["shock_evaluator", str(path), "--x-bounds", "-1", "2000"]
        ),
        patch("fludat_proc.shock_evaluator.plt.show", side_effect=exercise_drag),
    ):
        main()
