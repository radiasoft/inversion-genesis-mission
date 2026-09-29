from __future__ import annotations

import csv
import sys
from unittest.mock import patch

import numpy as np
import pytest

from fludat_proc import convergence
from fludat_proc.convergence import (
    ConvergenceSummary,
    ErrorDistribution,
    calculate_convergence,
    calculate_field_convergence,
    load_resolution_cgns_fields,
    load_resolution_lineouts,
    main,
    plot_convergence,
    pool_distributions,
    reference_pairs,
    relative_integrated_difference,
    summarize_convergence,
    write_convergence_csv,
)

from .conftest import write_density_cgns, write_lineout_file


def _write_three_resolutions(input_dir):
    write_lineout_file(input_dir / "0_1.txt", {"x-0-0": [(0.0, 0.0), (1.0, 3.0)]})
    write_lineout_file(input_dir / "0_05.txt", {"x-0-0": [(0.0, 0.0), (1.0, 2.0)]})
    write_lineout_file(input_dir / "0_02.txt", {"x-0-0": [(0.0, 0.0), (1.0, 1.0)]})


def test_reference_pairs():
    assert reference_pairs([3, 1, 2], grid_size=float, reference_mode="finest") == [
        (3, 1),
        (2, 1),
    ]
    assert reference_pairs([3, 1, 2], grid_size=float, reference_mode="adjacent") == [
        (3, 2),
        (2, 1),
    ]
    with pytest.raises(ValueError, match="Unknown convergence reference mode"):
        reference_pairs([1, 2], grid_size=float, reference_mode="nope")
    with pytest.raises(ValueError, match="At least two"):
        reference_pairs([1], grid_size=float, reference_mode="finest")


def test_relative_integrated_difference_interpolates_shared_grid():
    coarse = np.array([[0.0, 0.0], [1.0, 2.0]])
    finer = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    error, z_min, z_max = relative_integrated_difference(coarse, finer)

    assert error == pytest.approx(1.0)
    assert (z_min, z_max) == (0.0, 1.0)


def test_lineout_convergence_against_finest_and_adjacent(tmp_path):
    _write_three_resolutions(tmp_path)
    (tmp_path / "convergence_metrics.csv").write_text("ignored non-resolution file")
    resolutions = load_resolution_lineouts(tmp_path)

    finest = calculate_convergence(resolutions)
    adjacent = calculate_convergence(resolutions, reference_mode="adjacent")

    assert [(m.coarse_grid_size_mm, m.finer_grid_size_mm) for m in finest] == [
        (0.1, 0.02),
        (0.05, 0.02),
    ]
    assert [m.relative_error for m in finest] == [2.0, 1.0]
    assert [(m.coarse_grid_size_mm, m.finer_grid_size_mm) for m in adjacent] == [
        (0.1, 0.05),
        (0.05, 0.02),
    ]
    assert [m.relative_error for m in adjacent] == [0.5, 1.0]


def test_load_rejects_inconsistent_lineout_labels(tmp_path):
    write_lineout_file(tmp_path / "0_1.txt", {"x-0-0": [(0.0, 1.0), (1.0, 1.0)]})
    write_lineout_file(tmp_path / "0_05.txt", {"x-0-5": [(0.0, 1.0), (1.0, 1.0)]})

    with pytest.raises(ValueError, match="do not match"):
        load_resolution_lineouts(tmp_path)


def test_cgns_fields_support_x_clipping_and_triangulate_once(tmp_path):
    write_density_cgns(tmp_path / "0_3.cgns", density_scale=1.0)
    write_density_cgns(tmp_path / "0_2.cgns", density_scale=1.1)
    write_density_cgns(tmp_path / "0_1.cgns", density_scale=1.2)

    with patch(
        "fludat_proc.convergence.Delaunay", wraps=convergence.Delaunay
    ) as delaunay:
        metrics = calculate_field_convergence(
            load_resolution_cgns_fields(tmp_path), x_min=0.0, x_points=20, z_points=20
        )

    assert delaunay.call_count == 3
    assert [m.lineout_label for m in metrics] == ["field", "field"]
    assert (metrics[0].x_min_m, metrics[0].x_max_m) == (0.0, 1.0)
    assert metrics[0].relative_error == pytest.approx(0.2 / 1.2)
    assert metrics[1].relative_error == pytest.approx(0.1 / 1.2)
    assert metrics[0].local_relative_errors.mean == pytest.approx(0.2 / 1.2)
    assert metrics[0].peak_normalized_errors is not None


def test_cgns_convergence_supports_cubic_interpolation(tmp_path):
    write_density_cgns(tmp_path / "0_2.cgns", density_scale=1.0)
    write_density_cgns(tmp_path / "0_1.cgns", density_scale=1.1)

    metrics = calculate_field_convergence(
        load_resolution_cgns_fields(tmp_path),
        x_min=0.0,
        x_points=20,
        z_points=20,
        interpolation="cubic",
    )

    assert len(metrics) == 1
    assert metrics[0].relative_error > 0.0


def test_pool_distributions_is_exact():
    a = ErrorDistribution.from_samples([1.0, 3.0])
    b = ErrorDistribution.from_samples([6.0, 8.0, 10.0])

    pooled = pool_distributions([a, b])
    expected = ErrorDistribution.from_samples([1.0, 3.0, 6.0, 8.0, 10.0])

    assert pooled.sample_count == 5
    assert pooled.mean == pytest.approx(expected.mean)
    assert pooled.standard_deviation == pytest.approx(expected.standard_deviation)
    assert (pooled.minimum, pooled.maximum) == (1.0, 10.0)


def test_peak_normalized_errors_divide_by_reference_peak():
    coarse = np.array([[1.0, 3.0], [6.0, 8.0]])
    finer = np.array([[2.0, 5.0], [10.0, 14.0]])

    distribution = convergence._peak_normalized_errors(coarse, finer)

    assert distribution.sample_count == 4
    assert distribution.mean == pytest.approx(3.25 / 14.0)
    assert distribution.standard_deviation == pytest.approx(np.sqrt(3.6875) / 14.0)


def test_cgns_summary_pools_pointwise_errors_and_selects_plot_curves(tmp_path):
    write_density_cgns(tmp_path / "0_2.cgns", density_scale=1.0)
    write_density_cgns(tmp_path / "0_1.cgns", density_scale=1.0)
    fields = load_resolution_cgns_fields(tmp_path)
    coarse = fields[1]
    fields[1] = convergence.ResolutionField(
        coarse.grid_size_mm,
        coarse.path,
        coarse.x,
        coarse.z,
        coarse.density * (1 + 0.1 * coarse.x),
    )
    metrics = calculate_field_convergence(fields, x_min=0.0, x_points=20, z_points=20)
    summaries = summarize_convergence(metrics)

    assert len(summaries) == 1
    assert summaries[0].relative_error.standard_deviation > 0.0
    assert summaries[0].relative_error.mean != pytest.approx(metrics[0].relative_error)

    figures = {
        mode: plot_convergence(metrics, summaries, view="summary", field_error=mode)
        for mode in ("local-relative", "peak-normalized", "both")
    }
    for mode, expected_curves in (
        ("local-relative", 1),
        ("peak-normalized", 1),
        ("both", 2),
    ):
        (axis,) = figures[mode].axes
        assert axis.get_title() == "Mean field errors"
        assert len(axis.lines) == expected_curves
        assert len(axis.collections) == expected_curves


def test_summary_plot_uses_whole_decade_log_limits():
    summaries = [
        ConvergenceSummary(0.1, 0.01, ErrorDistribution(3, 0.02, 0.03, 0.0, 0.05))
    ]

    figure = plot_convergence([], summaries, view="summary")

    np.testing.assert_allclose(figure.axes[0].get_ylim(), (0.01, 0.1))


def test_csv_contains_per_lineout_and_aggregate_columns(tmp_path):
    _write_three_resolutions(tmp_path)
    metrics = calculate_convergence(load_resolution_lineouts(tmp_path))
    summaries = summarize_convergence(metrics)

    path = write_convergence_csv(tmp_path / "out" / "metrics.csv", metrics, summaries)

    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert float(rows[0]["relative_integrated_absolute_difference"]) == 2.0
    assert float(rows[0]["mean_relative_error"]) == 2.0
    assert rows[0]["mean_peak_normalized_absolute_error"] == ""


def test_reversed_cgns_bounds_are_rejected_clearly(tmp_path):
    write_density_cgns(tmp_path / "0_2.cgns")
    write_density_cgns(tmp_path / "0_1.cgns")
    fields = load_resolution_cgns_fields(tmp_path)

    with pytest.raises(ValueError, match="x bounds must be finite and increasing"):
        calculate_field_convergence(fields, x_min=1.0, x_max=0.0)
    argv = ["convergence", str(tmp_path), "--z-bounds", "1000", "0"]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        main()


def test_cli_writes_csv_next_to_output_but_not_into_input_dir(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    _write_three_resolutions(input_dir)
    output = tmp_path / "figures" / "conv.png"

    with patch.object(sys, "argv", ["convergence", str(input_dir), "-o", str(output)]):
        main()

    assert output.exists()
    assert output.with_suffix(".csv").exists()
    assert not (input_dir / "convergence_metrics.csv").exists()
