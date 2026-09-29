from __future__ import annotations

import numpy as np
import pytest

from fludat_proc.ansys_lineouts import (
    deduplicate_and_sort_profile,
    mirror_across_z0,
    parse_lineout_file,
    parse_x_position,
)
from fludat_proc.filenames import parse_grid_size_filename, parse_pressure_filename

from .conftest import write_lineout_file


def test_parser_returns_labeled_float64_profiles(tmp_path):
    path = write_lineout_file(tmp_path / "0_1.txt", {"x-0-0": [(0.0, 1.0), (1.0, 2.0)]})

    sections = parse_lineout_file(path)

    np.testing.assert_array_equal(sections["x-0-0"], [[0.0, 1.0], [1.0, 2.0]])
    assert sections["x-0-0"].dtype == np.float64


def test_parser_rejects_duplicate_labels_and_empty_files(tmp_path):
    path = write_lineout_file(
        tmp_path / "dup.txt", {"x-0-0": [(0.0, 1.0)], "x-0-0 ": [(1.0, 2.0)]}
    )
    path.write_text(path.read_text().replace('"x-0-0 "', '"x-0-0"'))
    with pytest.raises(ValueError, match="Duplicate lineout section"):
        parse_lineout_file(path)

    (tmp_path / "empty.txt").write_text("(title)\n")
    with pytest.raises(ValueError, match="No lineout sections"):
        parse_lineout_file(tmp_path / "empty.txt")


@pytest.mark.parametrize(
    ("label", "expected"),
    [("x-0-0", 0.0), ("x-1-5", 1.5), ("x-3", 3.0), ("l-x-2-0-y-0", 2.0)],
)
def test_parse_x_position(label, expected):
    assert parse_x_position(label) == expected


def test_parse_x_position_rejects_unknown_labels():
    with pytest.raises(ValueError, match="x position"):
        parse_x_position("axis")


def test_parse_pressure_filename_accepts_any_extension():
    assert parse_pressure_filename("5_bar.txt") == 5.0
    assert parse_pressure_filename("12.5_bar.cgns") == 12.5
    assert parse_pressure_filename("20_bar") == 20.0
    with pytest.raises(ValueError, match="backing pressure"):
        parse_pressure_filename("coarse.txt")


def test_parse_grid_size_filename():
    assert parse_grid_size_filename("0_075.txt") == 0.075
    assert parse_grid_size_filename("1.cgns") == 1.0
    with pytest.raises(ValueError, match="maximum grid size"):
        parse_grid_size_filename("coarse.txt")


def test_deduplicate_and_sort_profile_keeps_last_value():
    profile = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 3.0]])

    np.testing.assert_array_equal(
        deduplicate_and_sort_profile(profile), [[0.0, 0.0], [1.0, 3.0]]
    )
    with pytest.raises(ValueError, match="two distinct"):
        deduplicate_and_sort_profile(np.array([[1.0, 2.0], [1.0, 3.0]]))


def test_mirror_across_z0_discards_negative_z_and_mirrors_positive():
    profile = np.array([[-1.0, 9.0], [0.0, 1.0], [2.0, 3.0], [1.0, 2.0]])

    mirrored = mirror_across_z0(profile)

    np.testing.assert_array_equal(
        mirrored, [[-2.0, 3.0], [-1.0, 2.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    )
