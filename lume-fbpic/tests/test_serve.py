"""Tests for `lume_fbpic.serve.build_config` (the Runner config; no server is started)."""

from __future__ import annotations

import sys

import pytest

from lume_fbpic.actions import make_actions, make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.serve import build_config, expand_archives


@pytest.fixture()
def full_model(simulator) -> LUMEFBPICModel:
    return LUMEFBPICModel(
        simulator, [*make_actions(simulator), *make_descriptor_actions()], dummy_run=True
    )


def test_building_a_config_without_lume_pva_says_so(full_model, monkeypatch):
    monkeypatch.setitem(sys.modules, "lume_pva", None)
    monkeypatch.setitem(sys.modules, "lume_pva.runner", None)

    with pytest.raises(ImportError, match="lume-pva"):
        build_config(full_model)


def test_config_serves_only_read_only_variables_by_default(full_model):
    pytest.importorskip("lume_pva")

    config = build_config(full_model)

    names = set(config["variables"])
    assert "charge_pc" in names and "descriptor_mean_uz" in names
    assert "laser_energy" not in names
    assert all(str(entry["mode"]) == "ro" for entry in config["variables"].values())


def test_config_can_include_the_inputs_read_write(full_model):
    pytest.importorskip("lume_pva")

    config = build_config(full_model, include_inputs=True)

    assert str(config["variables"]["laser_energy"]["mode"]) == "rw"
    assert str(config["variables"]["charge_pc"]["mode"]) == "ro"


def test_config_uses_the_prefix_and_protocol(full_model):
    pytest.importorskip("lume_pva")

    config = build_config(full_model, prefix="X:", protocol=["pva"])

    assert config["prefix"] == "X:"
    assert config["protocol"] == ["pva"]


def test_synthesizing_a_bunch_needs_the_twin(full_model, tmp_path):
    pytest.importorskip("lume_pva")
    from lume_fbpic.serve import main

    full_model.archive(tmp_path / "a.h5")

    with pytest.raises(SystemExit):
        main([str(tmp_path / "a.h5"), "--synthesize-bunch"])


def test_the_twin_needs_particles_or_the_flag_to_synthesize_them(full_model, tmp_path, capsys):
    pytest.importorskip("lume_pva")
    from lume_fbpic.serve import main

    full_model.archive(tmp_path / "a.h5")  # no final particles

    with pytest.raises(SystemExit):
        main([str(tmp_path / "a.h5"), "--twin"])

    assert "--synthesize-bunch" in capsys.readouterr().err


def test_a_directory_expands_to_its_archives_in_name_order(tmp_path):
    for name in ("b.h5", "a.h5", "notes.txt"):
        (tmp_path / name).write_text("x")

    assert expand_archives([tmp_path]) == [tmp_path / "a.h5", tmp_path / "b.h5"]


def test_files_and_directories_can_be_mixed(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "x.h5").write_text("x")
    (tmp_path / "y.h5").write_text("x")

    assert expand_archives([tmp_path / "y.h5", tmp_path / "d"]) == [
        tmp_path / "y.h5",
        tmp_path / "d" / "x.h5",
    ]


def test_an_empty_directory_or_a_missing_path_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="no .h5"):
        expand_archives([tmp_path / "empty"])
    with pytest.raises(ValueError, match="no such"):
        expand_archives([tmp_path / "missing.h5"])


def test_archives_with_the_same_file_name_are_refused(full_model, tmp_path, capsys):
    pytest.importorskip("lume_pva")
    from lume_fbpic.serve import main

    for directory in ("one", "two"):
        (tmp_path / directory).mkdir()
        full_model.archive(tmp_path / directory / "run.h5")

    with pytest.raises(SystemExit):
        main([str(tmp_path / "one" / "run.h5"), str(tmp_path / "two" / "run.h5")])

    assert "unique" in capsys.readouterr().err


def test_the_twin_names_every_archive_that_lacks_particles(full_model, tmp_path, capsys):
    pytest.importorskip("lume_pva")
    from lume_fbpic.serve import main

    full_model.archive(tmp_path / "a.h5")
    full_model.archive(tmp_path / "b.h5")

    with pytest.raises(SystemExit):
        main([str(tmp_path), "--twin"])

    error = capsys.readouterr().err
    assert "'a'" in error and "'b'" in error and "--synthesize-bunch" in error


def test_the_selector_is_served_read_write(full_model):
    pytest.importorskip("lume_pva")
    from lume_fbpic.selector import ArchiveSelector

    selector = ArchiveSelector({"a": full_model})

    config = build_config(selector, serve_always={"LPA_Archive"}, prefix="X:")

    assert str(config["variables"]["LPA_Archive"]["mode"]) == "rw"
    assert "laser_energy" not in config["variables"]  # an LPA input is still not served
