"""Tests for `docs/htu-twin/twin/initial_sample_archives.py` on the repository's real dataset."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import pytest

from lume_fbpic.model import LUMEFBPICModel, _read_output_values

_TWIN = Path(__file__).resolve().parents[1] / "docs" / "htu-twin" / "twin"


def _recorded_values(path):
    """The recorded action values of the archive at `path`."""
    with h5py.File(path, "r") as f:
        return _read_output_values(f)


@pytest.fixture()
def script():
    sys.path.insert(0, str(_TWIN))
    try:
        import initial_sample_archives
    except ModuleNotFoundError:
        pytest.skip("docs/htu-twin/twin not importable")
    finally:
        sys.path.remove(str(_TWIN))
    if not initial_sample_archives.DEFAULT_DATASET.is_file():
        pytest.skip("sample_dataset.json is not in this checkout")
    return initial_sample_archives


@pytest.fixture()
def dataset(script) -> dict:
    return json.loads(script.DEFAULT_DATASET.read_text())


def test_archive_records_the_dataset_outputs_exactly(script, dataset, tmp_path):
    record = dataset["runs"]["sim_0003"]

    path = script.build_archive(record, dataset["metadata"], tmp_path / "s.h5")

    outputs = _recorded_values(path)
    for name, value in record["output"].items():
        if name == "total_beam_charge_c":  # the archive has the charge in pC
            assert outputs["descriptor_total_beam_charge_pc"] == pytest.approx(
                value * 1e12
            )
        else:
            assert outputs[f"descriptor_{name}"] == value


def test_archive_config_has_the_production_grid_and_the_runs_inputs(
    script, dataset, tmp_path
):
    record = dataset["runs"]["sim_0003"]  # zernike_astigmatism_4 = 8

    path = script.build_archive(record, dataset["metadata"], tmp_path / "s.h5")

    model = LUMEFBPICModel.from_archive(path, dummy_run=True)
    hyparams = model.simulator.hyparams
    assert (hyparams.nz, hyparams.nr, hyparams.nm) == (1500, 300, 5)
    assert model.simulator.laser.num_points == (600, 900)
    assert model.get(["zernike_astigmatism_4"])["zernike_astigmatism_4"] == 8.0
    assert (
        model.get(["laser_energy"])["laser_energy"] == record["input"]["laser_energy_J"]
    )


def test_loaded_archive_serves_the_recorded_descriptor_and_has_no_stats(
    script, dataset, tmp_path
):
    record = dataset["runs"]["sim_0000"]
    path = script.build_archive(record, dataset["metadata"], tmp_path / "s.h5")

    model = LUMEFBPICModel.from_archive(path, dummy_run=True)

    assert (
        model.get(["descriptor_mean_uz"])["descriptor_mean_uz"]
        == record["output"]["mean_uz"]
    )
    assert math.isnan(model.get(["charge_pc"])["charge_pc"])  # the dataset has no stats


def test_an_input_without_an_action_that_differs_from_the_baseline_is_an_error(
    script, dataset, tmp_path
):
    record = json.loads(json.dumps(dataset["runs"]["sim_0000"]))
    record["input"]["laser_wavelength_m"] = 7.0e-7

    with pytest.raises(ValueError, match="laser_wavelength_m"):
        script.build_archive(record, dataset["metadata"], tmp_path / "s.h5")


def test_an_unknown_input_is_an_error(script, dataset, tmp_path):
    record = json.loads(json.dumps(dataset["runs"]["sim_0000"]))
    record["input"]["mystery_knob"] = 1.0

    with pytest.raises(ValueError, match="mystery_knob"):
        script.build_archive(record, dataset["metadata"], tmp_path / "s.h5")


def test_descriptor_settings_must_match_the_dataset(script, dataset, tmp_path):
    metadata = {**dataset["metadata"], "uz_min": 10.0}

    with pytest.raises(ValueError, match="uz_min"):
        script.build_archive(dataset["runs"]["sim_0000"], metadata, tmp_path / "s.h5")
