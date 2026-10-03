"""Tests for LUMEFBPICModel."""

from __future__ import annotations

import pytest

from lume.exceptions import ReadOnlyError

from lume_fbpic.model import LUMEFBPICModel


def test_from_simulator_builds_default_action_set(simulator):
    model = LUMEFBPICModel.from_simulator(simulator)
    assert "laser_energy" in model.supported_variables
    assert "downramp_length" in model.supported_variables
    assert "final_particles" in model.supported_variables


def test_dummy_run_set_updates_parameter_without_running(model, simulator, mocker):
    run_mock = mocker.patch.object(simulator, "run")
    model.set({"laser_energy": 6.0})

    assert model.get("laser_energy") == 6.0
    run_mock.assert_not_called()


def test_set_runs_simulator_after_updating_parameter(simulator, mocker):
    model = LUMEFBPICModel.from_simulator(simulator)
    seen = []
    mocker.patch.object(
        simulator, "run", side_effect=lambda: seen.append(simulator.laser.energy)
    )
    model.set({"laser_energy": 6.0})

    assert seen == [6.0]  # run() saw the already-updated laser


def test_read_only_variable_cannot_be_set(model):
    with pytest.raises(ReadOnlyError):
        model.set({"charge_pc": 1.0})


def test_set_unknown_variable_raises(model):
    with pytest.raises(ValueError):
        model.set({"not_a_real_variable": 1.0})


def test_final_particles_property_delegates_to_simulator(model, simulator):
    simulator.final_particles = "placeholder"
    assert model.final_particles == "placeholder"
