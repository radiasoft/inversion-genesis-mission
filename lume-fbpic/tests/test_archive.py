"""Tests for the archive bundle: final particles flag, action definitions, and the action
input values at execution / output values after it."""

from __future__ import annotations

import math
import warnings

import h5py
import numpy
import pytest

from tests.downramp_actions import make_actions
from lume_fbpic.actions import LaserFieldAction, make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel, _read_output_values
from lume_fbpic.simulator import FBPICSimulator


def _recorded_values(path):
    """The recorded output values of the archive at `path`."""
    with h5py.File(path, "r") as f:
        return _read_output_values(f)


def _archive_with_outputs(model, path, outputs):
    """Archive `model` as a run whose recorded `outputs` (by action name) it never held."""
    model.simulator.stats = dict(outputs)
    try:
        model.archive(path)
    finally:
        model.simulator.stats = {}


@pytest.fixture()
def full_model(simulator) -> LUMEFBPICModel:
    """A dummy-run model with the default actions plus the 33 descriptor actions."""
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    return LUMEFBPICModel(simulator, actions, dummy_run=True)


def _pretend_run_finishes(simulator, mocker, particle_group, stats):
    """Replace `simulator.run` by one that 'completes' with the given results."""

    def run():
        simulator.final_particles = particle_group
        simulator.stats = dict(stats)
        simulator.finished = True

    mocker.patch.object(simulator, "run", side_effect=run)


def test_archive_does_not_store_particles_by_default(
    simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 1.0}

    simulator.archive(tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert "final_particles" not in f
        assert "stats" not in f
    restored = FBPICSimulator.from_archive(tmp_path / "a.h5")
    assert restored.final_particles is None
    assert restored.stats == {}


def test_archive_stores_final_particles_as_a_particle_group(
    simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 1.5, "energy_mean_mev": 2.5}

    simulator.archive(tmp_path / "a.h5", save_final_particles=True)

    restored = FBPICSimulator.from_archive(tmp_path / "a.h5")
    assert len(restored.final_particles) == len(particle_group)
    numpy.testing.assert_allclose(restored.final_particles.pz, particle_group.pz)
    numpy.testing.assert_allclose(
        restored.final_particles.weight, particle_group.weight
    )
    assert restored.stats == {"charge_pc": 1.5, "energy_mean_mev": 2.5}


def test_archive_with_particles_opens_directly_as_a_particle_file(
    simulator, particle_group, tmp_path
):
    from beamphysics import ParticleGroup

    simulator.final_particles = particle_group
    simulator.archive(tmp_path / "a.h5", save_final_particles=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a non-compliant openPMD layout warns
        opened = ParticleGroup(h5=str(tmp_path / "a.h5"))

    assert len(opened) == len(particle_group)
    numpy.testing.assert_allclose(opened.weight, particle_group.weight)


def test_saving_particles_before_any_run_is_an_error(simulator, tmp_path):
    with pytest.raises(ValueError, match="final particles"):
        simulator.archive(tmp_path / "a.h5", save_final_particles=True)


def test_model_archive_stores_the_action_definitions_in_order(full_model, tmp_path):
    full_model.archive(tmp_path / "m.h5")

    restored = LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)

    assert list(restored.supported_variables) == list(full_model.supported_variables)
    assert [type(a) for a in restored.supported_variables.values()] == [
        type(a) for a in full_model.supported_variables.values()
    ]
    assert [a.model_dump() for a in restored.supported_variables.values()] == [
        a.model_dump() for a in full_model.supported_variables.values()
    ]


def test_restored_model_sees_the_archived_config(full_model, tmp_path):
    full_model.set({"laser_energy": 6.0})
    full_model.archive(tmp_path / "m.h5")

    restored = LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)

    assert restored.get("laser_energy") == 6.0


def test_restored_model_returns_outputs_when_particles_were_saved(
    full_model, simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 7.0, "energy_mean_mev": 8.0, "energy_std_mev": 0.5}
    names = ["charge_pc", "descriptor_mean_uz", "descriptor_total_beam_charge_pc"]
    expected = full_model.get(names)
    full_model.archive(tmp_path / "m.h5", save_final_particles=True)

    restored = LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)

    got = restored.get(names)
    for name in names:
        assert got[name] == pytest.approx(expected[name])


def test_archive_stores_the_current_input_and_output_values(
    simulator, particle_group, mocker, tmp_path
):
    model = LUMEFBPICModel(simulator, make_actions(simulator))
    simulator.configure()
    _pretend_run_finishes(
        simulator, mocker, particle_group, {"charge_pc": 7.0, "energy_mean_mev": 8.0}
    )
    model.set({"laser_energy": 6.0})  # runs
    model.dummy_run = True
    model.set(
        {"laser_energy": 9.0}
    )  # changes the config after the run, without running
    model.archive(tmp_path / "m.h5")

    values = _recorded_values(tmp_path / "m.h5")

    assert values["charge_pc"] == 7.0
    assert values["energy_mean_mev"] == 8.0
    assert (
        "laser_energy" not in values
    )  # the inputs are in the config, not stored as values
    # the config holds the current input, not the 6.0 that ran
    assert (
        LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True).get(
            "laser_energy"
        )
        == 9.0
    )


def test_model_that_never_ran_stores_its_inputs_in_the_config_and_nan_outputs(
    full_model, tmp_path
):
    full_model.set({"laser_energy": 4.0})
    full_model.archive(tmp_path / "m.h5")

    values = _recorded_values(tmp_path / "m.h5")

    assert (
        LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True).get(
            "laser_energy"
        )
        == 4.0
    )
    assert math.isnan(values["charge_pc"])
    assert "final_particles" not in values  # no scalar value for particles


def test_from_archive_rejects_a_simulator_only_archive(simulator, tmp_path):
    simulator.archive(tmp_path / "s.h5")

    with pytest.raises(ValueError, match="no actions"):
        LUMEFBPICModel.from_archive(tmp_path / "s.h5")


def test_from_archive_rejects_an_unknown_action_class(full_model, tmp_path):
    full_model.archive(tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f["actions/0000"].attrs["class"] = "NoSuchAction"

    with pytest.raises(ValueError, match="NoSuchAction"):
        LUMEFBPICModel.from_archive(tmp_path / "m.h5")


def test_an_action_class_from_outside_lume_fbpic_actions_cannot_be_loaded(
    simulator, tmp_path
):
    class MyLaserAction(LaserFieldAction):
        pass

    model = LUMEFBPICModel(
        simulator,
        [MyLaserAction(name="mine", field_name="waist", unit="m")],
        dummy_run=True,
    )
    model.archive(tmp_path / "m.h5")  # a custom action can still be written

    with pytest.raises(ValueError, match="Unknown action class 'MyLaserAction'"):
        LUMEFBPICModel.from_archive(tmp_path / "m.h5")


def test_action_parameters_are_stored_as_attributes_not_json(full_model, tmp_path):
    full_model.archive(tmp_path / "m.h5")

    with h5py.File(tmp_path / "m.h5") as f:
        entry = f["actions/0000"]
        parameters = entry["parameters"].attrs
        assert entry.attrs["name"] == next(iter(full_model.supported_variables))
        assert parameters["name"] == entry.attrs["name"]
        assert not parameters["read_only"]  # h5py returns numpy.bool_
        assert "variable_class" not in parameters and "default_value" not in parameters
        assert "parameters" not in entry.attrs  # no JSON blob
        assert len(f["actions"]) == len(full_model.supported_variables)


def test_action_parameters_round_trip_with_their_types(
    simulator, particle_group, tmp_path
):
    from lume_fbpic.actions import make_descriptor_actions

    model = LUMEFBPICModel(
        simulator,
        [*make_actions(simulator), *make_descriptor_actions(uz_min=12.5)],
        dummy_run=True,
    )
    model.archive(tmp_path / "m.h5")

    restored = LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)

    assert [a.model_dump() for a in restored.supported_variables.values()] == [
        a.model_dump() for a in model.supported_variables.values()
    ]
    descriptor = restored.supported_variables["descriptor_mean_uz"]
    assert descriptor.uz_min == 12.5 and isinstance(descriptor.longitudinal_bins, int)


def test_an_action_with_an_unstorable_parameter_cannot_be_archived(simulator, tmp_path):
    class ListAction(LaserFieldAction):
        tags: list[str] = []

    model = LUMEFBPICModel(
        simulator,
        [ListAction(name="x", field_name="waist", unit="m", tags=["a"])],
        dummy_run=True,
    )

    with pytest.raises(TypeError, match="tags"):
        model.archive(tmp_path / "m.h5")


def test_outputs_recorded_in_the_simulators_stats_are_archived(full_model, tmp_path):
    full_model.set({"laser_energy": 4.0})

    _archive_with_outputs(
        full_model,
        tmp_path / "m.h5",
        {"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_pc": 510.0},
    )

    values = _recorded_values(tmp_path / "m.h5")
    assert values["descriptor_mean_uz"] == 171.6
    assert values["descriptor_total_beam_charge_pc"] == 510.0
    assert math.isnan(values["descriptor_cov_uz_uz"])  # not recorded


@pytest.fixture()
def recorded_model(full_model, tmp_path) -> LUMEFBPICModel:
    """A model loaded from an archive that has recorded outputs but no particles."""
    _archive_with_outputs(
        full_model,
        tmp_path / "m.h5",
        {"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_pc": 510.0},
    )
    return LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)


def test_loaded_model_serves_the_recorded_outputs_when_there_are_no_particles(
    recorded_model,
):
    got = recorded_model.get(["descriptor_mean_uz", "descriptor_total_beam_charge_pc"])

    assert got == {
        "descriptor_mean_uz": 171.6,
        "descriptor_total_beam_charge_pc": 510.0,
    }
    assert math.isnan(
        recorded_model.get(["descriptor_cov_uz_uz"])["descriptor_cov_uz_uz"]
    )


def test_recorded_outputs_go_stale_once_an_input_is_set(recorded_model):
    recorded_model.set({"laser_energy": 3.0})

    assert math.isnan(recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"])


def test_live_values_win_over_recorded_ones(recorded_model, particle_group):
    recorded_model.simulator.final_particles = particle_group

    live = recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"]

    assert live != 171.6
    assert 100 < live < 200  # the synthetic bunch's mean uz is ~150


def test_recorded_outputs_survive_archiving_again(recorded_model, tmp_path):
    recorded_model.archive(tmp_path / "again.h5")

    assert _recorded_values(tmp_path / "again.h5")["descriptor_mean_uz"] == 171.6


def test_reset_restores_the_initial_config_without_running(recorded_model, mocker):
    run = mocker.patch.object(recorded_model.simulator, "run")
    energy = recorded_model.get("laser_energy")
    recorded_model.set({"laser_energy": 9.0})
    assert recorded_model.get("laser_energy") == 9.0

    recorded_model.reset()  # ActionModel.reset() would fail here: default_value is None

    assert recorded_model.get("laser_energy") == energy
    run.assert_not_called()


def test_reset_brings_back_the_recorded_outputs(recorded_model):
    recorded_model.set({"laser_energy": 9.0})
    assert math.isnan(recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"])

    recorded_model.reset()

    assert recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"] == 171.6


def test_reset_discards_results_produced_after_construction(
    simulator, particle_group, mocker
):
    model = LUMEFBPICModel(simulator, make_actions(simulator))
    simulator.configure()
    _pretend_run_finishes(simulator, mocker, particle_group, {"charge_pc": 7.0})
    model.set({"laser_energy": 6.0})
    assert simulator.final_particles is particle_group

    model.reset()

    assert simulator.final_particles is None
    assert simulator.stats == {}
