"""Tests for the archive bundle: final particles flag, action definitions, and the action
input values at execution / output values after it."""

from __future__ import annotations

import json
import math
import warnings

import h5py
import numpy as np
import pytest

from lume_fbpic.actions import LaserFieldAction, make_actions, make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel, read_action_values, read_archive_metadata
from lume_fbpic.simulator import FBPICSimulator


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


def test_archive_does_not_store_particles_by_default(simulator, particle_group, tmp_path):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 1.0}

    simulator.archive(tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert "final_particles" not in f
        assert "stats" not in f
    restored = FBPICSimulator.from_archive(tmp_path / "a.h5")
    assert restored.final_particles is None
    assert restored.stats == {}


def test_archive_stores_final_particles_as_a_particle_group(simulator, particle_group, tmp_path):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 1.5, "energy_mean_mev": 2.5}

    simulator.archive(tmp_path / "a.h5", save_final_particles=True)

    restored = FBPICSimulator.from_archive(tmp_path / "a.h5")
    assert len(restored.final_particles) == len(particle_group)
    np.testing.assert_allclose(restored.final_particles.pz, particle_group.pz)
    np.testing.assert_allclose(restored.final_particles.weight, particle_group.weight)
    assert restored.stats == {"charge_pc": 1.5, "energy_mean_mev": 2.5}

    reloaded = FBPICSimulator.from_archive(tmp_path / "a.h5")
    reloaded.load_archive(tmp_path / "a.h5", configure=False)
    assert len(reloaded.final_particles) == len(particle_group)


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
    np.testing.assert_allclose(opened.weight, particle_group.weight)


def test_saving_particles_before_any_run_is_an_error(simulator, tmp_path):
    with pytest.raises(ValueError, match="final particles"):
        simulator.archive(tmp_path / "a.h5", save_final_particles=True)


def test_model_archive_stores_the_action_definitions_in_order(full_model, tmp_path):
    full_model.archive(tmp_path / "m.h5")

    restored = LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)

    assert [a.name for a in restored._actions] == [a.name for a in full_model._actions]
    assert [type(a) for a in restored._actions] == [type(a) for a in full_model._actions]
    assert [a.model_dump() for a in restored._actions] == [
        a.model_dump() for a in full_model._actions
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
    names = ["charge_pc", "descriptor_mean_uz", "descriptor_total_beam_charge_c"]
    expected = full_model.get(names)
    full_model.archive(tmp_path / "m.h5", save_final_particles=True)

    restored = LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)

    got = restored.get(names)
    for name in names:
        assert got[name] == pytest.approx(expected[name])


def test_values_are_recorded_at_execution_not_at_archive_time(
    simulator, particle_group, mocker, tmp_path
):
    model = LUMEFBPICModel.from_simulator(simulator)
    simulator.configure()
    _pretend_run_finishes(
        simulator, mocker, particle_group, {"charge_pc": 7.0, "energy_mean_mev": 8.0}
    )
    model.set({"laser_energy": 6.0})  # runs
    model.dummy_run = True
    model.set({"laser_energy": 9.0})  # changes the config after the run, without running
    model.archive(tmp_path / "m.h5")

    values = read_action_values(tmp_path / "m.h5")

    assert values["executed"] is True
    assert values["config_changed_since_execution"] is True
    assert values["inputs"]["laser_energy"] == 6.0  # as executed, not the current 9.0
    assert values["outputs"]["charge_pc"] == 7.0
    assert values["outputs"]["energy_mean_mev"] == 8.0
    assert LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True).get("laser_energy") == 9.0


def test_values_of_an_unchanged_executed_config(simulator, particle_group, mocker, tmp_path):
    model = LUMEFBPICModel.from_simulator(simulator)
    simulator.configure()
    _pretend_run_finishes(simulator, mocker, particle_group, {"charge_pc": 7.0})
    model.set({"laser_energy": 6.0})
    model.archive(tmp_path / "m.h5")

    values = read_action_values(tmp_path / "m.h5")

    assert values["executed"] is True
    assert values["config_changed_since_execution"] is False
    assert values["inputs"]["laser_energy"] == 6.0


def test_model_that_never_ran_stores_current_values_and_nan_outputs(full_model, tmp_path):
    full_model.set({"laser_energy": 4.0})
    full_model.archive(tmp_path / "m.h5")

    values = read_action_values(tmp_path / "m.h5")

    assert values["executed"] is False
    assert values["inputs"]["laser_energy"] == 4.0
    assert math.isnan(values["outputs"]["charge_pc"])
    assert "final_particles" not in values["outputs"]  # no scalar value for particles


def test_a_run_that_does_not_happen_is_not_recorded_as_executed(simulator, tmp_path):
    model = LUMEFBPICModel.from_simulator(simulator)  # not configured: run() is a no-op

    model.set({"laser_energy": 6.0})
    model.archive(tmp_path / "m.h5")

    assert read_action_values(tmp_path / "m.h5")["executed"] is False


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


def test_from_archive_accepts_extra_action_classes(simulator, tmp_path):
    class MyLaserAction(LaserFieldAction):
        pass

    model = LUMEFBPICModel(
        simulator,
        [MyLaserAction(name="mine", field_name="waist", unit="m")],
        dummy_run=True,
    )
    model.archive(tmp_path / "m.h5")

    with pytest.raises(ValueError, match="MyLaserAction"):
        LUMEFBPICModel.from_archive(tmp_path / "m.h5")
    restored = LUMEFBPICModel.from_archive(
        tmp_path / "m.h5", action_classes={"MyLaserAction": MyLaserAction}, dummy_run=True
    )
    assert isinstance(restored._actions[0], MyLaserAction)


def test_action_parameters_are_stored_as_json(full_model, tmp_path):
    full_model.archive(tmp_path / "m.h5")

    with h5py.File(tmp_path / "m.h5") as f:
        entry = f["actions/0000"]
        parameters = json.loads(entry.attrs["parameters"])
        assert parameters["name"] == full_model._actions[0].name
        assert f["actions"].attrs["count"] == len(full_model._actions)


def test_supplied_outputs_are_stored_as_the_recorded_run(full_model, tmp_path):
    full_model.set({"laser_energy": 4.0})

    full_model.archive(
        tmp_path / "m.h5",
        outputs={"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_c": 5.1e-10},
    )

    values = read_action_values(tmp_path / "m.h5")
    assert values["executed"] is True
    assert values["config_changed_since_execution"] is False
    assert values["inputs"]["laser_energy"] == 4.0
    assert values["outputs"]["descriptor_mean_uz"] == 171.6
    assert values["outputs"]["descriptor_total_beam_charge_c"] == 5.1e-10
    assert math.isnan(values["outputs"]["descriptor_cov_uz_uz"])  # not supplied
    with h5py.File(tmp_path / "m.h5") as f:
        assert bool(f["actions"].attrs["outputs_supplied"]) is True


def test_supplied_outputs_must_name_read_only_scalar_actions(full_model, tmp_path):
    with pytest.raises(ValueError, match="laser_energy"):
        full_model.archive(tmp_path / "m.h5", outputs={"laser_energy": 1.0})  # an input
    with pytest.raises(ValueError, match="nonsense"):
        full_model.archive(tmp_path / "m.h5", outputs={"nonsense": 1.0})


def test_metadata_is_stored_and_read_back(full_model, tmp_path):
    full_model.archive(
        tmp_path / "m.h5", metadata={"reconstructed": True, "source": "x.json", "run": ["a", 1]}
    )

    metadata = read_archive_metadata(tmp_path / "m.h5")

    assert metadata["reconstructed"] == True  # noqa: E712 -- h5py returns numpy.bool_
    assert metadata["source"] == "x.json"
    assert json.loads(metadata["run"]) == ["a", 1]
    full_model.archive(tmp_path / "n.h5")
    assert read_archive_metadata(tmp_path / "n.h5") == {}


@pytest.fixture()
def recorded_model(full_model, tmp_path) -> LUMEFBPICModel:
    """A model loaded from an archive that has recorded outputs but no particles."""
    full_model.archive(
        tmp_path / "m.h5",
        outputs={"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_c": 5.1e-10},
    )
    return LUMEFBPICModel.from_archive(tmp_path / "m.h5", dummy_run=True)


def test_loaded_model_serves_the_recorded_outputs_when_there_are_no_particles(recorded_model):
    got = recorded_model.get(["descriptor_mean_uz", "descriptor_total_beam_charge_c"])

    assert got == {"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_c": 5.1e-10}
    assert math.isnan(recorded_model.get(["descriptor_cov_uz_uz"])["descriptor_cov_uz_uz"])


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

    assert read_action_values(tmp_path / "again.h5")["outputs"]["descriptor_mean_uz"] == 171.6


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
    model = LUMEFBPICModel.from_simulator(simulator)
    simulator.configure()
    _pretend_run_finishes(simulator, mocker, particle_group, {"charge_pc": 7.0})
    model.set({"laser_energy": 6.0})
    assert simulator.final_particles is particle_group

    model.reset()

    assert simulator.final_particles is None
    assert simulator.stats == {}
    assert model._executed_inputs is None


def _full_descriptor_outputs(particle_group) -> dict[str, float]:
    """`descriptor_*` outputs, as a recorded run would carry, from a real descriptor."""
    from scipy.constants import c, e, m_e

    from inversion_fbpic.utils import distributions

    mc2 = m_e * c**2 / e
    phase_space = np.stack(
        [
            particle_group.x, particle_group.px / mc2, particle_group.y, particle_group.py / mc2,
            particle_group.z, particle_group.pz / mc2,
        ],
        axis=-1,
    )
    descriptor = distributions.compute_moment_descriptor(
        phase_space, np.asarray(particle_group.weight) / e
    )
    return {f"descriptor_{name}": value for name, value in descriptor.items()}


@pytest.fixture()
def reconstructed_model(full_model, particle_group, tmp_path) -> LUMEFBPICModel:
    """A model loaded from an archive with the full recorded descriptor and no particles."""
    full_model.archive(tmp_path / "r.h5", outputs=_full_descriptor_outputs(particle_group))
    return LUMEFBPICModel.from_archive(tmp_path / "r.h5", dummy_run=True)


def test_the_recorded_descriptor_is_the_33_scalars_without_the_prefix(reconstructed_model):
    descriptor = reconstructed_model.recorded_descriptor()

    assert len(descriptor) == 33
    assert "mean_uz" in descriptor and "descriptor_mean_uz" not in descriptor


def test_a_model_without_recorded_values_has_no_descriptor_to_synthesize(full_model):
    with pytest.raises(ValueError, match="no recorded descriptor"):
        full_model.synthesize_bunch()


def test_an_incomplete_recorded_descriptor_cannot_be_synthesized(recorded_model):
    with pytest.raises(ValueError, match="lacks"):
        recorded_model.synthesize_bunch()  # only two descriptor values were recorded


def test_a_synthesized_bunch_is_handed_downstream_while_the_simulator_has_none(reconstructed_model):
    assert reconstructed_model.final_particles is None

    bunch = reconstructed_model.synthesize_bunch(n_particles=2000)

    assert reconstructed_model.final_particles is bunch
    assert reconstructed_model.simulator.final_particles is None
    assert len(bunch) == 2000


def test_the_descriptor_outputs_stay_the_recorded_values_after_synthesizing(
    reconstructed_model, particle_group
):
    recorded = _full_descriptor_outputs(particle_group)
    reconstructed_model.synthesize_bunch(n_particles=2000)

    got = reconstructed_model.get(["descriptor_mean_uz", "descriptor_cov_uz_uz"])

    assert got["descriptor_mean_uz"] == recorded["descriptor_mean_uz"]  # not recomputed
    assert got["descriptor_cov_uz_uz"] == recorded["descriptor_cov_uz_uz"]


def test_real_particles_take_precedence_over_a_synthesized_bunch(reconstructed_model, particle_group):
    reconstructed_model.synthesize_bunch(n_particles=2000)
    reconstructed_model.simulator.final_particles = particle_group

    assert reconstructed_model.final_particles is particle_group


def test_a_synthesized_bunch_survives_reset(reconstructed_model):
    bunch = reconstructed_model.synthesize_bunch(n_particles=2000)

    reconstructed_model.reset()

    assert reconstructed_model.final_particles is bunch
