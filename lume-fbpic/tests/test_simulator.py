"""Tests for FBPICSimulator's config/lifecycle/archive paths -- no real fbpic run."""

from __future__ import annotations

import h5py
import pytest

from lume_fbpic.simulator import BaseSimulator, FBPICSimulator


def test_configure_accepts_complete_config(simulator):
    simulator.configure()
    assert simulator.configured is True


def test_configure_rejects_missing_densities(hyparams, laser, tmp_path):
    simulator = FBPICSimulator(
        hyparams, laser, [], target_species="electrons_flattop", working_directory=tmp_path
    )
    with pytest.raises(ValueError):
        simulator.configure()


def test_run_without_configure_is_a_noop(simulator):
    simulator.run()
    assert simulator.finished is False
    assert simulator.final_particles is None


def test_reset_restores_the_starting_state_without_rerunning(simulator):
    config = simulator.config()
    simulator.configure()
    simulator.stats = {"charge_pc": 1.0}
    simulator.final_particles = object()
    simulator.laser = None
    simulator.finished = True

    simulator.reset()

    assert simulator.stats == {}
    assert simulator.final_particles is None
    assert simulator.laser is config["laser"]
    assert simulator.densities == config["densities"]
    assert simulator.finished is False
    assert simulator.configured is True  # reset leaves it as it was


def test_reset_restores_the_archived_state_of_a_loaded_simulator(simulator, particle_group, tmp_path):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 7.0}
    simulator.archive(tmp_path / "a.h5", save_final_particles=True)
    restored = FBPICSimulator.from_archive(tmp_path / "a.h5")
    laser = restored.laser

    restored.laser = None
    restored.stats = {}
    restored.final_particles = None
    restored.reset()

    assert restored.laser is laser
    assert restored.stats == {"charge_pc": 7.0}
    assert len(restored.final_particles) == len(particle_group)


def test_stats_given_to_from_archive_are_the_starting_stats_of_an_archive_without_particles(
    simulator, tmp_path
):
    simulator.archive(tmp_path / "a.h5")

    restored = FBPICSimulator.from_archive(tmp_path / "a.h5", stats={"descriptor_mean_uz": 3.0})
    restored.stats = {}
    restored.reset()

    assert restored.stats == {"descriptor_mean_uz": 3.0}


def test_stats_given_to_from_archive_do_not_replace_those_of_an_archive_with_particles(
    simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 7.0}
    simulator.archive(tmp_path / "a.h5", save_final_particles=True)

    restored = FBPICSimulator.from_archive(tmp_path / "a.h5", stats={"descriptor_mean_uz": 3.0})

    assert restored.stats == {"charge_pc": 7.0}


def test_reset_needs_a_starting_state(simulator):
    simulator._initial_state = None

    with pytest.raises(RuntimeError, match="starting state"):
        simulator.reset()


def test_fingerprint_is_stable_for_the_same_config(simulator):
    assert simulator.fingerprint() == simulator.fingerprint()


def test_archive_round_trips_config_only(simulator, tmp_path):
    simulator.configure()
    archive_path = tmp_path / "archive.h5"
    simulator.archive(archive_path)

    restored = FBPICSimulator.from_archive(archive_path, working_directory=tmp_path)

    assert restored.hyparams.to_dict() == simulator.hyparams.to_dict()
    assert restored.laser.to_dict() == simulator.laser.to_dict()
    assert len(restored.densities) == len(simulator.densities)
    for original, loaded in zip(simulator.densities, restored.densities):
        assert loaded.to_dict() == original.to_dict()

    # A loaded simulator has to be configured before it runs.
    assert restored.configured is False
    assert restored.final_particles is None


def test_archive_stores_target_species_and_from_archive_restores_it(simulator, tmp_path):
    simulator.archive(tmp_path / "a.h5")

    restored = FBPICSimulator.from_archive(tmp_path / "a.h5", working_directory=tmp_path)

    assert restored.target_species == simulator.target_species
    assert restored.hyparams.to_dict() == simulator.hyparams.to_dict()
    assert restored.laser.to_dict() == simulator.laser.to_dict()
    assert [d.to_dict() for d in restored.densities] == [
        d.to_dict() for d in simulator.densities
    ]
    assert restored.working_directory == tmp_path


def test_load_results_reads_diagnostics_under_the_given_directory(simulator, tmp_path, mocker):
    update = mocker.patch.object(simulator, "_update_output")

    simulator.load_results(tmp_path)

    update.assert_called_once_with(tmp_path / simulator.hyparams.save_directory / "hdf5")
    assert simulator.finished is True


def test_run_runs_the_simulation_then_loads_results(simulator, mocker):
    simulator.configure()
    order = []
    mocker.patch.object(simulator, "run_simulation", side_effect=lambda: order.append("run"))
    mocker.patch.object(simulator, "load_results", side_effect=lambda: order.append("load"))

    simulator.run()

    assert order == ["run", "load"]


def test_archive_with_a_lume_fbpic_profile_loads_in_a_fresh_process(simulator, tmp_path):
    """A batch worker is a new process: `lume_fbpic`'s own profiles must register on import, and a
    bare-electron profile (`species=None`, stored with `ionization=1`) must load back."""
    import subprocess
    import sys

    from lume_fbpic.density_profiles import LinearRampFlattop

    profile = LinearRampFlattop(
        nominal_density=1.0e24,
        species=None,
        ionization=-1,
        p_nz=1,
        p_nr=1,
        p_nt=1,
        ramp_start=1.0e-5,
        ramp_length=1.0e-5,
        elec_name="electrons",
    )
    ramp = FBPICSimulator(
        simulator.hyparams,
        simulator.laser,
        [profile],
        target_species="electrons",
        working_directory=tmp_path,
    )
    ramp.archive(tmp_path / "ramp.h5")

    code = (
        "from lume_fbpic.simulator import FBPICSimulator; "
        f"s = FBPICSimulator.from_archive({str(tmp_path / 'ramp.h5')!r}); "
        "print(type(s.densities[0]).__name__, s.densities[0].species, s.densities[0].ionization)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-3:] == ["LinearRampFlattop", "None", "1"]


def test_archive_records_its_kind_and_base_from_archive_returns_the_matching_class(
    simulator, tmp_path
):
    simulator.archive(tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert f.attrs["config_kind"] == "lwfa"
    assert type(BaseSimulator.from_archive(tmp_path / "a.h5")) is FBPICSimulator


def test_from_archive_of_another_kind_raises(simulator, tmp_path):
    class Other(BaseSimulator):
        CONFIG_KIND = "other-kind"

    try:
        simulator.archive(tmp_path / "a.h5")
        with h5py.File(tmp_path / "a.h5", "a") as f:
            f.attrs["config_kind"] = "other-kind"

        with pytest.raises(ValueError, match="cannot load"):
            FBPICSimulator.from_archive(tmp_path / "a.h5")
    finally:
        BaseSimulator._registry.pop("other-kind")


def test_from_archive_of_an_unknown_kind_raises(simulator, tmp_path):
    simulator.archive(tmp_path / "a.h5")
    with h5py.File(tmp_path / "a.h5", "a") as f:
        f.attrs["config_kind"] = "nope"

    with pytest.raises(ValueError, match="unknown kind"):
        BaseSimulator.from_archive(tmp_path / "a.h5")


def test_config_and_set_config_round_trip(simulator):
    config = simulator.config()
    original = simulator.laser
    simulator.laser = None

    simulator.set_config(config)

    assert simulator.laser is original
    assert config["densities"] is not simulator.densities
