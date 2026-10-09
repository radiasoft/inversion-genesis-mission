"""Tests for the PWFA config classes, `PWFASimulator`, and its actions."""

from __future__ import annotations

import sys
from pathlib import Path

import attrs
import h5py
import numpy
import pytest

from inversion_fbpic.lib.serializable_config import SerializableConfig
from lume_fbpic.actions import make_pwfa_actions
from lume_fbpic.density_profiles import LinearRampFlattop, UpDownRampProfile
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.pwfa_config import FlatTopBunch, GaussianBunch, PWFAGrid
from lume_fbpic.simulator import FBPICSimulator, PWFASimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "examples"))
import pwfa_clara_febe as clara_example  # noqa: E402
import pwfa_gaussian as pwfa_gaussian_example  # noqa: E402


@pytest.fixture
def pwfa_model(tmp_path) -> LUMEFBPICModel:
    """A flat-top-driver PWFA model (no witness) on a coarse grid; `dummy_run`, so nothing runs."""
    grid = PWFAGrid(
        zmin=-10.0e-6,
        zmax=30.0e-6,
        nz=200,
        rmax=20.0e-6,
        nr=100,
        n_steps=400,
        write_period=20,
    )
    plasma = LinearRampFlattop(
        nominal_density=1.0e24,
        species=None,
        ionization=-1,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        ramp_start=0.0,
        ramp_length=50.0e-6,
    )
    driver = FlatTopBunch(
        gamma=2.0e3, density=5.0e24, radius=2.0e-6, zmin=15.0e-6, zmax=20.0e-6
    )
    simulator = PWFASimulator(grid, plasma, driver, working_directory=tmp_path)
    return LUMEFBPICModel(simulator, make_pwfa_actions(simulator), dummy_run=True)


def test_flat_top_bunch_charge_is_the_density_times_the_cylinder_volume():
    bunch = FlatTopBunch(
        gamma=10.0, density=1.0e24, radius=1.0e-6, zmin=0.0, zmax=1.0e-5
    )

    assert bunch.charge == pytest.approx(
        1.602176634e-19 * 1.0e24 * 3.141592653589793e-12 * 1.0e-5
    )


def test_flat_top_bunch_rejects_a_reversed_extent():
    with pytest.raises(ValueError, match="zmax"):
        FlatTopBunch(
            gamma=10.0, density=1.0e24, radius=1.0e-6, zmin=2.0e-6, zmax=1.0e-6
        )


def test_grid_defaults_to_one_cell_of_light_travel_per_step():
    grid = PWFAGrid(zmin=0.0, zmax=3.0e-5, nz=100, rmax=1.0e-5, nr=10, n_steps=5)

    assert grid.time_step == pytest.approx(3.0e-5 / 100 / 299792458.0)
    assert attrs.evolve(grid, dt=1.0e-15).time_step == 1.0e-15


@pytest.mark.parametrize(
    "changes",
    [{"z_boundary": "reflective"}, {"r_boundary": "periodic"}, {"zmax": -1.0}],
)
def test_grid_rejects_bad_values(changes):
    kwargs = dict(zmin=0.0, zmax=3.0e-5, nz=100, rmax=1.0e-5, nr=10, n_steps=5)
    with pytest.raises(ValueError):
        PWFAGrid(**{**kwargs, **changes})


def test_configure_rejects_a_target_species_that_does_not_exist(pwfa_model):
    pwfa_model.simulator.target_species = "witness"

    with pytest.raises(ValueError, match="target_species"):
        pwfa_model.simulator.configure()


def test_the_model_gets_the_pwfa_actions(pwfa_model):
    assert {"driver_gamma", "driver_density", "driver_radius", "plasma_density"} <= set(
        pwfa_model.supported_variables
    )
    assert "witness_gamma" not in pwfa_model.supported_variables
    assert "laser_energy" not in pwfa_model.supported_variables


def test_the_descriptor_actions_work_on_a_pwfa_simulator(pwfa_model, particle_group):
    from lume_fbpic.actions import make_descriptor_actions

    model = LUMEFBPICModel(
        pwfa_model.simulator, make_descriptor_actions(uz_min=1.0), dummy_run=True
    )
    pwfa_model.simulator.final_particles = particle_group

    values = model.get(["descriptor_mean_uz", "descriptor_total_beam_charge_pc"])

    assert (
        100 < values["descriptor_mean_uz"] < 200
    )  # the synthetic bunch's mean uz is ~150
    assert values["descriptor_total_beam_charge_pc"] > 0


def test_a_witness_adds_witness_actions(pwfa_model):
    simulator = pwfa_model.simulator
    witness = FlatTopBunch(
        gamma=500.0, density=1.0e23, radius=1.0e-6, zmin=0.0, zmax=1.0e-6
    )
    with_witness = PWFASimulator(
        simulator.grid,
        simulator.plasma,
        simulator.driver,
        witness,
        target_species="witness",
    )

    model = LUMEFBPICModel(
        with_witness, make_pwfa_actions(with_witness), dummy_run=True
    )

    assert {"witness_gamma", "witness_density"} <= set(model.supported_variables)


def test_setting_actions_replaces_the_frozen_configs(pwfa_model):
    original_driver = pwfa_model.simulator.driver

    pwfa_model.set({"driver_gamma": 1000.0, "plasma_density": 2.0e24})

    assert pwfa_model.simulator.driver is not original_driver
    assert pwfa_model.simulator.driver.gamma == 1000.0
    assert pwfa_model.simulator.plasma.nominal_density == 2.0e24
    assert pwfa_model.get(["driver_gamma"]) == {"driver_gamma": 1000.0}


def test_a_bunch_action_for_a_missing_witness_raises(pwfa_model):
    action = pwfa_model.supported_variables["driver_gamma"]
    missing = type(action)(name="w", bunch="witness", field_name="gamma")

    with pytest.raises(ValueError, match="no witness"):
        missing._get(pwfa_model.simulator)


def test_bunch_actions_come_back_from_an_archive(pwfa_model, tmp_path):
    pwfa_model.archive(tmp_path / "p.h5")

    restored = LUMEFBPICModel.from_archive(tmp_path / "p.h5")

    action = restored.supported_variables["driver_gamma"]
    assert type(action).__name__ == "BunchFieldAction"
    assert action.bunch == "driver" and action.field_name == "gamma"


def test_archive_round_trips_a_pwfa_model_and_picks_the_pwfa_class(
    pwfa_model, tmp_path
):
    pwfa_model.archive(tmp_path / "p.h5")

    with h5py.File(tmp_path / "p.h5") as f:
        assert f.attrs["config_kind"] == "pwfa"
    restored = LUMEFBPICModel.from_archive(tmp_path / "p.h5")

    assert type(restored.simulator) is PWFASimulator
    assert {k: v.to_dict() for k, v in restored.simulator.config().items()} == {
        k: v.to_dict() for k, v in pwfa_model.simulator.config().items()
    }
    assert restored.get(["driver_gamma"]) == pwfa_model.get(["driver_gamma"])


def test_an_lwfa_class_cannot_load_a_pwfa_archive(pwfa_model, tmp_path):
    pwfa_model.archive(tmp_path / "p.h5")

    with pytest.raises(ValueError, match="cannot load"):
        FBPICSimulator.from_archive(tmp_path / "p.h5")


def test_a_short_run_writes_the_driver_and_reads_it_back(tmp_path):
    simulator = _short_run_simulator(tmp_path)

    simulator.run()

    assert simulator.finished is True
    assert (tmp_path / "diags" / "hdf5" / "data00000002.h5").exists()
    assert simulator.stats["charge_pc"] == pytest.approx(
        simulator.driver.charge * 1e12, rel=0.05
    )
    assert simulator.stats["energy_mean_mev"] == pytest.approx(1021.5, rel=0.01)


def test_gaussian_bunch_charge_is_its_field_and_rejects_bad_values():
    bunch = GaussianBunch(
        gamma=100.0, charge=1.0e-10, sig_r=1.0e-6, sig_z=2.0e-6, zf=0.0
    )

    assert bunch.charge == 1.0e-10
    assert bunch.tf == 0.0 and bunch.n_emit == 0.0
    with pytest.raises(ValueError):
        GaussianBunch(gamma=100.0, charge=-1.0e-10, sig_r=1.0e-6, sig_z=2.0e-6, zf=0.0)


def test_the_gaussian_example_has_a_witness_and_gaussian_actions():
    model = pwfa_gaussian_example.build_model(dummy_run=True)

    names = set(model.supported_variables)
    assert {
        "driver_charge",
        "driver_sigma_z",
        "driver_position",
        "witness_charge",
    } <= names
    assert "driver_density" not in names
    assert model.simulator.target_species == "witness"
    assert model.simulator.grid.nz == 209 and model.simulator.grid.nr == 64
    assert model.simulator.grid.n_steps == 281

    model.set({"witness_position": 1.0e-4})
    assert model.simulator.witness.zf == 1.0e-4


def test_a_short_run_with_gaussian_bunches_reads_the_witness_back(tmp_path):
    model = pwfa_gaussian_example.build_model(n_steps=3)
    simulator = model.simulator
    simulator.working_directory = tmp_path
    simulator.grid = attrs.evolve(simulator.grid, nz=40, nr=16, write_period=2)
    simulator.driver = attrs.evolve(simulator.driver, n_macroparticles=200)
    simulator.witness = attrs.evolve(simulator.witness, n_macroparticles=100)
    simulator.configure()

    simulator.run()

    assert simulator.finished is True
    assert simulator.stats["charge_pc"] == pytest.approx(100.0, rel=1e-6)
    assert simulator.stats["energy_mean_mev"] == pytest.approx(1.0e4, rel=0.01)
    assert len(simulator.final_particles) == 100


def test_the_gaussian_example_plots_the_fields_of_a_short_run(tmp_path):
    pytest.importorskip("matplotlib")
    model = pwfa_gaussian_example.build_model(n_steps=3)
    simulator = model.simulator
    simulator.working_directory = tmp_path
    simulator.grid = attrs.evolve(simulator.grid, nz=40, nr=16, write_period=2)
    simulator.driver = attrs.evolve(simulator.driver, n_macroparticles=200)
    simulator.witness = attrs.evolve(simulator.witness, n_macroparticles=100)
    simulator.configure()
    simulator.run()

    files = pwfa_gaussian_example.plot_results(tmp_path, output=tmp_path / "plots")

    assert [f.name for f in files] == [
        "electron_density.png",
        "longitudinal_field.png",
        "transverse_force.png",
        "longitudinal_field_lineout.png",
    ]
    assert all(f.stat().st_size > 1000 for f in files)
    with pytest.raises(FileNotFoundError, match="no dumps"):
        pwfa_gaussian_example.plot_results(tmp_path / "plots")


def _short_run_simulator(tmp_path, **grid_changes) -> PWFASimulator:
    """A configured 3-step run of a flat-top driver in a bare-electron plasma, with dumps at step 2."""
    grid = PWFAGrid(
        zmin=-10.0e-6,
        zmax=30.0e-6,
        nz=40,
        rmax=20.0e-6,
        nr=20,
        n_steps=3,
        write_period=2,
        **grid_changes,
    )
    plasma = LinearRampFlattop(
        nominal_density=1.0e24,
        species=None,
        ionization=-1,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        ramp_start=0.0,
        ramp_length=5.0e-5,
    )
    driver = FlatTopBunch(
        gamma=2.0e3, density=5.0e24, radius=2.0e-6, zmin=15.0e-6, zmax=20.0e-6
    )
    simulator = PWFASimulator(grid, plasma, driver, working_directory=tmp_path)
    simulator.configure()
    return simulator


def _up_down_ramp(**changes) -> UpDownRampProfile:
    kwargs = dict(
        nominal_density=4.2e22,
        species=None,
        ionization=-1,
        p_nz=4,
        p_nr=4,
        p_nt=4,
        base_fraction=2.0 / 3.0,
        z_up=0.05,
        up_length=0.01,
        z_down=0.15,
        down_length=0.01,
        z_end=0.2,
    )
    return UpDownRampProfile(**{**kwargs, **changes})


def test_up_down_ramp_has_five_stages_and_no_plasma_outside_the_cell():
    import numpy

    dens = _up_down_ramp().build_density_function()
    z = numpy.array(
        [-1.0e-3, 0.0, 0.04, 0.055, 0.06, 0.1, 0.155, 0.16, 0.18, 0.2, 0.25]
    )

    relative = dens(z, numpy.zeros_like(z))

    numpy.testing.assert_allclose(
        relative, [0, 2 / 3, 2 / 3, 5 / 6, 1, 1, 5 / 6, 2 / 3, 2 / 3, 0, 0], atol=1e-12
    )


def test_up_down_ramp_has_no_end_by_default_and_round_trips_through_yaml():
    import numpy

    profile = _up_down_ramp(z_end=None)

    assert profile.build_density_function()(numpy.array([10.0]), numpy.array([0.0]))[
        0
    ] == pytest.approx(2 / 3)
    loaded = SerializableConfig.from_yaml(profile.to_yaml())
    assert loaded.to_dict() == profile.to_dict()


@pytest.mark.parametrize(
    "changes",
    [{"z_up": -0.1}, {"z_down": 0.055}, {"z_end": 0.155}, {"base_fraction": 1.5}],
)
def test_up_down_ramp_rejects_inconsistent_stages(changes):
    with pytest.raises(ValueError):
        _up_down_ramp(**changes)


@pytest.mark.parametrize(
    "witness, n0_fraction", [("longer", 2.8 / 4.2), ("shorter", 1.9 / 4.2)]
)
def test_the_clara_example_follows_the_papers_tables(witness, n0_fraction):
    model = clara_example.build_model(witness=witness, dummy_run=True)
    simulator = model.simulator

    assert simulator.grid.nr == 200 and simulator.grid.nm == 1
    assert (
        simulator.grid.r_boundary == "reflective"
        and simulator.grid.z_boundary == "open"
    )
    assert simulator.plasma.base_fraction == pytest.approx(n0_fraction)
    assert simulator.driver.charge == pytest.approx(150.0e-12)
    assert simulator.driver.zf - simulator.witness.zf == pytest.approx(90.0e-6)
    assert simulator.target_species == "witness"


@pytest.mark.parametrize(
    "arguments, nz, n_steps, time_step, plasma_ppc",
    [
        (
            {},
            100,
            133_333,
            1.5e-6 / 299792458.0,
            (2, 2, 4),
        ),  # the coarser 20 cm run; the paper: 500, 1 fs, (4, 4, 4)
        (
            {"dz": 0.30e-6, "plasma_ppc": (4, 4, 4)},
            500,
            666_667,
            1.0e-15,
            (4, 4, 4),
        ),  # the paper's
    ],
    ids=["defaults", "the_papers_resolution"],
)
def test_the_clara_example_resolution(arguments, nz, n_steps, time_step, plasma_ppc):
    simulator = clara_example.build_model(dummy_run=True, **arguments).simulator

    assert simulator.grid.nz == nz
    assert simulator.grid.n_steps == n_steps  # the whole 20 cm cell
    assert simulator.grid.time_step == pytest.approx(time_step, rel=1e-3)
    assert (
        simulator.plasma.p_nz,
        simulator.plasma.p_nr,
        simulator.plasma.p_nt,
    ) == plasma_ppc
    assert simulator.grid.write_plasma is False


def test_a_seeded_run_restores_the_global_random_state(tmp_path):
    simulator = _short_run_simulator(tmp_path / "seeded", random_seed=7)
    numpy.random.seed(123)
    before = numpy.random.get_state()

    simulator.run()

    after = numpy.random.get_state()
    assert after[0] == before[0] and after[2:] == before[2:]
    assert numpy.array_equal(after[1], before[1])


def test_a_grid_without_plasma_output_still_runs(tmp_path):
    simulator = _short_run_simulator(tmp_path, write_plasma=False)

    simulator.run()

    with h5py.File(tmp_path / "diags" / "hdf5" / "data00000002.h5") as f:
        assert sorted(f["data/2/particles"]) == ["driver"]


def _write_witness_dump(directory, iteration, time, *, outlier=False):
    """A fbpic-style dump at `time` [s]: a witness of 1000 equal-weight electrons (with `outlier`,
    one far outside the 5-sigma cut), a driver of 500, and small field arrays."""
    from scipy.constants import c, e, m_e

    rng = numpy.random.default_rng(iteration)
    path = directory / "diags" / "hdf5" / f"data{iteration:08d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        step = f.create_group(f"data/{iteration}")
        step.attrs["time"] = time
        for name, n, z_centre, sigma_r, sigma_z in (
            ("witness", 1000, -90.0e-6, 14.0e-6, 10.0e-6),
            ("driver", 500, -30.0e-6, 70.0e-6, 10.0e-6),
        ):
            x, y = rng.normal(0.0, sigma_r, n), rng.normal(0.0, sigma_r, n)
            z = rng.normal(z_centre, sigma_z, n)
            ux, uy, uz = (
                rng.normal(0.0, 0.01, n),
                rng.normal(0.0, 0.01, n),
                numpy.full(n, 500.0),
            )
            if outlier and name == "witness":
                x[0] = 1.0e-3
            particles = step.create_group(f"particles/{name}")
            for axis, values in (("x", x), ("y", y), ("z", z)):
                particles[f"position/{axis}"] = values
            for axis, values in (("x", ux), ("y", uy), ("z", uz)):
                particles[f"momentum/{axis}"] = values * m_e * c
            particles["weighting"] = numpy.full(n, 1.0e-12 / e / n)
        mesh = step.create_group("fields/E")
        mesh.attrs["gridSpacing"] = [0.75e-6, 1.5e-6]
        mesh.attrs["gridGlobalOffset"] = [0.0, -150.0e-6]
        for name in ("r", "t", "z"):
            mesh[name] = rng.normal(size=(1, 12, 100)) * 1.0e8
        step.create_group("fields/B")["t"] = rng.normal(size=(1, 12, 100))
    return path


def test_the_dump_metrics_follow_the_papers_cut(tmp_path):
    path = _write_witness_dump(tmp_path, 10, 1.0e-10, outlier=True)

    distance, energy, spread, charge, emittance, size = clara_example._dump_metrics(
        path, clara_example.CASES["longer"]
    )

    assert distance == pytest.approx(100.0 * 299792458.0 * 1.0e-10)
    assert charge == pytest.approx(
        1.0 * 999.0 / 1000.0, rel=1e-6
    )  # pC: the outlier is cut
    assert energy == pytest.approx(500.0 * 0.51099895, rel=1e-3)
    assert size == pytest.approx(14.0, rel=0.1)
    assert emittance == pytest.approx(
        14.0 * 0.01, rel=0.2
    )  # sigma_x * sigma_ux = 0.14 mm mrad
    assert spread < 0.1


def test_plot_results_writes_the_four_plots_from_the_dumps(tmp_path):
    pytest.importorskip("matplotlib")
    for iteration, time in ((0, 0.0), (10, 1.0e-10), (20, 2.0e-10)):
        _write_witness_dump(tmp_path, iteration, time)
    expected = [
        "transverse_size.png",
        "beam_quality_with_ramp_and_500MeV_markers.png",
        "beam_density_and_wakefield.png",
        "offaxis_transverse_wakefield.png",
    ]

    files = clara_example.plot_results(
        tmp_path, witness="longer", output=tmp_path / "plots"
    )

    assert [f.name for f in files] == expected
    assert all(f.stat().st_size > 1000 for f in files)
    shorter = clara_example.plot_results(
        tmp_path, witness="shorter", output=tmp_path / "plots_shorter"
    )
    assert [f.name for f in shorter] == expected


def test_plot_results_needs_dumps(tmp_path):
    with pytest.raises(FileNotFoundError, match="no dumps"):
        clara_example.plot_results(tmp_path)
