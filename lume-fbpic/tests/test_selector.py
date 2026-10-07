"""Tests for `ArchiveSelector` (docs/htu-twin/serving/selector.py)."""

from __future__ import annotations

import math

import numpy
import pytest
from scipy.constants import c, e, m_e

from tests.downramp_actions import make_actions
from lume_fbpic.actions import make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel
from selector import ArchiveSelector, _slice_charge_fractions, particles_from_descriptor

from inversion_fbpic.utils import distributions

from beamphysics import ParticleGroup

_MC2_EV = m_e * c**2 / e


def _bunch(uz_mean: float, charge_pc: float, seed: int = 0, n: int = 3000) -> ParticleGroup:
    rng = numpy.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.0, 2.0, n) * _MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * _MC2_EV,
            "pz": rng.normal(uz_mean, 30.0, n) * _MC2_EV,
            "t": numpy.zeros(n),
            "status": numpy.ones(n, dtype=int),
            "weight": numpy.full(n, charge_pc * 1.0e-12 / n),
            "species": "electron",
        }
    )


def _descriptor_outputs(pg) -> dict[str, float]:
    phase_space = numpy.stack(
        [pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1
    )
    descriptor = distributions.compute_moment_descriptor(phase_space, numpy.asarray(pg.weight) / e)
    return {f"descriptor_{k}": v for k, v in descriptor.items()}


def _reconstructed_model(simulator, tmp_path, name, uz_mean, charge_pc) -> LUMEFBPICModel:
    """A model loaded from an archive that records a descriptor and holds no particles."""
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    source = LUMEFBPICModel(simulator, actions, dummy_run=True)
    source.simulator.stats = _descriptor_outputs(_bunch(uz_mean, charge_pc))
    source.archive(tmp_path / f"{name}.h5")
    source.simulator.stats = {}
    return LUMEFBPICModel.from_archive(tmp_path / f"{name}.h5", dummy_run=True)


@pytest.fixture()
def selector(simulator, tmp_path) -> ArchiveSelector:
    models = {
        "run_a": _reconstructed_model(simulator, tmp_path, "run_a", 100.0, 200.0),
        "run_b": _reconstructed_model(simulator, tmp_path, "run_b", 200.0, 400.0),
    }
    return ArchiveSelector(models, synthetic_bunch_particles=2000)


def _charge(selector) -> float:
    return selector.get(["descriptor_total_beam_charge_pc"])["descriptor_total_beam_charge_pc"]


def test_the_first_run_is_active_at_the_start(selector):
    assert selector.active == "run_a"
    assert selector.get(["LPA_Archive"])["LPA_Archive"] == "run_a"


def test_the_enum_lists_the_runs_in_order(selector):
    variable = selector.supported_variables["LPA_Archive"]

    assert list(variable.options) == ["run_a", "run_b"]
    assert variable.read_only is False


def test_selecting_a_run_switches_the_outputs(selector):
    assert _charge(selector) == pytest.approx(200.0)

    selector.set({"LPA_Archive": "run_b"})

    assert selector.active == "run_b"
    assert _charge(selector) == pytest.approx(400.0)
    assert selector.get(["descriptor_mean_uz"])["descriptor_mean_uz"] == pytest.approx(200.0, rel=0.02)


def test_the_final_particles_follow_the_selection(selector):
    a = selector.final_particles
    selector.set({"LPA_Archive": "run_b"})
    b = selector.final_particles

    assert selector.models["run_a"].final_particles is None  # the runs hold none: these are built
    assert a.charge == pytest.approx(200.0e-12) and b.charge == pytest.approx(400.0e-12)


def test_an_unknown_run_is_refused_and_the_selection_is_unchanged(selector):
    with pytest.raises(ValueError, match="allowed"):
        selector.set({"LPA_Archive": "no_such_run"})

    assert selector.active == "run_a"


def test_the_variables_are_the_active_runs_plus_the_selector(selector):
    names = set(selector.supported_variables)

    assert names == set(selector.active_model.supported_variables) | {"LPA_Archive"}
    assert names == set(selector.models["run_b"].supported_variables) | {"LPA_Archive"}


def test_an_input_goes_to_the_active_run_only(selector):
    other_energy = selector.models["run_b"].get("laser_energy")

    selector.set({"laser_energy": 3.3})

    assert selector.models["run_a"].get("laser_energy") == 3.3
    assert selector.models["run_b"].get("laser_energy") == other_energy


def test_a_selection_and_an_input_can_be_set_together(selector):
    selector.set({"LPA_Archive": "run_b", "laser_energy": 4.4})

    assert selector.active == "run_b"
    assert selector.models["run_b"].get("laser_energy") == 4.4


def test_reset_makes_the_first_run_active_and_resets_every_run(selector):
    selector.set({"LPA_Archive": "run_b"})
    selector.set({"laser_energy": 9.0})

    selector.reset()

    assert selector.active == "run_a"
    assert selector.models["run_b"].get("laser_energy") != 9.0
    assert _charge(selector) == pytest.approx(200.0)


def test_a_single_run_is_a_selector_with_one_option(simulator, tmp_path):
    model = _reconstructed_model(simulator, tmp_path, "only", 150.0, 100.0)

    selector = ArchiveSelector({"only": model})

    assert list(selector.supported_variables["LPA_Archive"].options) == ["only"]
    assert selector.get(["LPA_Archive"])["LPA_Archive"] == "only"


def test_runs_with_different_variables_are_refused(simulator, tmp_path):
    a = _reconstructed_model(simulator, tmp_path, "a", 100.0, 100.0)
    fewer = LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True)  # no descriptors

    with pytest.raises(ValueError, match="different variables"):
        ArchiveSelector({"a": a, "fewer": fewer})


def test_no_models_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        ArchiveSelector({})


def test_the_selector_name_must_not_clash_with_a_variable(simulator, tmp_path):
    a = _reconstructed_model(simulator, tmp_path, "a", 100.0, 100.0)

    with pytest.raises(ValueError, match="already"):
        ArchiveSelector({"a": a}, selector_name="charge_pc")


def test_a_custom_selector_name_is_used(simulator, tmp_path):
    a = _reconstructed_model(simulator, tmp_path, "a", 100.0, 100.0)

    selector = ArchiveSelector({"a": a}, selector_name="Run")

    assert selector.get(["Run"])["Run"] == "a"
    assert "LPA_Archive" not in selector.supported_variables


def test_nan_outputs_stay_nan_for_runs_without_stats(selector):
    assert math.isnan(selector.get(["charge_pc"])["charge_pc"])


def test_a_run_without_particles_has_none_unless_a_synthetic_bunch_is_asked_for(simulator, tmp_path):
    model = _reconstructed_model(simulator, tmp_path, "run_a", 100.0, 200.0)

    assert ArchiveSelector({"a": model}).final_particles is None
    bunch = ArchiveSelector({"a": model}, synthetic_bunch_particles=2000).final_particles

    assert len(bunch) == 2000
    assert model.simulator.final_particles is None  # the model itself never holds the bunch
    assert model.final_particles is None


def test_the_synthetic_bunch_is_built_from_the_active_runs_descriptor(selector):
    first = selector.final_particles
    again = selector.final_particles
    charge_a = float(numpy.sum(first.weight))

    selector.set({"LPA_Archive": "run_b"})

    assert numpy.array_equal(first.x, again.x)  # built from the same recorded values each time
    assert float(numpy.sum(selector.final_particles.weight)) == pytest.approx(2.0 * charge_a, rel=1e-6)


def test_real_particles_take_precedence_over_a_synthetic_bunch(simulator, tmp_path):
    model = _reconstructed_model(simulator, tmp_path, "run_a", 100.0, 200.0)
    particles = _bunch(100.0, 200.0)
    model.simulator.final_particles = particles

    selector = ArchiveSelector({"a": model}, synthetic_bunch_particles=2000)

    assert selector.final_particles is particles


def test_the_descriptor_outputs_stay_the_recorded_values(selector):
    recorded = _descriptor_outputs(_bunch(100.0, 200.0))

    got = selector.get(["descriptor_mean_uz", "descriptor_cov_uz_uz"])

    assert got["descriptor_mean_uz"] == recorded["descriptor_mean_uz"]  # not recomputed from the bunch
    assert got["descriptor_cov_uz_uz"] == recorded["descriptor_cov_uz_uz"]


def test_a_run_with_no_recorded_descriptor_cannot_be_synthesized(simulator, tmp_path):
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    LUMEFBPICModel(simulator, actions, dummy_run=True).archive(tmp_path / "none.h5")
    model = LUMEFBPICModel.from_archive(tmp_path / "none.h5", dummy_run=True)

    with pytest.raises(ValueError, match="no recorded descriptor"):
        ArchiveSelector({"a": model}, synthetic_bunch_particles=2000)


def test_an_incomplete_recorded_descriptor_cannot_be_synthesized(simulator, tmp_path):
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    source = LUMEFBPICModel(simulator, actions, dummy_run=True)
    source.simulator.stats = {"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_pc": 510.0}
    source.archive(tmp_path / "some.h5")
    source.simulator.stats = {}
    model = LUMEFBPICModel.from_archive(tmp_path / "some.h5", dummy_run=True)

    with pytest.raises(ValueError, match="lacks"):
        ArchiveSelector({"a": model}, synthetic_bunch_particles=2000)


def test_a_synthetic_bunch_needs_at_least_two_particles(simulator, tmp_path):
    model = _reconstructed_model(simulator, tmp_path, "run_a", 100.0, 200.0)

    with pytest.raises(ValueError, match="at least 2"):
        ArchiveSelector({"a": model}, synthetic_bunch_particles=1)


def _lab_snapshot(n=4000, seed=0) -> ParticleGroup:
    rng = numpy.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.0, 2.0, n) * _MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * _MC2_EV,
            "pz": rng.normal(150.0, 40.0, n) * _MC2_EV,
            "t": numpy.zeros(n),
            "status": numpy.ones(n, dtype=int),
            "weight": numpy.full(n, 1.0e3 * e),
            "species": "electron",
        }
    )


def _descriptor_of(pg) -> dict[str, float]:
    phase_space = numpy.stack(
        [pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1
    )
    return distributions.compute_moment_descriptor(phase_space, numpy.asarray(pg.weight) / e)


def _phase_space(pg):
    return numpy.stack([pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1)


def test_a_synthetic_bunch_reproduces_the_descriptors_moments():
    descriptor = _descriptor_of(_lab_snapshot(n=20_000))

    synthetic = particles_from_descriptor(descriptor, n_particles=100_000, seed=3)

    again = distributions.compute_moment_descriptor(
        _phase_space(synthetic), numpy.asarray(synthetic.weight) / e
    )
    assert again["total_beam_charge_pc"] == pytest.approx(descriptor["total_beam_charge_pc"], rel=1e-9)
    assert again["mean_uz"] == pytest.approx(descriptor["mean_uz"], rel=0.01)
    for key in ("cov_uz_uz", "cov_x_x", "cov_y_y", "cov_ux_ux", "cov_uy_uy", "cov_z_z"):
        assert again[key] == pytest.approx(descriptor[key], rel=0.05), key
    for a, b, key in (("x", "ux", "cov_x_ux"), ("y", "uy", "cov_y_uy")):
        scale = numpy.sqrt(descriptor[f"cov_{a}_{a}"] * descriptor[f"cov_{b}_{b}"])
        assert abs(again[key] - descriptor[key]) < 0.02 * scale, key  # correlation within 0.02


def test_a_synthetic_bunch_is_in_the_bunch_frame_with_equal_charges():
    descriptor = _descriptor_of(_lab_snapshot())

    synthetic = particles_from_descriptor(descriptor, n_particles=1000)

    assert len(synthetic) == 1000
    assert numpy.allclose(numpy.asarray(synthetic.t), -numpy.asarray(synthetic.z) / c)
    assert numpy.all(numpy.asarray(synthetic.status) == 1)
    weight = numpy.asarray(synthetic.weight)
    assert numpy.allclose(weight, weight[0])
    assert weight.sum() == pytest.approx(descriptor["total_beam_charge_pc"] * 1e-12)
    assert numpy.asarray(synthetic.pz).min() / _MC2_EV >= 1.0  # no non-physical negative energies


def test_a_synthetic_bunch_depends_on_the_seed_only():
    descriptor = _descriptor_of(_lab_snapshot())

    a = particles_from_descriptor(descriptor, n_particles=500, seed=1)
    b = particles_from_descriptor(descriptor, n_particles=500, seed=1)
    other = particles_from_descriptor(descriptor, n_particles=500, seed=2)

    assert numpy.array_equal(numpy.asarray(a.x), numpy.asarray(b.x))
    assert not numpy.array_equal(numpy.asarray(a.x), numpy.asarray(other.x))


def test_energy_rises_along_z_when_the_descriptors_bins_do():
    descriptor = {**_descriptor_of(_lab_snapshot()), **{
        f"longitudinal_mean_uz_{i:02d}": m for i, m in enumerate((60.0, 90.0, 200.0, 400.0))
    }, **{f"longitudinal_rms_uz_{i:02d}": 25.0 for i in range(4)}}
    descriptor["mean_uz"] = 180.0
    descriptor["cov_uz_uz"] = 17_000.0

    synthetic = particles_from_descriptor(descriptor, n_particles=40_000, seed=0)

    z = numpy.asarray(synthetic.z)
    uz = numpy.asarray(synthetic.pz) / _MC2_EV
    assert numpy.corrcoef(z, uz)[0, 1] > 0.6  # the head (larger z) carries the high energies


def test_a_descriptor_missing_a_feature_is_an_error():
    descriptor = _descriptor_of(_lab_snapshot())
    del descriptor["cov_x_ux"]

    with pytest.raises(ValueError, match="cov_x_ux"):
        particles_from_descriptor(descriptor)


def test_slice_charge_fractions_reproduce_the_mean_and_variance():
    means, rms = numpy.array([62.0, 84.0, 205.0, 404.0]), numpy.array([23.0, 29.0, 72.0, 52.0])

    fractions = _slice_charge_fractions(means, rms, mean=171.6, variance=17_148.0)

    assert fractions.sum() == pytest.approx(1.0)
    assert numpy.all(fractions >= 0)
    assert fractions @ means == pytest.approx(171.6)
    assert fractions @ (rms**2 + means**2) - (fractions @ means) ** 2 == pytest.approx(17_148.0)


def test_inconsistent_bins_fall_back_to_equal_charge_with_a_warning():
    means, rms = numpy.array([10.0, 20.0, 30.0, 40.0]), numpy.array([1.0, 1.0, 1.0, 1.0])

    with pytest.warns(UserWarning, match="equal charge"):
        fractions = _slice_charge_fractions(means, rms, mean=500.0, variance=1.0)

    assert numpy.allclose(fractions, 0.25)

