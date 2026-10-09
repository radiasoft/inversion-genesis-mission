"""Tests for the moment-descriptor actions (`MomentDescriptorAction`, `make_descriptor_actions`)."""

from __future__ import annotations

import collections
import math

import numpy
import pytest
from scipy.constants import e
from lume_fbpic.simulator import ELECTRON_MC2_EV

from downramp_actions import make_actions
from lume_fbpic.actions import make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel

from inversion_fbpic.utils import distributions

from beamphysics import ParticleGroup

_ELECTRONS_PER_MACROPARTICLE = 1.0e3


def _particle_group(n=3000, seed=0, uz_mean=150.0, uz_std=30.0) -> ParticleGroup:
    rng = numpy.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(1.0e-6, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.5, 2.0, n) * ELECTRON_MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * ELECTRON_MC2_EV,
            "pz": rng.normal(uz_mean, uz_std, n) * ELECTRON_MC2_EV,
            "t": numpy.zeros(n),
            "status": numpy.ones(n, dtype=int),
            "weight": numpy.full(n, _ELECTRONS_PER_MACROPARTICLE * e),
            "species": "electron",
        }
    )


@pytest.fixture()
def descriptor_model(simulator) -> LUMEFBPICModel:
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    return LUMEFBPICModel(simulator, actions, dummy_run=True)


def _reference(pg, uz_min=30.0, central_fraction=0.95) -> dict[str, float]:
    """The descriptor computed directly with the library, as `build_dataset.py` would."""
    phase_space = numpy.stack(
        [
            pg.x,
            pg.px / ELECTRON_MC2_EV,
            pg.y,
            pg.py / ELECTRON_MC2_EV,
            pg.z,
            pg.pz / ELECTRON_MC2_EV,
        ],
        axis=-1,
    )
    weights = numpy.asarray(pg.weight) / e
    phase_space, weights = distributions.select_by_uz(
        phase_space, weights, uz_min=uz_min
    )
    phase_space, weights = distributions.crop_central_particles(
        phase_space, weights, central_fraction=central_fraction
    )
    return distributions.compute_moment_descriptor(phase_space, weights)


def test_actions_are_the_33_read_only_features_in_library_order():
    actions = make_descriptor_actions()

    keys = list(_reference(_particle_group()))
    assert [a.name for a in actions] == [f"descriptor_{k}" for k in keys]
    assert len(actions) == 33
    assert all(a.read_only for a in actions)
    assert actions[-1].unit == "pC"


def test_cache_is_recomputed_when_final_particles_change(
    descriptor_model, simulator, mocker
):
    spy = mocker.spy(distributions, "compute_moment_descriptor")
    mocker.patch("lume_fbpic.actions.compute_moment_descriptor", spy)
    names = ["descriptor_mean_uz", "descriptor_cov_uz_uz"]

    simulator.final_particles = _particle_group(seed=1)
    descriptor_model.get(names)
    descriptor_model.get(["descriptor_total_beam_charge_pc"])
    assert spy.call_count == 1  # one computation shared by every feature

    simulator.final_particles = _particle_group(seed=2)
    descriptor_model.get(names)
    assert spy.call_count == 2


def test_charge_is_the_selected_weights_in_picocoulombs(descriptor_model, simulator):
    pg = _particle_group()
    simulator.final_particles = pg

    charge = descriptor_model.get(["descriptor_total_beam_charge_pc"])[
        "descriptor_total_beam_charge_pc"
    ]

    selected = numpy.count_nonzero(numpy.asarray(pg.pz) / ELECTRON_MC2_EV >= 30.0)
    assert charge == pytest.approx(_reference(pg)["total_beam_charge_pc"], rel=1e-12)
    assert (
        0 < charge <= selected * _ELECTRONS_PER_MACROPARTICLE * e * 1e12
    )  # the crop only removes


def test_each_descriptor_action_has_the_unit_of_its_feature():
    units = {action.feature: action.unit for action in make_descriptor_actions()}

    assert units["mean_ux"] == units["mean_uz"] == "1"  # normalized momentum
    assert units["cov_x_x"] == units["cov_z_z"] == "m^2"
    assert units["cov_x_ux"] == units["cov_y_uz"] == units["cov_z_uz"] == "m"
    assert units["cov_ux_ux"] == units["cov_uy_uz"] == "1"
    assert units["longitudinal_mean_uz_00"] == units["longitudinal_rms_uz_03"] == "1"
    assert units["total_beam_charge_pc"] == "pC"
    assert collections.Counter(units.values()) == {"1": 17, "m": 9, "m^2": 6, "pC": 1}


def test_every_descriptor_unit_is_a_beamphysics_unit():
    from beamphysics.units import pmd_unit

    for action in make_descriptor_actions():
        pmd_unit(action.unit)  # raises for an unknown unit


def test_selection_parameters_change_the_result(simulator):
    pg = _particle_group()
    simulator.final_particles = pg
    name = "descriptor_total_beam_charge_pc"

    def charge(**selection):
        model = LUMEFBPICModel(
            simulator, make_descriptor_actions(**selection), dummy_run=True
        )
        return model.get([name])[name]

    everything = charge(uz_min=None, central_fraction=None)
    assert everything == pytest.approx(
        len(numpy.asarray(pg.x)) * _ELECTRONS_PER_MACROPARTICLE * e * 1e12
    )
    assert charge(uz_min=170.0, central_fraction=0.5) < everything


def test_values_match_the_library_descriptor(descriptor_model, simulator):
    pg = _particle_group()
    simulator.final_particles = pg
    expected = _reference(pg)

    got = descriptor_model.get([f"descriptor_{key}" for key in expected])

    for key, value in expected.items():
        assert got[f"descriptor_{key}"] == pytest.approx(value, rel=1e-12)


def test_values_are_nan_before_a_run(descriptor_model):
    value = descriptor_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"]

    assert math.isnan(value)


def test_values_are_nan_when_the_selection_leaves_too_few_particles(
    descriptor_model, simulator
):
    simulator.final_particles = _particle_group(
        uz_mean=2.0, uz_std=0.5
    )  # none reach uz >= 30

    value = descriptor_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"]

    assert math.isnan(value)


def test_descriptor_actions_cannot_be_set(descriptor_model):
    from lume.exceptions import ReadOnlyError

    with pytest.raises(ReadOnlyError):
        descriptor_model.set({"descriptor_mean_uz": 1.0})


def test_final_particles_can_be_read_before_any_result_exists(descriptor_model):
    """`lume-pva` reads every variable each cycle; a missing result must not make `get()` fail."""
    from lume_fbpic.actions import FinalParticlesAction

    model = LUMEFBPICModel(
        descriptor_model.simulator,
        [FinalParticlesAction(name="final_particles", read_only=True)],
        dummy_run=True,
    )

    assert model.get(["final_particles"])["final_particles"] is None
