"""Tests for the moment-descriptor actions (`MomentDescriptorAction`, `make_descriptor_actions`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from lume_fbpic.actions import make_actions, make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel

from inversion_fbpic.utils import distributions

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

_MC2_EV = m_e * c**2 / e
_ELECTRONS_PER_MACROPARTICLE = 1.0e3


def _particle_group(n=3000, seed=0, uz_mean=150.0, uz_std=30.0) -> ParticleGroup:
    rng = np.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(1.0e-6, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.5, 2.0, n) * _MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * _MC2_EV,
            "pz": rng.normal(uz_mean, uz_std, n) * _MC2_EV,
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, _ELECTRONS_PER_MACROPARTICLE * e),
            "species": "electron",
        }
    )


@pytest.fixture()
def descriptor_model(simulator) -> LUMEFBPICModel:
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    return LUMEFBPICModel(simulator, actions, dummy_run=True)


def _reference(pg, uz_min=30.0, central_fraction=0.95) -> dict[str, float]:
    """The descriptor computed directly with the library, as `build_dataset.py` would."""
    phase_space = np.stack(
        [pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1
    )
    weights = np.asarray(pg.weight) / e
    phase_space, weights = distributions.select_by_uz(phase_space, weights, uz_min=uz_min)
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
    assert actions[-1].unit == "C"


def test_cache_is_recomputed_when_final_particles_change(descriptor_model, simulator, mocker):
    spy = mocker.spy(distributions, "compute_moment_descriptor")
    mocker.patch("lume_fbpic.actions.compute_moment_descriptor", spy)
    names = ["descriptor_mean_uz", "descriptor_cov_uz_uz"]

    simulator.final_particles = _particle_group(seed=1)
    descriptor_model.get(names)
    descriptor_model.get(["descriptor_total_beam_charge_c"])
    assert spy.call_count == 1  # one computation shared by every feature

    simulator.final_particles = _particle_group(seed=2)
    descriptor_model.get(names)
    assert spy.call_count == 2


def test_charge_is_the_selected_weights_in_coulombs(descriptor_model, simulator):
    pg = _particle_group()
    simulator.final_particles = pg

    charge = descriptor_model.get(["descriptor_total_beam_charge_c"])[
        "descriptor_total_beam_charge_c"
    ]

    selected = np.count_nonzero(np.asarray(pg.pz) / _MC2_EV >= 30.0)
    assert charge == pytest.approx(_reference(pg)["total_beam_charge_c"], rel=1e-12)
    assert 0 < charge <= selected * _ELECTRONS_PER_MACROPARTICLE * e  # the crop only removes


def test_selection_parameters_change_the_result(simulator):
    pg = _particle_group()
    simulator.final_particles = pg
    name = "descriptor_total_beam_charge_c"

    def charge(**selection):
        model = LUMEFBPICModel(simulator, make_descriptor_actions(**selection), dummy_run=True)
        return model.get([name])[name]

    everything = charge(uz_min=None, central_fraction=None)
    assert everything == pytest.approx(len(np.asarray(pg.x)) * _ELECTRONS_PER_MACROPARTICLE * e)
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


def test_values_are_nan_when_the_selection_leaves_too_few_particles(descriptor_model, simulator):
    simulator.final_particles = _particle_group(uz_mean=2.0, uz_std=0.5)  # none reach uz >= 30

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
