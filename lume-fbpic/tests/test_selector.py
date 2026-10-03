"""Tests for `lume_fbpic.selector.ArchiveSelector`."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from lume_fbpic.actions import make_actions, make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.selector import ArchiveSelector

from inversion_fbpic.utils import distributions

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

_MC2_EV = m_e * c**2 / e


def _bunch(uz_mean: float, charge_pc: float, seed: int = 0, n: int = 3000) -> ParticleGroup:
    rng = np.random.default_rng(seed)
    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 3.0e-6, n),
            "y": rng.normal(0.0, 1.5e-6, n),
            "z": rng.normal(0.003, 4.0e-6, n),
            "px": rng.normal(0.0, 2.0, n) * _MC2_EV,
            "py": rng.normal(0.0, 1.0, n) * _MC2_EV,
            "pz": rng.normal(uz_mean, 30.0, n) * _MC2_EV,
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, charge_pc * 1.0e-12 / n),
            "species": "electron",
        }
    )


def _descriptor_outputs(pg) -> dict[str, float]:
    phase_space = np.stack(
        [pg.x, pg.px / _MC2_EV, pg.y, pg.py / _MC2_EV, pg.z, pg.pz / _MC2_EV], axis=-1
    )
    descriptor = distributions.compute_moment_descriptor(phase_space, np.asarray(pg.weight) / e)
    return {f"descriptor_{k}": v for k, v in descriptor.items()}


def _reconstructed_model(simulator, tmp_path, name, uz_mean, charge_pc) -> LUMEFBPICModel:
    """A model loaded from an archive that records a descriptor and holds no particles."""
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    source = LUMEFBPICModel(simulator, actions, dummy_run=True)
    source.archive(
        tmp_path / f"{name}.h5", outputs=_descriptor_outputs(_bunch(uz_mean, charge_pc))
    )
    return LUMEFBPICModel.from_archive(tmp_path / f"{name}.h5", dummy_run=True)


@pytest.fixture()
def selector(simulator, tmp_path) -> ArchiveSelector:
    models = {
        "run_a": _reconstructed_model(simulator, tmp_path, "run_a", 100.0, 200.0),
        "run_b": _reconstructed_model(simulator, tmp_path, "run_b", 200.0, 400.0),
    }
    for model in models.values():
        model.synthesize_bunch(n_particles=2000)
    return ArchiveSelector(models)


def _charge(selector) -> float:
    return selector.get(["descriptor_total_beam_charge_c"])["descriptor_total_beam_charge_c"]


def test_the_first_run_is_active_at_the_start(selector):
    assert selector.active == "run_a"
    assert selector.get(["LPA_Archive"])["LPA_Archive"] == "run_a"


def test_the_enum_lists_the_runs_in_order(selector):
    variable = selector.supported_variables["LPA_Archive"]

    assert list(variable.options) == ["run_a", "run_b"]
    assert variable.read_only is False


def test_selecting_a_run_switches_the_outputs(selector):
    assert _charge(selector) == pytest.approx(200.0e-12)

    selector.set({"LPA_Archive": "run_b"})

    assert selector.active == "run_b"
    assert _charge(selector) == pytest.approx(400.0e-12)
    assert selector.get(["descriptor_mean_uz"])["descriptor_mean_uz"] == pytest.approx(200.0, rel=0.02)


def test_the_final_particles_follow_the_selection(selector):
    a = selector.final_particles
    selector.set({"LPA_Archive": "run_b"})
    b = selector.final_particles

    assert a is selector.models["run_a"].final_particles
    assert b is selector.models["run_b"].final_particles
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
    assert _charge(selector) == pytest.approx(200.0e-12)


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
