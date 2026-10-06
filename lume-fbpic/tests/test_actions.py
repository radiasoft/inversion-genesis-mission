"""Tests for lume_fbpic action/variable classes."""

from __future__ import annotations

from tests.downramp_actions import make_actions
from lume_fbpic.actions import HyperparameterFieldAction, LaserFieldAction


def _action(simulator, name):
    return next(a for a in make_actions(simulator) if a.name == name)


def test_laser_field_action_get_matches_current_value(simulator):
    action = _action(simulator, "laser_energy")
    assert action._get(simulator) == simulator.laser.energy


def test_laser_field_action_set_replaces_object_via_attrs_evolve(simulator):
    action = _action(simulator, "laser_energy")
    original_laser = simulator.laser

    action._set(simulator, 7.5)

    assert simulator.laser is not original_laser
    assert simulator.laser.energy == 7.5
    assert action._get(simulator) == 7.5


def test_laser_field_action_set_on_non_amplitude_field_does_not_trip_energy_a0_check(
    simulator,
):
    # Regression test: every constructed _LaserPulse has *both* energy and a0 populated
    # (one given, one derived) from the moment it's built -- not just after an amplitude
    # field is set. Evolving an unrelated field (focal_position here) must still null out
    # whichever of the pair is currently derived, or attrs.evolve() carries over both and
    # trips "provide exactly one of energy or a0".
    action = _action(simulator, "laser_focal_position")

    action._set(simulator, 20.0e-6)

    assert simulator.laser.focal_position == 20.0e-6
    # energy is unchanged (it's still the amplitude source); a0 is freshly re-derived from
    # it by __attrs_post_init__, not left null -- nulling it going into evolve() only
    # avoids the "both provided" conflict, it doesn't mean it stays null afterward.
    assert simulator.laser.energy == 5.0
    assert simulator.laser.a0 is not None


def test_laser_field_action_set_a0_after_energy_still_allows_further_field_changes(
    simulator,
):
    # Regression test: switching the amplitude source (energy -> a0) must update
    # `_amplitude_source` so a *subsequent* unrelated field change still nulls the right
    # (now-derived) field.
    a0_action = LaserFieldAction(name="laser_a0", field_name="a0", unit=None)
    focal_action = _action(simulator, "laser_focal_position")

    a0_action._set(simulator, 3.0)
    focal_action._set(simulator, 20.0e-6)

    assert simulator.laser.a0 == 3.0
    assert simulator.laser.focal_position == 20.0e-6
    assert simulator.laser.energy is not None


def test_hyperparameter_field_action_get_set(simulator):
    # Not part of the HTU downramp action set (tests/downramp_actions.py deliberately
    # excludes grid/numerical hyperparameters as control variables), but exercised directly here
    # since HyperparameterFieldAction is available as a general-purpose class.
    action = HyperparameterFieldAction(
        name="gamma_boost", field_name="gamma_boost", unit=None
    )
    assert action._get(simulator) == simulator.hyparams.gamma_boost

    original_hyparams = simulator.hyparams
    action._set(simulator, 2.0)

    assert simulator.hyparams is not original_hyparams
    assert simulator.hyparams.gamma_boost == 2.0


def test_density_field_action_targets_correct_profile_by_index(simulator):
    action = _action(simulator, "downramp_length")

    action._set(simulator, 0.25e-3)

    assert simulator.densities[1].downramp_length == 0.25e-3
    assert simulator.densities[0].downramp_length != 0.25e-3


def test_stat_action_reads_from_cached_stats(simulator):
    action = _action(simulator, "charge_pc")
    simulator.stats = {"charge_pc": 12.3}
    assert action._get(simulator) == 12.3


def test_stat_action_defaults_to_nan_before_any_run(simulator):
    import math

    action = _action(simulator, "energy_mean_mev")
    assert math.isnan(action._get(simulator))


def test_final_particles_action_reads_from_cache(simulator):
    action = _action(simulator, "final_particles")
    simulator.final_particles = "placeholder"
    assert action._get(simulator) == "placeholder"
