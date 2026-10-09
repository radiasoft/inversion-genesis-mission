"""Tests for the units an action gets from the documentation of the config field it acts on."""

from __future__ import annotations

import pytest

from lume_fbpic.actions import (
    DensityFieldAction,
    DopantFractionAction,
    HyperparameterFieldAction,
    LaserFieldAction,
    StatAction,
)
from lume_fbpic.model import LUMEFBPICModel


def test_a_density_action_takes_the_unit_of_its_field(model):
    assert _unit(model, "flattop_density") == "m^-3"
    assert _unit(model, "downramp_density") == "m^-3"
    assert _unit(model, "downramp_length") == "m"


def test_a_dopant_fraction_is_dimensionless(simulator):
    action = DopantFractionAction(
        name="fraction", host_index=0, dopant_index=1, read_only=False
    )

    model = LUMEFBPICModel(simulator, [action], dummy_run=True)

    assert _unit(model, "fraction") == "1"


@pytest.mark.parametrize(
    "action_class, kwargs",
    [(DensityFieldAction, {"density_index": 0, "field_name": "no_such_field"})],
)
def test_a_field_that_does_not_exist_has_no_unit(simulator, action_class, kwargs):
    model = LUMEFBPICModel(
        simulator, [action_class(name="x", **kwargs)], dummy_run=True
    )

    assert _unit(model, "x") is None


def test_a_field_without_a_unit_tag_has_none(simulator):
    model = LUMEFBPICModel(
        simulator, [LaserFieldAction(name="a0", field_name="a0")], dummy_run=True
    )

    assert _unit(model, "a0") is None


def test_a_hyperparameter_action_takes_the_unit_of_its_field(simulator):
    model = LUMEFBPICModel(
        simulator,
        [
            HyperparameterFieldAction(name="start", field_name="zmin"),
            HyperparameterFieldAction(name="cells", field_name="nz"),
        ],
        dummy_run=True,
    )

    assert _unit(model, "start") == "m"
    assert _unit(model, "cells") is None  # a count: its tag is a type, `[int]`


def test_a_laser_action_takes_the_unit_of_its_field(model):
    assert _unit(model, "laser_energy") == "J"
    assert _unit(model, "laser_focal_position") == "m"
    assert _unit(model, "laser_temporal_width") == "s"


def test_a_pwfa_action_takes_the_unit_of_its_field():
    import pwfa_gaussian

    model = pwfa_gaussian.build_model(dummy_run=True)

    assert _unit(model, "plasma_density") == "m^-3"
    assert _unit(model, "driver_charge") == "C"
    assert _unit(model, "driver_sigma_r") == "m"
    assert _unit(model, "witness_position") == "m"
    assert _unit(model, "driver_gamma") is None


def test_a_stat_action_takes_the_unit_of_its_statistic(model):
    assert _unit(model, "charge_pc") == "pC"
    assert _unit(model, "energy_mean_mev") == "MeV"
    assert _unit(model, "energy_std_mev") == "MeV"


def test_a_unit_given_to_an_action_is_kept(simulator):
    model = LUMEFBPICModel(
        simulator,
        [
            LaserFieldAction(name="energy", field_name="energy", unit="mJ"),
            LaserFieldAction(name="focus", field_name="focal_position", unit=None),
            StatAction(name="charge", stat_name="charge_pc", unit="C", read_only=True),
        ],
        dummy_run=True,
    )

    assert _unit(model, "energy") == "mJ"
    assert _unit(model, "focus") is None
    assert _unit(model, "charge") == "C"


def test_a_zernike_coefficient_is_in_wavelengths():
    import ionization_injection

    model = ionization_injection.build_model()

    assert _unit(model, "zernike_coma_x") == "wavelengths"
    assert _unit(model, "nitrogen_dopant_fraction") == "1"


def test_the_derived_units_are_valid_and_survive_an_archive(model, tmp_path):
    from beamphysics.units import pmd_unit

    model.archive(tmp_path / "a.h5")
    loaded = LUMEFBPICModel.from_archive(tmp_path / "a.h5")

    for name, variable in model.supported_variables.items():
        unit = getattr(variable, "unit", None)
        assert _unit(loaded, name) == unit
        if unit is not None and unit != "wavelengths":
            pmd_unit(unit)  # raises for an unknown unit


def _unit(model: LUMEFBPICModel, name: str) -> str | None:
    return getattr(model.supported_variables[name], "unit", None)
