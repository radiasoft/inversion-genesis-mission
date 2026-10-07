"""
Tests for species and ionization validation on DensityProfile.

Covers the outer product of:

* species: ``None``, ``"H"``, and ``"He"`` (too few levels for high ionization)
* ionization: ``None``, ``-1`` (fully ionized), ``2``, and a value too large for the species

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_density_species.py -v
"""

from __future__ import annotations

import pytest

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.density_profiles import ExampleDensityProfile

# Minimal constructor kwargs shared by all ExampleDensityProfile instances.
_MINIMAL_PROFILE_KWARGS = {
    "nominal_density": 1.0e24,
    "length": 1.0e-5,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
}

# Ionization level that exceeds helium's capacity (2 levels) but is valid for heavier ions.
_IONIZATION_TOO_LARGE = 99

_SPECIES_IONIZATION_CASES: list[pytest.param] = [
    pytest.param(
        None,
        None,
        None,
        1,
        False,
        False,
        id="species_none-ionization_none",
    ),
    pytest.param(
        None,
        0,
        None,
        1,
        False,
        False,
        id="species_none-ionization_zero_is_the_default",
    ),
    pytest.param(
        None,
        -1,
        None,
        1,
        False,
        False,
        id="species_none-ionization_fully_ionized",
    ),
    pytest.param(
        None,
        1,
        None,
        1,
        False,
        False,
        id="species_none-ionization_one_is_the_stored_form",
    ),
    pytest.param(
        None,
        2,
        None,
        None,
        True,
        False,
        id="species_none-ionization_2",
    ),
    pytest.param(
        None,
        _IONIZATION_TOO_LARGE,
        None,
        None,
        True,
        False,
        id="species_none-ionization_too_large",
    ),
    pytest.param(
        "H",
        None,
        "H",
        0,
        False,
        False,
        id="species_h-ionization_none",
    ),
    pytest.param(
        "H",
        -1,
        "H",
        1,
        False,
        False,
        id="species_h-ionization_fully_ionized",
    ),
    pytest.param(
        "H",
        2,
        "H",
        None,
        True,
        False,
        id="species_h-ionization_2",
    ),
    pytest.param(
        "H",
        _IONIZATION_TOO_LARGE,
        "H",
        None,
        True,
        False,
        id="species_h-ionization_too_large",
    ),
    pytest.param(
        "He",
        None,
        "He",
        0,
        False,
        False,
        id="species_too_small-ionization_none",
    ),
    pytest.param(
        "He",
        -1,
        "He",
        2,
        False,
        False,
        id="species_too_small-ionization_fully_ionized",
    ),
    pytest.param(
        "He",
        2,
        "He",
        2,
        False,
        False,
        id="species_too_small-ionization_2",
    ),
    pytest.param(
        "He",
        _IONIZATION_TOO_LARGE,
        "He",
        None,
        True,
        False,
        id="species_too_small-ionization_too_large",
    ),
]


def _build_profile(
    species: str | None, ionization: int | None
) -> ExampleDensityProfile:
    return ExampleDensityProfile(
        species=species,
        ionization=ionization,
        **_MINIMAL_PROFILE_KWARGS,
    )


@pytest.mark.parametrize(
    (
        "species",
        "ionization",
        "expected_species",
        "expected_ionization",
        "expect_error",
        "expect_hydrogen_assumption_warning",
    ),
    _SPECIES_IONIZATION_CASES,
)
def test_species_ionization_outer_product(
    species: str | None,
    ionization: int | None,
    expected_species: str | None,
    expected_ionization: int | None,
    expect_error: bool,
    expect_hydrogen_assumption_warning: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate species/ionization normalization and error handling."""
    if expect_error:
        with pytest.raises(ValueError, match="ionization"):
            _build_profile(species, ionization)
        return

    profile = _build_profile(species, ionization)
    assert profile.species == expected_species
    assert profile.ionization == expected_ionization

    captured = capsys.readouterr()
    if expect_hydrogen_assumption_warning:
        assert "assuming Hydrogen" in captured.out
    else:
        assert "assuming Hydrogen" not in captured.out


def test_species_none_without_explicit_ionization_is_bare_electrons() -> None:
    """The default `ionization` (0) must not make `species=None` raise."""
    profile = ExampleDensityProfile(species=None, **_MINIMAL_PROFILE_KWARGS)
    assert profile.species is None
    assert profile.ionization == 1


@pytest.mark.parametrize(
    "species,expected_levels",
    [
        ("H", 1),
        ("He", 2),
    ],
)
def test_get_num_ionization_levels(species: str, expected_levels: int) -> None:
    assert _DensityProfile._get_num_ionization_levels(species) == expected_levels


def test_get_num_ionization_levels_unrecognized_species() -> None:
    with pytest.raises(ValueError, match="Error getting ionization levels for species"):
        _DensityProfile._get_num_ionization_levels("NotAnElement")


def test_hydrogen_too_small_for_ionization_two() -> None:
    """Hydrogen has one ionization level; ionization=2 must be rejected."""
    with pytest.raises(ValueError, match="Species H has only 1 ionization levels"):
        _build_profile("H", 2)


def test_helium_too_small_for_excessive_ionization() -> None:
    """Helium has two ionization levels; ionization=99 must be rejected."""
    with pytest.raises(ValueError, match="Species He has only 2 ionization levels"):
        _build_profile("He", _IONIZATION_TOO_LARGE)
