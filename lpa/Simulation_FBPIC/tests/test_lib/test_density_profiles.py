"""
Tests for density profile physics: build_density_function, extents,
get_plasma_wavelength, ModifiedDensityProfile, and MatchedRadialModifier.

Species/ionization validation lives in the sibling ``test_density.py``.

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_density_profiles.py -v
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import shutil


# ---------------------------------------------------------------------------
# Constructor kwargs (mirrored from conftest for importability)
# ---------------------------------------------------------------------------

_COMMON = {"nominal_density": 1.0e24, "p_nz": 1, "p_nr": 1, "p_nt": 1}

EXAMPLE_DENSITY_KWARGS = {**_COMMON, "length": 5.0e-6, "start_position": 0.0}

ASYMMETRIC_SINE_KWARGS = {
    **_COMMON,
    "peak_z0": 2.5e-3,
    "upramp_length": 2.5e-3,
    "downramp_length": 1.5e-3,
}

SMOOTH_SINE_FLATTOP_KWARGS = {
    **_COMMON,
    "flattop_width": 1.0e-3,
    "upramp_length": 0.5e-3,
    "downramp_length": 0.5e-3,
    "offset_length": 0.0,
}

GAUSSIAN_PLUS_TRIANGLE_KWARGS = {
    **_COMMON,
    "gauss_sigma": 3.0e-6,
    "gauss_z0": 10.0e-6,
    "tri_z0": 15.0e-6,
    "tri_left_width": 5.0e-6,
    "tri_right_width": 5.0e-6,
    "tri_height": 1.0,
}

GENERALIZED_GAUSSIAN_PLUS_TRIANGLE_KWARGS = {
    **_COMMON,
    "gauss_peak": 1.0,
    "gauss_alpha": 3.0e-6,
    "gauss_beta": 2.0,
    "gauss_z0": 10.0e-6,
    "tri_z0": 15.0e-6,
    "tri_left_width": 5.0e-6,
    "tri_right_width": 5.0e-6,
    "tri_height": 1.0,
}

H5_DENSITY_KWARGS = {
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    "density_name": "density",
    "lineout_axis": "z_m",
    "interpolation_points": {"x_mm": 1.0, "pressure_bar": 0.0},
}

GENERIC_CONICAL_TARGET_KWARGS = {
    **_COMMON,
    "start_position": 1.0e-3,
    "main_profile_type": "cosine_squared_flattop",
    "main_profile_parameters": {"fwhm": 2.0e-3, "ramp_length": 1.0e-3},
    "fringe_profile_type": "cosine_squared",
    "fringe_profile_parameters": {"fwhm": 0.25e-3},
    "fringe_center_offset": 1.0e-3,
    "fringe_relative_height": 0.4,
    "skew_rate": 0.0,
    "skew_mode": "exponential",
}

POWER_LAW_FLATTOP_KWARGS = {
    **_COMMON,
    "start_position": 1.0e-3,
    "flattop_width": 2.0e-3,
    "transition_length": 0.5e-3,
    "transition_exponent": 3.0,
    "skew_rate": 0.0,
    "skew_mode": "exponential",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_example():
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile

    return ExampleDensityProfile(**EXAMPLE_DENSITY_KWARGS)


def _build_asymmetric_sine():
    from inversion_fbpic.lib.density_profiles import AsymmetricSine

    return AsymmetricSine(**ASYMMETRIC_SINE_KWARGS)


def _build_smooth_sine_flattop():
    from inversion_fbpic.lib.density_profiles import SmoothSineFlattop

    return SmoothSineFlattop(**SMOOTH_SINE_FLATTOP_KWARGS)


def _build_gaussian_plus_triangle():
    from inversion_fbpic.lib.density_profiles import GaussianPlusTriangle

    return GaussianPlusTriangle(**GAUSSIAN_PLUS_TRIANGLE_KWARGS)


def _build_generalized_gaussian_plus_triangle():
    from inversion_fbpic.lib.density_profiles import GeneralizedGaussianPlusTriangle

    return GeneralizedGaussianPlusTriangle(**GENERALIZED_GAUSSIAN_PLUS_TRIANGLE_KWARGS)


def _build_generic_conical_target():
    from inversion_fbpic.lib.density_profiles import GenericConicalTarget

    return GenericConicalTarget(**GENERIC_CONICAL_TARGET_KWARGS)


def _build_power_law_flattop():
    from inversion_fbpic.lib.density_profiles import PowerLawFlattop

    return PowerLawFlattop(**POWER_LAW_FLATTOP_KWARGS)


def _build_generalized_lorentzian_sum():
    from inversion_fbpic.lib.density_profiles import (
        GeneralizedLorentzianParameters,
        GeneralizedLorentzianSum,
    )

    return GeneralizedLorentzianSum(
        parameters=[GeneralizedLorentzianParameters(2, 0, 1, 2, 1)],
        p_nz=1,
        p_nr=1,
        p_nt=1,
        density_cutoff_ratio=1e-4,
    )


def _build_modified():
    from inversion_fbpic.lib.density_profiles import SmoothSineFlattop
    from inversion_fbpic.lib.density_modifiers import (
        MatchedRadialModifier,
        ModifiedDensityProfile,
    )

    base = SmoothSineFlattop(**SMOOTH_SINE_FLATTOP_KWARGS)
    modifier = MatchedRadialModifier(matched_density=1.0e24, radial_extent=50.0e-6)
    return ModifiedDensityProfile(base_density_profile=base, modifiers=[modifier])


def _write_h5_density(tmp_path, dimension_labels: list[bytes] | None = None):
    filename = tmp_path / "density.h5"
    density = np.zeros((4, 2, 2))
    density[:, 1, 0] = [0.0, 1.0, 4.0, 0.0]

    with h5py.File(filename, "w") as h5_file:
        dataset = h5_file.create_dataset("density", data=density)
        if dimension_labels is not None:
            dataset.attrs["DIMENSION_LABELS"] = dimension_labels
        h5_file.create_dataset("z_m", data=[0.0, 1.0, 2.0, 3.0])
        h5_file.create_dataset("x_mm", data=[0.0, 1.0])
        h5_file.create_dataset("pressure_bar", data=[0.0, 1.0])

    return filename


def _write_oblique_h5_density(tmp_path):
    filename = tmp_path / "oblique_density.h5"
    z_m = np.array([0.0, 1.0, 2.0, 3.0])
    x_mm = np.array([0.0, 1.0, 2.0])
    pressure_bar = np.array([0.0, 1.0])
    density = np.empty((z_m.size, x_mm.size, pressure_bar.size))
    for z_index, z_value in enumerate(z_m):
        for x_index, x_value in enumerate(x_mm):
            density[z_index, x_index, 0] = z_value + x_value
            density[z_index, x_index, 1] = 2.0 * (z_value + x_value)

    with h5py.File(filename, "w") as h5_file:
        dataset = h5_file.create_dataset("density", data=density)
        dataset.attrs["DIMENSION_LABELS"] = [b"z_m", b"x_mm", b"pressure_bar"]
        h5_file.create_dataset("z_m", data=z_m)
        h5_file.create_dataset("x_mm", data=x_mm)
        h5_file.create_dataset("pressure_bar", data=pressure_bar)

    return filename


_PROFILE_BUILDERS = {
    "sine_squared_bump": _build_example,
    "asymmetric_sine": _build_asymmetric_sine,
    "smooth_sine_flattop": _build_smooth_sine_flattop,
    "gaussian_plus_triangle": _build_gaussian_plus_triangle,
    "generalized_gaussian_plus_triangle": _build_generalized_gaussian_plus_triangle,
    "generic_conical_target": _build_generic_conical_target,
    "power_law_flattop": _build_power_law_flattop,
    "generalized_lorentzian_sum": _build_generalized_lorentzian_sum,
    "modified_density_profile": _build_modified,
}


@pytest.fixture(params=list(_PROFILE_BUILDERS.keys()))
def profile_instance(request):
    """Parametrized fixture: one instance of each concrete profile subclass."""
    return _PROFILE_BUILDERS[request.param]()


# ===================================================================
# Unit tests — ExampleDensityProfile
# ===================================================================


class TestExampleDensityProfile:
    def test_peak_near_center(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z_center = profile.start_position + profile.length / 2.0
        assert dens(z_center, 0.0) == pytest.approx(1.0, abs=1e-10)

    def test_zero_outside_support(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z_before = profile.start_position - 1e-6
        z_after = profile.start_position + profile.length + 1e-6
        assert dens(z_before, 0.0) == pytest.approx(0.0, abs=1e-15)
        assert dens(z_after, 0.0) == pytest.approx(0.0, abs=1e-15)

    def test_radially_uniform(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z_center = profile.start_position + profile.length / 2.0
        val_r0 = dens(z_center, 0.0)
        val_r1 = dens(z_center, 1e-3)
        assert val_r0 == pytest.approx(val_r1)

    def test_array_input(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z = np.linspace(
            profile.start_position - 1e-6,
            profile.start_position + profile.length + 1e-6,
            50,
        )
        r = np.zeros_like(z)
        result = dens(z, r)
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape
        assert np.all(result >= 0.0)


# ===================================================================
# Unit tests — Generalized Lorentzian profiles
# ===================================================================


class TestGeneralizedLorentzianParameters:
    def test_exposes_mapping_parameters(self) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        parameters = GeneralizedLorentzianParameters(2, 3, 4, 5, 6)

        assert (
            parameters.A,
            parameters.c,
            parameters.w,
            parameters.b,
            parameters.m,
        ) == (
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        )
        assert list(parameters) == ["A", "c", "w", "b", "m"]
        assert list(parameters.values()) == [2.0, 3.0, 4.0, 5.0, 6.0]
        assert parameters["A"] == 2.0
        assert parameters["m"] == 6.0
        assert dict(parameters) == {
            "A": 2.0,
            "c": 3.0,
            "w": 4.0,
            "b": 5.0,
            "m": 6.0,
        }

    @pytest.mark.parametrize(("parameter", "value"), [("b", 0.0), ("m", -1.0)])
    def test_rejects_non_positive_exponents(self, parameter, value) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        values = {"A": 2.0, "c": 3.0, "w": 4.0, "b": 2.0, "m": 1.0}
        values[parameter] = value

        with pytest.raises(ValueError):
            GeneralizedLorentzianParameters(**values)

    def test_z_extent_at_cutoff_ratio(self) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        parameters = GeneralizedLorentzianParameters(A=5, c=10, w=2, b=2, m=1)

        assert parameters.z_extent(0.2) == pytest.approx((6.0, 14.0))

    def test_density_function_evaluates_scalar_and_array_inputs(self) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        parameters = GeneralizedLorentzianParameters(A=5, c=10, w=2, b=2, m=1)

        assert parameters.dens_func(10.0) == pytest.approx(5.0)
        np.testing.assert_allclose(
            parameters.dens_func(np.array([10.0, 14.0])), [5.0, 1.0]
        )


class TestGeneralizedLorentzianSum:
    def _build(self, parameters, **overrides):
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianSum

        return GeneralizedLorentzianSum(
            parameters=parameters,
            p_nz=1,
            p_nr=1,
            p_nt=1,
            **overrides,
        )

    def test_sets_nominal_density_to_largest_term_amplitude(self) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        profile = self._build(
            [
                GeneralizedLorentzianParameters(2, 0, 1, 2, 1),
                GeneralizedLorentzianParameters(5, 3, 1, 2, 1),
            ]
        )

        assert profile.nominal_density == pytest.approx(5.0)
        assert profile.get_r_extent() is None

    def test_serializes_parameters_as_keyed_mappings(self) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        profile = self._build([GeneralizedLorentzianParameters(2, 3, 4, 5, 6)])

        assert profile.to_dict()["parameters"]["parameters"] == [
            {"A": 2.0, "c": 3.0, "w": 4.0, "b": 5.0, "m": 6.0}
        ]

    def test_json_round_trip(self) -> None:
        from inversion_fbpic.lib.density_core import _DensityProfile
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        profile = self._build([GeneralizedLorentzianParameters(2, 3, 4, 5, 6)])

        reloaded = _DensityProfile.from_json(profile.to_json())

        assert type(reloaded) is type(profile)
        assert reloaded.parameters == profile.parameters

    def test_z_extent_combines_term_extents(self) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        profile = self._build(
            [
                GeneralizedLorentzianParameters(2, 0, 1, 2, 1),
                GeneralizedLorentzianParameters(2, 6, 1, 2, 1),
            ],
            density_cutoff_ratio=0.2,
        )

        assert profile.get_z_extent() == pytest.approx((-2.0, 8.0))

    def test_density_function_sums_terms_and_clips_negative_values(self) -> None:
        from inversion_fbpic.lib.density_profiles import GeneralizedLorentzianParameters

        profile = self._build(
            [
                GeneralizedLorentzianParameters(2, 0, 1, 2, 1),
                GeneralizedLorentzianParameters(-3, 3, 1, 2, 1),
            ]
        )
        density = profile.build_density_function()
        z = np.array([0.0, 3.0])

        np.testing.assert_allclose(density(z, np.zeros_like(z)), [0.85, 0.0])
        assert density(0.0, 0.0) == pytest.approx(density(0.0, 1.0))


# ===================================================================
# Unit tests — GenericConicalTarget
# ===================================================================


class TestGenericConicalTarget:
    def _build(self, **overrides):
        from inversion_fbpic.lib.density_profiles import GenericConicalTarget

        return GenericConicalTarget(**(GENERIC_CONICAL_TARGET_KWARGS | overrides))

    @pytest.mark.parametrize(
        ("main_type", "main_parameters"),
        [
            ("supergaussian", {"fwhm": 2.0e-3, "beta": 4.0}),
            ("cosine_squared_flattop", {"fwhm": 2.0e-3, "ramp_length": 1.0e-3}),
            (
                "power_law_flattop",
                {
                    "flattop_width": 2.0e-3,
                    "transition_length": 0.5e-3,
                    "transition_exponent": 3.0,
                },
            ),
            (
                "lorentzian_flattop",
                {
                    "flattop_width": 2.0e-3,
                    "transition_length": 0.5e-3,
                    "coordinate_exponent": 4.0,
                    "profile_exponent": 2.0,
                },
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("fringe_type", "fringe_parameters"),
        [
            ("supergaussian", {"fwhm": 0.25e-3, "beta": 4.0}),
            ("cosine_squared", {"fwhm": 0.25e-3}),
            (
                "power_law",
                {
                    "transition_length": 0.1e-3,
                    "transition_exponent": 3.0,
                },
            ),
            (
                "lorentzian",
                {
                    "transition_length": 0.1e-3,
                    "coordinate_exponent": 4.0,
                    "profile_exponent": 2.0,
                },
            ),
            (None, {}),
        ],
    )
    def test_all_main_and_fringe_pairs_build(
        self, main_type, main_parameters, fringe_type, fringe_parameters
    ) -> None:
        overrides = {
            "main_profile_type": main_type,
            "main_profile_parameters": main_parameters,
            "fringe_profile_type": fringe_type,
            "fringe_profile_parameters": fringe_parameters,
            "fringe_center_offset": 1.0e-3 if fringe_type is not None else None,
            "fringe_relative_height": 0.4 if fringe_type is not None else 1.0,
        }
        profile = self._build(**overrides)
        density = profile.build_density_function()
        z_min, z_max = profile.get_z_extent()
        values = density(np.linspace(z_min, z_max, 100), np.zeros(100))

        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    def test_power_law_fringe_has_no_flattop(self) -> None:
        profile = self._build(
            fringe_profile_type="power_law",
            fringe_profile_parameters={
                "transition_length": 1.0,
                "transition_exponent": 3.0,
            },
            fringe_center_offset=3.0,
            fringe_relative_height=1.0,
        )
        density = profile.build_density_function()
        fringe_center = profile.centroid + profile.fringe_center_offset

        assert density(fringe_center, 0.0) == pytest.approx(1.0)
        assert density(fringe_center + 0.5, 0.0) == pytest.approx((1.0 - 0.5**3) ** 3)

    def test_lorentzian_flattop_keeps_fringe_tails_inside_composite_support(
        self,
    ) -> None:
        cutoff = 1.0e-3
        coordinate_exponent = 4.0
        profile_exponent = 2.0
        cutoff_coordinate = (cutoff ** (-1.0 / profile_exponent) - 1.0) ** (
            1.0 / coordinate_exponent
        )
        fringe_relative_height = 0.4
        fringe_cutoff_coordinate = (
            (cutoff / fringe_relative_height) ** (-1.0 / profile_exponent) - 1.0
        ) ** (1.0 / coordinate_exponent)
        profile = self._build(
            main_profile_type="lorentzian_flattop",
            main_profile_parameters={
                "flattop_width": 4.0,
                "transition_length": 2.0,
                "coordinate_exponent": coordinate_exponent,
                "profile_exponent": profile_exponent,
                "density_cutoff_ratio": cutoff,
            },
            fringe_profile_type="lorentzian",
            fringe_profile_parameters={
                "transition_length": 1.0,
                "coordinate_exponent": coordinate_exponent,
                "profile_exponent": profile_exponent,
                "density_cutoff_ratio": cutoff,
            },
            fringe_center_offset=10.0,
            fringe_relative_height=fringe_relative_height,
        )
        density = profile.build_density_function()
        main_transition = profile.centroid + 3.0
        fringe_center = profile.centroid + profile.fringe_center_offset
        main_cutoff = profile.centroid + 2.0 + 2.0 * cutoff_coordinate
        fringe_cutoff = fringe_center + fringe_cutoff_coordinate
        fringe_tail_at_main_cutoff = fringe_relative_height * (
            1.0 + abs(main_cutoff - fringe_center) ** coordinate_exponent
        ) ** (-profile_exponent)

        assert density(profile.centroid, 0.0) == pytest.approx(1.0)
        assert density(main_transition, 0.0) == pytest.approx((1.0 + 0.5**4) ** -2)
        assert density(main_cutoff, 0.0) == pytest.approx(
            cutoff + fringe_tail_at_main_cutoff
        )
        main_tail_at_fringe_center = (
            1.0
            + ((fringe_center - profile.centroid - 2.0) / 2.0) ** coordinate_exponent
        ) ** (-profile_exponent)
        assert density(fringe_center, 0.0) == pytest.approx(
            fringe_relative_height + main_tail_at_fringe_center
        )
        main_tail_at_fringe_offset = (
            1.0
            + ((fringe_center + 0.5 - profile.centroid - 2.0) / 2.0)
            ** coordinate_exponent
        ) ** (-profile_exponent)
        assert density(fringe_center + 0.5, 0.0) == pytest.approx(
            fringe_relative_height * (1.0 + 0.5**4) ** -2 + main_tail_at_fringe_offset
        )
        main_tail_at_fringe_cutoff = (
            1.0
            + ((fringe_cutoff - profile.centroid - 2.0) / 2.0) ** coordinate_exponent
        ) ** (-profile_exponent)
        assert density(fringe_cutoff, 0.0) == pytest.approx(
            cutoff + main_tail_at_fringe_cutoff
        )
        assert density(fringe_cutoff + 1.0e-9, 0.0) == pytest.approx(
            density(fringe_cutoff, 0.0), rel=1.0e-6
        )

    def test_low_amplitude_lorentzian_fringe_does_not_expand_support(self) -> None:
        main_parameters = {"fwhm": 4.0, "ramp_length": 2.0}
        profile = self._build(
            main_profile_parameters=main_parameters,
            fringe_profile_type="lorentzian",
            fringe_profile_parameters={
                "transition_length": 1.0,
                "coordinate_exponent": 4.0,
                "profile_exponent": 2.0,
                "density_cutoff_ratio": 1.0e-3,
            },
            fringe_center_offset=10.0,
            fringe_relative_height=1.0e-4,
        )
        main_only = self._build(
            main_profile_parameters=main_parameters,
            fringe_profile_type=None,
            fringe_profile_parameters={},
            fringe_center_offset=None,
            fringe_relative_height=1.0,
        )

        assert profile.get_z_extent() == pytest.approx(main_only.get_z_extent())
        assert profile.build_density_function()(
            profile.centroid + 10.0, 0.0
        ) == pytest.approx(1.0e-4)

    def test_supergaussian_fringe_tail_remains_inside_main_support(self) -> None:
        profile = self._build(
            main_profile_parameters={"fwhm": 10.0, "ramp_length": 2.0},
            fringe_profile_type="supergaussian",
            fringe_profile_parameters={"fwhm": 1.0, "beta": 2.0},
            fringe_center_offset=2.0,
            fringe_relative_height=0.1,
        )
        density = profile.build_density_function()
        fringe_coordinate = profile.centroid + 4.0

        assert fringe_coordinate < profile.get_z_extent()[1]
        assert density(fringe_coordinate, 0.0) > 0.0

    def test_lorentzian_density_remains_continuous_outside_metadata_extent(
        self,
    ) -> None:
        profile = self._build(
            main_profile_type="lorentzian_flattop",
            main_profile_parameters={
                "flattop_width": 2.0,
                "transition_length": 1.0,
                "coordinate_exponent": 4.0,
                "profile_exponent": 2.0,
                "density_cutoff_ratio": 1.0e-3,
            },
            fringe_profile_type=None,
            fringe_profile_parameters={},
            fringe_center_offset=None,
            fringe_relative_height=1.0,
        )
        density = profile.build_density_function()
        _, z_max = profile.get_z_extent()

        assert density(z_max + 1.0e-6, 0.0) > 0.0

    def test_high_amplitude_lorentzian_fringe_expands_to_relative_cutoff(self) -> None:
        cutoff = 1.0e-3
        relative_height = 2.0
        coordinate_exponent = 4.0
        profile_exponent = 2.0
        old_half_extent = (cutoff ** (-1.0 / profile_exponent) - 1.0) ** (
            1.0 / coordinate_exponent
        )
        effective_half_extent = (
            (cutoff / relative_height) ** (-1.0 / profile_exponent) - 1.0
        ) ** (1.0 / coordinate_exponent)
        profile = self._build(
            fringe_profile_type="lorentzian",
            fringe_profile_parameters={
                "transition_length": 1.0,
                "coordinate_exponent": coordinate_exponent,
                "profile_exponent": profile_exponent,
                "density_cutoff_ratio": cutoff,
            },
            fringe_center_offset=10.0,
            fringe_relative_height=relative_height,
        )
        fringe_edge = profile.centroid + 10.0 + effective_half_extent

        assert effective_half_extent > old_half_extent
        assert profile.get_z_extent()[1] == pytest.approx(fringe_edge)
        assert profile.build_density_function()(fringe_edge, 0.0) == pytest.approx(
            cutoff
        )

    @pytest.mark.parametrize("fringe_side", ["left", "right", "both"])
    def test_fringe_side_controls_components_and_support(self, fringe_side) -> None:
        profile = self._build(
            fringe_side=fringe_side,
            fringe_center_offset=2.0e-3,
            fringe_profile_parameters={"fwhm": 0.1e-3},
        )
        density = profile.build_density_function()
        z_min, z_max = profile.get_z_extent()
        right_center = profile.centroid + profile.fringe_center_offset
        left_center = profile.centroid - profile.fringe_center_offset

        assert z_min == profile.start_position
        if fringe_side in ("left", "both"):
            assert density(left_center, 0.0) > 0.0
        if fringe_side == "right":
            assert density(left_center, 0.0) == pytest.approx(0.0)
        if fringe_side in ("right", "both"):
            assert density(right_center, 0.0) > 0.0
        if fringe_side == "left":
            assert density(right_center, 0.0) == pytest.approx(0.0)
        assert density(z_min - 1.0e-9, 0.0) == pytest.approx(0.0)
        assert density(z_max + 1.0e-9, 0.0) == pytest.approx(0.0)

    def test_skew_is_applied_after_component_sum(self) -> None:
        profile = self._build(
            main_profile_parameters={"fwhm": 4.0, "ramp_length": 2.0},
            fringe_profile_parameters={"fwhm": 1.0},
            fringe_center_offset=1.5,
            fringe_relative_height=0.4,
            skew_rate=0.2,
            skew_mode="finite_supergaussian",
        )
        density = profile.build_density_function()
        shifted_z = 1.0
        expected_unskewed = 1.0 + 0.4 * np.cos(np.pi / 4.0) ** 2
        z_min, z_max = profile.get_z_extent()
        support_width = z_max - z_min

        assert density(profile.centroid + shifted_z, 0.0) == pytest.approx(
            expected_unskewed
            * np.exp(
                profile.skew_rate * shifted_z
                - 4.0 * np.log(2.0) * shifted_z**2 / support_width**2
            )
        )

    def test_exponential_skew_remains_available(self) -> None:
        profile = self._build(
            main_profile_parameters={"fwhm": 4.0, "ramp_length": 2.0},
            fringe_profile_parameters={"fwhm": 1.0},
            fringe_center_offset=1.5,
            fringe_relative_height=0.4,
            skew_rate=0.2,
            skew_mode="exponential",
        )
        density = profile.build_density_function()
        shifted_z = 1.0
        expected_unskewed = 1.0 + 0.4 * np.cos(np.pi / 4.0) ** 2

        assert density(profile.centroid + shifted_z, 0.0) == pytest.approx(
            expected_unskewed * np.exp(profile.skew_rate * shifted_z)
        )

    def test_saturated_skew_preserves_local_rate_and_reciprocal_limits(self) -> None:
        from inversion_fbpic.lib._density_implementations.generic_conical_target import (
            _skew_multiplier,
        )

        profile = self._build(
            main_profile_parameters={"fwhm": 4.0, "ramp_length": 2.0},
            fringe_profile_type=None,
            fringe_profile_parameters={},
            fringe_center_offset=None,
            fringe_relative_height=1.0,
            skew_rate=0.2,
            skew_mode="saturated",
        )
        left_span, right_span = profile._profile_support_spans()
        saturation_length = 3.0 * max(left_span, right_span)
        coordinate = 20.0 * saturation_length
        multiplier, reverse_multiplier = _skew_multiplier(
            np.asarray([coordinate, -coordinate]),
            left_span,
            right_span,
            profile.skew_rate,
            "saturated",
        )
        step = 1.0e-6
        density = profile.build_density_function()
        center_density = density(profile.centroid, 0.0)
        local_log_slope = (
            np.log(density(profile.centroid + step, 0.0))
            - np.log(density(profile.centroid - step, 0.0))
        ) / (2.0 * step)

        assert center_density == pytest.approx(1.0)
        assert local_log_slope == pytest.approx(profile.skew_rate, rel=1.0e-5)
        assert multiplier == pytest.approx(
            np.exp(profile.skew_rate * saturation_length)
        )
        assert multiplier * reverse_multiplier == pytest.approx(1.0)

    def test_auto_skew_selects_saturated_for_slow_tail_components(self) -> None:
        lorentzian = self._build(
            main_profile_type="lorentzian_flattop",
            main_profile_parameters={
                "flattop_width": 2.0,
                "transition_length": 1.0,
                "coordinate_exponent": 4.0,
                "profile_exponent": 2.0,
            },
            skew_mode="auto",
        )
        slow_supergaussian = self._build(
            main_profile_type="supergaussian",
            main_profile_parameters={"fwhm": 1.0, "beta": 1.0},
            skew_mode="auto",
        )
        fast_supergaussian = self._build(
            main_profile_type="supergaussian",
            main_profile_parameters={"fwhm": 1.0, "beta": 2.0},
            skew_mode="auto",
        )

        assert lorentzian.skew_mode == "saturated"
        assert slow_supergaussian.skew_mode == "saturated"
        assert fast_supergaussian.skew_mode == "finite_supergaussian"
        assert lorentzian.to_dict()["parameters"]["skew_mode"] == "saturated"

    def test_finite_supergaussian_skew_permits_beta_at_most_one(self) -> None:
        profile = self._build(
            main_profile_type="supergaussian",
            main_profile_parameters={"fwhm": 1.0, "beta": 1.0},
            skew_rate=1.0,
            skew_mode="finite_supergaussian",
        )

        assert np.isfinite(profile.build_density_function()(profile.centroid, 0.0))

    def test_finite_supergaussian_peak_may_lie_outside_support(self) -> None:
        profile = self._build(
            main_profile_parameters={"fwhm": 4.0, "ramp_length": 2.0},
            fringe_profile_type=None,
            fringe_profile_parameters={},
            fringe_center_offset=None,
            fringe_relative_height=1.0,
            skew_rate=2.0,
            skew_mode="finite_supergaussian",
        )
        z_min, z_max = profile.get_z_extent()
        support_width = z_max - z_min
        peak = profile.centroid + profile.skew_rate * support_width**2 / (
            8.0 * np.log(2.0)
        )
        density = profile.build_density_function()

        assert peak > z_max
        assert np.all(np.isfinite(density(np.linspace(z_min, z_max, 101), 0.0)))
        assert density(z_max + 1.0e-9, 0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {
                "main_profile_type": "lorentzian_flattop",
                "main_profile_parameters": {
                    "flattop_width": 2.0,
                    "transition_length": 1.0,
                    "coordinate_exponent": 4.0,
                    "profile_exponent": 2.0,
                },
                "skew_rate": 0.1,
                "skew_mode": "exponential",
            },
            {
                "fringe_profile_type": "lorentzian",
                "fringe_profile_parameters": {
                    "transition_length": 1.0,
                    "coordinate_exponent": 4.0,
                    "profile_exponent": 2.0,
                },
                "fringe_center_offset": 2.0,
                "skew_rate": 0.1,
                "skew_mode": "exponential",
            },
        ],
    )
    def test_rejects_unguarded_exponential_skew_for_lorentzians(
        self, overrides
    ) -> None:
        with pytest.raises(ValueError, match="not supported"):
            self._build(**overrides)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"main_profile_parameters": {"fwhm": 1.0}}, "missing"),
            (
                {"main_profile_parameters": {"fwhm": 1.0, "ramp_length": 2.0}},
                "less than or equal",
            ),
            (
                {"fringe_profile_type": None, "fringe_center_offset": 1.0},
                "require fringe_profile_type",
            ),
            ({"fringe_center_offset": None}, "fringe_center_offset is required"),
            (
                {
                    "main_profile_type": "supergaussian",
                    "main_profile_parameters": {"fwhm": 1.0, "beta": 1.0},
                    "skew_rate": 1.0,
                    "skew_mode": "exponential",
                },
                "beta <= 1.0",
            ),
            ({"skew_mode": "unsupported"}, "skew_mode"),
        ],
    )
    def test_rejects_invalid_component_configuration(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            self._build(**overrides)


# ===================================================================
# Unit tests — PowerLawFlattop
# ===================================================================


class TestPowerLawFlattop:
    def _build(self, **overrides):
        from inversion_fbpic.lib.density_profiles import PowerLawFlattop

        return PowerLawFlattop(**(POWER_LAW_FLATTOP_KWARGS | overrides))

    def test_constant_core_and_power_law_transition(self) -> None:
        profile = self._build()
        density = profile.build_density_function()
        flat_edge = profile.centroid + profile.flattop_width / 2.0
        transition_midpoint = flat_edge + profile.transition_length / 2.0

        assert density(profile.centroid, 0.0) == pytest.approx(1.0)
        assert density(flat_edge, 0.0) == pytest.approx(1.0)
        assert density(transition_midpoint, 0.0) == pytest.approx(
            (1.0 - 0.5**profile.transition_exponent) ** 3
        )

    def test_support_and_centroid_are_derived_from_start_position(self) -> None:
        profile = self._build()
        density = profile.build_density_function()
        z_min, z_max = profile.get_z_extent()

        assert z_min == profile.start_position
        assert profile.centroid == pytest.approx((z_min + z_max) / 2.0)
        assert density(z_min, 0.0) == pytest.approx(0.0)
        assert density(z_max, 0.0) == pytest.approx(0.0)
        assert density(z_min - 1.0e-9, 0.0) == pytest.approx(0.0)
        assert density(z_max + 1.0e-9, 0.0) == pytest.approx(0.0)

    def test_requires_exponent_greater_than_two(self) -> None:
        with pytest.raises(ValueError, match="transition_exponent"):
            self._build(transition_exponent=2.0)


# ===================================================================
# Unit tests — InterpolateFromH5Profile
# ===================================================================


class TestInterpolateFromH5Profile:
    def _build(self, tmp_path, **overrides):
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        return InterpolateFromH5Profile(
            filename=filename, **(H5_DENSITY_KWARGS | overrides)
        )

    @pytest.mark.parametrize("extension", ["yaml", "json"])
    def test_file_round_trip_resolves_h5_relative_to_config_source(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        extension: str,
    ) -> None:
        """A config and its sibling data tree remain loadable after relocation."""
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile
        from inversion_fbpic.lib.serializable_config import SerializableConfig

        work_dir = tmp_path / "work"
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True)
        filename = _write_h5_density(
            data_dir, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        run_dir = work_dir / "out"
        run_dir.mkdir()
        monkeypatch.chdir(work_dir)
        profile = InterpolateFromH5Profile(
            filename=filename.relative_to(work_dir), **H5_DENSITY_KWARGS
        )
        config_path = run_dir / f"profile.{extension}"
        if extension == "yaml":
            profile.to_yaml_file(config_path, comments=False)
        else:
            profile.to_json_file(config_path)

        serialized = config_path.read_text(encoding="utf-8")
        if extension == "yaml":
            import yaml

            payload = yaml.safe_load(serialized)
        else:
            import json

            payload = json.loads(serialized)
        assert payload["parameters"]["filename"] == "../data/density.h5"

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)

        relocated_work_dir = tmp_path / "relocated"
        shutil.move(str(work_dir), relocated_work_dir)
        relocated_config_path = relocated_work_dir / "out" / config_path.name
        relocated_filename = relocated_work_dir / "data" / filename.name

        loaded = SerializableConfig.from_file(relocated_config_path)
        assert isinstance(loaded, InterpolateFromH5Profile)
        assert loaded.filename == relocated_filename.resolve()
        assert loaded.nominal_density == pytest.approx(profile.nominal_density)

    def test_interpolates_normalizes_and_derives_metadata(self, tmp_path) -> None:
        profile = self._build(tmp_path)
        density = profile.build_density_function()

        assert density(1.0, 0.0) == pytest.approx(1.0)
        assert density(np.array([-1.0, 0.0, 1.0, 2.0]), 0.0) == pytest.approx(
            [0.0, 0.25, 1.0, 0.0]
        )
        assert profile.nominal_density == pytest.approx(4.0)
        assert profile.centroid == pytest.approx(0.8)
        assert profile.get_z_extent() == pytest.approx((0.0, 1.0))

    def test_offset_and_broadcasting(self, tmp_path) -> None:
        profile = self._build(tmp_path, longitudinal_offset=10.0)
        density = profile.build_density_function()

        assert density(11.0, np.zeros(3)) == pytest.approx(np.ones(3))
        assert density(np.array([10.0, 11.0]), np.array([0.0, 1.0])) == pytest.approx(
            [0.25, 1.0]
        )
        assert density(9.0, 0.0) == pytest.approx(0.0)
        assert profile.centroid == pytest.approx(10.8)
        assert profile.get_z_extent() == pytest.approx((10.0, 11.0))

    def test_warns_when_data_bounds_truncate_cutoff_support(self, tmp_path) -> None:
        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["density"][:, 1, 0] = [2.0, 4.0, 2.0, 1.0]

        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.warns(
            UserWarning, match="extent is truncated by the available data"
        ):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_preserves_nonuniform_scalar_axis_samples(self, tmp_path) -> None:
        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["z_m"][:] = [0.0, 0.5, 2.0, 3.0]

        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)
        assert profile.lineout.grid[0] == pytest.approx([-0.5, 0.0, 1.5, 2.5])

    @pytest.mark.parametrize(
        ("centering_mode", "expected_centroid", "expected_extent"),
        [
            ("centroid", 10.0, (9.2, 10.2)),
            ("left", 10.8, (10.0, 11.0)),
            ("right", 9.8, (9.0, 10.0)),
        ],
    )
    def test_centers_profile_at_selected_anchor(
        self, tmp_path, centering_mode, expected_centroid, expected_extent
    ) -> None:
        profile = self._build(
            tmp_path,
            centering_mode=centering_mode,
            longitudinal_offset=10.0,
        )

        assert profile.centroid == pytest.approx(expected_centroid)
        assert profile.get_z_extent() == pytest.approx(expected_extent)

    def test_samples_oblique_affine_path_and_preserves_centering(
        self, tmp_path
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(
            filename=_write_oblique_h5_density(tmp_path),
            p_nz=1,
            p_nr=1,
            p_nt=1,
            density_name="density",
            neutral=False,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": 1.0},
            },
            interpolation_points={"pressure_bar": 0.0},
            centering_mode="centroid",
            longitudinal_offset=10.0,
        )
        density = profile.build_density_function()

        assert profile.lineout.grid[0].size == 7
        assert profile.centroid == pytest.approx(10.0)
        assert density(profile.lineout.grid[0], 0.0) == pytest.approx(
            [
                1.0 / 3.0,
                4.0 / 9.0,
                5.0 / 9.0,
                2.0 / 3.0,
                7.0 / 9.0,
                8.0 / 9.0,
                1.0,
            ]
        )
        assert density(profile.lineout.grid[0][0] - 1.0e-6, 0.0) == pytest.approx(0.0)
        assert density(profile.lineout.grid[0][-1] + 1.0e-6, 0.0) == pytest.approx(0.0)

    def test_samples_affine_path_with_fixed_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(
            filename=_write_h5_density(
                tmp_path,
                dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"],
            ),
            p_nz=1,
            p_nr=1,
            p_nt=1,
            density_name="density",
            neutral=False,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": 0.0},
            },
            interpolation_points={"pressure_bar": 0.0},
        )
        density = profile.build_density_function()

        assert profile.lineout.grid[0] == pytest.approx([-1.0, 0.0, 1.0, 2.0])
        assert density(profile.lineout.grid[0], 0.0) == pytest.approx(
            [0.0, 0.25, 1.0, 0.0]
        )

    def test_rejects_out_of_bounds_origin_for_fixed_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.raises(ValueError, match="coordinate dataset bounds"):
            InterpolateFromH5Profile(
                filename=_write_h5_density(
                    tmp_path,
                    dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"],
                ),
                p_nz=1,
                p_nr=1,
                p_nt=1,
                density_name="density",
                neutral=False,
                lineout_axis={
                    "z_m": {"origin": 0.0, "coefficient": 1.0},
                    "x_mm": {"origin": 99.0, "coefficient": 0.0},
                },
                interpolation_points={"pressure_bar": 0.0},
            )

    def test_centroid_axis_uses_a_fixed_slice(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(
            filename=_write_oblique_h5_density(tmp_path),
            p_nz=1,
            p_nr=1,
            p_nt=1,
            density_name="density",
            neutral=False,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": -1.0},
            },
            centroid_axis="z_m",
            interpolation_points={"pressure_bar": 0.0},
            centering_mode="centroid",
            longitudinal_offset=10.0,
        )

        # At x_mm=1, the z_m density weights are [1, 2, 3, 4], centering at t=2.
        assert profile.centroid == pytest.approx(10.0)
        assert profile.lineout.grid[0] == pytest.approx(
            [8.0 + sample_index / 6.0 for sample_index in range(7)]
        )

    def test_rejects_out_of_bounds_origin_for_centroid_slice(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.raises(ValueError, match="lineout_axis origins"):
            InterpolateFromH5Profile(
                filename=_write_oblique_h5_density(tmp_path),
                p_nz=1,
                p_nr=1,
                p_nt=1,
                density_name="density",
                neutral=False,
                lineout_axis={
                    "z_m": {"origin": 0.0, "coefficient": 1.0},
                    "x_mm": {"origin": 3.0, "coefficient": -1.0},
                },
                centroid_axis="z_m",
                interpolation_points={"pressure_bar": 0.0},
                centering_mode="centroid",
            )

    def test_rejects_fixed_centroid_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.raises(ValueError, match="centroid_axis must have a nonzero"):
            InterpolateFromH5Profile(
                filename=_write_h5_density(
                    tmp_path,
                    dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"],
                ),
                p_nz=1,
                p_nr=1,
                p_nt=1,
                density_name="density",
                neutral=False,
                lineout_axis={
                    "z_m": {"origin": 0.0, "coefficient": 0.0},
                    "x_mm": {"origin": 1.0, "coefficient": 1.0},
                },
                centroid_axis="z_m",
                interpolation_points={"pressure_bar": 0.0},
            )

    def test_serializes_affine_path(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_core import _DensityProfile

        profile = self._build(
            tmp_path,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": -1.0},
            },
            centroid_axis="z_m",
            interpolation_points={"pressure_bar": 0.0},
        )

        restored = _DensityProfile.from_yaml(profile.to_yaml(comments=False))
        assert restored.lineout_axis == profile.lineout_axis
        assert restored.centroid_axis == profile.centroid_axis

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"interpolation_points": {"x_mm": 1.0}}, "missing interpolation"),
            (
                {
                    "interpolation_points": {
                        "x_mm": 1.0,
                        "pressure_bar": 0.0,
                        "y_mm": 0.0,
                    }
                },
                "unexpected interpolation",
            ),
            ({"lineout_axis": "y_mm"}, "not in DIMENSION_LABELS"),
            ({"centroid_axis": "y_mm"}, "centroid_axis.*not in DIMENSION_LABELS"),
            (
                {"centroid_axis": "x_mm"},
                "centroid_axis must be included in lineout_axis",
            ),
            ({"lineout_axis": {}}, "non-empty dictionary"),
            (
                {
                    "lineout_axis": {
                        "z_m": {"origin": 0.0, "coefficient": 0.0},
                    },
                    "interpolation_points": {"x_mm": 1.0, "pressure_bar": 0.0},
                },
                "must contain at least one nonzero coefficient",
            ),
            (
                {
                    "lineout_axis": {
                        "z_m": {"origin": 0.0, "coefficient": 1.0},
                        "x_mm": {"origin": float("nan"), "coefficient": 1.0},
                    },
                    "interpolation_points": {"pressure_bar": 0.0},
                },
                "finite values",
            ),
            ({"density_cutoff_ratio": -0.1}, "must be >= 0.0"),
            ({"density_cutoff_ratio": 1.0}, "must be < 1.0"),
            (
                {
                    "interpolation_points": {
                        "x_mm": 2.0,
                        "pressure_bar": 0.0,
                    }
                },
                "coordinate dataset bounds",
            ),
        ],
    )
    def test_rejects_invalid_configuration(self, tmp_path, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            self._build(tmp_path, **kwargs)

    def test_rejects_missing_dimension_labels(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(tmp_path)
        with pytest.raises(ValueError, match="must define DIMENSION_LABELS"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    @pytest.mark.parametrize(
        ("dimension_labels", "match"),
        [
            ([b"z_m", b"x_mm"], "one label for each density dimension"),
            (
                [b"z_m", b"x_mm", b"z_m"],
                "must not contain duplicate labels",
            ),
        ],
    )
    def test_rejects_invalid_dimension_labels(
        self, tmp_path, dimension_labels, match
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(tmp_path, dimension_labels=dimension_labels)
        with pytest.raises(ValueError, match=match):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_non_finite_density_samples(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["density"][0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="must contain only finite values"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_negative_density_samples(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["density"][0, 0, 0] = -1.0

        with pytest.raises(ValueError, match="must not contain negative values"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_single_point_coordinate_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = tmp_path / "single_point_axis.h5"
        with h5py.File(filename, "w") as h5_file:
            density = h5_file.create_dataset("density", data=np.ones((1, 2, 2)))
            density.attrs["DIMENSION_LABELS"] = [b"z_m", b"x_mm", b"pressure_bar"]
            h5_file.create_dataset("z_m", data=[0.0])
            h5_file.create_dataset("x_mm", data=[0.0, 1.0])
            h5_file.create_dataset("pressure_bar", data=[0.0, 1.0])

        with pytest.raises(
            ValueError,
            match="Coordinate dataset 'z_m' must contain at least two values",
        ):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    @pytest.mark.parametrize(
        ("coordinate", "replacement", "match"),
        [
            ("z_m", [0.0, 1.0, 1.0, 3.0], "strictly increasing"),
            ("x_mm", [0.0, np.inf], "must be finite"),
        ],
    )
    def test_rejects_invalid_coordinate_values(
        self, tmp_path, coordinate, replacement, match
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file[coordinate][:] = replacement

        with pytest.raises(ValueError, match=match):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    @pytest.mark.parametrize(
        ("coordinate", "replacement", "match"),
        [
            ("z_m", [0.0, 1.0], "length 4"),
            ("x_mm", [[0.0, 1.0]], "one-dimensional"),
        ],
    )
    def test_rejects_malformed_coordinate_datasets(
        self, tmp_path, coordinate, replacement, match
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            del h5_file[coordinate]
            h5_file.create_dataset(coordinate, data=replacement)

        with pytest.raises(ValueError, match=match):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_missing_coordinate_dataset(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            del h5_file["pressure_bar"]

        with pytest.raises(ValueError, match="Coordinate dataset 'pressure_bar'"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)


# ===================================================================
# Unit tests — get_plasma_wavelength
# ===================================================================


class TestGetPlasmaWavelength:
    def test_nominal(self) -> None:
        profile = _build_example()
        expected = 3.3e7 / np.sqrt(profile.nominal_density)
        assert profile.get_plasma_wavelength() == pytest.approx(expected, rel=1e-6)

    def test_position_dependent(self) -> None:
        profile = _build_example()
        z_center = profile.start_position + profile.length / 2.0
        result = profile.get_plasma_wavelength(z=z_center, r=0.0)
        assert result > 0.0
        assert np.isfinite(result)

    def test_zero_density_returns_inf(self) -> None:
        profile = _build_example()
        z_outside = profile.start_position - 1e-3
        result = profile.get_plasma_wavelength(z=z_outside, r=0.0)
        assert result == float("inf")


# ===================================================================
# Unit tests — MatchedRadialModifier
# ===================================================================


class TestMatchedRadialModifier:
    def test_modify_increases_density_with_r(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        def flat_z_profile(z, r):
            return np.ones_like(np.asarray(z), dtype=float)

        modifier = MatchedRadialModifier(matched_density=1.0e24, radial_extent=50e-6)
        modified = modifier.modify_density_function(flat_z_profile)

        val_r0 = modified(0.0, 0.0)
        val_r_nonzero = modified(0.0, 20e-6)
        assert val_r_nonzero > val_r0

    def test_get_r_extent_returns_value(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        modifier = MatchedRadialModifier(matched_density=1.0e24, radial_extent=50e-6)
        assert modifier.get_r_extent() == pytest.approx(50e-6)

    def test_get_r_extent_none(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        modifier = MatchedRadialModifier(matched_density=1.0e24)
        assert modifier.get_r_extent() is None

    def test_get_z_extent_is_none(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        modifier = MatchedRadialModifier(matched_density=1.0e24)
        assert modifier.get_z_extent() is None


# ===================================================================
# Unit tests — ModifiedDensityProfile
# ===================================================================


class TestModifiedDensityProfile:
    def test_inherits_base_attributes(self) -> None:
        profile = _build_modified()
        base = profile.resolved_base_density_profile
        assert profile.nominal_density == base.nominal_density
        assert profile.p_nz == base.p_nz
        assert profile.species == base.species
        assert profile.ionization == base.ionization

    def test_z_extent_at_least_as_wide_as_base(self) -> None:
        profile = _build_modified()
        base_extent = profile.resolved_base_density_profile.get_z_extent()
        mod_extent = profile.get_z_extent()
        assert mod_extent[0] <= base_extent[0]
        assert mod_extent[1] >= base_extent[1]

    def test_r_extent_from_modifier(self) -> None:
        profile = _build_modified()
        assert profile.get_r_extent() == pytest.approx(50e-6)

    def test_build_density_function_callable(self) -> None:
        profile = _build_modified()
        dens = profile.build_density_function()
        z = np.linspace(0.0, 1e-3, 20)
        r = np.zeros_like(z)
        result = dens(z, r)
        assert result.shape == z.shape
        assert np.all(np.isfinite(result))

    def test_modified_density_has_radial_dependence(self) -> None:
        profile = _build_modified()
        dens = profile.build_density_function()
        z_mid = (
            SMOOTH_SINE_FLATTOP_KWARGS["offset_length"]
            + SMOOTH_SINE_FLATTOP_KWARGS["upramp_length"]
            + SMOOTH_SINE_FLATTOP_KWARGS["flattop_width"] / 2.0
        )
        val_r0 = dens(z_mid, 0.0)
        val_r_nonzero = dens(z_mid, 20e-6)
        if val_r0 > 0:
            assert val_r_nonzero > val_r0

    def test_from_dict_round_trip(self) -> None:
        profile = _build_modified()
        d = profile.to_dict()

        from inversion_fbpic.lib.density_modifiers import ModifiedDensityProfile

        reloaded = ModifiedDensityProfile.from_dict(d)
        assert reloaded.nominal_density == profile.nominal_density
        assert reloaded.get_z_extent() == profile.get_z_extent()


# ===================================================================
# Parametrized whole-class instance tests
# ===================================================================


class TestAllDensityProfiles:
    def test_build_density_function_returns_callable(self, profile_instance) -> None:
        dens = profile_instance.build_density_function()
        z = np.linspace(-1e-3, 5e-3, 100)
        r = np.zeros_like(z)
        result = dens(z, r)
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape

    def test_density_non_negative(self, profile_instance) -> None:
        dens = profile_instance.build_density_function()
        z_min, z_max = profile_instance.get_z_extent()
        z = np.linspace(z_min, z_max, 200)
        r = np.zeros_like(z)
        result = dens(z, r)
        assert np.all(result >= -1e-15), f"Negative density found: {result.min()}"

    def test_density_zero_far_outside_support(self, profile_instance) -> None:
        z_min, z_max = profile_instance.get_z_extent()
        span = z_max - z_min
        dens = profile_instance.build_density_function()
        z_far_left = z_min - 10 * span
        z_far_right = z_max + 10 * span
        assert dens(z_far_left, 0.0) == pytest.approx(0.0, abs=1e-6)
        assert dens(z_far_right, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_get_z_extent_returns_ordered_pair(self, profile_instance) -> None:
        z_min, z_max = profile_instance.get_z_extent()
        assert z_min < z_max

    def test_get_r_extent_none_or_positive(self, profile_instance) -> None:
        r_ext = profile_instance.get_r_extent()
        assert r_ext is None or r_ext > 0.0

    def test_yaml_round_trip(self, profile_instance) -> None:
        yaml_str = profile_instance.to_yaml(comments=False)
        from inversion_fbpic.lib.density_core import _DensityProfile

        reloaded = _DensityProfile.from_yaml(yaml_str)
        assert type(reloaded) is type(profile_instance)
        orig_params = profile_instance.to_dict()["parameters"]
        reload_params = reloaded.to_dict()["parameters"]
        for key in orig_params:
            val = orig_params[key]
            if val is None or isinstance(val, (dict, list)):
                continue
            assert reload_params[key] == pytest.approx(
                val, rel=1e-6
            ), f"Mismatch on {key}: {val} vs {reload_params[key]}"

    def test_get_plasma_wavelength_nominal(self, profile_instance) -> None:
        wl = profile_instance.get_plasma_wavelength()
        expected = 3.3e7 / np.sqrt(profile_instance.nominal_density)
        assert wl == pytest.approx(expected, rel=1e-6)
