from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Iterator

import attrs
import numpy as np
import numpy.typing as npt

from inversion_fbpic.lib.density_core import _DensityProfile, DensityCallable

from ._density_implementations.interpolate_from_h5_profile import (
    InterpolateFromH5Profile as InterpolateFromH5Profile,
)
from ._density_implementations.generic_conical_target import (
    GenericConicalTarget as GenericConicalTarget,
    PowerLawFlattop as PowerLawFlattop,
)


# NOTE: THIS IS AN EXMAMPLE IMPLEMENTATION OF A DENSITY PROFILE. IT IS NOT INTENDED TO BE USED IN PRACTICE.
# NOTE: EACH NOTE MUST BE CONSIDERED WHEN IMPLEMENTING A NEW DENSITY PROFILE.
@attrs.define(kw_only=True, slots=False, frozen=True)
class ExampleDensityProfile(_DensityProfile):
    """Finite sine-squared density bump profile.

    Args:
        length: (float) [m] Total length of the finite sine-squared support in meters.
            The profile is non-zero only over this interval.
        start_position: (float) [m] |OPTIONAL| z position in meters where the profile support starts. Defaults to 0.0.
    """

    # NOTE: Docstring must follow this format in order to be parsed correctly for yaml comments (aside from the extra blank line in the Args block -- that is a stylistic choice)
    # NOTE: Parameters with a None option or a default value are optional in yaml.

    # Keep this string stable once used in stored JSON payloads.
    # NOTE: NEW IMPLEMENTATIONS MUST OVERRIDE SUBCLASS (NOT CONFIG_TYPE).
    SUBCLASS: ClassVar[str] = "sine_squared_bump"

    # Attrs fields with `init=True` become serialized "parameters". `init=False` means the field is not serialized nor is it taken as input.
    # NOTE: THIS IS WHERE THE INPUTS FOR THE CONSTRUCTOR LIVE
    length: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    start_position: float = attrs.field(default=0.0, converter=float)

    # NOTE: ANY SUBCLASS THAT REQUIRES FURTHER INITIALIZATION CHECKS OR ADVANCED SETUP MUST OVERRIDE THIS METHOD AND CALL SUPER().__ATTRS_POST_INIT__
    # NOTE: IN THIS CASE, NO ADDITIONAL SETUP IS NEEDED; THIS METHOD DEFINITION COULD BE OMITTED.
    # NOTE: BECAUSE THIS CLASS IS FROZEN, CHANGES MUST BE MADE VIA `object.__setattr__`.
    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()  # NOTE: ALL SUBCLASSES MUST CALL THEIR SUPERCLASS'S __ATTRS_POST_INIT__ IF THEY OVERRIDE IT

    # NOTE: IMPLEMENTATIONS MUST OVERRIDE THESE METHODS, DETERMINE THE Z AND R EXTENTS OF THE PROFILE.
    def get_z_extent(self) -> tuple[float, float]:
        """
        Get the longitudinal extent of the density profile in meters.
        """
        return (self.start_position, self.start_position + self.length)

    def get_r_extent(self) -> float | None:
        """
        Get the radial extent of the density profile in meters.
        """
        return None

    # NOTE: IMPLEMENTATIONS MUST OVERRIDE THIS, RETURN A CALLABLE THAT RETURNS THE RELATIVE DENSITY PROFILE.
    def build_density_function(self) -> DensityCallable:
        """
        Return a finite sine-squared n(z, r) profile.

        Returns:
            A callable function that returns the relative density at position (z, r).
            The function takes two arguments:
                z: Array-like of z positions
                r: Array-like of r positions
            and returns an array-like of density values.
        """

        # Subclasses should return a callable with signature dens_func(z, r)
        # that produces a relative density profile compatible with FBPIC.
        def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
            z_arr = np.asarray(z)

            n = np.zeros_like(z_arr, dtype=float)
            in_support = (z_arr >= self.start_position) & (
                z_arr <= self.start_position + self.length
            )
            phase = np.pi * (z_arr[in_support] - self.start_position) / self.length
            n[in_support] = np.sin(phase) ** 2

            return n

        return dens_func


# ------------------------------------------------------------
# Implemented Density Profiles
# ------------------------------------------------------------


@attrs.define(kw_only=True, slots=False, frozen=True)
class AsymmetricSine(_DensityProfile):
    """
    Asymmetric sine-squared density profile.
    The profile is a combination of a sine-squared upramp of a specified length and a corresponding downramp of another specified length.
    Args:
        peak_z0: (float) [m] Position of the maximum of the profile.
        upramp_length: (float) [m] Length of the upramp.
        downramp_length: (float) [m] Length of the downramp.
    """

    SUBCLASS: ClassVar[str] = "asymmetric_sine"

    peak_z0: float = attrs.field(converter=float)
    upramp_length: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    downramp_length: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )

    def get_z_extent(self) -> tuple[float, float]:
        return (
            self.peak_z0 - self.upramp_length,
            self.peak_z0 + self.downramp_length,
        )

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        try:
            from inversion_fbpic.density_profiles.downramp_injection import (
                build_asymmetric_cosine_blade_profile,
            )
        except ImportError:
            from downramp_injection import build_asymmetric_cosine_blade_profile  # type: ignore

        return build_asymmetric_cosine_blade_profile(
            upramp_length=self.upramp_length,
            downramp_length=self.downramp_length,
            offset_length=self.peak_z0 - self.upramp_length,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class SmoothSineFlattop(_DensityProfile):
    """
    Smooth flattop density profile. The upramp and downramp on either side are cosine-squared functions.
    The profile is nonzero, starting at offset_length; the flattop starts at offset_length + upramp_length.
    Args:
        flattop_width: (float) [m] Width of the flattop.
        upramp_length: (float) [m] Length of the upramp.
        downramp_length: (float) [m] Length of the downramp.
        offset_length: (float) [m] |OPTIONAL| Offset of the profile. Defaults to 0.0.
    """

    SUBCLASS: ClassVar[str] = "smooth_sine_flattop"

    flattop_width: float = attrs.field(
        converter=float, validator=attrs.validators.ge(0.0)
    )
    upramp_length: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    downramp_length: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    offset_length: float = attrs.field(default=0.0, converter=float)

    def get_z_extent(self) -> tuple[float, float]:
        return (
            self.offset_length,
            self.offset_length
            + self.upramp_length
            + self.flattop_width
            + self.downramp_length,
        )

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        try:
            from inversion_fbpic.density_profiles.downramp_injection import (
                build_smooth_flattop_profile,
            )
        except ImportError:
            from downramp_injection import build_smooth_flattop_profile  # type: ignore

        return build_smooth_flattop_profile(
            flattop_width=self.flattop_width,
            upramp_length=self.upramp_length,
            downramp_length=self.downramp_length,
            offset_length=self.offset_length,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class GaussianPlusTriangle(_DensityProfile):
    """
    Gaussian plus triangle density profile in z.

    Args:
        gauss_sigma: (float) [m] Standard deviation of the Gaussian in z.
        gauss_z0: (float) [m] Center position of the Gaussian in z.
        tri_z0: (float) [m] Center (tip) position of the triangle in z.
        tri_left_width: (float) [m] Length scale of the triangle to the left of the tip.
        tri_right_width: (float) [m] Length scale of the triangle to the right of the tip.
        tri_height: (float) [m] |OPTIONAL| Height of the triangle at the tip with respect to the Gaussian amplitude. Defaults to 1.0.
        num_sigma_extent: (float) |OPTIONAL| Number of sigma's to account for in the longitudinal extent of the Gaussian component. Defaults to 3.0.
    """

    # Keep this string stable once used in stored JSON payloads.
    SUBCLASS: ClassVar[str] = "gaussian_plus_triangle"

    gauss_sigma: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    gauss_z0: float = attrs.field(converter=float)
    tri_z0: float = attrs.field(converter=float)
    tri_left_width: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    tri_right_width: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    tri_height: float = attrs.field(
        default=1.0, converter=float, validator=attrs.validators.gt(0.0)
    )
    num_sigma_extent: float = attrs.field(
        default=3.0, converter=float, validator=attrs.validators.gt(0.0)
    )

    def get_z_extent(self) -> tuple[float, float]:
        triangle_extent = (
            self.tri_z0 - self.tri_left_width,
            self.tri_z0 + self.tri_right_width,
        )
        gaussian_extent = (
            self.gauss_z0 - self.num_sigma_extent * self.gauss_sigma,
            self.gauss_z0 + self.num_sigma_extent * self.gauss_sigma,
        )
        return (
            min(triangle_extent[0], gaussian_extent[0]),
            max(triangle_extent[1], gaussian_extent[1]),
        )

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        try:
            from inversion_fbpic.density_profiles.downramp_injection import (
                build_gaussian_plus_triangle_z_density_function,
            )
        except ImportError:
            from downramp_injection import build_gaussian_plus_triangle_z_density_function  # type: ignore

        return build_gaussian_plus_triangle_z_density_function(
            sigma=self.gauss_sigma,
            center_location=self.gauss_z0,
            z_tip=self.tri_z0,
            left_width=self.tri_left_width,
            right_width=self.tri_right_width,
            triangle_height=self.tri_height,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class GeneralizedGaussianPlusTriangle(_DensityProfile):
    """
    Generalized Gaussian plus triangle density profile in z.

    Args:
        gauss_peak: (float) Peak value of the generalized Gaussian.
        gauss_alpha: (float) [m] Scale parameter of the generalized Gaussian.
        gauss_beta: (float) Shape parameter of the generalized Gaussian.
        gauss_z0: (float) [m] Center position of the generalized Gaussian in z.
        tri_z0: (float) [m] Center (tip) position of the triangle in z.
        tri_left_width: (float) [m] Length scale of the triangle to the left of the tip.
        tri_right_width: (float) [m] Length scale of the triangle to the right of the tip.
        tri_height: (float) [m] Height of the triangle at the tip with respect to the generalized Gaussian amplitude.
        num_sigma_extent: (float) |OPTIONAL| Number of sigma's to account for in the longitudinal extent of the Gaussian component. Defaults to 3.0.
    """

    # Keep this string stable once used in stored JSON payloads.
    SUBCLASS: ClassVar[str] = "generalized_gaussian_plus_triangle"

    # Attrs fields with init=True become serialized "parameters".
    gauss_peak: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    gauss_alpha: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    gauss_beta: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    gauss_z0: float = attrs.field(converter=float)
    tri_z0: float = attrs.field(converter=float)
    tri_left_width: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    tri_right_width: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    tri_height: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    num_sigma_extent: float = attrs.field(
        default=3.0, converter=float, validator=attrs.validators.gt(0.0)
    )

    def get_z_extent(self) -> tuple[float, float]:
        # get equivalent width of Gaussian where value is e^(-num_sigma_extent**2/2)
        gen_gaus_nsig_width = (
            np.power(self.num_sigma_extent**2 / 2.0, 1.0 / self.gauss_beta)
            * self.gauss_alpha
        )

        triangle_extent = (
            self.tri_z0 - self.tri_left_width,
            self.tri_z0 + self.tri_right_width,
        )
        gaussian_extent = (
            self.gauss_z0 - gen_gaus_nsig_width,
            self.gauss_z0 + gen_gaus_nsig_width,
        )

        return (
            min(triangle_extent[0], gaussian_extent[0]),
            max(triangle_extent[1], gaussian_extent[1]),
        )

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        try:
            from inversion_fbpic.density_profiles.downramp_injection import (
                build_generalized_gaussian_with_downramp,
            )
        except ImportError:
            from downramp_injection import build_generalized_gaussian_with_downramp  # type: ignore

        return build_generalized_gaussian_with_downramp(
            gauss_peak=self.gauss_peak,
            gauss_alpha=self.gauss_alpha,
            gauss_beta=self.gauss_beta,
            gauss_z0=self.gauss_z0,
            ramp_z0=self.tri_z0,
            ramp_left_tau=self.tri_left_width,
            ramp_right_width=self.tri_right_width,
            ramp_height=self.tri_height,
        )


@attrs.define(slots=False, frozen=True)
class GeneralizedLorentzianParameters(Mapping[str, float]):
    """Parameters for a generalized Lorentzian term:
    A[1+|(s-c)/w|^b]^{-m}

    Args:
        A: (float) [m^-3] amplitude
        c: (float) [m] center
        w: (float) [m] width
        b: (float) inner exponent
        m: (float) outer exponent
    """

    A: float = attrs.field(converter=float)
    c: float = attrs.field(converter=float)
    w: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    b: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    m: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))

    _FIELD_NAMES: ClassVar[tuple[str, ...]] = ("A", "c", "w", "b", "m")

    @classmethod
    def from_mapping(
        cls, parameters: "GeneralizedLorentzianParameters | Mapping[str, float]"
    ) -> "GeneralizedLorentzianParameters":
        if isinstance(parameters, cls):
            return parameters
        return cls(**parameters)

    @classmethod
    def normalize(
        cls,
        parameters: (
            "GeneralizedLorentzianParameters"
            " | Mapping[str, float]"
            " | list[GeneralizedLorentzianParameters | Mapping[str, float]]"
        ),
    ) -> list["GeneralizedLorentzianParameters"]:
        if isinstance(parameters, (list, tuple)):
            return [cls.from_mapping(parameter) for parameter in parameters]
        return [cls.from_mapping(parameters)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._FIELD_NAMES)

    def __len__(self) -> int:
        return len(self._FIELD_NAMES)

    def __getitem__(self, key: str) -> float:
        if key not in self._FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def z_extent(self, density_cutoff_ratio: float = 1e-3) -> tuple[float, float]:
        """Get the z extent of the generalized Lorentzian term based on the density cutoff ratio."""

        # Calculate the z extent based on the density cutoff ratio
        if density_cutoff_ratio >= 1.0:
            return (self.c, self.c)
        else:
            delta_z = abs(
                (density_cutoff_ratio ** (-1.0 / self.m) - 1.0) ** (1.0 / self.b)
                * self.w
            )
            return (self.c - delta_z, self.c + delta_z)

    def dens_func(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """Return the density function for the generalized Lorentzian term. May contain negative values!"""
        z_arr = np.asarray(z)
        return self.A * (1.0 + np.abs((z_arr - self.c) / self.w) ** self.b) ** (-self.m)


@attrs.define(kw_only=True, slots=False, frozen=True)
class GeneralizedLorentzianSum(_DensityProfile):
    """
    Summed series of generalized Lorentzian terms:
    n(z) = sum_i A_i [1 + |(z - c_i)/w_i|^b_i]^{-m_i}

    Args:
        parameters: (list[GeneralizedLorentzianParameters]|GeneralizedLorentzianParameters) Parameters or list of parameters for each generalized Lorentzian term.
        density_cutoff_ratio: (float) |OPTIONAL| Ratio of the density cutoff to the maximum density. The guaranteed minimum density ratio relative to the maximal component at the edges is the number of terms times this value. Defaults to 1e-3.
        nominal_density: (float) [m^-3] |NOT REFERENCED| The maximal density contribution for an individual Lorentzian for this profile. This quantity is determined by the dataset used.
    """

    SUBCLASS: ClassVar[str] = "generalized_lorentzian_sum"

    parameters: list[GeneralizedLorentzianParameters] = attrs.field(
        converter=GeneralizedLorentzianParameters.normalize
    )
    density_cutoff_ratio: float = attrs.field(
        default=1e-3, converter=float, validator=attrs.validators.gt(0.0)
    )
    nominal_density: float = attrs.field(init=False, default=0.0)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        # Determine the nominal density as the maximum of the sum of the generalized Lorentzian terms at their centers
        if [p.A for p in self.parameters if p.A > 0.0] == []:
            raise ValueError("At least one parameter must have a positive amplitude A.")
        object.__setattr__(self, "nominal_density", max(p.A for p in self.parameters))

    def get_z_extent(self) -> tuple[float, float]:
        extents = [
            p.z_extent(self.density_cutoff_ratio * abs(self.nominal_density / p.A))
            for p in self.parameters
            if p.A != 0.0
        ]
        return (
            min(extent[0] for extent in extents),
            max(extent[1] for extent in extents),
        )

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        return (
            lambda z, r: np.maximum(sum(p.dens_func(z) for p in self.parameters), 0.0)
            / self.nominal_density
        )
