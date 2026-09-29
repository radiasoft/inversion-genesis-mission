"""Composable finite density profiles for FBPIC simulations."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Literal

import attrs
import numpy as np
import numpy.typing as npt

from inversion_fbpic.lib.density_core import _DensityProfile, DensityCallable


SkewMode = Literal["auto", "finite_supergaussian", "exponential", "saturated"]
ConcreteSkewMode = Literal["finite_supergaussian", "exponential", "saturated"]
_SKEW_MODE_VALUES = ("auto", "finite_supergaussian", "exponential", "saturated")


def _finite_float(
    _instance: object, attribute: attrs.Attribute[float], value: float
) -> None:
    del _instance
    if not np.isfinite(value):
        raise ValueError(f"{attribute.name} must be finite.")


def _supergaussian(
    shifted_z: npt.NDArray[np.float64], mean: float, fwhm: float, beta: float
) -> npt.NDArray[np.float64]:
    return np.exp(
        -np.log(2.0) * (4.0 * (shifted_z - mean) ** 2 / fwhm**2) ** (beta / 2.0)
    )


def _supergaussian_half_span(
    fwhm: float, beta: float, num_sigma_extent: float
) -> float:
    return fwhm / 2.0 * (num_sigma_extent**2 / (2.0 * np.log(2.0))) ** (1.0 / beta)


def _cosine_squared_flattop(
    shifted_z: npt.NDArray[np.float64], main_fwhm: float, ramp_length: float
) -> npt.NDArray[np.float64]:
    flat_half_width = (main_fwhm - ramp_length) / 2.0
    ramp_half_width = (main_fwhm + ramp_length) / 2.0
    distance = np.abs(shifted_z)
    result = np.zeros_like(shifted_z, dtype=float)
    result[distance <= flat_half_width] = 1.0
    in_ramp = (distance > flat_half_width) & (distance <= ramp_half_width)
    phase = np.pi * (distance[in_ramp] - flat_half_width) / (2.0 * ramp_length)
    result[in_ramp] = np.cos(phase) ** 2
    return result


def _cosine_squared_fringe(
    shifted_z: npt.NDArray[np.float64], center: float, fwhm: float
) -> npt.NDArray[np.float64]:
    normalized_distance = (shifted_z - center) / fwhm
    result = np.zeros_like(shifted_z, dtype=float)
    in_fringe = np.abs(normalized_distance) <= 1.0
    result[in_fringe] = np.cos(np.pi * normalized_distance[in_fringe] / 2.0) ** 2
    return result


def _power_law_flattop(
    shifted_z: npt.NDArray[np.float64],
    flattop_width: float,
    transition_length: float,
    transition_exponent: float,
) -> npt.NDArray[np.float64]:
    flat_half_width = flattop_width / 2.0
    transition_coordinate = (np.abs(shifted_z) - flat_half_width) / transition_length
    result = np.zeros_like(shifted_z, dtype=float)
    result[transition_coordinate <= 0.0] = 1.0
    in_transition = (transition_coordinate > 0.0) & (transition_coordinate <= 1.0)
    result[in_transition] = (
        1.0 - transition_coordinate[in_transition] ** transition_exponent
    ) ** 3
    return result


def _power_law(
    shifted_z: npt.NDArray[np.float64],
    transition_length: float,
    transition_exponent: float,
) -> npt.NDArray[np.float64]:
    transition_coordinate = np.abs(shifted_z) / transition_length
    result = np.zeros_like(shifted_z, dtype=float)
    in_transition = transition_coordinate <= 1.0
    result[in_transition] = (
        1.0 - transition_coordinate[in_transition] ** transition_exponent
    ) ** 3
    return result


def _lorentzian_cutoff_coordinate(
    coordinate_exponent: float,
    profile_exponent: float,
    density_cutoff_ratio: float,
) -> float:
    return (density_cutoff_ratio ** (-1.0 / profile_exponent) - 1.0) ** (
        1.0 / coordinate_exponent
    )


def _shape_half_extent(profile_type: str, parameters: dict[str, float]) -> float:
    if profile_type == "supergaussian":
        return _supergaussian_half_span(
            parameters["fwhm"], parameters["beta"], parameters["num_sigma_extent"]
        )
    if profile_type == "cosine_squared_flattop":
        return (parameters["fwhm"] + parameters["ramp_length"]) / 2.0
    if profile_type == "power_law_flattop":
        return parameters["flattop_width"] / 2.0 + parameters["transition_length"]
    if profile_type == "power_law":
        return parameters["transition_length"]
    if profile_type == "lorentzian_flattop":
        return parameters["flattop_width"] / 2.0 + parameters[
            "transition_length"
        ] * _lorentzian_cutoff_coordinate(
            parameters["coordinate_exponent"],
            parameters["profile_exponent"],
            parameters["density_cutoff_ratio"],
        )
    if profile_type == "lorentzian":
        return parameters["transition_length"] * _lorentzian_cutoff_coordinate(
            parameters["coordinate_exponent"],
            parameters["profile_exponent"],
            parameters["density_cutoff_ratio"],
        )
    if profile_type == "cosine_squared":
        return parameters["fwhm"]
    raise ValueError(f"Unsupported profile type: {profile_type!r}.")


def _shape_support_cutoff_ratio(
    profile_type: str, parameters: dict[str, float]
) -> float:
    if profile_type == "supergaussian":
        return np.exp(-parameters["num_sigma_extent"] ** 2 / 2.0)
    if profile_type in {"lorentzian_flattop", "lorentzian"}:
        return parameters["density_cutoff_ratio"]
    return 0.0


def _shape_half_extent_at_cutoff(
    profile_type: str,
    parameters: dict[str, float],
    density_cutoff_ratio: float,
) -> float:
    if profile_type == "supergaussian":
        return (
            parameters["fwhm"]
            / 2.0
            * (-np.log(density_cutoff_ratio) / np.log(2.0))
            ** (1.0 / parameters["beta"])
        )
    if profile_type == "lorentzian_flattop":
        return parameters["flattop_width"] / 2.0 + parameters[
            "transition_length"
        ] * _lorentzian_cutoff_coordinate(
            parameters["coordinate_exponent"],
            parameters["profile_exponent"],
            density_cutoff_ratio,
        )
    if profile_type == "lorentzian":
        return parameters["transition_length"] * _lorentzian_cutoff_coordinate(
            parameters["coordinate_exponent"],
            parameters["profile_exponent"],
            density_cutoff_ratio,
        )
    return _shape_half_extent(profile_type, parameters)


def _effective_fringe_cutoff_ratio(
    profile_type: str,
    parameters: dict[str, float],
    fringe_relative_height: float,
) -> float | None:
    if fringe_relative_height <= 0.0:
        return None
    cutoff_ratio = _shape_support_cutoff_ratio(profile_type, parameters)
    if fringe_relative_height < cutoff_ratio:
        return None
    return cutoff_ratio / fringe_relative_height


def _fringe_half_extent(
    profile_type: str,
    parameters: dict[str, float],
    fringe_relative_height: float,
) -> float | None:
    cutoff_ratio = _effective_fringe_cutoff_ratio(
        profile_type, parameters, fringe_relative_height
    )
    if cutoff_ratio is None:
        return None
    return _shape_half_extent_at_cutoff(profile_type, parameters, cutoff_ratio)


def _shape_density(
    profile_type: str,
    parameters: dict[str, float],
    shifted_z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if profile_type == "supergaussian":
        return _supergaussian(shifted_z, 0.0, parameters["fwhm"], parameters["beta"])
    if profile_type == "cosine_squared_flattop":
        return _cosine_squared_flattop(
            shifted_z, parameters["fwhm"], parameters["ramp_length"]
        )
    if profile_type == "power_law_flattop":
        return _power_law_flattop(
            shifted_z,
            parameters["flattop_width"],
            parameters["transition_length"],
            parameters["transition_exponent"],
        )
    if profile_type == "power_law":
        return _power_law(
            shifted_z,
            parameters["transition_length"],
            parameters["transition_exponent"],
        )
    if profile_type == "lorentzian_flattop":
        transition_coordinate = (
            np.maximum(np.abs(shifted_z) - parameters["flattop_width"] / 2.0, 0.0)
            / parameters["transition_length"]
        )
        return (1.0 + transition_coordinate ** parameters["coordinate_exponent"]) ** (
            -parameters["profile_exponent"]
        )
    if profile_type == "lorentzian":
        transition_coordinate = np.abs(shifted_z) / parameters["transition_length"]
        return (1.0 + transition_coordinate ** parameters["coordinate_exponent"]) ** (
            -parameters["profile_exponent"]
        )
    if profile_type == "cosine_squared":
        return _cosine_squared_fringe(shifted_z, 0.0, parameters["fwhm"])
    raise ValueError(f"Unsupported profile type: {profile_type!r}.")


def _validated_shape_parameters(
    profile_type: str, parameters: dict[str, float], *, role: str
) -> dict[str, float]:
    schemas = {
        "supergaussian": ({"fwhm", "beta"}, {"num_sigma_extent": 3.0}),
        "cosine_squared_flattop": ({"fwhm", "ramp_length"}, {}),
        "power_law_flattop": (
            {"flattop_width", "transition_length", "transition_exponent"},
            {},
        ),
        "power_law": ({"transition_length", "transition_exponent"}, {}),
        "lorentzian_flattop": (
            {
                "flattop_width",
                "transition_length",
                "coordinate_exponent",
                "profile_exponent",
            },
            {"density_cutoff_ratio": 1.0e-3},
        ),
        "lorentzian": (
            {"transition_length", "coordinate_exponent", "profile_exponent"},
            {"density_cutoff_ratio": 1.0e-3},
        ),
        "cosine_squared": ({"fwhm"}, {}),
    }
    if profile_type not in schemas:
        raise ValueError(f"Unsupported {role}_profile_type: {profile_type!r}.")
    if not isinstance(parameters, dict):
        raise ValueError(f"{role}_profile_parameters must be a dictionary.")

    required, defaults = schemas[profile_type]
    expected = required | set(defaults)
    missing = sorted(required - set(parameters))
    unexpected = sorted(set(parameters) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"Invalid {role}_profile_parameters: {'; '.join(details)}.")

    normalized = dict(defaults)
    for name, value in parameters.items():
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{role}_profile_parameters[{name!r}] must be a finite number."
            ) from error
        if not np.isfinite(normalized[name]):
            raise ValueError(
                f"{role}_profile_parameters[{name!r}] must be a finite number."
            )

    for name in (
        "fwhm",
        "ramp_length",
        "flattop_width",
        "transition_length",
        "beta",
        "num_sigma_extent",
        "profile_exponent",
    ):
        if name in normalized and normalized[name] <= 0.0:
            raise ValueError(
                f"{role}_profile_parameters[{name!r}] must be greater than zero."
            )
    if "coordinate_exponent" in normalized and normalized["coordinate_exponent"] <= 2.0:
        raise ValueError(
            f"{role}_profile_parameters['coordinate_exponent'] must be greater than 2.0."
        )
    if (
        "density_cutoff_ratio" in normalized
        and not 0.0 < normalized["density_cutoff_ratio"] < 1.0
    ):
        raise ValueError(
            f"{role}_profile_parameters['density_cutoff_ratio'] must be between zero and one."
        )
    if "transition_exponent" in normalized and normalized["transition_exponent"] <= 2.0:
        raise ValueError(
            f"{role}_profile_parameters['transition_exponent'] must be greater than 2.0."
        )
    if (
        profile_type == "cosine_squared_flattop"
        and normalized["ramp_length"] > normalized["fwhm"]
    ):
        raise ValueError(
            f"{role}_profile_parameters['ramp_length'] must be less than or equal to fwhm."
        )
    return normalized


def _skew_multiplier(
    shifted_z: npt.NDArray[np.float64],
    left_span: float,
    right_span: float,
    skew_rate: float,
    skew_mode: SkewMode,
) -> npt.NDArray[np.float64]:
    if skew_mode == "exponential":
        return np.exp(skew_rate * shifted_z)
    if skew_mode == "saturated":
        saturation_length = 3.0 * max(left_span, right_span)
        return np.exp(
            skew_rate * saturation_length * np.tanh(shifted_z / saturation_length)
        )

    support_width = left_span + right_span
    return np.exp(
        skew_rate * shifted_z - 4.0 * np.log(2.0) * shifted_z**2 / support_width**2
    )


@attrs.define(kw_only=True, slots=False, frozen=True)
class _FiniteSkewedProfile(_DensityProfile):
    """Shared extent metadata and post-composition skew for density profiles."""

    start_position: float = attrs.field(converter=float, validator=_finite_float)
    skew_rate: float = attrs.field(
        default=0.0, converter=float, validator=_finite_float
    )
    skew_mode: SkewMode = attrs.field(
        default="auto",
        validator=attrs.validators.in_(_SKEW_MODE_VALUES),
    )

    def __attrs_post_init__(self) -> None:
        self._validate_profile()
        object.__setattr__(self, "skew_mode", self._resolve_skew_mode())
        self._validate_skew_mode()
        super().__attrs_post_init__()
        left_span, _ = self._profile_support_spans()
        object.__setattr__(self, "centroid", self.start_position + left_span)

    def get_z_extent(self) -> tuple[float, float]:
        left_span, right_span = self._profile_support_spans()
        return (
            self.start_position,
            self.start_position + left_span + right_span,
        )

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        left_span, right_span = self._profile_support_spans()

        def density(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.NDArray[np.float64]:
            del r
            z_values = np.asarray(z, dtype=float)
            shifted_z = z_values - self.centroid
            return self._unskewed_density(shifted_z) * _skew_multiplier(
                shifted_z,
                left_span,
                right_span,
                self.skew_rate,
                self.skew_mode,
            )

        return density

    def _validate_profile(self) -> None:
        pass

    def _resolve_skew_mode(self) -> ConcreteSkewMode:
        if self.skew_mode == "auto":
            return "finite_supergaussian"
        return self.skew_mode

    def _validate_skew_mode(self) -> None:
        pass

    @abstractmethod
    def _profile_half_extent(self) -> float: ...

    def _profile_support_spans(self) -> tuple[float, float]:
        half_extent = self._profile_half_extent()
        return (half_extent, half_extent)

    @abstractmethod
    def _unskewed_density(
        self, shifted_z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


@attrs.define(kw_only=True, slots=False, frozen=True)
class PowerLawFlattop(_FiniteSkewedProfile):
    """Finite flattop with a second-order-continuous power-law edge.

    The relative density is one across the central flattop. On each edge it
    follows ``(1 - s**transition_exponent)**3`` for normalized coordinate
    ``s`` from zero to one. Requiring ``transition_exponent > 2`` makes the
    profile and its first two derivatives continuous at the flattop boundary;
    the cubic outer factor makes the same derivatives vanish at finite support.

    Args:
        start_position: (float) [m] Longitudinal start of the profile extent metadata. The density remains continuously evaluable outside this interval.
        flattop_width: (float) [m] Full width of the constant-density central region.
        transition_length: (float) [m] Length of each power-law edge transition.
        transition_exponent: (float) Power-law exponent. Must be greater than 2.
        skew_rate: (float) [m^-1] |OPTIONAL| Longitudinal skew rate. It is the centroid log-density slope for every skew mode. Defaults to 0.0.
        skew_mode: (Literal[str]) |OPTIONAL| Skew equation. auto resolves to finite_supergaussian for this compact-edge profile. saturated approaches reciprocal finite factors derived from skew_rate without adding a Gaussian tail envelope. exponential preserves the legacy multiplier. Defaults to auto.
    """

    SUBCLASS: ClassVar[str] = "power_law_flattop"

    flattop_width: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    transition_length: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    transition_exponent: float = attrs.field(
        converter=float, validator=attrs.validators.gt(2.0)
    )
    centroid: float = attrs.field(init=False)

    def _profile_half_extent(self) -> float:
        return self.flattop_width / 2.0 + self.transition_length

    def _unskewed_density(
        self, shifted_z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return _power_law_flattop(
            shifted_z,
            self.flattop_width,
            self.transition_length,
            self.transition_exponent,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class GenericConicalTarget(_FiniteSkewedProfile):
    """Target with finite extent metadata composed from a main profile and fringes.

    ``main_profile_parameters`` and ``fringe_profile_parameters`` are validated
    dictionaries for the selected type. Supported main types are
    ``supergaussian``, ``cosine_squared_flattop``, ``power_law_flattop``, and
    ``lorentzian_flattop``.
    Supported fringe types are ``supergaussian``, ``cosine_squared``,
    ``power_law``, and ``lorentzian``.
    Skew is applied once after adding every selected component.

    Args:
        start_position: (float) [m] Longitudinal start of the full target extent metadata. The composed density remains continuously evaluable outside this interval.
        main_profile_type: (Literal[str]) Type of the centered main profile.
        main_profile_parameters: (dict[str, float]) Parameters for the main profile type.
        fringe_profile_type: (Literal[str]|None) |OPTIONAL| Type of the fringe profile. Defaults to None.
        fringe_profile_parameters: (dict[str, float]) |OPTIONAL| Parameters for the fringe profile type. Defaults to {}.
        fringe_center_offset: (float|None) [m] |OPTIONAL| Positive distance from the main center to each fringe center. Required when a fringe type is selected.
        fringe_relative_height: (float) |OPTIONAL| Amplitude of every fringe relative to the main profile. Fringe support is included where its amplitude relative to the unit main lobe is at or above the component cutoff. Defaults to 1.0.
        fringe_side: (Literal[str]) |OPTIONAL| Select ``left``, ``right``, or ``both`` fringes. Defaults to ``both``.
        skew_rate: (float) [m^-1] |OPTIONAL| Longitudinal skew rate applied after combining components. It is the centroid log-density slope for every skew mode. Defaults to 0.0.
        skew_mode: (Literal[str]) |OPTIONAL| Skew equation. auto resolves to saturated for any Lorentzian component or SuperGaussian with beta <= 1; otherwise it resolves to finite_supergaussian. saturated approaches reciprocal finite factors alpha and 1/alpha, where alpha is derived from skew_rate and profile extent. exponential preserves the legacy multiplier. Defaults to auto.
    """

    SUBCLASS: ClassVar[str] = "generic_conical_target"

    main_profile_type: Literal[
        "supergaussian",
        "cosine_squared_flattop",
        "power_law_flattop",
        "lorentzian_flattop",
    ] = attrs.field(
        validator=attrs.validators.in_(
            [
                "supergaussian",
                "cosine_squared_flattop",
                "power_law_flattop",
                "lorentzian_flattop",
            ]
        )
    )
    main_profile_parameters: dict[str, float] = attrs.field()
    fringe_profile_type: (
        Literal["supergaussian", "cosine_squared", "power_law", "lorentzian"] | None
    ) = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.in_(
                ["supergaussian", "cosine_squared", "power_law", "lorentzian"]
            )
        ),
    )
    fringe_profile_parameters: dict[str, float] = attrs.field(factory=dict)
    fringe_center_offset: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    fringe_relative_height: float = attrs.field(
        default=1.0, converter=float, validator=attrs.validators.ge(0.0)
    )
    fringe_side: Literal["left", "right", "both"] = attrs.field(
        default="both", validator=attrs.validators.in_(["left", "right", "both"])
    )
    centroid: float = attrs.field(init=False)

    def _validate_profile(self) -> None:
        main_parameters = _validated_shape_parameters(
            self.main_profile_type, self.main_profile_parameters, role="main"
        )
        object.__setattr__(self, "main_profile_parameters", main_parameters)

        if self.fringe_profile_type is None:
            if self.fringe_profile_parameters or self.fringe_center_offset is not None:
                raise ValueError(
                    "fringe_profile_parameters and fringe_center_offset require fringe_profile_type."
                )
            if self.fringe_relative_height != 1.0:
                raise ValueError("fringe_relative_height requires fringe_profile_type.")
            return

        if self.fringe_center_offset is None:
            raise ValueError(
                "fringe_center_offset is required when fringe_profile_type is set."
            )
        fringe_parameters = _validated_shape_parameters(
            self.fringe_profile_type, self.fringe_profile_parameters, role="fringe"
        )
        object.__setattr__(self, "fringe_profile_parameters", fringe_parameters)

    def _resolve_skew_mode(self) -> ConcreteSkewMode:
        if self.skew_mode != "auto":
            return self.skew_mode
        components = [(self.main_profile_type, self.main_profile_parameters)]
        if self.fringe_profile_type is not None:
            components.append(
                (self.fringe_profile_type, self.fringe_profile_parameters)
            )
        if any(
            profile_type in {"lorentzian_flattop", "lorentzian"}
            or (profile_type == "supergaussian" and parameters["beta"] <= 1.0)
            for profile_type, parameters in components
        ):
            return "saturated"
        return "finite_supergaussian"

    def _validate_skew_mode(self) -> None:
        if self.skew_mode != "exponential" or self.skew_rate == 0.0:
            return
        components = [(self.main_profile_type, self.main_profile_parameters, "main")]
        if self.fringe_profile_type is not None:
            components.append(
                (
                    self.fringe_profile_type,
                    self.fringe_profile_parameters,
                    "fringe",
                )
            )
        for profile_type, parameters, role in components:
            if profile_type == "supergaussian" and parameters["beta"] <= 1.0:
                raise ValueError(
                    f"Bare exponential-skewed super-Gaussian {role} profiles with beta <= 1.0 are not supported."
                )
            if profile_type == "lorentzian_flattop" or profile_type == "lorentzian":
                raise ValueError(
                    f"Bare exponential-skewed Lorentzian {role} profiles are not supported."
                )

    def _profile_half_extent(self) -> float:
        return max(self._profile_support_spans())

    def _profile_support_spans(self) -> tuple[float, float]:
        main_span = _shape_half_extent(
            self.main_profile_type, self.main_profile_parameters
        )
        if self.fringe_profile_type is None:
            return (main_span, main_span)

        fringe_span = _fringe_half_extent(
            self.fringe_profile_type,
            self.fringe_profile_parameters,
            self.fringe_relative_height,
        )
        if fringe_span is None:
            return (main_span, main_span)
        fringe_extent = self.fringe_center_offset + fringe_span
        left_span = (
            max(main_span, fringe_extent)
            if self.fringe_side in ("left", "both")
            else main_span
        )
        right_span = (
            max(main_span, fringe_extent)
            if self.fringe_side in ("right", "both")
            else main_span
        )
        return (left_span, right_span)

    def _unskewed_density(
        self, shifted_z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        density = _shape_density(
            self.main_profile_type,
            self.main_profile_parameters,
            shifted_z,
        )
        if self.fringe_profile_type is None:
            return density

        if self.fringe_side in ("left", "both"):
            density += self.fringe_relative_height * _shape_density(
                self.fringe_profile_type,
                self.fringe_profile_parameters,
                shifted_z + self.fringe_center_offset,
            )
        if self.fringe_side in ("right", "both"):
            density += self.fringe_relative_height * _shape_density(
                self.fringe_profile_type,
                self.fringe_profile_parameters,
                shifted_z - self.fringe_center_offset,
            )
        return density
