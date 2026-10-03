"""`_DensityProfile` subclasses defined by `lume_fbpic` itself, not `inversion_fbpic`.

`_DensityProfile` subclass registration goes through `SerializableConfig.__init_subclass__`
(a plain attrs/Python mechanism keyed on the `SUBCLASS` class attribute), which doesn't care
what package defines the subclass -- so a new density-profile shape can be added here,
composing on top of `inversion_fbpic` exactly like the rest of `lume_fbpic`, without
modifying it.
"""

from __future__ import annotations

from typing import ClassVar

import attrs
import numpy as np
import numpy.typing as npt

from inversion_fbpic.lib.density_core import DensityCallable, _DensityProfile


@attrs.define(kw_only=True, slots=False, frozen=True)
class LinearRampFlattop(_DensityProfile):
    """Linear-ramp-then-flat density profile, reproducing fbpic's own
    `docs/source/example_input/lwfa_script.py` `dens_func` exactly: zero relative density
    before `ramp_start`, a linear ramp up to full relative density over `ramp_length`, then
    flat (relative density 1) indefinitely after that. Unlike `SmoothSineFlattop` or
    `AsymmetricSine`, there is no downramp and the plateau has no declared end -- it really
    is unbounded, matching upstream's own `dens_func`, which never tapers off either.

    `get_z_extent()` reports only the ramp's own span (`ramp_start` to
    `ramp_start + ramp_length`). The flat region beyond it is still physically real --
    `build_density_function()`'s `dens_func` returns 1 there unconditionally, and fbpic
    loads particles across the whole simulation box regardless of this declared extent --
    it's just not counted towards `Simulation`'s auto-derived interaction length
    (`z_extent span + SimulationHyperparameters.right_buffer`). Set `right_buffer`
    explicitly to control how far past the ramp the run actually goes, mirroring upstream's
    own `L_interact`, which is likewise a literal constant independent of the density
    profile's shape.

    Args:
        ramp_start: (float) [m] z position where the linear ramp begins; relative density is
            0 before this point.
        ramp_length: (float) [m] Length of the linear ramp from 0 to full relative density.
    """

    SUBCLASS: ClassVar[str] = "linear_ramp_flattop"

    ramp_start: float = attrs.field(converter=float)
    ramp_length: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

    def get_z_extent(self) -> tuple[float, float]:
        return (self.ramp_start, self.ramp_start + self.ramp_length)

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        ramp_start = self.ramp_start
        ramp_length = self.ramp_length

        def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
            z_arr = np.asarray(z)
            n = np.ones_like(z_arr, dtype=float)
            n = np.where(
                z_arr < ramp_start + ramp_length, (z_arr - ramp_start) / ramp_length, n
            )
            n = np.where(z_arr < ramp_start, 0.0, n)
            return n

        return dens_func


@attrs.define(kw_only=True, slots=False, frozen=True)
class GeneralizedGaussianProfile(_DensityProfile):
    """Generalized-normal (generalized Gaussian) density bump in z, wrapping
    `inversion_fbpic.density_profiles.downramp_injection.build_generalized_gaussian_profile`
    exactly: `dens_func(z, r) = gauss_peak * exp(-(|z - z0| / alpha) ** beta)`, independent
    of r. `beta=2` is a standard Gaussian; `beta<2` is more sharply peaked with heavier
    tails, `beta>2` is flatter near the peak with a steeper falloff. This is the shape
    `runs/initial_sample/ionization_injection_template.py` uses for its He/N gas-jet
    target -- no existing `inversion_fbpic.lib.density_profiles` class wraps this function
    (only `AsymmetricSine`/`SmoothSineFlattop`/etc.'s sine-squared ramps do), so this
    composes it the same way `LinearRampFlattop` composes the linear-ramp `dens_func`.

    `get_z_extent()` reports `z0 +/- z_extent_alphas * alpha` -- `z_extent_alphas` defaults
    to 2.8, matching the exact constant `ionization_injection_template.py` itself uses for
    its own `interaction_length = 2.8 * density_alpha_m + density_center_location_m`. Note
    that upstream quantity is a one-sided offset from z=0 (the box's own right edge in that
    script), not a symmetric span the way `get_z_extent()` returns here -- matching the
    upstream run's total interaction length via `SimulationHyperparameters.right_buffer`
    therefore requires accounting for that asymmetry (see the `lume-fbpic` conversion script
    for the worked calculation), not just copying the "2.8" constant.

    Args:
        gauss_peak: (float) Peak relative-density multiplier at `z0`. Not itself normalized
            to 1 -- the upstream script's own default is 1.1, a small intentional density
            overshoot, not exactly 1.0.
        z0: (float) [m] Location of the profile's peak.
        alpha: (float) [m] Scale parameter (must be > 0).
        beta: (float) Shape parameter (must be > 0).
        z_extent_alphas: (float) |OPTIONAL| Number of `alpha`-widths on each side of `z0`
            reported by `get_z_extent()`. Defaults to 2.8 (see above).
    """

    SUBCLASS: ClassVar[str] = "generalized_gaussian"

    gauss_peak: float = attrs.field(converter=float)
    z0: float = attrs.field(converter=float)
    alpha: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    beta: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    z_extent_alphas: float = attrs.field(
        default=2.8, converter=float, validator=attrs.validators.gt(0.0)
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

    def get_z_extent(self) -> tuple[float, float]:
        half_width = self.z_extent_alphas * self.alpha
        return (self.z0 - half_width, self.z0 + half_width)

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        from inversion_fbpic.density_profiles.downramp_injection import (
            build_generalized_gaussian_profile,
        )

        return build_generalized_gaussian_profile(
            gauss_peak=self.gauss_peak,
            gauss_z0=self.z0,
            gauss_alpha=self.alpha,
            gauss_beta=self.beta,
        )
