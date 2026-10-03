"""PV-facing Action/Variable classes for lume_fbpic.

`inversion_fbpic`'s config objects (`SimulationHyperparameters`, `_LaserPulse`,
`_DensityProfile` subclasses) are frozen `attrs` classes, so writable actions here replace
the whole held object via `attrs.evolve()` rather than mutating a field in place -- unlike
`lume-cheetah`'s actions, which `setattr()` directly on Cheetah's plain mutable elements.
"""

from __future__ import annotations

from typing import Any

import attrs
import numpy as np
from scipy.constants import c, e, m_e

from lume.actions import Action, ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ParticleGroupVariable, ScalarVariable

from lume_fbpic.simulator import FBPICSimulator

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.utils.distributions import (
    COORD_NAMES,
    LONGITUDINAL_PROFILE_BINS,
    SPLINE,
    compute_moment_descriptor,
    crop_central_particles,
    select_by_uz,
)

# Electron rest energy [eV]: converts a ParticleGroup's px/py/pz [eV/c] to normalized u = p/(m c).
_MC2_EV = m_e * c**2 / e


class LaserFieldAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable scalar mapped onto one field of the simulator's `laser` config object.

    Every constructed `_LaserPulse` has *both* `energy` and `a0` populated -- whichever
    wasn't given at construction is derived and stored too (tracked on the private
    `_amplitude_source` attribute). A plain `attrs.evolve()` on *any* field -- not just
    `energy`/`a0` themselves -- would otherwise carry over both and re-trip
    `_LaserPulse`'s "provide exactly one" check. So every `_set()` here nulls out
    whichever of the pair should not survive the change: the sibling when setting
    `energy`/`a0` directly, or whichever one is currently the derived one otherwise.
    """

    field_name: str

    def _get(self, simulator: FBPICSimulator) -> Any:
        return getattr(simulator.laser, self.field_name)

    def _set(self, simulator: FBPICSimulator, value: Any) -> None:
        laser = simulator.laser
        changes: dict[str, Any] = {self.field_name: value}
        if self.field_name in ("energy", "a0"):
            changes["a0" if self.field_name == "energy" else "energy"] = None
        else:
            derived = "a0" if laser._amplitude_source == "energy" else "energy"
            changes[derived] = None
        simulator.laser = attrs.evolve(laser, **changes)


class ZernikeCoefficientAction(LaserFieldAction):
    """Writable scalar mapped onto ONE entry of the laser's `zernike_coefficients` dict (for
    example `coefficient="coma_x"`), leaving the other coefficients as they are. Only
    `LasyLaserPulse` has this field; the energy/`a0` handling of `LaserFieldAction._set` carries
    over unchanged. A coefficient name the laser does not accept is rejected when the new
    config is built."""

    coefficient: str
    field_name: str = "zernike_coefficients"

    def _get(self, simulator: FBPICSimulator) -> Any:
        return super()._get(simulator).get(self.coefficient, 0.0)

    def _set(self, simulator: FBPICSimulator, value: Any) -> None:
        coefficients = {**super()._get(simulator), self.coefficient: value}
        super()._set(simulator, coefficients)


class HyperparameterFieldAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable scalar mapped onto one field of the simulator's `hyparams` config object."""

    field_name: str

    def _get(self, simulator: FBPICSimulator) -> Any:
        return getattr(simulator.hyparams, self.field_name)

    def _set(self, simulator: FBPICSimulator, value: Any) -> None:
        simulator.hyparams = attrs.evolve(simulator.hyparams, **{self.field_name: value})


class DensityFieldAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable scalar mapped onto one field of one of the simulator's density profiles,
    addressed by index into `simulator.densities`."""

    density_index: int
    field_name: str

    def _get(self, simulator: FBPICSimulator) -> Any:
        return getattr(simulator.densities[self.density_index], self.field_name)

    def _set(self, simulator: FBPICSimulator, value: Any) -> None:
        densities = list(simulator.densities)
        densities[self.density_index] = attrs.evolve(
            densities[self.density_index], **{self.field_name: value}
        )
        simulator.densities = densities


class DopantFractionAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable dopant fraction of a two-gas mixture, held in two density profiles of
    `simulator.densities` (`host_index` for the background gas, `dopant_index` for the dopant).

    The fraction is `dopant atoms / (host atoms + dopant atoms)`. A profile's atom density is
    its `nominal_density / num_ionization_levels(species)` -- `_DensityProfile.add_to_simulation`
    divides by the level count -- so setting a new fraction rewrites BOTH `nominal_density`
    values, scaled back up by each species' level count, while keeping the total atom density
    (host + dopant) constant. That is the scan variable `nitrogen_dopant_fraction` of the
    `initial_sample` dataset, where `n_N = n_gas * fraction` and `n_He = n_gas * (1 - fraction)`
    with `n_gas` fixed. Both profiles must have a `species`, and the fraction must lie strictly
    between 0 and 1 (a profile's `nominal_density` has to stay positive).
    """

    dopant_index: int
    host_index: int

    def _atom_densities(self, simulator: FBPICSimulator) -> tuple[float, float]:
        """Return `(host, dopant)` neutral-atom densities in m^-3."""
        host = simulator.densities[self.host_index]
        dopant = simulator.densities[self.dopant_index]
        return (
            host.nominal_density / _num_levels(host),
            dopant.nominal_density / _num_levels(dopant),
        )

    def _get(self, simulator: FBPICSimulator) -> Any:
        host, dopant = self._atom_densities(simulator)
        return dopant / (host + dopant)

    def _set(self, simulator: FBPICSimulator, value: Any) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError(f"Dopant fraction must be strictly between 0 and 1, got {value}.")
        host_atoms, dopant_atoms = self._atom_densities(simulator)
        total = host_atoms + dopant_atoms
        densities = list(simulator.densities)
        host = densities[self.host_index]
        dopant = densities[self.dopant_index]
        densities[self.host_index] = attrs.evolve(
            host, nominal_density=_num_levels(host) * total * (1.0 - value)
        )
        densities[self.dopant_index] = attrs.evolve(
            dopant, nominal_density=_num_levels(dopant) * total * value
        )
        simulator.densities = densities


class MomentDescriptorAction(ReadOnlyActionMixin[FBPICSimulator], ScalarVariable):
    """Read-only scalar: ONE feature of the moment descriptor of `simulator.final_particles`.

    The descriptor is `inversion_fbpic.utils.distributions.compute_moment_descriptor` -- the
    33 features (3 momentum centroids, 21 covariance entries, 2 x `longitudinal_bins`
    longitudinal `uz` features, charge) that `build_dataset.py` writes for each simulation.
    The particles are selected the way `build_dataset.py` selects them first: `uz >= uz_min`,
    then the central `central_fraction` on each of the six phase-space axes (either `None`
    skips that step). `make_descriptor_actions()` builds the whole set. The value is NaN
    before a run, or when fewer than two particles survive the selection.

    The descriptor is computed once per set of selection parameters and cached on the
    simulator against the current `final_particles`, so reading all 33 features costs one
    computation.
    """

    central_fraction: float | None = 0.95
    feature: str
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS
    uz_min: float | None = 30.0

    def _get(self, simulator: FBPICSimulator) -> Any:
        descriptor = _descriptor(
            simulator, self.uz_min, self.central_fraction, self.longitudinal_bins
        )
        return descriptor.get(self.feature, float("nan"))


class StatAction(ReadOnlyActionMixin[FBPICSimulator], ScalarVariable):
    """Read-only scalar sourced from `simulator.stats` (populated after a run)."""

    stat_name: str

    def _get(self, simulator: FBPICSimulator) -> Any:
        return simulator.stats.get(self.stat_name, float("nan"))


class FinalParticlesAction(ReadOnlyActionMixin[FBPICSimulator], ParticleGroupVariable):
    """Read-only ParticleGroup sourced from `simulator.final_particles` (populated after a
    run).

    The value is `None` while there are no particles -- before a run, or in a model loaded from
    an archive written without them. `validate_value` accepts that, as LUME's own check would
    reject it and with it every `model.get()` that includes this variable (including the one
    `lume-pva` makes for all variables on each cycle).
    """

    def validate_value(self, value: Any) -> None:
        if value is not None:
            super().validate_value(value)

    def _get(self, simulator: FBPICSimulator) -> Any:
        return simulator.final_particles


def action_from_config(config: dict[str, Any], extra_classes: dict[str, type] | None = None) -> Action:
    """Rebuild an action from `action_to_config()`'s output.

    `config["class"]` is looked up among this module's action classes, plus `extra_classes`
    (class name -> class) for subclasses defined elsewhere.

    Raises:
        ValueError: If the class name is not known.
    """
    classes = {**_action_classes(), **(extra_classes or {})}
    name = config["class"]
    if name not in classes:
        raise ValueError(
            f"Unknown action class {name!r}; known: {sorted(classes)}. Pass it in "
            "`extra_classes` if it is defined outside lume_fbpic.actions."
        )
    return classes[name](**config["parameters"])


def action_to_config(action: Action) -> dict[str, Any]:
    """JSON-compatible description of an action: `{"class": <class name>, "parameters": {...}}`."""
    return {"class": type(action).__name__, "parameters": action.model_dump(mode="json")}


def make_actions(simulator: FBPICSimulator) -> list[Action]:
    """Build the default HTU downramp-injection control/output variable set.

    Matches `demo_downramp_simulation`'s config shape: one `GaussianLaserPulse` and two
    `SmoothSineFlattop` density profiles (index 0 = flattop, index 1 = downramp).

    Control variables (writable): laser energy, focal position, temporal width; flattop and
    downramp nominal densities; downramp length.
    Outputs (read-only): charge, mean/std energy, and `final_particles`.
    """
    return [
        LaserFieldAction(name="laser_energy", field_name="energy", unit="J"),
        LaserFieldAction(
            name="laser_focal_position", field_name="focal_position", unit="m"
        ),
        LaserFieldAction(
            name="laser_temporal_width", field_name="tau_fwhm", unit="s"
        ),
        DensityFieldAction(
            name="flattop_density",
            density_index=0,
            field_name="nominal_density",
            unit="m^-3",
        ),
        DensityFieldAction(
            name="downramp_density",
            density_index=1,
            field_name="nominal_density",
            unit="m^-3",
        ),
        DensityFieldAction(
            name="downramp_length",
            density_index=1,
            field_name="downramp_length",
            unit="m",
        ),
        StatAction(
            name="charge_pc", stat_name="charge_pc", unit="pC", read_only=True
        ),
        StatAction(
            name="energy_mean_mev",
            stat_name="energy_mean_mev",
            unit="MeV",
            read_only=True,
        ),
        StatAction(
            name="energy_std_mev",
            stat_name="energy_std_mev",
            unit="MeV",
            read_only=True,
        ),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]


def make_descriptor_actions(
    *,
    central_fraction: float | None = 0.95,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
    prefix: str = "descriptor_",
    uz_min: float | None = 30.0,
) -> list[Action]:
    """Build one read-only `MomentDescriptorAction` per descriptor feature (33 by default).

    The names are `prefix` + the feature name (`descriptor_mean_uz`,
    `descriptor_cov_x_ux`, `descriptor_total_beam_charge_c`, ...), in the order
    `compute_moment_descriptor` returns them. The default selection (`uz_min=30`,
    `central_fraction=0.95`) is the `build_dataset.py` default, tuned to the
    ionization-injection beams; a lower-energy bunch (the LWFA example's, for instance, has a
    mean `uz` near 2) needs a lower `uz_min`.
    """
    return [
        MomentDescriptorAction(
            name=f"{prefix}{feature}",
            feature=feature,
            central_fraction=central_fraction,
            longitudinal_bins=longitudinal_bins,
            read_only=True,
            uz_min=uz_min,
            unit="C" if feature == "total_beam_charge_c" else None,
        )
        for feature in _descriptor_features(longitudinal_bins)
    ]


def _action_classes() -> dict[str, type]:
    """Name -> class for every action class defined in this module."""
    classes = (
        DensityFieldAction,
        DopantFractionAction,
        FinalParticlesAction,
        HyperparameterFieldAction,
        LaserFieldAction,
        MomentDescriptorAction,
        StatAction,
        ZernikeCoefficientAction,
    )
    return {cls.__name__: cls for cls in classes}


def _descriptor(
    simulator: FBPICSimulator,
    uz_min: float | None,
    central_fraction: float | None,
    longitudinal_bins: int,
) -> dict[str, float]:
    """Moment descriptor of `simulator.final_particles` for one selection (empty if none).

    Cached on the simulator against the identity of the current `final_particles` and the
    selection parameters, so the 33 `MomentDescriptorAction`s of one set share one computation.
    """
    particles = simulator.final_particles
    if particles is None:
        return {}
    key = (uz_min, central_fraction, longitudinal_bins)
    cached = getattr(simulator, "_descriptor_cache", None)
    if cached is not None and cached[0] is particles and cached[1] == key:
        return cached[2]

    phase_space = np.stack(
        [
            particles.x,
            particles.px / _MC2_EV,
            particles.y,
            particles.py / _MC2_EV,
            particles.z,
            particles.pz / _MC2_EV,
        ],
        axis=-1,
    )
    # The library's weights are electron counts (it multiplies by e for the charge); the
    # ParticleGroup's are coulombs.
    weights = np.asarray(particles.weight, dtype=np.float64) / e
    try:
        if uz_min is not None:
            phase_space, weights = select_by_uz(phase_space, weights, uz_min=uz_min)
        if central_fraction is not None:
            phase_space, weights = crop_central_particles(
                phase_space, weights, central_fraction=central_fraction
            )
        result = compute_moment_descriptor(
            phase_space, weights, longitudinal_mode=SPLINE, longitudinal_bins=longitudinal_bins
        )
    except ValueError:  # fewer than two usable particles after the selection
        result = {}
    simulator._descriptor_cache = (particles, key, result)
    return result


def _descriptor_features(longitudinal_bins: int) -> list[str]:
    """Feature names of the spline-mode descriptor, in `compute_moment_descriptor`'s order."""
    features = ["mean_ux", "mean_uy", "mean_uz"]
    for row in range(6):
        for column in range(row, 6):
            features.append(f"cov_{COORD_NAMES[row]}_{COORD_NAMES[column]}")
    for index in range(longitudinal_bins):
        features.append(f"longitudinal_mean_uz_{index:02d}")
        features.append(f"longitudinal_rms_uz_{index:02d}")
    features.append("total_beam_charge_c")
    return features


def _num_levels(profile: Any) -> int:
    """Ionization-level count of a density profile's species (the divisor `add_to_simulation`
    applies to the profile's `nominal_density`)."""
    if profile.species is None:
        raise ValueError("A dopant fraction needs profiles with a species; got species=None.")
    return _DensityProfile._get_num_ionization_levels(profile.species)
