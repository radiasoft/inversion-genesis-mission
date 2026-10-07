"""PV-facing Action/Variable classes for lume_fbpic.

`inversion_fbpic`'s config objects (`SimulationHyperparameters`, `_LaserPulse`,
`_DensityProfile` subclasses) are frozen `attrs` classes, so writable actions here replace
the whole held object via `attrs.evolve()` rather than mutating a field in place -- unlike
`lume-cheetah`'s actions, which `setattr()` directly on Cheetah's plain mutable elements.
"""

from __future__ import annotations

import typing

import attrs
import numpy
from scipy.constants import c, e, m_e

from lume.actions import Action, ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ParticleGroupVariable, ScalarVariable

from lume_fbpic.pwfa_config import FlatTopBunch, GaussianBunch
from lume_fbpic.simulator import BaseSimulator, FBPICSimulator, PWFASimulator

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


class BunchFieldAction(WritableActionMixin[PWFASimulator], ScalarVariable):
    """Writable scalar mapped onto one field of a PWFA simulator's `driver` or `witness` bunch."""

    bunch: str
    field_name: str

    def _bunch(self, simulator: PWFASimulator) -> typing.Any:
        if self.bunch not in ("driver", "witness"):
            raise ValueError(f"bunch must be 'driver' or 'witness', not {self.bunch!r}")
        bunch = getattr(simulator, self.bunch)
        if bunch is None:
            raise ValueError(f"the simulator has no {self.bunch}")
        return bunch

    def _get(self, simulator: PWFASimulator) -> typing.Any:
        return getattr(self._bunch(simulator), self.field_name)

    def _set(self, simulator: PWFASimulator, value: typing.Any) -> None:
        setattr(
            simulator, self.bunch, attrs.evolve(self._bunch(simulator), **{self.field_name: value})
        )


class DensityFieldAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable scalar mapped onto one field of one of the simulator's density profiles,
    addressed by index into `simulator.densities`."""

    density_index: int
    field_name: str

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return getattr(simulator.densities[self.density_index], self.field_name)

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
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
            host.nominal_density / self._num_levels(host),
            dopant.nominal_density / self._num_levels(dopant),
        )

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        host, dopant = self._atom_densities(simulator)
        return dopant / (host + dopant)

    @staticmethod
    def _num_levels(profile: typing.Any) -> int:
        """Ionization-level count of a density profile's species (the divisor `add_to_simulation`
        applies to the profile's `nominal_density`)."""
        if profile.species is None:
            raise ValueError("A dopant fraction needs profiles with a species; got species=None.")
        return _DensityProfile._get_num_ionization_levels(profile.species)

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError(f"Dopant fraction must be strictly between 0 and 1, got {value}.")
        host_atoms, dopant_atoms = self._atom_densities(simulator)
        total = host_atoms + dopant_atoms
        densities = list(simulator.densities)
        host = densities[self.host_index]
        dopant = densities[self.dopant_index]
        densities[self.host_index] = attrs.evolve(
            host, nominal_density=self._num_levels(host) * total * (1.0 - value)
        )
        densities[self.dopant_index] = attrs.evolve(
            dopant, nominal_density=self._num_levels(dopant) * total * value
        )
        simulator.densities = densities


class FinalParticlesAction(ReadOnlyActionMixin[BaseSimulator], ParticleGroupVariable):
    """Read-only ParticleGroup sourced from `simulator.final_particles` (populated after a
    run).

    The value is `None` while there are no particles -- before a run, or in a model loaded from
    an archive written without them. `validate_value` accepts that, as LUME's own check would
    reject it and with it every `model.get()` that includes this variable (including the one
    `lume-pva` makes for all variables on each cycle).
    """

    def validate_value(self, value: typing.Any) -> None:
        if value is not None:
            super().validate_value(value)

    def _get(self, simulator: BaseSimulator) -> typing.Any:
        return simulator.final_particles


class HyperparameterFieldAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable scalar mapped onto one field of the simulator's `hyparams` config object."""

    field_name: str

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return getattr(simulator.hyparams, self.field_name)

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        simulator.hyparams = attrs.evolve(simulator.hyparams, **{self.field_name: value})


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

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return getattr(simulator.laser, self.field_name)

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        laser = simulator.laser
        changes: dict[str, typing.Any] = {self.field_name: value}
        if self.field_name in ("energy", "a0"):
            changes["a0" if self.field_name == "energy" else "energy"] = None
        else:
            derived = "a0" if laser._amplitude_source == "energy" else "energy"
            changes[derived] = None
        simulator.laser = attrs.evolve(laser, **changes)


class MomentDescriptorAction(ReadOnlyActionMixin[FBPICSimulator], ScalarVariable):
    """Read-only scalar: ONE feature of the moment descriptor of `simulator.final_particles`.

    The descriptor is `inversion_fbpic.utils.distributions.compute_moment_descriptor` -- the
    33 features (3 momentum centroids, 21 covariance entries, 2 x `longitudinal_bins`
    longitudinal `uz` features, charge) that `build_dataset.py` writes for each simulation.
    The particles are selected the way `build_dataset.py` selects them first: `uz >= uz_min`,
    then the central `central_fraction` on each of the six phase-space axes (either `None`
    skips that step). `make_descriptor_actions()` builds the whole set. The value is NaN
    before a run, or when fewer than two particles survive the selection. A simulator that has
    no particles but holds recorded output values in `stats` (one loaded from an archive of a
    reconstructed run) gives the recorded value for this action's name.

    The descriptor is computed once per set of selection parameters and cached on the
    simulator against the current `final_particles`, so reading all 33 features costs one
    computation.
    """

    central_fraction: float | None = 0.95
    feature: str
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS
    uz_min: float | None = 30.0

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        if simulator.final_particles is None:
            return simulator.stats.get(self.name, float("nan"))
        descriptor = _descriptor(
            simulator, self.uz_min, self.central_fraction, self.longitudinal_bins
        )
        return descriptor.get(self.feature, float("nan"))


class PlasmaFieldAction(WritableActionMixin[PWFASimulator], ScalarVariable):
    """Writable scalar mapped onto one field of a PWFA simulator's plasma density profile."""

    field_name: str

    def _get(self, simulator: PWFASimulator) -> typing.Any:
        return getattr(simulator.plasma, self.field_name)

    def _set(self, simulator: PWFASimulator, value: typing.Any) -> None:
        simulator.plasma = attrs.evolve(simulator.plasma, **{self.field_name: value})


class StatAction(ReadOnlyActionMixin[BaseSimulator], ScalarVariable):
    """Read-only scalar sourced from `simulator.stats` (populated after a run).

    A simulator loaded from an archive without particles holds the archive's recorded output
    values in `stats` too, by action name; if there is no `stat_name` entry, the value under this
    action's own name is used.
    """

    stat_name: str

    def _get(self, simulator: BaseSimulator) -> typing.Any:
        if self.stat_name in simulator.stats:
            return simulator.stats[self.stat_name]
        return simulator.stats.get(self.name, float("nan"))


class ZernikeCoefficientAction(LaserFieldAction):
    """Writable scalar mapped onto ONE entry of the laser's `zernike_coefficients` dict (for
    example `coefficient="coma_x"`), leaving the other coefficients as they are. Only
    `LasyLaserPulse` has this field; the energy/`a0` handling of `LaserFieldAction._set` carries
    over unchanged. A coefficient name the laser does not accept is rejected when the new
    config is built."""

    coefficient: str
    field_name: str = "zernike_coefficients"

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return super()._get(simulator).get(self.coefficient, 0.0)

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        coefficients = {**super()._get(simulator), self.coefficient: value}
        super()._set(simulator, coefficients)


def make_descriptor_actions(
    *,
    central_fraction: float | None = 0.95,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
    prefix: str = "descriptor_",
    uz_min: float | None = 30.0,
) -> list[Action]:
    """Build one read-only `MomentDescriptorAction` per descriptor feature (33 by default).

    The names are `prefix` + the feature name (`descriptor_mean_uz`,
    `descriptor_cov_x_ux`, `descriptor_total_beam_charge_pc`, ...), in the order
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
            unit="pC" if feature == "total_beam_charge_pc" else None,
        )
        for feature in _descriptor_features(longitudinal_bins)
    ]


def make_pwfa_actions(simulator: PWFASimulator) -> list[Action]:
    """Build the default control/output variable set for a PWFA simulator.

    Control variables (writable): the plasma density, and for the driver and, if there is
    one, the witness (named `driver_*` / `witness_*`): the Lorentz factor, and for a
    `FlatTopBunch` the density and radius, for a `GaussianBunch` the charge, the RMS radius and
    length and the position `zf`.
    Outputs (read-only): charge and mean/std energy of the `target_species`, and
    `final_particles`.
    """
    actions: list[Action] = [
        PlasmaFieldAction(name="plasma_density", field_name="nominal_density", unit="m^-3")
    ]
    actions += _bunch_actions("driver", simulator.driver)
    if simulator.witness is not None:
        actions += _bunch_actions("witness", simulator.witness)
    return actions + [
        StatAction(name="charge_pc", stat_name="charge_pc", unit="pC", read_only=True),
        StatAction(
            name="energy_mean_mev", stat_name="energy_mean_mev", unit="MeV", read_only=True
        ),
        StatAction(
            name="energy_std_mev", stat_name="energy_std_mev", unit="MeV", read_only=True
        ),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]


def _bunch_actions(bunch: str, config: typing.Any) -> list[Action]:
    """The writable actions of one PWFA bunch (`driver` or `witness`), by the kind of bunch."""
    fields = [("gamma", "gamma", None)]
    if isinstance(config, FlatTopBunch):
        fields += [("density", "density", "m^-3"), ("radius", "radius", "m")]
    elif isinstance(config, GaussianBunch):
        fields += [
            ("charge", "charge", "C"),
            ("sigma_r", "sig_r", "m"),
            ("sigma_z", "sig_z", "m"),
            ("position", "zf", "m"),
        ]
    return [
        BunchFieldAction(name=f"{bunch}_{name}", bunch=bunch, field_name=field, unit=unit)
        for name, field, unit in fields
    ]


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

    phase_space = numpy.stack(
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
    weights = numpy.asarray(particles.weight, dtype=numpy.float64) / e
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
    features.append("total_beam_charge_pc")
    return features
