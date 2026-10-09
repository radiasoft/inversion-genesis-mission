"""Adapter that lets the HTU transport twin (`geecs-lume-twin`) be the stage after `lume-fbpic`.

`lume.staged_model.StagedModel` passes each stage's `final_particles` (a `ParticleGroup`) to the
next stage's `initial_particles` setter. The twin's model is a plain `LUMECheetahModel`, which has
no such setter, so `TwinStage` wraps it::

    from lume.staged_model import StagedModel
    from htu.model import build_htu_model

    chain = StagedModel([lpa_model, TwinStage(build_htu_model())])
    chain.set({"EMQ1H_Current": 0.68})   # injects the LPA bunch, then sets the magnet

The twin's repository is not modified. The adapter drives the twin through the same internals its
own source puts use (`htu.model._apply_machine_state` and the beam attributes of its
`CheetahSimulator`), so it follows that code; `htu` is imported only when a `TwinStage` is built.

What the injection does, in order: if the incoming `ParticleGroup` is a lab-frame snapshot (every
particle at the same `t`) it is selected and moved to the bunch frame (`bunch_frame_particles`,
with this stage's `uz_min` and `central_fraction`); a group that already has arrival times is used
as it is. If `plasma_exit_z` is given, the bunch is then drifted ballistically from where it is
(its charge-weighted mean `z`) to that plane (`drift_particles`) -- the twin's source plane is its
`PlasmaExit` screen, and the 5.2 cm from there to the first magnet is the twin's own `SrcToPMQ1`
drift, so only the offset between the snapshot and the plane is added here. It becomes a Cheetah
`ParticleBeam` (`ParticleBeam.from_openpmd_particlegroup`, reference
energy the charge-weighted mean total energy), replaces the twin's source beam, and the twin
re-derives its magnet settings at the new beam energy -- its rule that magnet PVs hold their
currents through energy changes. The re-track is deferred to the next `get()` or `set()` of the
stage, so that a `StagedModel.set()`, which injects and then sets the twin's own values, tracks the
(slow) 45,000-particle bunch once, not twice.

The twin's `Source_*` PVs are then set to the bunch's moments (`beam_moments`), so they read what
was injected, and by default (`lock_source=True`) they are read-only: the LPA is the twin's source,
and the only way to change it is through the LPA model. With `lock_source=False` they stay
writable, and putting one regenerates a parametric Gaussian beam from them, replacing the injected
bunch. Until a bunch has been injected the twin still tracks the default source beam it was built
with (`external_beam` says which). `TwinStage.reset()` restores that beam.
"""

from __future__ import annotations

import dataclasses
import warnings
import typing

import numpy
import torch
from scipy.constants import c
from lume_fbpic.simulator import ELECTRON_MC2_EV

from lume.model import LUMEModel
from lume.staged_model import InitialParticlesMixIn


from beamphysics import ParticleGroup

# Names of the twin's source-parameter variables.
_SOURCE_PREFIX = "Source_"

# SourceParams fields that `beam_moments()` fills.
_SOURCE_FIELDS = (
    "energy_mev",
    "energy_spread_pct",
    "charge_pc",
    "beta_x_mm",
    "beta_y_mm",
    "alpha_x",
    "alpha_y",
    "norm_emit_x_um",
    "norm_emit_y_um",
    "x_um",
    "y_um",
    "xp_mrad",
    "yp_mrad",
)


class TwinStage(InitialParticlesMixIn, LUMEModel):
    """The twin's `LUMECheetahModel` as a stage that accepts `initial_particles`.

    Everything else -- `supported_variables`, `get`, `set` -- is the wrapped model's.

    Args:
        model: The twin's model; `htu.model.build_htu_model()` if omitted.
        central_fraction: Central fraction kept per axis when the incoming particles are a lab
            snapshot (see `bunch_frame_particles`); `None` keeps all.
        uz_min: Smallest `uz` kept from a lab snapshot; `None` keeps all.
        plasma_exit_z: Lab-frame `z` [m] of the twin's source plane (the plasma exit). The bunch
            is drifted from its own mean `z` to this plane before injection; `None` (the
            default) injects it where it is. `plasma_exit_z()` gives the end
            of a simulation's density profiles. A positive drift means the snapshot was taken
            before the bunch reached the plane, still in the gas; the vacuum drift then ignores
            the rest of the plasma, and a warning says so.
        lock_source: Serve the twin's `Source_*` variables read-only, so that the LPA bunch is
            its only source. They still read the injected bunch's moments. `False` leaves them
            writable.
        screen_binning: Make every camera image `screen_binning` times coarser in each direction
            (1024 x 1024 becomes 256 x 256 for 4), over the same field of view. Only applies
            when this stage builds the twin model itself (`model=None`). The twin's default
            resolution puts about 0.02 macroparticles in a pixel for a 20,000-particle bunch, so
            the images are mostly shot noise; coarser pixels give a histogram that can be read.
            Must divide the image sizes.
    """

    def __init__(
        self,
        model=None,
        *,
        central_fraction: float | None = 0.95,
        uz_min: float | None = 30.0,
        plasma_exit_z: float | None = None,
        lock_source: bool = True,
        screen_binning: int = 1,
    ) -> None:
        super().__init__()
        if model is None:
            geometries = (
                binned_screen_geometries(screen_binning)
                if screen_binning != 1
                else None
            )
            model = _import_htu().build_htu_model(screen_geometries=geometries)
        elif screen_binning != 1:
            raise ValueError(
                "screen_binning only applies when the stage builds the twin model"
            )
        self.screen_binning = screen_binning
        self._model = model
        self.central_fraction = central_fraction
        self.uz_min = uz_min
        self.plasma_exit_z = plasma_exit_z
        self.lock_source = lock_source
        self._variables = {
            name: (
                variable.model_copy(update={"read_only": True})
                if lock_source and name.startswith(_SOURCE_PREFIX)
                else variable
            )
            for name, variable in model.supported_variables.items()
        }
        self._drift_m: float | None = None
        self._tracking_pending = (
            False  # a bunch was injected and the twin not yet re-tracked
        )
        self._particles: ParticleGroup | None = None
        self._injected_beam = (
            None  # the beam object put into the twin by the last injection
        )
        simulator = model.simulator
        self._original_beam = simulator.initial_beam_distribution.clone()
        self._original_source_params = dataclasses.replace(simulator.source_params)

    @property
    def drift_length(self) -> float | None:
        """Metres the last injected bunch was drifted to reach the source plane (negative: back
        toward the plasma); None if it was not drifted."""
        return self._drift_m

    @property
    def external_beam(self) -> bool:
        """True while the injected bunch (not a parametric source beam) is the twin's source beam.

        Putting a `Source_*` PV regenerates a parametric beam and makes this False."""
        return (
            self._particles is not None
            and self._model.simulator.initial_beam_distribution is self._injected_beam
        )

    @property
    def initial_particles(self) -> ParticleGroup | None:
        """The bunch last injected, as it went in (bunch frame, selected and drifted to the source
        plane if `plasma_exit_z` is set); None if none."""
        return self._particles

    @initial_particles.setter
    def initial_particles(self, particles: ParticleGroup) -> None:
        htu_model = _import_htu()
        from cheetah import ParticleBeam

        if (
            float(_spread(particles.t)) == 0.0
        ):  # a lab snapshot: all particles at one instant
            particles = bunch_frame_particles(
                particles, central_fraction=self.central_fraction, uz_min=self.uz_min
            )
        drift_m = None
        if self.plasma_exit_z is not None:
            z_mean = float(
                numpy.average(particles.z, weights=numpy.abs(particles.weight))
            )
            drift_m = float(self.plasma_exit_z) - z_mean
            if drift_m > 0:
                warnings.warn(
                    f"the bunch (mean z = {z_mean:.6g} m) is {drift_m * 1e3:.3g} mm short of the "
                    "plasma exit: drifting it there in vacuum ignores the remaining plasma",
                    stacklevel=2,
                )
            particles = drift_particles(particles, drift_m)
        simulator = self._model.simulator
        dtype = simulator.beam_distribution.particles.dtype
        weights = particles.weight
        energy = torch.tensor(
            float((particles.energy * abs(weights)).sum() / abs(weights).sum()),
            dtype=dtype,
        )
        # Everything that can raise runs before any state is touched.
        beam = ParticleBeam.from_openpmd_particlegroup(
            particles, energy=energy, dtype=dtype
        )
        moments = beam_moments(particles)
        source_params = dataclasses.replace(
            simulator.source_params,
            num_particles=int(moments["num_particles"]),
            **{name: moments[name] for name in _SOURCE_FIELDS},
        )

        saved = (
            simulator.source_params,
            simulator.initial_beam_distribution,
            simulator.initial_beam_distribution_charge,
            simulator.beam_distribution,
            simulator.energies,
        )
        try:
            simulator.source_params = source_params
            simulator.initial_beam_distribution = beam.clone()
            simulator.initial_beam_distribution_charge = beam.particle_charges.clone()
            simulator.beam_distribution = beam.clone()
            simulator.energies = simulator.get_energy()
            htu_model._apply_machine_state(simulator)
        except Exception:
            (
                simulator.source_params,
                simulator.initial_beam_distribution,
                simulator.initial_beam_distribution_charge,
                simulator.beam_distribution,
                simulator.energies,
            ) = saved
            raise
        self._particles = particles
        self._drift_m = drift_m
        self._injected_beam = simulator.initial_beam_distribution
        self._tracking_pending = True

    @property
    def supported_variables(self) -> dict[str, typing.Any]:
        return self._variables

    def reset(self) -> None:
        """Restore the source beam the twin had when it was wrapped, then reset the twin."""
        simulator = self._model.simulator
        simulator.source_params = dataclasses.replace(self._original_source_params)
        simulator.initial_beam_distribution = self._original_beam.clone()
        simulator.initial_beam_distribution_charge = (
            self._original_beam.particle_charges.clone()
        )
        self._model.reset()  # clones the (restored) initial beam, re-tracks, refreshes the state
        self._tracking_pending = False
        self._particles = None
        self._drift_m = None
        self._injected_beam = None

    def _get(self, names: list[str]) -> dict[str, typing.Any]:
        self._track_if_pending()
        return self._model.get(names)

    def _set(self, values: dict[str, typing.Any]) -> None:
        self._model.set(values)  # the twin re-tracks after applying values
        if values:
            self._tracking_pending = False
        else:
            self._track_if_pending()

    def _track_if_pending(self) -> None:
        """Re-track the twin (and refresh its state) if a bunch was injected since the last track."""
        if self._tracking_pending:
            self._model.simulator.track()
            self._model.update_state()
            self._tracking_pending = False


def beam_moments(particles: ParticleGroup) -> dict[str, float]:
    """Charge-weighted energy, charge and per-plane Twiss of a bunch.

    Keys: `energy_mev` (mean total energy), `energy_spread_pct` (rms, percent of the mean),
    `charge_pc`, `num_particles`, and for each plane `p` in `x`, `y`: `beta_<p>_mm`, `alpha_<p>`,
    `norm_emit_<p>_um` (normalized emittance in mm mrad), `<p>_um` (centroid) and `<p>p_mrad`
    (mean angle). Angles are `px / p0c`, with `p0c` the mean momentum, as in Cheetah, and
    `alpha` follows the MAD convention (positive when converging). A zero-emittance plane gets
    `beta = alpha = 0`.

    Raises:
        ValueError: If the bunch has fewer than two particles or no charge.
    """
    weights = numpy.abs(numpy.asarray(particles.weight, dtype=numpy.float64))
    if len(weights) < 2 or weights.sum() <= 0:
        raise ValueError("need at least two particles with charge")
    weights = weights / weights.sum()

    def mean(values):
        return float(numpy.sum(weights * values))

    energy = numpy.asarray(particles.energy, dtype=numpy.float64)
    mean_energy = mean(energy)
    p0c = float(numpy.sqrt(mean_energy**2 - ELECTRON_MC2_EV**2))
    moments = {
        "energy_mev": mean_energy / 1.0e6,
        "energy_spread_pct": 100.0
        * float(numpy.sqrt(mean((energy - mean_energy) ** 2)))
        / mean_energy,
        "charge_pc": float(numpy.sum(numpy.abs(particles.weight))) * 1.0e12,
        "num_particles": float(len(weights)),
    }
    for plane in ("x", "y"):
        position = numpy.asarray(getattr(particles, plane), dtype=numpy.float64)
        angle = (
            numpy.asarray(getattr(particles, f"p{plane}"), dtype=numpy.float64) / p0c
        )
        centroid, mean_angle = mean(position), mean(angle)
        dx, da = position - centroid, angle - mean_angle
        sxx, sxa, saa = mean(dx * dx), mean(dx * da), mean(da * da)
        emittance = float(numpy.sqrt(max(sxx * saa - sxa * sxa, 0.0)))
        beta, alpha = (
            (sxx / emittance, -sxa / emittance) if emittance > 0 else (0.0, 0.0)
        )
        moments[f"beta_{plane}_mm"] = beta * 1.0e3
        moments[f"alpha_{plane}"] = alpha
        moments[f"norm_emit_{plane}_um"] = emittance * (p0c / ELECTRON_MC2_EV) * 1.0e6
        moments[f"{plane}_um"] = centroid * 1.0e6
        moments[f"{plane}p_mrad"] = mean_angle * 1.0e3
    return moments


def binned_screen_geometries(factor: int) -> dict[str, typing.Any]:
    """The twin's screen geometries (`htu.screens.ScreenGeometry`) with every image `factor`
    times coarser in each direction and the same field of view: the size is divided by `factor`
    and the metres per pixel multiplied by it. Includes the twin's own per-screen overrides (the
    chicane slit camera's 20 um pixels) and scales a crosshair position if a geometry has one.

    Raises:
        ValueError: If `factor` is not a positive integer dividing every image size.
    """
    if int(factor) != factor or factor < 1:
        raise ValueError(f"the binning factor must be a positive integer, got {factor}")
    factor = int(factor)
    from htu.lattice import DEFAULT_SCREEN_GEOMETRY_OVERRIDES, SCREEN_NAMES
    from htu.screens import DEFAULT_GEOMETRY, ScreenGeometry

    geometries = {}
    for name in SCREEN_NAMES:
        base = DEFAULT_SCREEN_GEOMETRY_OVERRIDES.get(name, DEFAULT_GEOMETRY)
        if base.size_x % factor or base.size_y % factor:
            raise ValueError(
                f"binning {factor} does not divide the {base.size_x} x {base.size_y} image of {name}"
            )
        geometries[name] = ScreenGeometry(
            size_x=base.size_x // factor,
            size_y=base.size_y // factor,
            calibration_m=base.calibration_m * factor,
            crosshair_x=None if base.crosshair_x is None else base.crosshair_x / factor,
            crosshair_y=None if base.crosshair_y is None else base.crosshair_y / factor,
        )
    return geometries


def build_chain(lpa_model, twin_model=None, **stage_kwargs):
    """`StagedModel([lpa_model, TwinStage(twin_model, **stage_kwargs)])`: the LPA bunch becomes the
    twin's source. `twin_model` defaults to the twin's own `build_htu_model()`; `stage_kwargs` go
    to `TwinStage`. The stage is `chain.lume_model_instances[1]`.

    `StagedModel` needs variable names to be unique across the two models; the `lume-fbpic`
    names do not collide with the twin's.
    """
    from lume.staged_model import StagedModel

    return StagedModel([lpa_model, TwinStage(twin_model, **stage_kwargs)])


def bunch_frame_particles(
    particles: ParticleGroup,
    *,
    central_fraction: float | None = 0.95,
    uz_min: float | None = 30.0,
) -> ParticleGroup:
    """Select the bunch from a lab-frame snapshot and put it in the bunch frame.

    Selection follows the moment descriptor (`inversion_fbpic.utils.distributions`): keep
    `uz = pz / (m_e c) >= uz_min`, then the central `central_fraction` of the charge-weighted
    distance from the mean on each of the six phase-space axes (`None` skips a step).

    The returned `ParticleGroup` has `t = -(z - <z>) / c`, `<z>` the charge-weighted mean, so that a
    particle ahead of the bunch centre (larger `z`) has a negative arrival time, as in Cheetah's
    `tau = c t`, and `status = 1`. `x, y, z, px, py, pz` and the charge weights are unchanged.

    Raises:
        ValueError: If fewer than two particles remain after the selection.
    """
    x, y, z = (
        numpy.asarray(getattr(particles, name), dtype=numpy.float64) for name in "xyz"
    )
    ux, uy, uz = (
        numpy.asarray(getattr(particles, f"p{name}"), dtype=numpy.float64)
        / ELECTRON_MC2_EV
        for name in "xyz"
    )
    weight = numpy.asarray(particles.weight, dtype=numpy.float64)

    keep = numpy.ones(len(x), dtype=bool)
    if uz_min is not None:
        keep &= uz >= uz_min
    if central_fraction is not None:
        if not 0 < central_fraction <= 1:
            raise ValueError("central_fraction must be in (0, 1]")
        phase_space = numpy.stack([x, ux, y, uy, z, uz], axis=-1)[keep]
        if len(phase_space) >= 2:
            centre = numpy.average(phase_space, axis=0, weights=numpy.abs(weight[keep]))
            distance = numpy.abs(phase_space - centre)
            limit = numpy.quantile(distance, central_fraction, axis=0)
            inside = numpy.all(distance <= limit, axis=1)
            central = numpy.zeros(len(x), dtype=bool)
            central[numpy.flatnonzero(keep)[inside]] = True
            keep = central
    if keep.sum() < 2:
        raise ValueError("fewer than two particles remain after the selection")

    z_mean = numpy.average(z[keep], weights=numpy.abs(weight[keep]))
    return ParticleGroup(
        data={
            "x": x[keep],
            "y": y[keep],
            "z": z[keep],
            "px": numpy.asarray(particles.px, dtype=numpy.float64)[keep],
            "py": numpy.asarray(particles.py, dtype=numpy.float64)[keep],
            "pz": numpy.asarray(particles.pz, dtype=numpy.float64)[keep],
            "t": -(z[keep] - z_mean) / c,
            "status": numpy.ones(int(keep.sum()), dtype=int),
            "weight": weight[keep],
            "species": "electron",
        }
    )


def drift_particles(particles: ParticleGroup, length: float) -> ParticleGroup:
    """Move a bunch ballistically along `z` by `length` metres (negative moves it back).

    Each particle travels in a straight line at its own angle `px / pz`, so `x` and `y` change by
    `length * px / pz` and `z` by `length`. The arrival time `t` changes by the extra time of
    flight relative to the reference particle -- the charge-weighted mean energy `E0`, on axis --
    which is `(length / c) * (E / (pz c) - E0 / p0c)` per particle (`pz c` and `p0c` in eV):
    higher-energy particles gain on the reference when `length` is positive, and the
    charge-weighted mean `t` is not held at zero. This is the same ballistic drift as Cheetah's
    `Drift` with `tracking_method="drift_kick_drift"`; there are no fields, no space charge and no
    scattering. Momenta, charges and `status` are unchanged.
    """
    px, py, pz = (
        numpy.asarray(getattr(particles, f"p{n}"), dtype=numpy.float64) for n in "xyz"
    )
    energy = numpy.asarray(particles.energy, dtype=numpy.float64)
    weight = numpy.abs(numpy.asarray(particles.weight, dtype=numpy.float64))
    e0 = float(numpy.average(energy, weights=weight))
    p0c = float(numpy.sqrt(e0**2 - ELECTRON_MC2_EV**2))
    return ParticleGroup(
        data={
            "x": numpy.asarray(particles.x, dtype=numpy.float64) + length * px / pz,
            "y": numpy.asarray(particles.y, dtype=numpy.float64) + length * py / pz,
            "z": numpy.asarray(particles.z, dtype=numpy.float64) + length,
            "px": px,
            "py": py,
            "pz": pz,
            "t": numpy.asarray(particles.t, dtype=numpy.float64)
            + (length / c) * (energy / pz - e0 / p0c),
            "status": numpy.asarray(particles.status),
            "weight": numpy.asarray(particles.weight, dtype=numpy.float64),
            "species": particles.species,
        }
    )


def plasma_exit_z(densities) -> float:
    """The nominal plasma-exit plane [m, lab `z`]: the downstream end of the longest density
    extent among `densities` (each profile's `get_z_extent()`).

    That end is a convention of the profile (for `GeneralizedGaussianProfile`, `z0 + 2.8 alpha`),
    not a sharp edge -- the gas density there is not zero -- so pass an explicit plane to
    `TwinStage` when the machine's actual exit is known.
    """
    return float(max(density.get_z_extent()[1] for density in densities))


def _import_htu():
    try:
        import htu.model as htu_model
    except ImportError as error:
        raise ImportError(
            "TwinStage needs the HTU twin package (geecs-lume-twin, import name `htu`), which is "
            "not importable."
        ) from error
    return htu_model


def _spread(values) -> float:
    values = numpy.asarray(values, dtype=numpy.float64)
    return float(values.max() - values.min()) if len(values) else 0.0
