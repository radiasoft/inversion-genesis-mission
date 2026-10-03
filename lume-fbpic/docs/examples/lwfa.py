"""lume_fbpic conversion of fbpic's `lwfa_script.py` upstream example
(fbpic/docs/source/example_input/lwfa_script.py): a non-boosted, single-species LWFA case.

Picked over the other three upstream examples because it's the cleanest fit for what
inversion_fbpic supports today without modification:
- ionization_script.py's pattern is already demonstrated by demo_ionization_simulation.
- boosted_frame_script.py needs an externally-injected relativistic bunch
  (`fbpic.lpa_utils.bunch.add_particle_bunch`), which inversion_fbpic has no equivalent of --
  every electron population goes through `_DensityProfile`'s ionization-of-a-neutral-gas
  path, not direct bunch loading.
- parametric_script.py's per-rank-independent-scan technique needs `use_all_mpi_ranks=False`,
  which `Simulation.setup_simulation()` hard-wires equal to `use_mpi` -- not reachable
  through this wrapper (see prior session findings).

Also useful because its shape (one laser, ONE density profile) differs from the two-profile
HTU downramp case `lume_fbpic.actions.make_actions()` is hard-coded for -- this script builds
its own small action list directly from the same building-block Action classes, showing
they're reusable beyond that one hard-coded case.

Notable substitutions/approximations made converting it (not a bit-exact reproduction):

1. Density profile shape and interaction length: RESOLVED. The original's `dens_func` is a
   bare linear ramp then an unbounded flat plateau (no downramp, never tapers off), run only
   until an independent literal `L_interact=50e-6` (decoupled from the density profile's own
   extent, which is much longer: `p_zmax=500e-6`). No `_DensityProfile` subclass wrapping
   this shape existed in `inversion_fbpic` -- `lume_fbpic.density_profiles.LinearRampFlattop`
   (defined in *this* package, not `inversion_fbpic` -- subclass registration doesn't care
   about package boundaries) reproduces the exact `dens_func` math. Its `get_z_extent()`
   deliberately reports only the ramp's own span, not the (unbounded) plateau -- `hyparams.
   right_buffer` (an existing, general `SimulationHyperparameters` hook) is set below to
   make `Simulation`'s auto-derived interaction length (`density z_extent span +
   right_buffer`) equal the original's literal `L_interact(50e-6) + box width(40e-6) = 90e-6`
   exactly, without needing any independent-length knob on `Simulation` itself. One more
   knob was needed for an exact step count: the original hardcodes `v_window=c`, but
   `SimulationHyperparameters.beta_window` defaults to `None`, which auto-calculates a
   physically slightly-more-accurate (but not bit-identical) group velocity via
   `calculate_mean_group_beta()` -- giving 1801 steps instead of upstream's exact 1800.
   `beta_window=1.0` below forces the same simplification upstream makes, bit-matching
   `num_steps` exactly (verified: `dt` was already identical between the two; only
   `T_interact` via `v_window` differed).

2. Species: RESOLVED. The original loads bare pre-ionized electrons (`q=-e, m=m_e`, no ion
   species at all). That is `species=None` on `_DensityProfile`, which now means exactly
   that: a single electron species at the full `nominal_density` and no ion species. Only
   `ionization` of None or -1 is accepted with it (both mean fully ionized; -1 is used
   below as the clearer spelling). This is not supported in a boosted-frame simulation,
   where `add_to_simulation()` raises instead of assuming an ion background -- fine here,
   since this case is non-boosted. Verified against real fbpic: `add_to_simulation()`
   returns `ions is None` and adds exactly one species with `q=-e, m=m_e`.

3. Laser duration: RESOLVED. The original passes `tau=16e-15` directly as fbpic's internal
   (sigma-like) `tau` parameter. `GaussianLaserPulse` takes `tau_fwhm` and converts it via
   `calculate_laser_tau_from_fwhm_intensity(tau_fwhm) = tau_fwhm / SIGMA_TO_FWHM_INTENSITY /
   LASER_TAU_TO_SIGMA` -- a clean invertible formula, so `TAU_FWHM` below is computed as its
   exact inverse (verified: round-tripping it back through
   `calculate_laser_tau_from_fwhm_intensity` reproduces `16e-15` to full float precision),
   not an approximate estimate.

4. Laser amplitude: `a0=4.0` is the one case among the fbpic examples where `a0` genuinely
   is the script's native, literal control parameter (not derived from an energy) -- maps
   directly onto `GaussianLaserPulse(a0=...)`, no inversion needed.
"""

from __future__ import annotations

from typing import Any

import attrs

from lume_fbpic.actions import (
    DensityFieldAction,
    FinalParticlesAction,
    LaserFieldAction,
    StatAction,
    make_descriptor_actions,
)
from lume_fbpic.density_profiles import LinearRampFlattop
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import FBPICSimulator

from inversion_fbpic.lib.simulation import SimulationHyperparameters
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.utils.simulation_setup_tools import (
    LASER_TAU_TO_SIGMA,
    SIGMA_TO_FWHM_INTENSITY,
    calculate_laser_tau_from_fwhm_intensity,
)

# Exact inverse of calculate_laser_tau_from_fwhm_intensity(), so that GaussianLaserPulse's
# tau_fwhm reproduces the original's literal tau=16e-15 -- see module docstring, point 3.
_UPSTREAM_TAU = 16.0e-15
TAU_FWHM = _UPSTREAM_TAU * SIGMA_TO_FWHM_INTENSITY * LASER_TAU_TO_SIGMA
assert calculate_laser_tau_from_fwhm_intensity(TAU_FWHM) == _UPSTREAM_TAU


def build_model(working_directory: str | None = None) -> LUMEFBPICModel:
    hyparams = SimulationHyperparameters(
        zmin=-10.0e-6,
        zmax=30.0e-6,
        rmax=20.0e-6,
        nz=800,
        nr=50,
        nm=2,
        use_mpi=False,
        number_dumps=10,
        gamma_boost=None,  # non-boosted, matches lwfa_script.py
        field_diagnostics=["rho", "E", "B"],
        random_seed=None,  # lwfa_script.py doesn't set one either
        # Matches the original's L_interact(50e-6) + box width(40e-6) = 90e-6 total slide
        # distance exactly: density z_extent span (ramp_length=40e-6) + right_buffer(50e-6)
        # = 90e-6 -- see module docstring, point 1.
        right_buffer=50.0e-6,
        # The original hardcodes v_window=c (beta_window=1.0) exactly, ignoring the small
        # plasma-dispersion correction to the group velocity. Left at the default (None),
        # Simulation auto-calculates a physically slightly-more-accurate beta_window via
        # calculate_mean_group_beta(), which gives 1801 steps instead of upstream's exact
        # 1800 -- forcing 1.0 here bit-matches upstream's num_steps instead.
        beta_window=1.0,
    )

    laser = GaussianLaserPulse(
        a0=4.0,
        z0=15.0e-6,
        wavelength=0.8e-6,  # not set in the original -> fbpic's own default
        tau_fwhm=TAU_FWHM,  # exact inverse of the original's raw tau=16e-15, see point 3
        cep=0.0,
        waist=5.0e-6,
        focal_position=15.0e-6,  # no zf in the original -> defaults to z0
        polarization=0.0,
    )

    # Exact reproduction of the original's dens_func -- see module docstring, point 1.
    density = LinearRampFlattop(
        nominal_density=4.0e18 * 1.0e6,  # n_e, converted cm^-3 -> m^-3
        species=None,  # bare electrons, no ion species -- see point 2
        ionization=-1,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        ramp_start=30.0e-6,
        ramp_length=40.0e-6,
        elec_name="electrons",
        elec_select={"uz": [1.0, None]},  # matches the original's ParticleDiagnostic select
    )

    simulator = FBPICSimulator(
        hyparams,
        laser,
        [density],
        target_species="electrons",
        working_directory=working_directory,
    )

    # A one-density-profile action list, built directly from the same building-block Action
    # classes make_actions() uses for the two-profile HTU case -- not make_actions() itself,
    # since that's hard-coded for that different shape.
    actions = [
        LaserFieldAction(name="laser_a0", field_name="a0", unit=None),
        LaserFieldAction(name="laser_focal_position", field_name="focal_position", unit="m"),
        DensityFieldAction(
            name="plasma_density",
            density_index=0,
            field_name="nominal_density",
            unit="m^-3",
        ),
        StatAction(name="charge_pc", stat_name="charge_pc", unit="pC", read_only=True),
        StatAction(
            name="energy_mean_mev", stat_name="energy_mean_mev", unit="MeV", read_only=True
        ),
        FinalParticlesAction(name="final_particles", read_only=True),
        # Moment descriptor of the bunch. This bunch is ~1 MeV (mean uz ~ 2), so the
        # build_dataset.py default of uz >= 30 would select nothing; use uz >= 1 here.
        *make_descriptor_actions(uz_min=1.0),
    ]

    return LUMEFBPICModel(simulator, actions)


if __name__ == "__main__":
    model = build_model()
    model.simulator.configure()
    model.set({"laser_a0": 4.0})  # set() applies the value, then runs the simulation
    print(model.get(["charge_pc", "energy_mean_mev"]))
