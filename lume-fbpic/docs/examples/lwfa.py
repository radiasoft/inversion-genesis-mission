"""lume_fbpic conversion of fbpic's `lwfa_script.py`
(fbpic/docs/source/example_input/lwfa_script.py): a non-boosted, single-species LWFA with one
Gaussian laser and one density profile. The action list is built here from the building-block
action classes in `lume_fbpic.actions`.

Substitutions made in the conversion:

- Density: the original's `dens_func` (a linear ramp, then an unbounded plateau) is
  `lume_fbpic.density_profiles.LinearRampFlattop`. `right_buffer` is set so the interaction
  length equals the original's `L_interact` plus the box width, and `beta_window=1.0` matches
  its `v_window=c`, which gives the same number of steps (1800).
- Species: bare pre-ionized electrons and no ions, as `species=None` on the profile.
- Laser duration: `TAU_FWHM` is the exact inverse of the conversion from the original's `tau`
  to `tau_fwhm`.
- `a0 = 4.0` maps directly onto `GaussianLaserPulse(a0=...)`.
- `p_rmax` is not reproduced: `_DensityProfile` does not pass it on, so macroparticles load out
  to `rmax`.
"""

from __future__ import annotations

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
# tau_fwhm reproduces the original's literal tau=16e-15 -- see the module docstring.
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
        # = 90e-6 -- see the module docstring.
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
        tau_fwhm=TAU_FWHM,  # exact inverse of the original's raw tau=16e-15
        cep=0.0,
        waist=5.0e-6,
        focal_position=15.0e-6,  # no zf in the original -> defaults to z0
        polarization=0.0,
    )

    # Exact reproduction of the original's dens_func -- see the module docstring.
    density = LinearRampFlattop(
        nominal_density=4.0e18 * 1.0e6,  # n_e, converted cm^-3 -> m^-3
        species=None,  # bare electrons, no ion species
        ionization=-1,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        ramp_start=30.0e-6,
        ramp_length=40.0e-6,
        elec_name="electrons",
        elec_select={
            "uz": [1.0, None]
        },  # matches the original's ParticleDiagnostic select
    )

    simulator = FBPICSimulator(
        hyparams,
        laser,
        [density],
        target_species="electrons",
        working_directory=working_directory,
    )

    # The model's inputs and outputs, for one laser and one density profile.
    actions = [
        LaserFieldAction(name="laser_a0", field_name="a0"),
        LaserFieldAction(name="laser_focal_position", field_name="focal_position"),
        DensityFieldAction(
            name="plasma_density",
            density_index=0,
            field_name="nominal_density",
        ),
        StatAction(name="charge_pc", stat_name="charge_pc", read_only=True),
        StatAction(name="energy_mean_mev", stat_name="energy_mean_mev", read_only=True),
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
