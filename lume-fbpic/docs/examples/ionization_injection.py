"""lume_fbpic conversion of `runs/initial_sample/ionization_injection_runscript_00.py`
(`sim_0000` from `runs/initial_sample/sample_dataset/sample_dataset.json`): a boosted-frame
helium/nitrogen ionization-injection LPA simulation driven by a real Zernike-aberrated
super-Gaussian LASY laser pulse (here, all Zernike coefficients are 0 -- an idealized,
unaberrated super-Gaussian, matching `sim_0000`'s own baseline configuration exactly).

Requires `lume_fbpic.density_profiles.GeneralizedGaussianProfile` (added alongside this
script) and the real `LasyLaserPulse` implementation on this `lasy-laser-work` branch --
NOT present on `main` yet, where `LasyLaserPulse` is still an unimplemented stub.

Notable substitutions/approximations made converting it (not a bit-exact reproduction):

1. Density profile shape: RESOLVED. `build_generalized_gaussian_profile` (a generalized
   normal/"generalized Gaussian" bump in z, `n = gauss_peak * exp(-(|z-z0|/alpha)**beta)`)
   has no existing `inversion_fbpic.lib.density_profiles` class wrapping it -- only
   sine-squared-ramp shapes (`SmoothSineFlattop`/`AsymmetricSine`/etc.) exist there.
   `lume_fbpic.density_profiles.GeneralizedGaussianProfile` composes it the same way
   `LinearRampFlattop` composes the linear-ramp `dens_func`.

2. Two-gas mixture (He background + N dopant), same density shape: the original calls
   `sim.add_new_species()` four times directly (He atoms, N atoms, and two *empty* electron
   species that only gain weight via `make_ionizable()`) -- not through `_DensityProfile` at
   all. This maps cleanly onto TWO `GeneralizedGaussianProfile` instances sharing the same
   `z0`/`alpha`/`beta`/`gauss_peak` (the same physical density shape), differing only in
   `nominal_density`/`species`/`elec_name` -- exactly the same "one profile per species, list
   of profiles" pattern `demo_downramp_simulation`'s flattop+downramp case already uses.
   `elec_name=None` on the He profile (never diagnosed upstream either); `target_species=
   "electrons_n"` on `FBPICSimulator`, matching upstream's sole diagnosed species,
   `nitrogen_electrons`.

3. Ion (neutral-atom) density scaling: RESOLVED, but requires a real correction, not just a
   value copy. `_DensityProfile.add_to_simulation()` sets the ion species' density to
   `nominal_density / num_ionization_levels` (He: 2 levels; N: 7 levels, both confirmed via
   `_DensityProfile._get_num_ionization_levels`) -- the original script's raw
   `sim.add_new_species(n=n_gas*(1 - frac), ...)` / `n=n_gas*frac` calls do NOT divide by the
   level count at all. So `nominal_density` here is set to the *desired atom density times
   its species' level count* (He: `2 * n_gas*(1-frac)`; N: `7 * n_gas*frac`) to compensate
   and land on the exact same physical atom density upstream uses. `ionization=0` for both
   (matching `level_start=0`) means the electron-species density formula
   (`nominal_density * ionization/num_levels`) is 0 regardless -- an empty species that only
   gains weight dynamically via ionization, exactly matching upstream's own bare
   `sim.add_new_species(q=-e, m=m_e)` (no density/dens_func at all) for both electron species.

4. `p_zmin`/`p_rmax` particle-loading restriction: NOT reproduced, a real, acknowledged gap.
   The original restricts particle loading to `p_zmin=0.0, p_rmax=90e-6` (out of a much
   larger `rmax=200e-6` box) via raw `add_new_species(p_zmin=..., p_rmax=...)` kwargs.
   `_DensityProfile.add_to_simulation()` does not pass `p_zmin`/`p_rmax` through to
   `add_new_species()` at all -- only `p_nz`/`p_nr`/`p_nt`. So this conversion loads
   macroparticles across the *entire* grid instead of the intended restricted region --
   not wrong physics (the generalized-Gaussian `dens_func` is already ~0 far from `z0`, and
   there's no r-dependence to restrict), but more macroparticles than necessary, some
   entirely wasted (loaded where density is negligible or beyond `p_rmax` where the beam
   never reaches). A real, scoped gap in `inversion_fbpic.lib.density_core`, not something
   routable around from `lume_fbpic`.

5. `beta_window` (moving-window velocity): RESOLVED. The original computes `v_window` via
   its own simplified formula (`c * (1 - 0.5*n_gas/n_crit_800nm)`, using `n_crit=1.75e27`)
   rather than `Simulation`'s own more general `calculate_mean_group_beta()` -- computed
   below to match it exactly.

6. `right_buffer`/interaction length: NOT bit-matched, a real, acknowledged approximation
   (unlike the `lwfa_script.py` conversion's exact step-count match). The original's own
   `interaction_length = 2.8*alpha + z0` is a ONE-SIDED offset from `z=0` (the box's own
   right edge here), while `GeneralizedGaussianProfile.get_z_extent()` returns a SYMMETRIC
   span (`z0 +/- 2.8*alpha`) -- so matching the original's total interaction length exactly
   would require a *negative* `right_buffer` (the two-sided span is naturally longer than the
   one-sided formula). `SimulationHyperparameters.right_buffer` has a `> 0.0` validator (no
   such restriction applied to `lwfa_script.py`'s conversion, where the needed value happened
   to be positive), so an exact match isn't reachable here at all -- `right_buffer` is left at
   its default (`None`, i.e. the box width), giving a somewhat *longer* total interaction
   length than `sim_0000`'s own tuned value, not a contrived near-zero workaround.

7. Laser: uses the real `LasyLaserPulse` directly, no approximation needed -- confirmed a
   drop-in for `GaussianLaserPulse` in `FBPICSimulator`/`Simulation.setup_simulation()`
   (which now calls `laser.prepare(comm, relative_to=working_directory)` automatically for
   every laser before adding pulses). `z0` (informational only) is set the same way
   `demos/demo_lasy_laser_simulation` sets it: `z0_antenna - c*(t_start + 3*tau_fwhm)`.

Reproducing the nz=1000 CPU run: the constants below are the coarsened configuration that
was actually run on one node (14 threads, no GPU) from
`~/save/inversion/simple_sample00/nz1000/`, not `sim_0000`'s own production grid. Relative to
the production script it uses `nz=1000, nr=100, nm=2`, a (300, 450) / 2-mode LASY grid,
`p_nt=4`, 10 lab-frame dumps (`write_period=10`), and NO `uz>=10` selection on
`electrons_n` (at `nz=400` that filter returned 0 particles; kept off here so the output
matches). Output particles are therefore every nitrogen electron, as in that run -- apply
`uz_min` downstream (`select_by_uz`/`build_dataset.py --uz-min`). Known remaining differences
from that run, none of which could be removed from `lume_fbpic`: (a) the species is named
`electrons_n` here but `nitrogen_electrons` in the original's openPMD output; (b) points 4
and 6 above (`p_zmin`/`p_rmax` and the interaction length) still apply, so the run is about 15%
longer (~13,600 steps against the original's 11,813: interaction length 6.23 mm against
5.38 mm through the same `boost.interaction_time` and grid -- estimated from the length
ratio, not run) and loads more macroparticles; (c) `Simulation` passes
`n_order=-1` (infinite-order stencil) on a single rank, where the original passed
`n_order=32`; (d) `Simulation` always requests `use_cuda=True`, which fbpic downgrades to CPU
when CUDA is not installed. At `nz=400` and `nz=800` the laser is under-driven on the PIC
grid and no electrons are accelerated, so `NZ` should not be lowered.
"""

from __future__ import annotations

from pathlib import Path

from scipy.constants import c

from lume_fbpic.actions import (
    DensityFieldAction,
    DopantFractionAction,
    FinalParticlesAction,
    LaserFieldAction,
    StatAction,
    ZernikeCoefficientAction,
    make_descriptor_actions,
)
from lume_fbpic.density_profiles import GeneralizedGaussianProfile
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import FBPICSimulator

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.laser import LasyLaserPulse
from inversion_fbpic.lib.simulation import SimulationHyperparameters

# --- physical parameters (sim_0000 from sample_dataset.json) ---
PEAK_PLASMA_DENSITY_CM3 = 4.0e18
NITROGEN_DOPANT_FRACTION = 0.03
DENSITY_PEAK = 1.1
DENSITY_ALPHA_M = 1.1e-3
DENSITY_BETA = 1.1
DENSITY_CENTER_LOCATION_M = 2.3e-3
LASER_WAVELENGTH_M = 800e-9
LASER_ENERGY_J = 2.5
LASER_PULSE_DURATION_FWHM_S = 30e-15
LASER_SPOT_SIZE_M = 24e-6
LASER_SUPER_GAUSSIAN_ORDER = 3.0
LASER_FOCAL_POSITION_M = 3.5e-3

# --- hyperparameters: the coarsened 1-node CPU configuration (see module docstring,
# "Reproducing the nz=1000 CPU run"). sim_0000's own values were NZ, NR, N_AZIMUTHAL_MODES =
# 1500, 300, 5, NUMBER_DUMPS = WRITE_PERIOD = 50, P_NT = 15.
NZ, NR, N_AZIMUTHAL_MODES = 1000, 100, 2
ZMIN, ZMAX, RMAX = -70.0e-6, 0.0, 200.0e-6
GAMMA_BOOST = 1.5
NUMBER_DUMPS = 10
WRITE_PERIOD = 10
P_NZ, P_NR, P_NT = 2, 2, 4

# --- LASY laser grid, coarsened to match (sim_0000: 5 modes, (600, 900) points) ---
LASER_N_AZIMUTHAL_MODES = 2
LASER_NUM_POINTS = (300, 450)
Z0_ANTENNA = 0.0
T_START = 0.0


def build_model(working_directory: str | None = None) -> LUMEFBPICModel:
    # See module docstring, point 5: match the original's own v_window formula (n_crit at
    # 800 nm ~= 1.75e27 m^-3, its own approximation), not Simulation's more general
    # calculate_mean_group_beta(). right_buffer is left at its default -- see point 6.
    n_plasma = PEAK_PLASMA_DENSITY_CM3 * 1.0e6  # cm^-3 -> m^-3
    n_gas = n_plasma / 2.0
    beta_window = 1.0 - 0.5 * n_gas / 1.75e27

    hyparams = SimulationHyperparameters(
        zmin=ZMIN,
        zmax=ZMAX,
        rmax=RMAX,
        nz=NZ,
        nr=NR,
        nm=N_AZIMUTHAL_MODES,
        use_mpi=False,
        number_dumps=NUMBER_DUMPS,
        write_period=WRITE_PERIOD,
        gamma_boost=GAMMA_BOOST,
        field_diagnostics=["rho", "E", "B"],
        random_seed=0,
        beta_window=beta_window,
    )

    laser = LasyLaserPulse(
        energy=LASER_ENERGY_J,
        wavelength=LASER_WAVELENGTH_M,
        tau_fwhm=LASER_PULSE_DURATION_FWHM_S,
        waist=LASER_SPOT_SIZE_M,
        focal_position=LASER_FOCAL_POSITION_M,
        super_gaussian_order=LASER_SUPER_GAUSSIAN_ORDER,
        zernike_coefficients={},  # all zero -- sim_0000's own baseline (see module docstring)
        z0=Z0_ANTENNA - c * (T_START + 3 * LASER_PULSE_DURATION_FWHM_S),
        z0_antenna=Z0_ANTENNA,
        t_start=T_START,
        n_azimuthal_modes=LASER_N_AZIMUTHAL_MODES,
        num_points=LASER_NUM_POINTS,
        hi_range=8.0,
        center_and_remove_tilt=True,
        centering_angles=72,
        lasy_file=Path("diags") / "lasy_laser",
    )

    # See module docstring, points 2-3: two profiles, same shape, ion density scaled by
    # each species' own ionization-level count to compensate for add_to_simulation()'s
    # nominal_density / num_ionization_levels formula.
    he_levels = _DensityProfile._get_num_ionization_levels("He")
    n_levels = _DensityProfile._get_num_ionization_levels("N")
    he_atom_density = n_gas * (1.0 - NITROGEN_DOPANT_FRACTION)
    n_atom_density = n_gas * NITROGEN_DOPANT_FRACTION

    density_he = GeneralizedGaussianProfile(
        nominal_density=he_levels * he_atom_density,
        species="He",
        ionization=0,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        gauss_peak=DENSITY_PEAK,
        z0=DENSITY_CENTER_LOCATION_M,
        alpha=DENSITY_ALPHA_M,
        beta=DENSITY_BETA,
        elec_name=None,  # never diagnosed upstream either
    )
    density_n = GeneralizedGaussianProfile(
        nominal_density=n_levels * n_atom_density,
        species="N",
        ionization=0,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        gauss_peak=DENSITY_PEAK,
        z0=DENSITY_CENTER_LOCATION_M,
        alpha=DENSITY_ALPHA_M,
        beta=DENSITY_BETA,
        elec_name="electrons_n",
        elec_select=None,  # the nz=1000 run had the template's uz>=10 filter removed
    )

    simulator = FBPICSimulator(
        hyparams,
        laser,
        [density_he, density_n],
        target_species="electrons_n",
        working_directory=working_directory,
    )

    actions = [
        LaserFieldAction(name="laser_energy", field_name="energy", unit="J"),
        LaserFieldAction(
            name="laser_focal_position", field_name="focal_position", unit="m"
        ),
        LaserFieldAction(
            name="laser_temporal_width", field_name="tau_fwhm", unit="s"
        ),
        DensityFieldAction(
            name="nitrogen_dopant_density",
            density_index=1,
            field_name="nominal_density",
            unit="m^-3",
        ),
        # Scan variable `nitrogen_dopant_fraction` of sample_dataset.json: rewrites BOTH gas
        # densities at constant total atom density (profile 0 = He, profile 1 = N).
        DopantFractionAction(
            name="nitrogen_dopant_fraction", host_index=0, dopant_index=1, unit=None
        ),
        # One action per Zernike coefficient, named like the dataset's `zernike_<name>` inputs
        # (for example `zernike_astigmatism_4`, `zernike_coma_x`); phase amplitudes in radians.
        *[
            ZernikeCoefficientAction(
                name=f"zernike_{coefficient}", coefficient=coefficient, unit="rad"
            )
            for coefficient in laser.zernike_coefficients
        ],
        StatAction(name="charge_pc", stat_name="charge_pc", unit="pC", read_only=True),
        StatAction(
            name="energy_mean_mev", stat_name="energy_mean_mev", unit="MeV", read_only=True
        ),
        StatAction(
            name="energy_std_mev", stat_name="energy_std_mev", unit="MeV", read_only=True
        ),
        FinalParticlesAction(name="final_particles", read_only=True),
        # The 33 moment-descriptor scalars build_dataset.py writes (uz >= 30, central 95%).
        *make_descriptor_actions(),
    ]

    return LUMEFBPICModel(simulator, actions)


if __name__ == "__main__":
    model = build_model()
    model.simulator.configure()
    model.set({"laser_energy": LASER_ENERGY_J})  # set() applies the value, then runs
    print(model.get(["charge_pc", "energy_mean_mev"]))
