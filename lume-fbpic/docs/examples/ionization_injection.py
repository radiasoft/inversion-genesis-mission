"""lume_fbpic approximation of `runs/initial_sample/ionization_injection_runscript_00.py`
(`sim_0000` of `runs/initial_sample/sample_dataset/sample_dataset.json`): a boosted-frame
helium/nitrogen ionization-injection LPA driven by an unaberrated (all Zernike coefficients 0)
super-Gaussian LASY laser pulse. Needs the real `LasyLaserPulse`, which is not on `main` yet.

Substitutions, none of which makes it a bit-exact reproduction:

- Gas jets: the original calls `add_new_species()` directly for He, N and two empty electron
  species. Here two `GeneralizedGaussianProfile`s (He and N, same shape) do it, with
  `nominal_density` set to the atom density times the species' ionization levels, because
  `_DensityProfile` divides by the level count.
- Particle loading: `p_zmin`/`p_rmax` are not passed through by `_DensityProfile`, so
  macroparticles load over the whole grid. Same physics, more macroparticles.
- Moving window: `beta_window` is computed from the original's simplified `v_window` formula.
- Interaction length: the original's would need a negative `right_buffer`, which
  `SimulationHyperparameters` rejects. `right_buffer` is left at its default, so the run is
  about 15% longer (~13,600 steps against 11,813).
- Laser: `LasyLaserPulse` is used as is; `z0` is set as in `demo_lasy_laser_simulation`.
- Species name: `electrons_n` here, `nitrogen_electrons` in the original's output.
- `Simulation` passes `n_order=-1` (the original: 32) and always requests CUDA, which fbpic
  downgrades to CPU when it is not installed.

The constants below are the coarsened configuration that was run on a multi-core CPU machine
(14 threads, no GPU), not `sim_0000`'s production grid: `nz=1000`, `nr=100`, `nm=2`, a (300, 450)
/ 2-mode LASY grid, `p_nt=4`, 10 lab-frame dumps, and no `uz >= 10` selection (at `nz=400` it
left no particles), so apply `uz_min` downstream. At `nz=400` no electrons were accelerated
(maximum `uz` 0.3), so `NZ` should not be lowered.
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
    # Match the original's own v_window formula (n_crit at
    # 800 nm ~= 1.75e27 m^-3, its own approximation), not Simulation's more general
    # calculate_mean_group_beta(). right_buffer is left at its default -- see the module docstring.
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

    # Two profiles (see the module docstring): same shape, ion density scaled by
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
