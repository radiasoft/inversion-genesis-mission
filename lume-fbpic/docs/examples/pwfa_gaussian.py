"""A lab-frame, beam-driven plasma wakefield accelerator (PWFA) with nominal FACET-II parameters.

A Gaussian drive bunch (3 nC, gamma 1.957e4, sigma_r = 0.4 / k_p, sigma_z = 1.2 / k_p) and a
Gaussian witness bunch (100 pC, sigma_r = 2 um, sigma_z = 6 um) 150 um behind it, with the same
gamma, cross a 4e16 cm^-3 electron plasma that starts as a sharp step. The box is two plasma
wavelengths long and one in radius, with cells sized from the plasma wavenumber and the bunch
sizes, and about 280 steps, enough for the wake to form behind the drive bunch. The model's
inputs are the drive bunch charge and the other bunch and plasma fields from
`make_pwfa_actions()`; its charge and energy outputs describe the witness.

Setup notes:

- The plasma is bare electrons (`species=None`), so no ion species and no protons.
- The step is a `LinearRampFlattop` with a 1 nm ramp at twice the box length, and the bunches
  start behind it.
- The grid is open in z and reflective in r.
- `random_seed` is set so a run repeats.
"""

from __future__ import annotations

import numpy
from scipy.constants import c, e, epsilon_0, m_e, pi

from lume_fbpic.actions import make_pwfa_actions
from lume_fbpic.density_profiles import LinearRampFlattop
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.pwfa_config import GaussianBunch, PWFAGrid
from lume_fbpic.simulator import PWFASimulator

N_PLASMA = 4.0e16 * 100**3  # [m^-3]
OMEGA_P = numpy.sqrt(N_PLASMA * e**2 / (m_e * epsilon_0))
K_P = OMEGA_P / c
LAMBDA_P = 2.0 * pi / K_P

DRIVE_SIGMA_R = 0.4 / K_P
DRIVE_SIGMA_Z = 1.2 / K_P
DRIVE_Q = 3.0e-9  # [C]
DRIVE_GAMMA = 1.957e4
WITNESS_SIGMA_R = 2.0e-6
WITNESS_SIGMA_Z = 6.0e-6
WITNESS_Q = 1.0e-10  # [C]
TRAILING_DISTANCE = 150.0e-6

DOMAIN_LENGTH = 2.0 * LAMBDA_P
DOMAIN_RADIUS = LAMBDA_P
DELTA_R = min(0.244 * DRIVE_SIGMA_R, 0.2 * LAMBDA_P)
DELTA_Z = min(DELTA_R, min(0.05 * DRIVE_SIGMA_Z, 0.1 * LAMBDA_P))
DT = numpy.sqrt(DELTA_Z**2 + DELTA_R**2) / c
SIM_TIME = 2.5 * DOMAIN_LENGTH / c
DUMP_PERIOD = (int(SIM_TIME / DT) + 8) // 8
N_STEPS = 8 * DUMP_PERIOD + 1


def build_model(*, dummy_run: bool = False, n_steps: int = N_STEPS) -> LUMEFBPICModel:
    grid = PWFAGrid(
        zmin=0.0,
        zmax=DOMAIN_LENGTH,
        nz=int(numpy.rint(DOMAIN_LENGTH / DELTA_Z)),
        rmax=DOMAIN_RADIUS,
        nr=int(numpy.rint(DOMAIN_RADIUS / DELTA_R)),
        nm=1,
        dt=DT,
        n_steps=n_steps,
        write_period=DUMP_PERIOD,
        r_boundary="reflective",
        smoother_passes=1,
        smoother_compensator=False,
        random_seed=0,
    )
    plasma = LinearRampFlattop(
        nominal_density=N_PLASMA,
        species=None,
        ionization=-1,
        p_nz=4,
        p_nr=4,
        p_nt=4,
        ramp_start=2.0 * DOMAIN_LENGTH,
        ramp_length=1.0e-9,
    )
    driver = GaussianBunch(
        charge=DRIVE_Q,
        gamma=DRIVE_GAMMA,
        sig_r=DRIVE_SIGMA_R,
        sig_z=DRIVE_SIGMA_Z,
        sig_gamma=1.0,
        n_emit=0.0,
        n_macroparticles=16000,
        tf=0.0,
        zf=0.75 * DOMAIN_LENGTH,
    )
    witness = GaussianBunch(
        charge=WITNESS_Q,
        gamma=DRIVE_GAMMA,
        sig_r=WITNESS_SIGMA_R,
        sig_z=WITNESS_SIGMA_Z,
        sig_gamma=1.0,
        n_emit=0.0,
        n_macroparticles=1600,
        tf=0.0,
        zf=0.75 * DOMAIN_LENGTH - TRAILING_DISTANCE,
    )
    simulator = PWFASimulator(grid, plasma, driver, witness, target_species="witness")
    return LUMEFBPICModel(simulator, make_pwfa_actions(simulator), dummy_run=dummy_run)


if __name__ == "__main__":
    model = build_model()
    model.simulator.configure()
    model.set({"driver_charge": DRIVE_Q})  # set() applies the value, then runs the simulation
    print(model.get(["charge_pc", "energy_mean_mev", "energy_std_mev"]))
