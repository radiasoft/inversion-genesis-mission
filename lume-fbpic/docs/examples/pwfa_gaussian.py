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

Running it prints the witness's charge, mean energy and energy spread, then plots the last dump
(`plot_results()`): the electron density, the longitudinal field `E_z` as a map and as a lineout
near the axis, and the transverse force `E_r - c B_theta`. Needs matplotlib.
"""

from __future__ import annotations

import pathlib

import h5py
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


def plot_results(
    directory: str | pathlib.Path, *, output: str | pathlib.Path = "."
) -> list[pathlib.Path]:
    """Plot the last dump of the run in `directory` (its working directory, holding
    `diags/hdf5`) and return the files written to `output`:

    - `electron_density.png`: the electron density of the plasma and the bunches [cm^-3].
    - `longitudinal_field.png`: `E_z` [GV/m], with zero at the middle of the colour scale.
    - `longitudinal_field_lineout.png`: `E_z` along the axis.
    - `transverse_force.png`: `E_r - c B_theta` [GV/m], the force on a relativistic electron.

    The horizontal axis is `k_p zeta`, with `zeta = z - z_driver` (zero at the drive bunch's
    centre, the wake behind it at negative values) and the radial one is `k_p r`. Needs
    matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.colors import TwoSlopeNorm

    snapshot = _read_last_dump(pathlib.Path(directory))
    zeta, radius = K_P * snapshot["zeta"], K_P * snapshot["r"]
    extent = [zeta[0], zeta[-1], radius[0], radius[-1]]
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = []

    def field_map(name, data, label, cmap, norm=None):
        figure, axis = pyplot.subplots()
        image = axis.imshow(
            data, extent=extent, cmap=cmap, origin="lower", norm=norm, aspect="auto"
        )
        axis.set(xlabel=r"$k_p \zeta$", ylabel=r"$k_p r$")
        figure.colorbar(image, ax=axis, orientation="horizontal", label=label)
        figure.tight_layout()
        files.append(output / f"{name}.png")
        figure.savefig(files[-1], dpi=120)
        pyplot.close(figure)

    def centred(data):
        return TwoSlopeNorm(
            0.0, vmin=min(data.min(), -1e-30), vmax=max(data.max(), 1e-30)
        )

    field_map("electron_density", snapshot["n_e"], r"$n_e$ [cm$^{-3}$]", "viridis")
    field_map(
        "longitudinal_field",
        snapshot["E_z"],
        r"$E_z$ [GV/m]",
        "RdBu",
        centred(snapshot["E_z"]),
    )
    field_map(
        "transverse_force",
        snapshot["F_r"],
        r"$E_r - c B_\theta$ [GV/m]",
        "RdBu",
        centred(snapshot["F_r"]),
    )

    figure, axis = pyplot.subplots()
    axis.plot(zeta, snapshot["E_z"][0])
    axis.set(xlabel=r"$k_p \zeta$", ylabel=r"$E_z$ on axis [GV/m]")
    figure.tight_layout()
    files.append(output / "longitudinal_field_lineout.png")
    figure.savefig(files[-1], dpi=120)
    pyplot.close(figure)
    return files


def _read_last_dump(directory: pathlib.Path) -> dict:
    """The last dump of a run: `zeta` and `r` [m], and `n_e` [cm^-3], `E_z` and `F_r = E_r - c B_theta`
    [GV/m] on the grid, shape (r, z)."""
    paths = sorted((directory / "diags" / "hdf5").glob("data*.h5"))
    if not paths:
        raise FileNotFoundError(f"no dumps under {directory / 'diags' / 'hdf5'}")
    with h5py.File(paths[-1], "r") as f:
        step = f[f"data/{int(paths[-1].name[4:12])}"]
        mesh = step["fields/E"]
        dr, dz = mesh.attrs["gridSpacing"]
        z0 = mesh.attrs["gridGlobalOffset"][1]
        e_z, e_r, b_t = (
            step["fields/E/z"][0],
            step["fields/E/r"][0],
            step["fields/B/t"][0],
        )
        rho = step["fields/rho"][0]
        z_driver = numpy.average(
            step["particles/driver/position/z"][:],
            weights=step["particles/driver/weighting"][:],
        )
    n_r, n_z = e_z.shape
    return {
        "zeta": z0 + dz * numpy.arange(n_z) - z_driver,
        "r": dr * (numpy.arange(n_r) + 0.5),
        "n_e": -rho
        / e
        / 100.0**3,  # rho is the charge density of everything: plasma and bunches
        "E_z": e_z * 1.0e-9,
        "F_r": (e_r - c * b_t) * 1.0e-9,
    }


if __name__ == "__main__":
    model = build_model()
    model.set(
        {"driver_charge": DRIVE_Q}
    )  # set() applies the value, then runs the simulation
    print(model.get(["charge_pc", "energy_mean_mev", "energy_std_mev"]))
    for path in plot_results(
        model.simulator.working_directory, output=model.simulator.working_directory
    ):
        print(f"wrote {path}")
