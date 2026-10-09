"""
LPA simulation driven by an aberrated super-Gaussian LASY laser pulse.

Compared with the Gaussian-laser demos, the laser here is a ``LasyLaserPulse``:
the pulse is built with LASY at focus (super-Gaussian transverse profile plus
Zernike phase aberrations), back-propagated to the simulation start plane,
written to an HDF5 file, and emitted into the FBPIC box by a stationary laser
antenna. Its ``a0`` is not an analytic input: it is measured numerically from
the LASY field at focus when the pulse is prepared.
"""

from inversion_fbpic.lib import density_profiles as dn, laser as ls, simulation as sm
from mpi4py import MPI
from scipy.constants import c
import matplotlib.pyplot as plt
import os
from pathlib import Path
from inversion_fbpic.utils.simulation_setup_tools import (
    calculate_acceleration_gradient,
    calculate_plasma_wavelength,
)
from inversion_fbpic.utils.plotting import plot_from_hdf5_series
from inversion_fbpic.utils.make_movie import make_movie
from inversion_fbpic.utils import analysis


MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()

if MPI_SIZE <= 1:
    USE_MPI = False
else:
    USE_MPI = True

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    DIAGS_DIR = "diags"
    PLOTS_DIR = Path("plots")
    # define parameters
    TARGET_ENERGY = 430e6  # eV
    LASER_ENERGY = 4.5  # J
    WAVELENGTH = 800e-9  # m
    LASER_WAIST = 30e-6  # m
    TAU_FWHM = 40e-15  # s
    FOCAL_POSITION = 3.0e-3  # m
    SUPER_GAUSSIAN_ORDER = 3.0  # 2.0 would be a plain Gaussian
    # Zernike phase amplitudes at focus (radians). Missing names default to 0.
    ZERNIKE_COEFFICIENTS = {
        "astigmatism_2": 0.4,
        "coma_x": 0.3,
        "spherical_3": 0.2,
    }
    FLATTOP_PLASMA_DENSITY = 1.0e18 * 1e6  # m^-3

    # LASY pulses are emitted by an antenna. FBPIC resets the LASY time axis to
    # zero, so the peak leaves the antenna at t = T_START + PEAK_DELAY, where
    # PEAK_DELAY pins the peak's position in the LASY time window (3 * TAU_FWHM
    # is also what the default transform-limited window gives). The emitted
    # pulse trails the antenna position by the window length, 2 * PEAK_DELAY * c.
    Z0_ANTENNA = -5e-6  # m, just inside the front of the box
    T_START = 0.0  # s
    PEAK_DELAY = 3 * TAU_FWHM  # s
    LASER_CENTROID = Z0_ANTENNA - c * (T_START + PEAK_DELAY)  # m, informational
    WINDOW_SIZE = max(
        3.0 * calculate_plasma_wavelength(FLATTOP_PLASMA_DENSITY),
        2 * PEAK_DELAY * c + 2 * abs(Z0_ANTENNA),
    )  # m

    laser = ls.LasyLaserPulse(
        energy=LASER_ENERGY,
        wavelength=WAVELENGTH,
        tau_fwhm=TAU_FWHM,
        waist=LASER_WAIST,
        focal_position=FOCAL_POSITION,
        super_gaussian_order=SUPER_GAUSSIAN_ORDER,
        zernike_coefficients=ZERNIKE_COEFFICIENTS,
        z0=LASER_CENTROID,
        z0_antenna=Z0_ANTENNA,
        t_start=T_START,
        peak_delay_from_file_start=PEAK_DELAY,
        # The start-plane re-centering step interpolates the field on a polar
        # grid, which is not accurate enough for aberrated beams (it shifts the
        # measured focus a0 by several percent). The coma-induced centroid
        # offset here is ~1 um, so re-centering is unnecessary.
        center_and_remove_tilt=False,
        lasy_file=Path(DIAGS_DIR) / "lasy_laser",  # relative to working_directory
    )

    # Same energy and duration in an ideal Gaussian, for comparison plots.
    reference_laser = ls.GaussianLaserPulse(
        energy=LASER_ENERGY,
        wavelength=WAVELENGTH,
        waist=LASER_WAIST,
        tau_fwhm=TAU_FWHM,
        z0=LASER_CENTROID,
        cep=0.0,
        focal_position=FOCAL_POSITION,
        polarization=0.0,
    )

    # Build the LASY file now (rank 0 builds, the other ranks wait and receive
    # the file path and a0) so a0 is available to size the plasma. Simulation
    # would otherwise do this itself during setup_simulation().
    laser.prepare(MPI.COMM_WORLD if USE_MPI else None, relative_to=SCRIPT_DIR)
    if laser.a0 is None:
        raise ValueError(
            "LASY laser a0 was not measured. Please check the laser parameters."
        )
    if MPI_RANK == 0:
        print(f"LASY file: {laser.lasy_file_path}")
        print(
            f"Measured a0 at focus: {laser.a0:.3f} "
            f"(ideal Gaussian with the same energy: {reference_laser.a0:.3f})"
        )

    accel_gradient = calculate_acceleration_gradient(laser.a0, FLATTOP_PLASMA_DENSITY)
    flattop_length = TARGET_ENERGY / accel_gradient
    if MPI_RANK == 0:
        print(f"Flattop length: {flattop_length * 1e6} um")

    flattop_profile = dn.SmoothSineFlattop(
        nominal_density=FLATTOP_PLASMA_DENSITY,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        elec_name="electrons_flattop",
        elec_select={"uz": [10.0, None]},
        flattop_width=flattop_length,
        upramp_length=100e-6,
        downramp_length=100e-6,
    )

    hyparams = sm.SimulationHyperparameters(
        zmin=-WINDOW_SIZE,
        zmax=0.0,
        rmax=120e-6,
        nz=1024,
        nr=300,
        nm=5,
        use_mpi=USE_MPI,
        number_dumps=100,
        gamma_boost=3,
        field_diagnostics=["E", "B", "rho"],
    )

    # save yamls
    if not USE_MPI or MPI_RANK == 0:
        os.makedirs("cfgs", exist_ok=True)
        hyparams.to_yaml_file("cfgs/hyparams.yaml")
        hyparams.grid_parameters_yaml("cfgs/grid_parameters.yaml")
        laser.to_yaml_file("cfgs/laser.yaml")  # records the measured out_a0
        reference_laser.to_yaml_file("cfgs/reference_gaussian_laser.yaml")
        flattop_profile.to_yaml_file("cfgs/flattop_profile.yaml")

    # plot the laser and the density profile
    if not USE_MPI or MPI_RANK == 0:
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # Start-plane envelope lineout and face-on |E| map of the LASY pulse.
        laser.plot(output_path=PLOTS_DIR / "lasy_laser.png")

        # Overlay the LASY lineout with the ideal Gaussian of equal energy. Note
        # the LASY curve is the defocused on-axis field at the start plane, while
        # GaussianLaserPulse.plot draws its focal amplitude; the legend gives the
        # focus a0 of both.
        fig, ax = plt.subplots()
        laser.plot(
            mode="lineout", ax=ax, label=f"LASY at start plane, focus a0={laser.a0:.2f}"
        )
        reference_laser.plot(
            mode="lineout",
            ax=ax,
            label=f"Gaussian focal amplitude, a0={reference_laser.a0:.2f}",
        )
        ax.legend()
        ax.grid()
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("On-axis envelope amplitude (a0)")
        ax.set_title("On-axis laser envelopes")
        fig.savefig(PLOTS_DIR / "laser_comparison.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        flattop_profile.plot_z_profile(ax=ax, label="Flattop", num=600)
        ax.legend()
        ax.grid()
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Density (m^-3)")
        ax.set_title("Density Profile")
        fig.savefig(PLOTS_DIR / "density_profiles.png")
        plt.close(fig)

    sim = sm.Simulation(elements=[hyparams, laser, flattop_profile])

    sim.setup_simulation(working_directory=SCRIPT_DIR)
    sim.run_simulation()

    # analysis
    if not USE_MPI or MPI_RANK == 0:
        # plot some movies
        for field in ["rho", "eme"]:
            stills_dir, prefix = plot_from_hdf5_series(
                Path(DIAGS_DIR) / "hdf5",
                PLOTS_DIR / "stills",
                field_name=field if field == "rho" else None,
                component=field if field == "eme" else None,
                vminmax=(0, 1e18) if field == "rho" else None,
                cmap="magma",
            )
            movie = make_movie(stills_dir, prefix, filename=field, movie_dir=PLOTS_DIR)

        # do a beam analysis
        bd = analysis.load_beam_data(Path(DIAGS_DIR) / "hdf5", "electrons_flattop")
        ba = analysis.analyze_beam(*bd[:7], bd[7])
        analysis.print_beam_summary(ba, PLOTS_DIR / "beam_summary.txt")
        analysis.plot_beam_analysis(
            *bd[:7], ba, save_path=PLOTS_DIR / "beam_analysis.png"
        )
