"""
This script is a simple launch script for the template in the same directory.

Demonstrates configuring and launching one ionization-injection simulation.
Use ``build-dataset`` and ``plot-phase-space-moments`` after the run completes
to calculate dataset descriptors and inspect the resulting phase space.
"""

from pathlib import Path

from ionization_injection_template import IonizationInjectionSimulation


RUN_DIRECTORY = Path(__file__).parent
DESCRIPTION: str = "sim_0000"

# FBPIC numerical and execution settings.
HYPERPARAMETERS = {
    "use_cuda": True,
    "n_order": 32,
    "gamma_boost": 1.5,
    "nz": 1500,
    "nr": 300,
    "zmax": 0.0,
    "zmin": -70.0e-6,
    "rmax": 200.0e-6,
    "n_azimuthal_modes": 5,
    "p_zmin": 0.0,
    "p_rmax": 90e-6,
    "p_nz": 2,
    "p_nr": 2,
    "p_nt": 15,
    "number_dumps_lab": 50,
    "write_period": 50,
    "save_checkpoints": False,
    "checkpoint_period": 100,
    "use_restart": False,
    "track_electrons": False,
    "random_seed": 0,
    "lasy_file": RUN_DIRECTORY / "data" / DESCRIPTION / "experimental_laser",
    "lasy_t_start": 0.0,
    "lasy_antenna_position": 0.0,
    "lab_diagnostic_directory": RUN_DIRECTORY / "data" / DESCRIPTION / "lab_diags",
    "laser_polarization": (1, 0),
    "laser_n_azimuthal_modes": 5,
    "laser_num_points": (600, 900),
    "laser_hi_range": 8.0,
    "laser_center_and_remove_tilt": True,
    "laser_centering_angles": 72,
}

# Physical laser and gas parameters.
PHYSICAL_PARAMETERS = {
    "peak_plasma_density_cm3": 4.0e18,
    "nitrogen_dopant_fraction": 0.03,
    "density_peak": 1.1,
    "density_alpha_m": 1.1e-3,
    "density_beta": 1.1,
    "density_center_location_m": 2.3e-3,
    "laser_wavelength_m": 800e-9,
    "laser_energy_J": 2.5,
    "laser_pulse_duration_fwhm_s": 30e-15,
    "laser_spot_size_m": 24e-6,
    "laser_super_gaussian_order": 3,
    "laser_focal_position_m": 3.5e-3,
    "zernike_astigmatism_2": 0.0,
    "zernike_astigmatism_4": 0.0,
    "zernike_coma_y": 0.0,
    "zernike_coma_x": 0.0,
    "zernike_trefoil_y": 0.0,
    "zernike_trefoil_x": 0.0,
    "zernike_spherical_3": 0.0,
    "zernike_astigmatism_6": 0.0,
    "zernike_coma_5_y": 0.0,
    "zernike_coma_5_x": 0.0,
    "zernike_secondary_trefoil_y": 0.0,
    "zernike_secondary_trefoil_x": 0.0,
}


def main() -> None:
    """Construct and run the configured FBPIC simulation."""
    simulation = IonizationInjectionSimulation(
        PHYSICAL_PARAMETERS,
        HYPERPARAMETERS,
    )
    simulation.run()


if __name__ == "__main__":
    main()