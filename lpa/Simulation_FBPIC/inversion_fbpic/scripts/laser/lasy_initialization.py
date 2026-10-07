"""Initialize, diagnose, and export an aberrated LASY pulse for FBPIC."""

from pathlib import Path
from time import perf_counter

from inversion_fbpic.utils.laser import HighOrderLasyLaser

try:
    from .lasy_propagation import run_propagation_diagnostics
except ImportError:
    from lasy_propagation import run_propagation_diagnostics


OUTPUT_DIRECTORY = Path("diags")
OUTPUT_PREFIX = "experimental_laser"
SHOW_PLOTS = True

# Physical laser values.
PHYSICAL_PARAMETERS = {
    "laser_wavelength_m": 800e-9,
    "laser_energy_J": 2.5,
    "laser_pulse_duration_fwhm_s": 30e-15, # 30e-15,
    # ``"auto"`` derives the transform-limited Gaussian bandwidth from the
    # pulse duration. Use a fixed numeric value to scan duration independently.
    "laser_spectral_bandwidth_rad_s": "auto",
    "laser_cep_phase_rad": 0.0,
    # At 30 fs, +/-1 corresponds to +/-5e-28 s^2 GDD and +/-1e-41 s^3 TOD.
    # Coefficients scale with duration squared/cubed for other pulse widths.
    "laser_gdd_relative": 0.0,
    "laser_tod_relative": 0.0,
    "laser_fod_s4": 0.0,
    "laser_spot_size_m": 24e-6,
    "laser_super_gaussian_order": 3,
    "laser_focal_position_m": 3.5e-3,
    "zernike_astigmatism_2": 0.0, #  +/- 0.5
    "zernike_astigmatism_4": 0.0,
    "zernike_coma_x": 0.0,  #+/- 0.25
    "zernike_coma_y": 0.0,
    "zernike_trefoil_x": 0.0, # +/- 0.75
    "zernike_trefoil_y": 0.0,
    "zernike_spherical_3": 0.0,  # +/- 0.25
    # the following are higher order terms that I will ignore for the present study
    "zernike_astigmatism_6": 0.00,  # Keep at 0
    "zernike_coma_5_x": 0.0,  # Keep at 0
    "zernike_coma_5_y": 0.0,  # Keep at 0
    "zernike_secondary_trefoil_x": 0.0,  # Keep at 0
    "zernike_secondary_trefoil_y": 0.0,  # Keep at 0
}

# Numerical representation settings.
HYPERPARAMETERS = {
    "polarization": (1, 0),
    "n_azimuthal_modes": 5,
    "num_points": (1200, 600),  # (r, z)
    "hi_range": 20.0,
    "center_and_remove_tilt": True,
    "centering_angles": 72,
    "spectral_time_window_factor": 6.0,
}


def main() -> None:
    """Construct, prepare, export, and optionally diagnose the laser pulse."""
    time_start = perf_counter()
    high_order_laser = HighOrderLasyLaser(PHYSICAL_PARAMETERS, HYPERPARAMETERS)
    print(
        "Initialization and back propagation took %.2f seconds."
        % (perf_counter() - time_start)
    )

    time_start = perf_counter()
    output_path = high_order_laser.save(OUTPUT_DIRECTORY / OUTPUT_PREFIX)
    print("Write took %.2f seconds." % (perf_counter() - time_start))

    if SHOW_PLOTS:
        run_propagation_diagnostics(
            output_path,
            PHYSICAL_PARAMETERS["laser_focal_position_m"],
        )


if __name__ == "__main__":
    main()
