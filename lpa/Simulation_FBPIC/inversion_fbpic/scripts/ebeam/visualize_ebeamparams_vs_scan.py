"""
This script is currently designed to work with FBPIC parameter scans taken using the scripts found in
  "simulations/aws_set1_simulations/htu_scans".  Each of the scanning scripts is numbered 1-6, and the
  respective results of these scans are stored in "analysis_X" folders numbered 1-6.  When given a
  scan within one of these analysis folders, this script loops through all simulations performed and
  plots relevant ebeam parameters vs the scanned parameter.  The scanned parameter is read from the prefix
  in front of the "diags" folder.

Usage:
    Command-line usage:
        python visualize_ebeamparams_vs_scan.py [OPTIONS]

    Optional Arguments:
        --directory-folder PATH
            Path to the directory containing all scans. If not provided, uses the default
            value defined in the script.

        --analysis-folder STR
            Directory name containing the specific scan set of interest, typically "analysis_X".
            If not provided, uses the default value defined in the script.

        --set-name STR
            Particular 1D scan to analyze within ANALYSIS_FOLDER. If not provided, uses the
            default value defined in the script.

        -s, --species STR
            Name of the particle species to load (e.g., "electrons" or "n_elec").
            Default: "electrons"

        --sub-directory STR
            Name of the folder where h5 files are stored. If not provided, defaults to
            the species name.

        --longitudinal-threshold FLOAT|None
            Minimum z position of ebeam particles to consider. This is the default value
            used if there is no match in `get_longitudinal_threshold_setting()`.
            Use 'None' or 'null' to explicitly set to None.
            Default: 5396e-6

        --factor FLOAT
            Factor to multiply the scanned parameter in plots. Purely for visualization.
            Default: 1.0

        --units STR
            Label to use for the scanned parameter in plots. Purely for visualization.
            Default: ""

        --plot-phase-spaces
            Flag to plot the phase space of each electron beam. Default value is set by
            DEFAULT_PLOT_PHASE_SPACES in the script.

        --no-plot-phase-spaces
            Flag to skip plotting the phase space of each electron beam. This overrides
            the default value and sets it to False.

    Examples:
        # Use all defaults
        python visualize_ebeamparams_vs_scan.py

        # Specify a different directory and analysis folder
        python visualize_ebeamparams_vs_scan.py --directory-folder /path/to/scans --analysis-folder analysis_1

        # Analyze a specific scan with custom settings
        python visualize_ebeamparams_vs_scan.py --set-name my_scan --factor 1000 --units "mm"

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Union, Optional

from inversion_fbpic.utils.analysis import (
    analyze_beam,
    load_beam_data,
    apply_cut,
    plot_beam_analysis,
)
from inversion_fbpic.utils.argparse_utils import float_or_none
from pathlib import Path

# Default configuration variables
DEFAULT_DIRECTORY_FOLDER: Optional[Path] = None  # Directory containing all scans
DEFAULT_ANALYSIS_FOLDER: str = (
    "analysis"  # Set of scans of interest, typically "analysis_X"
)
DEFAULT_SET_NAME: Optional[str] = None  # Particular 1D scan to analyze
DEFAULT_SPECIES: str = (
    "electrons"  # Name of the particle species to load (e.g., "electrons" or "n_elec")
)
DEFAULT_SUB_DIRECTORY: Optional[str] = (
    None  # Name of the folder where h5 files are stored. If None, uses species name.
)

# Set the minimum z positions of ebeam particles according to the input data (change as needed)
# This is the default value used if there is no match in `get_longitudinal_threshold_setting()`
DEFAULT_LONGITUDINAL_THRESHOLD: Optional[float] = None

DEFAULT_FACTOR: float = 1.0  # Factor to multiply the scanned parameter in plots
DEFAULT_UNITS: str = ""  # Label to use for the scanned parameter in plots

DEFAULT_PLOT_PHASE_SPACES: bool = (
    True  # Set as False to skip plotting the phase space of each electron beam
)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze and plot how ebeam parameters vary across a parameter scan."
    )

    parser.add_argument(
        "--directory-folder",
        type=Path,
        default=DEFAULT_DIRECTORY_FOLDER,
        help="Path to the directory containing all scans",
    )

    parser.add_argument(
        "--analysis-folder",
        type=str,
        default=DEFAULT_ANALYSIS_FOLDER,
        help="Directory name containing the specific scan set of interest, typically 'analysis_X'",
    )

    parser.add_argument(
        "--set-name",
        type=str,
        default=DEFAULT_SET_NAME,
        help="Particular 1D scan to analyze within ANALYSIS_FOLDER",
    )

    parser.add_argument(
        "-s",
        "--species",
        type=str,
        default=DEFAULT_SPECIES,
        help="Name of the particle species to load (e.g., 'electrons' or 'n_elec')",
    )

    parser.add_argument(
        "--sub-directory",
        type=str,
        default=None,
        help="Name of the folder where h5 files are stored. If not provided, uses species name.",
    )

    parser.add_argument(
        "--longitudinal-threshold",
        type=float_or_none,
        default=DEFAULT_LONGITUDINAL_THRESHOLD,
        help="Minimum z position of ebeam particles to consider. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--factor",
        type=float,
        default=DEFAULT_FACTOR,
        help="Factor to multiply the scanned parameter in plots",
    )

    parser.add_argument(
        "--units",
        type=str,
        default=DEFAULT_UNITS,
        help="Label to use for the scanned parameter in plots",
    )

    parser.add_argument(
        "--plot-phase-spaces",
        action="store_true",
        default=DEFAULT_PLOT_PHASE_SPACES,
        help="Flag to plot the phase space of each electron beam",
    )

    parser.add_argument(
        "--no-plot-phase-spaces",
        action="store_false",
        dest="plot_phase_spaces",
        help="Flag to skip plotting the phase space of each electron beam (overrides default)",
    )

    return parser.parse_args()


def get_longitudinal_threshold_setting(
    directory_folder: Path, analysis_folder: str, default_threshold: Optional[float]
) -> Optional[float]:
    """
    If a particular set of scans has been calibrated to use a specific longitudinal threshold for the ROI,
    then return that value.  Otherwise, return the default longitudinal threshold for the minimum z value
    to consider for ebeam statistics.

    Args:
        directory_folder: Path to the directory containing all scans.
        analysis_folder: Directory name containing the specific scan set of interest.
        default_threshold: Default threshold value to use if no calibration match is found.

    Returns:
        The minimum value of z to use for ebeam statistics, either a default value or from a given calibration.
    """

    threshold: Optional[float] = None

    # Below are specific calibrated cases:
    if str(directory_folder.parent.name) in ["aws_set1_simulations"]:
        if analysis_folder in ["analysis_4"]:
            threshold: float = 4110e-6
        elif analysis_folder in ["analysis_5", "analysis_6"]:
            threshold: float = 5420e-6
    elif str(directory_folder.parent.name) in ["aws_set2_simulations"]:
        if analysis_folder in ["analysis2"]:
            threshold: float = 5420e-6

    return default_threshold if threshold is None else threshold


def process(args: argparse.Namespace) -> None:
    """
    Script entry point for analyzing and plotting how ebeam parameters vary across a scan.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    # Determine sub_directory: use provided value or default to species name
    if args.directory_folder is None:
        raise ValueError(
            "Must specify a directory containing all scans using '--directory-folder'"
        )
    elif not Path(args.directory_folder).is_dir():
        raise FileNotFoundError(f"Directory '{args.directory_folder}' does not exist")

    if args.set_name is None:
        raise ValueError(
            "Must specify str for the name of the scan set using '--set-name'"
        )

    sub_directory = (
        args.sub_directory if args.sub_directory is not None else args.species
    )

    case_values: Union[list[float], np.ndarray] = (
        []
    )  # To be read from folder names:  XXX_diags
    central_energies: Union[list[float], np.ndarray] = []
    energy_stds: Union[list[float], np.ndarray] = []
    energy_fwhms: Union[list[float], np.ndarray] = []
    total_charges: Union[list[float], np.ndarray] = []
    emittance_x: Union[list[float], np.ndarray] = []
    emittance_y: Union[list[float], np.ndarray] = []
    sigma_z: Union[list[float], np.ndarray] = []
    peak_current: Union[list[float], np.ndarray] = []
    set_directory = args.directory_folder / args.analysis_folder / args.set_name
    for case_directory in set_directory.iterdir():
        if case_directory.is_dir():
            data_folder = case_directory / sub_directory
            if data_folder.is_dir() and any(data_folder.iterdir()):
                x, y, z, ux, uy, uz, w, q, ts = load_beam_data(
                    data_folder, args.species, iteration=-1
                )
                arrs = {"x": x, "y": y, "z": z, "ux": ux, "uy": uy, "uz": uz, "w": w}
                arrs = apply_cut(
                    arrs,
                    "z",
                    get_longitudinal_threshold_setting(
                        args.directory_folder,
                        args.analysis_folder,
                        args.longitudinal_threshold,
                    ),
                    op="gt",
                )

                x, y, z, ux, uy, uz, w = (
                    arrs["x"],
                    arrs["y"],
                    arrs["z"],
                    arrs["ux"],
                    arrs["uy"],
                    arrs["uz"],
                    arrs["w"],
                )
                if len(z) > 0:

                    results = analyze_beam(x, y, z, ux, uy, uz, w, q, bins=200)

                    if args.plot_phase_spaces:
                        sup_title = f"{case_directory.name}"
                        plot_beam_analysis(
                            x,
                            y,
                            z,
                            ux,
                            uy,
                            uz,
                            w,
                            results,
                            supertitle=sup_title,
                        )

                    # Extract N value from folder name
                    case = float(case_directory.name.split("_")[0])
                    case_values.append(case)
                    central_energies.append(
                        results["energy_parameters"]["central_energy_mev"]
                    )
                    energy_stds.append(results["energy_parameters"]["energy_std_mev"])
                    energy_fwhms.append(results["energy_parameters"]["energy_fwhm_mev"])
                    total_charges.append(results["total_charge_c"] * 1e12)  # pC
                    emittance_x.append(results["emittance"]["x"] * 1e6)  # mm-mrad
                    emittance_y.append(results["emittance"]["y"] * 1e6)  # mm-mrad
                    sigma_z.append(results["beam_sizes"]["sigma_z_m"] * 1e6)  # um
                    peak_current.append(np.max(results["current"]) * 1e-3)  # kA
    # Sort by case
    case_values = np.array(case_values) * args.factor
    sort_idx = np.argsort(case_values)
    case_values = case_values[sort_idx]
    central_energies = np.array(central_energies)[sort_idx]
    energy_stds = np.array(energy_stds)[sort_idx]
    energy_fwhms = np.array(energy_fwhms)[sort_idx]
    total_charges = np.array(total_charges)[sort_idx]
    emittance_x = np.array(emittance_x)[sort_idx]
    emittance_y = np.array(emittance_y)[sort_idx]
    sigma_z = np.array(sigma_z)[sort_idx]
    peak_current = np.array(peak_current)[sort_idx]
    # Plot results
    plt.figure(figsize=(10, 8))
    # Top-left: Central Energy and sigma_z
    ax1 = plt.subplot(2, 2, 1)
    ln1 = ax1.plot(
        case_values,
        central_energies,
        marker="o",
        color="tab:blue",
        label="Central Energy (MeV)",
    )
    ax1.set_xlabel(f"{args.set_name} ({args.units})")
    ax1.set_ylabel("Central Energy (MeV)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(f"Central Energy & σz vs {args.set_name}")
    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        case_values, sigma_z, marker="s", color="tab:orange", label="σz (μm)"
    )
    ax2.set_ylabel("σz (μm)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    lns = ln1 + ln2
    labs = [line.get_label() for line in lns]
    ax1.legend(lns, labs, loc="best")

    # Top-right: Energy spread
    ax5 = plt.subplot(2, 2, 2)
    ln5 = ax5.plot(
        case_values, energy_fwhms, marker="o", color="tab:blue", label="FWHM"
    )
    ln6 = ax5.plot(
        case_values, energy_stds, marker="x", color="tab:orange", label="RMS"
    )
    ax5.set_xlabel(f"{args.set_name} ({args.units})")
    ax5.set_ylabel("Energy Spread (MeV)", color="tab:blue")
    ax5.tick_params(axis="y", labelcolor="tab:blue")
    ax5.set_title(f"Energy Spread vs {args.set_name}")
    lns3 = ln5 + ln6
    labs3 = [line.get_label() for line in lns3]
    ax5.legend(lns3, labs3, loc="best")

    # Bottom-left: Total charge and peak current
    ax3 = plt.subplot(2, 2, 3)
    ln3 = ax3.plot(
        case_values,
        total_charges,
        marker="o",
        color="tab:green",
        label="Total Charge (pC)",
    )
    ax3.set_xlabel(f"{args.set_name} ({args.units})")
    ax3.set_ylabel("Total Charge (pC)", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")
    ax3.set_title(f"Total Charge & Peak Current vs {args.set_name}")
    ax4 = ax3.twinx()
    ln4 = ax4.plot(
        case_values,
        peak_current,
        marker="^",
        color="tab:red",
        label="Peak Current (kA)",
    )
    ax4.set_ylabel("Peak Current (kA)", color="tab:red")
    ax4.tick_params(axis="y", labelcolor="tab:red")
    lns2 = ln3 + ln4
    labs2 = [line.get_label() for line in lns2]
    ax3.legend(lns2, labs2, loc="best")

    # Bottom-right: Emittance
    plt.subplot(2, 2, 4)
    plt.plot(case_values, emittance_x, marker="o", label="x")
    plt.plot(case_values, emittance_y, marker="x", label="y")
    plt.xlabel(f"{args.set_name} ({args.units})")
    plt.ylabel("Emittance (mm-mrad)")
    plt.title(f"Emittance vs {args.set_name}")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
