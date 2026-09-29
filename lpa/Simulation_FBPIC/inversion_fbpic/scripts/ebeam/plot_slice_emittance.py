"""Electron beam slice emittance analysis and visualization.

This module provides functionality to analyze electron beam statistics across energy
slices, calculating charge density and normalized emittance for each slice.

Usage:
    First, extract the ebeam from the full .h5 output file using either
    "extract_ebeam_from_hdf5.py" or "extract_ebeam_from_set.py". The extracted ebeam
    will be located in trimmed-down h5 files in an "ebeam" directory.

    Command-line usage:
        python plot_slice_emittance.py [OPTIONS]

    Optional Arguments:
        -d, --diag-folder PATH
            Path to the ebeam .h5 files directory. Required.

        -i, --iteration-number INT
            Which dump to load from DIAG_FOLDER. Set to "-1" for final dump.
            Default: -1

        -s, --species STR
            Which species to load from DIAG_FOLDER. Typically "electrons".
            Default: "electrons"

        --min-z FLOAT|None
            Minimum z position when selecting electrons. Use 'None' or 'null' to
            explicitly set to None.
            Default: 8860e-6

        --max-z FLOAT|None
            Maximum z position when selecting electrons. Use 'None' or 'null' to
            explicitly set to None.
            Default: None

    Examples:
        # Analyze a diagnostics folder
        python plot_slice_emittance.py -d /path/to/hdf5

        # Load a specific iteration and set z window
        python plot_slice_emittance.py -i 100 --min-z 8000e-6 --max-z 10000e-6
"""

from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const

import inversion_fbpic.utils.analysis as an
from inversion_fbpic.utils.argparse_utils import float_or_none

# Default configuration variables
DEFAULT_DIAG_FOLDER: Optional[Path] = None
DEFAULT_MIN_Z: Optional[float] = 8860e-6  # Minimum z when selecting electrons (meters)
DEFAULT_MAX_Z: Optional[float] = None  # Maximum z when selecting electrons
DEFAULT_ITERATION_NUMBER: int = -1  # Which dump to load from DIAG_FOLDER
DEFAULT_SPECIES: str = "electrons"  # Which species to load from DIAG_FOLDER


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze electron beam statistics across energy slices."
    )

    parser.add_argument(
        "-d",
        "--diag-folder",
        type=Path,
        default=DEFAULT_DIAG_FOLDER,
        required=True,
        help="Path to the ebeam .h5 files directory",
    )

    parser.add_argument(
        "-i",
        "--iteration-number",
        type=int,
        default=DEFAULT_ITERATION_NUMBER,
        help="Which dump to load from DIAG_FOLDER. Set to '-1' for final dump",
    )

    parser.add_argument(
        "-s",
        "--species",
        type=str,
        default=DEFAULT_SPECIES,
        help="Which species to load from DIAG_FOLDER. Typically 'electrons'",
    )

    parser.add_argument(
        "--min-z",
        type=float_or_none,
        default=DEFAULT_MIN_Z,
        help="Minimum z position when selecting electrons. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--max-z",
        type=float_or_none,
        default=DEFAULT_MAX_Z,
        help="Maximum z position when selecting electrons. Use 'None' or 'null' to explicitly set to None",
    )

    return parser.parse_args()


def load_and_filter_beam_data(
    diag_folder: Path,
    species: str,
    iteration_number: int,
    min_z: Optional[float],
    max_z: Optional[float],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load and filter electron beam data.

    Args:
        diag_folder: Path to the ebeam .h5 files directory.
        species: Which species to load.
        iteration_number: Which dump to load.
        min_z: Minimum z position when selecting electrons.
        max_z: Maximum z position when selecting electrons.

    Returns:
        Tuple containing filtered beam arrays (x, y, z, ux, uy, uz, w).

    Raises:
        FileNotFoundError: If beam data cannot be found.
        Exception: If beam data cannot be loaded.
    """
    try:
        x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
            diag_folder, species, iteration=iteration_number
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data in {diag_folder}")
    except Exception as e:
        raise Exception(f"Error loading beam data: {e}")

    arrs: Dict[str, np.ndarray] = {
        "x": x,
        "y": y,
        "z": z,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "w": w,
    }

    if min_z is not None:
        arrs = an.apply_cut(arrs, "z", min_z, op="gt")
    if max_z is not None:
        arrs = an.apply_cut(arrs, "z", max_z, op="lt")

    return (
        arrs["x"],
        arrs["y"],
        arrs["z"],
        arrs["ux"],
        arrs["uy"],
        arrs["uz"],
        arrs["w"],
    )


def calculate_energy_slices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
    n_bins: int = 70,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate energy slice statistics.

    Args:
        x: Particle x positions in meters.
        y: Particle y positions in meters.
        z: Particle z positions in meters.
        ux: Normalized x-momenta (dimensionless).
        uy: Normalized y-momenta (dimensionless).
        uz: Normalized z-momenta (dimensionless).
        w: Particle weights.
        n_bins: Number of energy bins to create.

    Returns:
        Tuple containing (slice_energies, slice_charges, slice_emittance_x,
                         slice_emittance_y).
    """
    uz_min, uz_max = np.min(uz), np.max(uz)
    uz_bins = np.linspace(uz_min, uz_max, n_bins + 1)

    slice_energies: List[float] = []
    slice_charges: List[float] = []
    slice_emittance_x: List[float] = []
    slice_emittance_y: List[float] = []

    for i in range(n_bins):
        slice_mask = (uz >= uz_bins[i]) & (uz < uz_bins[i + 1])

        if not np.any(slice_mask):
            print(f"Skipping bad slice: {i}")
            continue

        # Extract slice data
        x_slice = x[slice_mask]
        y_slice = y[slice_mask]
        ux_slice = ux[slice_mask]
        uy_slice = uy[slice_mask]
        uz_slice = uz[slice_mask]
        w_slice = w[slice_mask]

        # Calculate energy for this slice
        gamma_slice = np.sqrt(1 + ux_slice**2 + uy_slice**2 + uz_slice**2)
        energy_mev = (gamma_slice - 1) * const.m_e * const.c**2 / const.e / 1e6
        energy_min = np.min(energy_mev)
        energy_max = np.max(energy_mev)
        energy_width = energy_max - energy_min

        try:
            # Calculate charge for this slice
            total_charge = np.sum(w_slice * const.e)

            # Calculate emittance for this slice
            energy_params = an.calculate_energy_parameters(
                ux_slice, uy_slice, uz_slice, w_slice
            )
            geometric_emittance = an.calculate_geometric_emittance(
                x_slice, y_slice, ux_slice, uy_slice, uz_slice, w_slice
            )
            normalized_emittance = {
                "x": geometric_emittance["x"] * energy_params["gamma_beam"],
                "y": geometric_emittance["y"] * energy_params["gamma_beam"],
            }

            # Record the charge and x and y emittance values for each bin
            slice_energies.append((energy_max + energy_min) / 2)
            slice_charges.append(
                (total_charge * 1e12) / energy_width if energy_width > 0 else 0
            )
            slice_emittance_x.append(
                normalized_emittance["x"] * 1e6
            )  # Convert to um⋅rad
            slice_emittance_y.append(
                normalized_emittance["y"] * 1e6
            )  # Convert to um⋅rad

        except Exception as e:
            print(f"Warning: Could not analyze slice {i}: {e}")
            continue

    return (
        np.array(slice_energies),
        np.array(slice_charges),
        np.array(slice_emittance_x),
        np.array(slice_emittance_y),
    )


def create_energy_slice_plot(
    slice_energies: np.ndarray,
    slice_charges: np.ndarray,
    slice_emittance_x: np.ndarray,
    slice_emittance_y: np.ndarray,
) -> None:
    """Create and display the energy slice analysis plot.

    Args:
        slice_energies: Energy values for each slice in MeV.
        slice_charges: Charge density for each slice in pC/MeV.
        slice_emittance_x: X-plane normalized emittance for each slice in μm⋅rad.
        slice_emittance_y: Y-plane normalized emittance for each slice in μm⋅rad.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: charge density
    color1 = "tab:blue"
    ax1.set_xlabel("Energy (MeV)")
    ax1.set_ylabel("Charge Density (pC/MeV)", color=color1)
    line1 = ax1.plot(
        slice_energies, slice_charges, "o-", color=color1, label="Charge Density"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Right y-axis: emittance
    ax2 = ax1.twinx()
    color2 = "tab:red"
    color3 = "tab:green"
    ax2.set_ylabel("Normalized Slice Emittance (um⋅rad)", color=color2)
    line2 = ax2.plot(slice_energies, slice_emittance_x, "s-", color=color2, label="εx")
    line3 = ax2.plot(slice_energies, slice_emittance_y, "^-", color=color3, label="εy")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add legend
    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Calculate average energy slice width
    energy_range = (
        np.max(slice_energies) - np.min(slice_energies)
        if len(slice_energies) > 0
        else 0
    )
    avg_slice_width = (
        energy_range / len(slice_energies) if len(slice_energies) > 0 else 0
    )

    plt.title(f"Electron Beam Statistics vs Energy Slices (Avg Slice Width: {avg_slice_width:.2f} MeV)")
    plt.tight_layout()
    plt.show()


def print_analysis_summary(
    slice_energies: np.ndarray,
    slice_charges: np.ndarray,
    slice_emittance_x: np.ndarray,
    slice_emittance_y: np.ndarray,
) -> None:
    """Print summary statistics of the energy slice analysis.

    Args:
        slice_energies: Energy values for each slice in MeV.
        slice_charges: Charge density for each slice in pC/MeV.
        slice_emittance_x: X-plane normalized emittance for each slice in μm⋅rad.
        slice_emittance_y: Y-plane normalized emittance for each slice in μm⋅rad.
    """
    print("\n=== ENERGY SLICE ANALYSIS SUMMARY ===")
    print(f"Number of energy slices analyzed: {len(slice_energies)}")
    print(
        f"Energy range: {np.min(slice_energies):.1f} - {np.max(slice_energies):.1f} MeV"
    )
    print(f"Average charge density: {np.mean(slice_charges):.2f} pC/MeV")
    print(f"Peak charge density: {np.max(slice_charges):.2f} pC/MeV")
    print(f"Average εx: {np.mean(slice_emittance_x):.2f} um⋅rad")
    print(f"Average εy: {np.mean(slice_emittance_y):.2f} um⋅rad")


def process(args: argparse.Namespace) -> None:
    """
    Main function to perform electron beam slice emittance analysis.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    if args.diag_folder is None:
        raise RuntimeError("Must specify a beam data path with --diag-folder.")

    try:
        x, y, z, ux, uy, uz, w = load_and_filter_beam_data(
            args.diag_folder,
            args.species,
            args.iteration_number,
            args.min_z,
            args.max_z,
        )
    except (FileNotFoundError, Exception) as e:
        print(f"Error: {e}")
        return

    # Calculate energy slice statistics
    slice_energies, slice_charges, slice_emittance_x, slice_emittance_y = (
        calculate_energy_slices(x, y, z, ux, uy, uz, w)
    )

    # Create and display the plot
    create_energy_slice_plot(
        slice_energies, slice_charges, slice_emittance_x, slice_emittance_y
    )

    # Print summary statistics
    print_analysis_summary(
        slice_energies, slice_charges, slice_emittance_x, slice_emittance_y
    )


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
