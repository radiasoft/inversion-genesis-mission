"""
Uses the `analysis` module to perform basic ebeam analysis on a specified diagnostics directory.  Generates
  a plot with the current profile, z-energy phase space, x-x' phase space, and y-y' phase space.

Usage:
    First, it is recommended to extract the ebeam from the full .h5 output file.  Run either
      "extract_ebeam_from_hdf5.py" or "extract_ebeam_from_set.py" and the extracted ebeam will
      be located in trimmed-down h5 files in a "ebeam" directory

    Command-line usage:
        python plot_ebeam_analysis.py [OPTIONS]

    Optional Arguments:
        -d, --diag-folder PATH
            Path to the ebeam .h5 files directory. If not provided, uses the default
            value defined in the script.

        -i, --iteration-number INT
            Which dump to load from DIAG_FOLDER. Set to "-1" for final dump.
            Default: -1

        -s, --species STR
            Which species to load from DIAG_FOLDER. Typically "electrons".
            Default: "n_elec"

        --min-uz FLOAT|None
            Minimum uz value for particle selection. If not provided, uses the default
            value defined in the script. Use 'None' or 'null' to explicitly set to None.

        --max-uz FLOAT|None
            Maximum uz value for particle selection. If not provided, uses the default
            value defined in the script. Use 'None' or 'null' to explicitly set to None.

        --min-z FLOAT|None
            Minimum z value for particle selection. If not provided, uses the default
            value defined in the script. Use 'None' or 'null' to explicitly set to None.

        --max-z FLOAT|None
            Maximum z value for particle selection. If not provided, uses the default
            value defined in the script. Use 'None' or 'null' to explicitly set to None.

    Examples:
        # Use all defaults
        python plot_ebeam_analysis.py

        # Specify a different diagnostics folder
        python plot_ebeam_analysis.py -d /path/to/hdf5

        # Load a specific iteration and species
        python plot_ebeam_analysis.py -i 100 -s electrons

        # Specify particle selection criteria
        python plot_ebeam_analysis.py --min-uz 200 --min-z 45018e-6

        # Explicitly set min-z to None (override default)
        python plot_ebeam_analysis.py --min-z None

"""

from pathlib import Path
import argparse
import inversion_fbpic.utils.analysis as an
from typing import Optional
from inversion_fbpic.utils.argparse_utils import float_or_none, build_selection_dict

DEFAULT_DIAG_FOLDER: Optional[Path] = None  # Path to ebeam .h5 files

DEFAULT_MIN_UZ: Optional[float] = None  # Minimum uz value for particle selection
DEFAULT_MAX_UZ: Optional[float] = None  # Maximum uz value for particle selection
DEFAULT_MIN_Z: Optional[float] = None  # Minimum z value for particle selection
DEFAULT_MAX_Z: Optional[float] = None  # Maximum z value for particle selection

DEFAULT_ITERATION_NUMBER: int = (
    -1
)  # Which dump to load from DIAG_FOLDER.  Set to "-1" for final dump

DEFAULT_SPECIES: str = (
    "n_elec"  # Which species to load from DIAG_FOLDER.  Typically "electrons"
)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Perform basic ebeam analysis and plotting using the analysis module."
    )

    parser.add_argument(
        "-d",
        "--diag-folder",
        type=Path,
        default=DEFAULT_DIAG_FOLDER,
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
        "--min-uz",
        type=float_or_none,
        default=DEFAULT_MIN_UZ,
        help="Minimum uz value for particle selection. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--max-uz",
        type=float_or_none,
        default=DEFAULT_MAX_UZ,
        help="Maximum uz value for particle selection. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--min-z",
        type=float_or_none,
        default=DEFAULT_MIN_Z,
        help="Minimum z value for particle selection. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--max-z",
        type=float_or_none,
        default=DEFAULT_MAX_Z,
        help="Maximum z value for particle selection. Use 'None' or 'null' to explicitly set to None",
    )

    return parser.parse_args()


def process(args: argparse.Namespace) -> None:
    """
    Perform basic ebeam analysis and plotting using the analysis module.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    if args.diag_folder is None:
        raise RuntimeError(
            "Must specify path to the ebeam .h5 files directory using: -d, --diag-folder"
        )

    # Build selection dictionary from individual parameters
    selection = build_selection_dict(
        min_uz=args.min_uz,
        max_uz=args.max_uz,
        min_z=args.min_z,
        max_z=args.max_z,
    )

    # Load data
    try:
        x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
            args.diag_folder,
            args.species,
            iteration=args.iteration_number,
            select=selection,
        )
    except FileNotFoundError:
        print(f"Error: Could not find data in {args.diag_folder}")
        return
    except Exception as e:
        print(f"Error loading beam data: {e}")
        return

    # Analyze
    analysis = an.analyze_beam(x, y, z, ux, uy, uz, w, q, bins=200)
    an.print_beam_summary(analysis)

    # Plot
    an.plot_beam_analysis(x, y, z, ux, uy, uz, w, analysis)

def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
