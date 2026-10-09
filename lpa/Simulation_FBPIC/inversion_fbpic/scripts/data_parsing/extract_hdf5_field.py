#!/usr/bin/env python3
"""Extract and save numpy binary files of the given scalar field from HDF5 files.
Can be run either in the command line or through python interpreter

See below for examples using the command line options
    --field-name    : Specify the field name to extract from the HDF5 file.
    --set           : Specify the given path is instead a directory containing multiple diags folders
    --component     : Specify the component of the field to extract. Automatically converts 'x' to 'r', 'y' to 't', and 'z' to 'z' for cylindrical coordinates. Defaults to None (scalar field).

1. Command line - single simulation directory:
   python extract_hdf5_field.py <diags_folder> --field-name "rho_electrons"

2. Command line - multiple simulations in a set folder:
   python extract_hdf5_field.py --set <set_folder> --field-name "rho_electrons"

3. Python interpreter - single simulation (using global variables):
   Set DIAGS_FOLDER and FIELD_NAME variables, then run main()

4. Python interpreter - multiple simulations (using global variables):
   Set SET_FOLDER and FIELD_NAME variables, then run main()

Images are saved to
 - "analysis/<diags_folder_name>/<field_name>_<component>/" if component is not None
 - "analysis/<diags_folder_name>/<field_name>/" if component is None
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union, Literal

logger = logging.getLogger(__name__)

# Import hdf5_funcs either through virtual environment or local copy
try:
    from inversion_fbpic.utils.hdf5_funcs import extract_field_slice
except ImportError:
    from hdf5_funcs import extract_field_slice  # type: ignore


# Global variables for use in Python interpreter
DEFAULT_FIELD_NAME: str = "rho"
SET_FOLDER: Optional[str] = None
DIAGS_FOLDER: Optional[str] = None


def process_single_simulation(
    diags_folder: Union[str, Path],
    field_name: str = DEFAULT_FIELD_NAME,
    component: Literal["x", "y", "z"] | None = None,
    verbose: bool = False,
) -> Path | None:
    """Process a single simulation directory and extract the scalar field from the HDF5 files, saving the results as .npy files.

    Args:
        diags_folder (str|Path): Path to the diagnostics folder.
        field_name (str): Name of the scalar field to extract.
        component (Literal["x", "y", "z"] | None): Component of the field to extract. Automatically converts "x" to "r", "y" to "t", and "z" to "z" for cylindrical coordinates. Defaults to None (scalar field).
        verbose (bool): Whether to print verbose output. Defaults to False.

    Returns:
        Path to the output directory, or None if the operation failed.
    """
    diags_path = Path(diags_folder)
    if not (diags_path.exists() and diags_path.is_dir()):
        logger.warning("Path `%s` is not a directory, skipping.", diags_path)
        return None

    source = diags_path / "hdf5"
    if not source.is_dir():
        logger.warning("Path `%s` is not a valid hdf5 directory, skipping.", source)
        return None

    component_suffix = "_" + component if component is not None else ""
    destination = (
        diags_path.parent
        / Path("analysis")
        / diags_path.name
        / f"{field_name}{component_suffix}"
    )
    destination.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Processing single simulation: {diags_path}")
    extract_field_slice(
        source_folder=diags_path,
        destination_folder=destination,
        field_name=field_name,
        component=component,
        verbose=verbose,
    )
    return destination


def clean_single_simulation(
    diags_folder: Union[str, Path],
    field_name: str = DEFAULT_FIELD_NAME,
    component: Literal["x", "y", "z"] | None = None,
    file_extension: str = "npy",
    verbose: bool = False,
) -> None:
    """Clean up the diagnostics folder by removing the files created by `process_single_simulation` with the given extension.

    Args:
        diags_folder (str|Path): Path to the diagnostics folder.
        field_name (str): Name of the scalar field to extract.
        component (Literal["x", "y", "z"] | None): Component of the field to extract. Automatically converts "x" to "r", "y" to "t", and "z" to "z" for cylindrical coordinates. Defaults to None (scalar field).
        file_extension (str): Extension of the files to clean up. Default is "npy".
        verbose (bool): Whether to print verbose output. Default is False.
    """
    diags_path = Path(diags_folder)
    if diags_path.exists() and diags_path.is_dir():
        component_suffix = "_" + component if component is not None else ""
        files_dir = (
            diags_path.parent
            / "analysis"
            / diags_path.name
            / f"{field_name}{component_suffix}"
        )
        if files_dir.exists() and files_dir.is_dir():
            for file in files_dir.glob(f"*.{file_extension}"):
                file.unlink()
            if verbose:
                print(f"Cleaned up {files_dir}")
        else:
            if verbose:
                print(f"No {file_extension} files found in {files_dir} to clean up.")


def process_simulation_set(
    set_folder: Union[str, Path],
    field_name: str = DEFAULT_FIELD_NAME,
    component: Literal["x", "y", "z"] | None = None,
    verbose: bool = False,
) -> list[Path]:
    """Process multiple simulations in a set folder.

    Args:
        set_folder (str|Path): Path to the set folder containing multiple simulation directories.
        field_name (str): Name of the scalar field to extract.
        component (Literal["x", "y", "z"] | None): Component of the field to extract. Automatically converts "x" to "r", "y" to "t", and "z" to "z" for cylindrical coordinates. Defaults to None (scalar field).
        verbose (bool): Whether to print verbose output. Defaults to False.

    Returns:
        List of paths to the output directories.
    """
    set_path = Path(set_folder)
    if not (set_path.exists() and set_path.is_dir()):
        logger.error("Path `%s` is not a directory.", set_path)
        sys.exit(1)

    output_dirs: list[Path] = []
    if verbose:
        print(f"Processing simulation set: {set_folder}")

    for subfolder in set_path.iterdir():
        if not subfolder.is_dir():
            continue
        result = process_single_simulation(
            diags_folder=subfolder,
            field_name=field_name,
            component=component,
            verbose=verbose,
        )
        if result is not None:
            output_dirs.append(result)

    return output_dirs


def main() -> None:
    """Main function that handles system arguments and/or global variables"""
    # Check if running from command line
    if len(sys.argv) > 1:
        # Command line usage
        parser = argparse.ArgumentParser(
            description="Extract scalar field from FBPIC HDF5 files",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
            Examples:
              # Single simulation
              python extract_hdf5_field.py /test_simulation/diags
              
              # Multiple simulations
              python extract_hdf5_field.py --set /test_simulation_set/
            """,
        )

        # Create mutually exclusive group for single vs set processing
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "diags_folder", nargs="?", help="Path to single simulation diags folder"
        )
        group.add_argument(
            "--set",
            dest="set_folder",
            help="Path to folder containing multiple simulation diags folders",
        )

        parser.add_argument(
            "--field-name",
            default=DEFAULT_FIELD_NAME,
            help=f"Name of the scalar field to extract (default: {DEFAULT_FIELD_NAME})",
        )

        parser.add_argument(
            "--component",
            default=None,
            choices=["x", "y", "z"],
            type=str.lower,
            help="Component of the field to extract. Automatically converts 'x' to 'r', 'y' to 't', and 'z' to 'z' for cylindrical coordinates. Defaults to None (scalar field).",
        )

        args = parser.parse_args()

        if args.set_folder:
            # Use case 2: Command line - multiple simulations
            process_simulation_set(args.set_folder, args.field_name, args.component)
        else:
            # Use case 1: Command line - single simulation
            if not args.diags_folder:
                parser.error("diags_folder is required when not using --set")
            process_single_simulation(
                args.diags_folder, args.field_name, args.component
            )

    else:
        # Python interpreter usage - check global variables
        if SET_FOLDER is not None:
            # Use case 4: Python interpreter - multiple simulations
            print(f"Using global SET_FOLDER: {SET_FOLDER}")
            print(f"Using global FIELD_NAME: {DEFAULT_FIELD_NAME}")
            process_simulation_set(SET_FOLDER, DEFAULT_FIELD_NAME)
        if DIAGS_FOLDER is not None:
            # Use case 3: Python interpreter - single simulation
            print(f"Using global DIAGS_FOLDER: {DIAGS_FOLDER}")
            print(f"Using global FIELD_NAME: {DEFAULT_FIELD_NAME}")
            process_single_simulation(DIAGS_FOLDER, DEFAULT_FIELD_NAME)
        if SET_FOLDER is None and DIAGS_FOLDER is None:
            print("No arguments provided and no global variables set.")
            print("Please either:")
            print("1. Provide command line arguments")
            print(
                "2. Set DIAGS_FOLDER (and optionally FIELD_NAME) for single simulation"
            )
            print("3. Set SET_FOLDER for multiple simulations")
            sys.exit(1)


if __name__ == "__main__":
    main()
