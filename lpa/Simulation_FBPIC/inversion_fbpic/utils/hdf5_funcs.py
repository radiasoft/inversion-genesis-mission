"""
hdf5_funcs.py

Utility functions for extracting, filtering, and analyzing FBPIC data from HDF5 files
  produced by FBPIC.  Includes routines for:
- Extracting electron species data with optional filtering by longitudinal position and energy.
- Extracting 2D scalar field information from .h5 files and into a binary .npy file
- Removing field data from HDF5 files to reduce file size.
- Repacking HDF5 files to reclaim disk space after deletions.
- Printing and analyzing the structure and size of HDF5 file contents.

See "extract_ebeam_from_hdf5.py" for usage on ebeam extraction, and "extract_field_slice" for usage
  on scalar field extraction.
"""

from typing import Optional, Union, Literal
from pathlib import Path
import h5py
import shutil
import numpy as np
from scipy.constants import m_e, c, e
import tempfile


# Methods to extract high energy electrons from a full .h5 file and into a trimmed-down .h5 file


def ebeam_extract_particles_only(
    source: Path,
    destination: Path,
    species_name: str,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
    min_uz: Optional[float] = None,
    max_uz: Optional[float] = None,
    last_dump_only: bool = False,
) -> None:
    """
    Extracts electron beam particles only, removing fields from the output.  Electron beam particles are filtered
    from the rest of the particles by selecting electrons within a longitudinal phase space window, defined by
    "min_z", "max_z", "min_uz", "max_uz".  Typically, specifying a "min_uz" is sufficient to trim out all low energy
    plasma electrons.

    This function processes each file individually, extracting filtered electrons and removing fields,
    then repacking each file and moving to a separate directory determined by the "destination" argument

    Args:
        source (Path): Path to a file or directory containing HDF5 files.
        destination (Path): Directory to save the extracted files.
        species_name (str): Name of the electron species in the source file(s).
        min_z (Optional[float], optional): Minimum z position. Only electrons with z >= min_z are extracted. Defaults to None.
        max_z (Optional[float], optional): Maximum z position. Only electrons with z <= max_z are extracted. Defaults to None.
        min_uz (Optional[float], optional): Minimum kinetic energy (MeV). Only electrons with KE >= min_uz are extracted. Defaults to None.
        max_uz (Optional[float], optional): Maximum kinetic energy (MeV). Only electrons with KE <= max_uz are extracted. Defaults to None.
        last_dump_only (bool, optional): If True and ``source`` is a directory, only the last file in sorted
            ``*.h5`` order (final dump for typical FBPIC naming) is processed. Ignored when ``source`` is a single file.

    Returns:
        None

    Raises:
        ValueError: If the source is not a file or directory.
    """
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def process_single_file(file_path: Path) -> None:
        """Process a single HDF5 file: extract electrons, remove fields, and repack."""
        print(f"Processing file {file_path}")

        with h5py.File(file_path, "r") as f:
            # Find the iteration group (usually 'data/#####')
            data_group = [k for k in f["data"].keys()][0]
            group_path = f"data/{data_group}/particles/{species_name}"

            # Load arrays
            z = f[f"{group_path}/position/z"][:]
            uz = f[f"{group_path}/momentum/z"][:]

            # Calculate kinetic energy in MeV
            elec_p = np.array([uz])
            elec_E = np.sqrt((elec_p * c) ** 2 + (m_e * c**2) ** 2)
            elec_KE = elec_E - m_e * c**2
            elec_KE_MeV = elec_KE / (1e6 * e)

            # Build mask
            mask = np.ones(z.shape, dtype=bool)
            if min_z is not None:
                mask &= z >= min_z
            if max_z is not None:
                mask &= z <= max_z
            if min_uz is not None:
                mask &= elec_KE_MeV.squeeze() >= min_uz
            if max_uz is not None:
                mask &= elec_KE_MeV.squeeze() <= max_uz

            if not np.any(mask):
                print(f" No electrons found in {file_path} matching the criteria.")
                return

            # Prepare output file path
            out_file = destination / file_path.name

            # Copy the whole file structure, then overwrite the electron datasets with filtered data
            shutil.copy(file_path, out_file)
            with h5py.File(out_file, "a") as fout:
                for field in [
                    "position/x",
                    "position/y",
                    "position/z",
                    "momentum/x",
                    "momentum/y",
                    "momentum/z",
                    "weighting",
                ]:
                    dpath = f"{group_path}/{field}"
                    if dpath in fout:
                        data = fout[dpath][:]
                        attrs = dict(fout[dpath].attrs)
                        del fout[dpath]
                        dset = fout.create_dataset(dpath, data=data[mask])
                        for k, v in attrs.items():
                            dset.attrs[k] = v
                fout[group_path].attrs["numParticles"] = np.sum(mask)

            # Now remove the fields group and repack
            print(
                f"File size before removing fields: {out_file.stat().st_size / (1024 * 1024):.2f} MB"
            )
            with h5py.File(out_file, "a") as fout:
                if "data" in fout:
                    for data_group in fout["data"]:
                        if "fields" in fout["data"][data_group]:
                            print(f"Removing fields from {data_group}")
                            del fout["data"][data_group]["fields"]
                        else:
                            print(f"No fields found in {data_group}")
                fout.flush()

            # Repack the file to reclaim disk space
            repack_hdf5_file(out_file)

    # Process single file or all files in folder
    if source.is_file():
        process_single_file(source)
    elif source.is_dir():
        files = sorted(source.glob("*.h5"))
        if last_dump_only and files:
            files = [files[-1]]
        for file_path in files:
            process_single_file(file_path)
    else:
        raise ValueError(f"Source {source} is not a file or directory.")


def repack_hdf5_file(file_path: Path) -> None:
    """
    Repacks an HDF5 file to reclaim space from deleted data.  Otherwise, the HDF5 file will have the same size
    before and after removing data.

    Args:
        file_path (Path): Path to the HDF5 file to repack.

    Returns:
        None
    """
    print(f"Repacking {file_path} to reclaim disk space...")
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    with h5py.File(file_path, "r") as src:
        with h5py.File(tmp_path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for key in src.keys():
                if key == "data":
                    dst.create_group("data")
                    for data_group in src["data"]:
                        dst["data"].create_group(data_group)
                        for attr_key, attr_value in src["data"][
                            data_group
                        ].attrs.items():
                            dst["data"][data_group].attrs[attr_key] = attr_value
                        for subkey in src["data"][data_group].keys():
                            if subkey != "fields":
                                src["data"][data_group].copy(
                                    subkey, dst["data"][data_group]
                                )
                else:
                    src.copy(key, dst)
    shutil.move(tmp_path, file_path)
    print(
        f"Repacking complete. New file size: {file_path.stat().st_size / (1024 * 1024):.2f} MB"
    )


# Method to extract scalar fields from an .h5 file and export to binary .npy file


def extract_field_slice(
    source_folder: Union[str, Path],
    destination_folder: Union[str, Path],
    field_name: str = "rho_electrons",
    component: Literal["x", "y", "z"] | None = None,
    verbose: bool = False,
):
    """
    Extracts a scalar field from all HDF5 files in a diagnostics folder and saves as .npy files.

    Args:
        source_folder (str|Path): Path to the diagnostics folder containing HDF5 files.
        destination_folder (str|Path): Path to the output directory for .npy files.
        field_name (str): Name of the scalar field to extract.
        component (Literal["x", "y", "z"] | None): Component of the field to extract. Automatically converts "x" to "r", "y" to "t", and "z" to "z" for cylindrical coordinates. Defaults to None (scalar field).
        verbose (bool): Whether to print verbose output. Defaults to False.

    Returns:
        None
    """
    diags_folder = Path(source_folder)
    hdf5_dir = diags_folder / "hdf5"
    if not hdf5_dir.exists():
        print(f"HDF5 directory not found: {hdf5_dir}")
        return

    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    cyl_component = None
    if component is not None:
        if component == "x":
            cyl_component = "r"
        elif component == "y":
            cyl_component = "t"
        elif component == "z":
            cyl_component = "z"
        else:
            raise ValueError(f"Invalid component: {component}")

    # Find all HDF5 files
    h5_files = sorted(hdf5_dir.glob("*.h5"))
    if not h5_files:
        print(f"No HDF5 files found in {hdf5_dir}")
        return
    for h5file in h5_files:
        with h5py.File(h5file, "r") as f:
            # Find the iteration group (usually one per file)
            data_group = list(f["data"].keys())[0]
            group_path = f"data/{data_group}/fields/{field_name}"
            component_suffix = "/" + cyl_component if cyl_component is not None else ""
            group_component_path = f"{group_path}{component_suffix}"
            if group_component_path not in f:
                print(f"{group_component_path} not found in {h5file}")
                continue
            field_data = f[group_component_path][:]
            # Always select the first slice for plotting (FBPIC: 0=physical mode)
            if field_data.ndim == 3:
                field_data = field_data[0, :, :]
            # Try to reconstruct axes from gridGlobalOffset, gridSpacing, and axisLabels
            try:
                attrs = f[group_path].attrs
                axis_labels = [
                    label.decode() if isinstance(label, bytes) else label
                    for label in attrs.get("axisLabels", [])
                ]
                offset = attrs.get("gridGlobalOffset")
                spacing = attrs.get("gridSpacing")
                if offset is not None and spacing is not None and len(axis_labels) == 2:
                    shape = field_data.shape
                    axes = []
                    for i in range(2):
                        axes.append(offset[i] + np.arange(shape[i]) * spacing[i])
                    # Assign axes based on axisLabels
                    if axis_labels[0] == "r" and axis_labels[1] == "z":
                        y = axes[0]
                        x = axes[1]
                    elif axis_labels[0] == "z" and axis_labels[1] == "r":
                        x = axes[0]
                        y = axes[1]
                    else:
                        # Unexpected order, default to previous logic
                        x = axes[1]
                        y = axes[0]
                    physical_units = True
                else:
                    raise KeyError
            except Exception:
                # Fallback to previous logic (datasets, attributes, or pixel indices)
                try:
                    z_path = f"data/{data_group}/fields/z"
                    r_path = f"data/{data_group}/fields/r"
                    if z_path in f and r_path in f:
                        x = f[z_path][:]
                        y = f[r_path][:]
                        physical_units = True
                    else:
                        raise KeyError
                except Exception:
                    try:
                        axis_x = f[group_path].attrs.get("axis1")
                        axis_y = f[group_path].attrs.get("axis2")
                        if axis_x is not None and axis_y is not None:
                            x = np.array(axis_x)
                            y = np.array(axis_y)
                            physical_units = True
                        else:
                            raise KeyError
                    except Exception:
                        try:
                            x = f[f"data/{data_group}/fields"].attrs["axis1"]
                            y = f[f"data/{data_group}/fields"].attrs["axis2"]
                            physical_units = True
                        except Exception:
                            x = np.arange(field_data.shape[1])
                            y = np.arange(field_data.shape[0])
                            physical_units = False

            # Print the physical window extent
            if physical_units and verbose:
                print(
                    f"{h5file.name}: z range = [{x[0]:.3e}, {x[-1]:.3e}] m, r range = [{y[0]:.3e}, {y[-1]:.3e}] m"
                )
            elif verbose:
                print(f"{h5file.name}: z and r axes are in pixel units.")

            # Mirror the image along r=0 (vertical axis)
            field_mirrored = np.vstack([np.flipud(field_data), field_data])
            y_mirrored = np.concatenate([-y[::-1], y])

            # Add extra row and column for axis values
            arr = np.zeros(
                (field_mirrored.shape[0] + 1, field_mirrored.shape[1] + 1),
                dtype=field_mirrored.dtype,
            )
            arr[:-1, 1:] = field_mirrored
            arr[-1, 1:] = x  # x-axis (z) values in last row, skip [0]
            arr[:-1, 0] = y_mirrored  # y-axis (r) values in first column, skip [-1]
            arr[-1, 0] = np.nan  # lower-left corner

            # Save as .npy file
            component_suffix = "_" + component if component is not None else ""
            out_path = (
                Path(destination_folder)
                / f"{field_name}{component_suffix}_{data_group}.npy"
            )
            np.save(out_path, arr)
            if verbose:
                print(f"Saved {out_path}")


# Debugging methods used to print h5 file organization and size


def print_hdf5_structure(filename: Path) -> None:
    """
    Debugging method to recursively print the dataset structure of an HDF5 file.

    Args:
        filename (Path): Path to the HDF5 file to inspect.

    Returns:
        None
    """
    with h5py.File(filename, "r") as f:

        def print_attrs(name: str, obj: Union[h5py.Group, h5py.Dataset]) -> None:
            print(name)

        f.visititems(print_attrs)


def analyze_hdf5_sizes(filename: Path) -> None:
    """
    Debugging method to print the size of different components in an HDF5 file.

    Args:
        filename (Path): Path to the HDF5 file to analyze.

    Returns:
        None
    """
    with h5py.File(filename, "r") as f:

        def get_group_size(name: str, obj: Union[h5py.Group, h5py.Dataset]) -> None:
            if isinstance(obj, h5py.Dataset):
                size_mb = obj.size * obj.dtype.itemsize / (1024 * 1024)
                print(f"{name}: {size_mb:.3f} MB")
            elif isinstance(obj, h5py.Group):
                print(f"{name}: (group)")

        print(f"Analyzing file: {filename}")
        print("Dataset sizes:")
        f.visititems(get_group_size)
