#!/usr/bin/env python3
"""Build a nozzle density cube from ANSYS lineout exports, one file per backing pressure.

Each raw file holds one ``(z, density)`` lineout per transverse position ``x``.
Every lineout is mirrored across ``z = 0``, scaled, resampled onto the z grid of the
first lineout, and stacked into ``density[z, x, pressure]``.

Example (conda env inv-fbpic):

    python -m fludat_proc.lineouts_to_density_hdf5 1.0e20 data/density_lineouts_raw/400_um \\
        --output data/density_lineouts/400_um.h5 --density-units cm^-3
"""

from __future__ import annotations

if __package__ in (
    None,
    "",
):  # run as a plain script, e.g. `python fludat_proc/plot_density.py`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_proc"

import argparse
from pathlib import Path

import numpy as np

from .ansys_lineouts import mirror_across_z0, parse_lineout_file, parse_x_position
from .density_cube import DensityCube, write_density_cube
from .filenames import parse_pressure_filename


def collect_pressure_files(input_path: Path) -> list[tuple[float, Path]]:
    """Return ``(pressure_bar, path)`` pairs for one raw file or one nozzle directory."""
    input_path = input_path.resolve()
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
        if not paths:
            raise ValueError(f"No lineout files found in {input_path}")
    else:
        raise FileNotFoundError(input_path)

    pressure_files = sorted(
        (parse_pressure_filename(path.name), path) for path in paths
    )
    pressures = [pressure for pressure, _ in pressure_files]
    if len(set(pressures)) != len(pressures):
        raise ValueError(f"Duplicate backing pressure file in {input_path}")
    return pressure_files


def build_density_cube(
    input_path: Path,
    *,
    density_scale: float,
    density_units: str,
    nozzle: str | None = None,
) -> DensityCube:
    """Parse, mirror, and stack one nozzle's lineouts into a density cube.

    All pressure files must provide the same set of x positions. The cube's z grid is
    the first lineout's grid restricted to the z range every lineout covers, so no
    lineout is extrapolated.
    """
    pressure_files = collect_pressure_files(input_path)

    profiles: dict[tuple[float, float], np.ndarray] = {}
    x_positions: list[float] | None = None
    for pressure, path in pressure_files:
        parsed_sections = sorted(
            (parse_x_position(label), data)
            for label, data in parse_lineout_file(path).items()
        )
        current_x_positions = [x_position for x_position, _ in parsed_sections]
        if len(set(current_x_positions)) != len(current_x_positions):
            raise ValueError(f"Duplicate x position in {path}")
        if x_positions is None:
            x_positions = current_x_positions
        elif x_positions != current_x_positions:
            raise ValueError(
                f"x positions in {path} do not match other pressures: "
                f"{x_positions} vs {current_x_positions}"
            )
        for x_position, data in parsed_sections:
            profile = mirror_across_z0(data)
            profile[:, 1] *= density_scale
            profiles[pressure, x_position] = profile

    if x_positions is None:
        raise ValueError(f"No lineout files found in {input_path}")
    pressures = np.array([pressure for pressure, _ in pressure_files], dtype=np.float64)
    z_min = max(float(profile[0, 0]) for profile in profiles.values())
    z_max = min(float(profile[-1, 0]) for profile in profiles.values())
    reference_z = profiles[pressures[0], x_positions[0]][:, 0]
    z = reference_z[(reference_z >= z_min) & (reference_z <= z_max)].copy()
    if z.size < 2:
        raise ValueError("Lineouts do not share a z range with at least two samples")

    density = np.empty((z.size, len(x_positions), pressures.size), dtype=np.float64)
    for ip, pressure in enumerate(pressures):
        for ix, x_position in enumerate(x_positions):
            profile_z, profile_density = profiles[pressure, x_position].T
            density[:, ix, ip] = np.interp(z, profile_z, profile_density)

    input_path = input_path.resolve()
    return DensityCube(
        nozzle=nozzle or (input_path.name if input_path.is_dir() else input_path.stem),
        z=z,
        x=np.array(x_positions, dtype=np.float64),
        pressure=pressures,
        density=density,
        density_units=density_units,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror ANSYS density lineouts across z=0 and stack them into one "
            "HDF5 density cube indexed by (z, x, backing pressure)."
        )
    )
    parser.add_argument(
        "density_scale",
        type=float,
        help="Factor to multiply raw file densities by before saving",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="One nozzle directory of <pressure>_bar.txt lineout files, or one such file",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output HDF5 path"
    )
    parser.add_argument(
        "--density-units",
        required=True,
        help="Units of the scaled density dataset, for example cm^-3 or m^-3",
    )
    parser.add_argument(
        "--nozzle",
        default=None,
        help="Nozzle name stored in the output (default: input directory name)",
    )
    args = parser.parse_args()

    cube = build_density_cube(
        args.input,
        density_scale=args.density_scale,
        density_units=args.density_units,
        nozzle=args.nozzle,
    )
    output = write_density_cube(
        args.output,
        cube,
        density_scale=args.density_scale,
        attributes={
            "source_format": "ANSYS lineouts",
            "source_z_symmetry": "z >= 0 lineouts mirrored across z=0",
        },
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
