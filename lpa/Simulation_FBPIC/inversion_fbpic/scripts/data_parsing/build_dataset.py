#!/usr/bin/env python3
"""Build a compact JSON dataset from FBPIC simulation diagnostics.

Each ``sim_*`` directory beneath the supplied root must contain ``input.ini``
with a ``[PhysicalParameters]`` section and particle diagnostics under
``lab_diags/hdf5``. The output records the physical inputs and a phase-space
descriptor for the final diagnostic in each simulation.

Sample usage:
build-dataset /path/to/raw_runs --output /path/to/dataset.json

"""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

from inversion_fbpic.utils.distributions import (
    MOMENTS,
    OFF,
    SPLINE,
    compute_moment_descriptor,
    crop_central_particles,
    load_openpmd_particles,
    select_by_uz,
)

LONGITUDINAL_MODES = {"off": OFF, "moments": MOMENTS, "spline": SPLINE}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sample raw FBPIC runs into an input/output JSON dataset."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing sim_*/input.ini folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--uz-min",
        type=float,
        default=30.0,
        help="Retain particles with uz at or above this value (default: 30)",
    )
    parser.add_argument(
        "--species",
        default="nitrogen_electrons",
        help="openPMD particle species used to calculate outputs",
    )
    parser.add_argument(
        "--central-fraction",
        type=float,
        default=0.95,
        help=(
            "Central fraction retained independently on each phase-space axis "
            "(default: 0.95)"
        ),
    )
    parser.add_argument(
        "--longitudinal-mode",
        choices=LONGITUDINAL_MODES,
        default="spline",
        help="Longitudinal descriptor representation (default: spline)",
    )
    parser.add_argument(
        "--longitudinal-bins",
        type=int,
        default=4,
        help="Fixed number of spline longitudinal bins (default: 4)",
    )
    return parser.parse_args()


def parse_ini_value(value: str) -> bool | int | float | str:
    """Parse scalar INI values while preserving non-numeric configuration strings."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        numeric = float(value)
    except ValueError:
        return value
    return int(numeric) if numeric.is_integer() else numeric


def read_input(path: Path) -> dict[str, dict[str, bool | int | float | str]]:
    """Read every INI section and convert scalar values where possible."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    with path.open(encoding="utf-8") as input_file:
        parser.read_file(input_file)
    return {
        section: {key: parse_ini_value(value) for key, value in parser[section].items()}
        for section in parser.sections()
    }


def build_output(
    diagnostic_file: Path,
    species: str,
    uz_min: float,
    central_fraction: float,
    longitudinal_mode: int,
    longitudinal_bins: int,
) -> dict[str, float]:
    """Calculate the shared 33-scalar descriptor from an openPMD diagnostic."""
    particles, weights = load_openpmd_particles(diagnostic_file, species)
    particles, weights = select_by_uz(particles, weights, uz_min=uz_min)
    particles, weights = crop_central_particles(
        particles, weights, central_fraction=central_fraction
    )
    return compute_moment_descriptor(
        particles,
        weights,
        longitudinal_mode=longitudinal_mode,
        longitudinal_bins=longitudinal_bins,
    )


def main() -> None:
    args = parse_args()
    longitudinal_mode = LONGITUDINAL_MODES[args.longitudinal_mode]
    records: dict[str, object] = {
        "metadata": {
            "species": args.species,
            "uz_min": args.uz_min,
            "central_fraction": args.central_fraction,
            "longitudinal_mode": args.longitudinal_mode,
            "longitudinal_bins": args.longitudinal_bins,
        },
        "runs": {},
    }
    runs = records["runs"]
    assert isinstance(runs, dict)
    for run_dir in sorted(path for path in args.root.glob("sim_*") if path.is_dir()):
        diagnostic_files = sorted((run_dir / "lab_diags" / "hdf5").glob("data*.h5"))
        if not diagnostic_files:
            raise ValueError(f"{run_dir}: no particle diagnostics found")

        input_sections = read_input(run_dir / "input.ini")
        if "PhysicalParameters" not in input_sections:
            raise ValueError(f"{run_dir}: missing [PhysicalParameters] in input.ini")
        runs[run_dir.name] = {
            "input": input_sections["PhysicalParameters"],
            "output": build_output(
                diagnostic_files[-1],
                args.species,
                args.uz_min,
                args.central_fraction,
                longitudinal_mode,
                args.longitudinal_bins,
            ),
        }

    if not runs:
        raise SystemExit(f"No sim_* folders found below {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    print(f"Wrote {args.output} with {len(records)} simulations")


if __name__ == "__main__":
    main()