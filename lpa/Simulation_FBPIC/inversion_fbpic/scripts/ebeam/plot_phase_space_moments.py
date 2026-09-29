#!/usr/bin/env python3
"""Plot cropped phase-space moments directly from one FBPIC openPMD HDF5 file.

Example:

    plot-phase-space-moments /path/to/lab_diags/hdf5/data00000049.h5 \
        --species nitrogen_electrons --uz-min 30 --all \
        --output phase_space_moments.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from inversion_fbpic.utils.distributions import (
    MOMENTS,
    OFF,
    SPLINE,
    crop_central_particles,
    load_openpmd_particles,
    select_by_uz,
)

LONGITUDINAL_MODES = {"off": OFF, "moments": MOMENTS, "spline": SPLINE}
from inversion_fbpic.utils.distributions_plotting import (
    plot_all_phase_space_moments,
    plot_phase_space_moments,
)


def main() -> None:
    """Load one openPMD HDF5 particle diagnostic and save the shared figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("--species", required=True)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("phase_space_moments.png")
    )
    parser.add_argument(
        "--uz-min", type=float, default=30.0, help="Minimum retained uz"
    )
    parser.add_argument(
        "--crop-central-fraction",
        type=float,
        default=0.95,
        help="Retain this central fraction on each phase-space axis before plotting",
    )
    parser.add_argument(
        "--central-fraction",
        type=float,
        default=0.999,
        help="Central fraction displayed in each plot axis",
    )
    parser.add_argument("--bins", type=int, default=150)
    parser.add_argument(
        "--all", action="store_true", help="Plot all unique phase-space projections"
    )
    parser.add_argument(
        "--gram-charlier", type=int, choices=(2, 3, 4, 5), default=None
    )
    parser.add_argument(
        "--longitudinal-mode",
        choices=LONGITUDINAL_MODES,
        default="spline",
        help="Longitudinal representation used by the moment model (default: spline)",
    )
    parser.add_argument(
        "--longitudinal-bins",
        type=int,
        default=4,
        help="Fixed number of spline longitudinal bins (default: 4)",
    )
    args = parser.parse_args()
    particles, weights = load_openpmd_particles(
        args.h5_file, args.species, args.iteration
    )
    particles, weights = select_by_uz(particles, weights, uz_min=args.uz_min)
    particles, weights = crop_central_particles(
        particles, weights, central_fraction=args.crop_central_fraction
    )
    plotter = plot_all_phase_space_moments if args.all else plot_phase_space_moments
    figure = plotter(
        particles,
        weights,
        uz_min=None,
        central_fraction=args.central_fraction,
        bins=args.bins,
        title=f"Cropped particle phase space: {args.h5_file}",
        gram_charlier_order=args.gram_charlier,
        longitudinal_mode=LONGITUDINAL_MODES[args.longitudinal_mode],
        longitudinal_bins=args.longitudinal_bins,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()