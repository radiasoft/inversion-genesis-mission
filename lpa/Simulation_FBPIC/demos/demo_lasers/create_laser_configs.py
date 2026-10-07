"""Write laser configs with ``--format yaml|json|hdf5`` (default: YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

from scipy.constants import pi

from inversion_fbpic.lib.laser import GaussianLaserPulse, _LaserPulse

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR / "cfg"
FORMAT_SUFFIXES = {"yaml": ".yaml", "json": ".json", "hdf5": ".h5"}

COMMON_LASER_KWARGS: dict = dict(
    energy=5.0,
    z0=-30e-6,
    wavelength=800e-9,
    tau_fwhm=38e-15,
    cep=0.0,
    waist=28e-6,
    focal_position=3e-3,
)


def build_example_profiles() -> dict[str, _LaserPulse]:
    """Instantiate each example laser pulse variant."""
    profiles = {
        name: GaussianLaserPulse(**COMMON_LASER_KWARGS, polarization=polarization)
        for name, polarization in {
            "gaussian_linear": 0.0,
            "gaussian_linear_45deg": pi / 4,
            "gaussian_left_circular": "left",
            "gaussian_elliptical": [pi / 8, pi / 2],
        }.items()
    }
    profiles["gaussian_antenna"] = GaussianLaserPulse(
        **COMMON_LASER_KWARGS,
        polarization=0.0,
        method="antenna",
        z0_antenna=0.0,
        v_antenna=0.0,
    )
    return profiles


def create_laser_config_files(
    output_dir: Path | None = None, *, serialization_format: str = "yaml"
) -> dict[str, Path]:
    """Write each laser case as YAML, JSON, or HDF5."""
    if serialization_format not in FORMAT_SUFFIXES:
        raise ValueError(f"Unsupported serialization format: {serialization_format!r}")
    output_dir = output_dir or CFG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = FORMAT_SUFFIXES[serialization_format]

    paths: dict[str, Path] = {}
    for name, pulse in build_example_profiles().items():
        path = output_dir / f"{name}{suffix}"
        if serialization_format == "hdf5":
            pulse.to_hdf5_file(path, include_nones=False, overwrite=True)
        elif serialization_format == "json":
            pulse.to_json_file(path, include_nones=False)
        else:
            pulse.to_yaml_file(path, include_nones=False)
        paths[name] = path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=tuple(FORMAT_SUFFIXES), default="yaml")
    args = parser.parse_args()
    paths = create_laser_config_files(serialization_format=args.format)
    print(f"Wrote {len(paths)} {args.format.upper()} files to {CFG_DIR}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
