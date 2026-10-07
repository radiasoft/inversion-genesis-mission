"""Write density configs with ``--format yaml|json|hdf5`` (default: YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.lib.density_modifiers import (
    MatchedRadialModifier,
    ModifiedDensityProfile,
)
from inversion_fbpic.lib.density_profiles import (
    AsymmetricSine,
    ExampleDensityProfile,
    GaussianPlusTriangle,
    GeneralizedGaussianPlusTriangle,
    SmoothSineFlattop,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR / "cfg"
FORMAT_SUFFIXES = {"yaml": ".yaml", "json": ".json", "hdf5": ".h5"}

NOMINAL_DENSITY = 1e18 * 1e6  # m^-3
PARTICLE_COUNTS = dict(p_nz=2, p_nr=2, p_nt=2)
DEFAULT_R_MAX = 50e-6  # m


def build_example_profiles() -> dict[str, _DensityProfile]:
    """Instantiate some standalone concrete density profiles with representative parameters."""
    common = dict(nominal_density=NOMINAL_DENSITY, **PARTICLE_COUNTS)

    return {
        "example_density_profile": ExampleDensityProfile(
            length=5e-3,
            start_position=0.0,
            **common,
        ),
        "asymmetric_sine": AsymmetricSine(
            peak_z0=2.5e-3,
            upramp_length=2.5e-3,
            downramp_length=1.5e-3,
            **common,
        ),
        "smooth_sine_flattop": SmoothSineFlattop(
            flattop_width=3e-3,
            upramp_length=0.5e-3,
            downramp_length=1e-3,
            offset_length=-0.5e-3,
            **common,
        ),
        "gaussian_plus_triangle": GaussianPlusTriangle(
            gauss_sigma=0.4e-3,
            gauss_z0=2.0e-3,
            tri_z0=2.035e-3,
            tri_left_width=0.5e-3,
            tri_right_width=0.07e-3,
            tri_height=1.2,
            **common,
        ),
        "generalized_gaussian_plus_triangle": GeneralizedGaussianPlusTriangle(
            gauss_peak=1.1,
            gauss_alpha=1.1e-3,
            gauss_beta=1.1,
            gauss_z0=2.3e-3,
            tri_z0=2.035e-3,
            tri_left_width=0.5e-3,
            tri_right_width=0.07e-3,
            tri_height=1.2,
            **common,
        ),
    }


def _write_config(
    profile: SerializableConfig, path: Path, serialization_format: str
) -> None:
    if serialization_format == "hdf5":
        profile.to_hdf5_file(path, include_nones=False, overwrite=True)
    elif serialization_format == "json":
        profile.to_json_file(path, include_nones=False)
    else:
        profile.to_yaml_file(path, include_nones=False)


def _build_modified_profiles(
    output_dir: Path, serialization_format: str = "yaml"
) -> dict[str, _DensityProfile]:
    """Build modified density profiles referencing configs in the selected format."""
    suffix = FORMAT_SUFFIXES[serialization_format]
    modifier_path = output_dir / f"matched_radial_modifier{suffix}"

    modifier = MatchedRadialModifier(
        matched_density=NOMINAL_DENSITY,
        radial_extent=DEFAULT_R_MAX,
    )
    _write_config(modifier, modifier_path, serialization_format)

    return {
        "modified_density_profile": ModifiedDensityProfile(
            base_density_profile=output_dir / f"smooth_sine_flattop{suffix}",
            modifiers=[modifier_path],
        ),
    }


def create_density_config_files(
    output_dir: Path | None = None, *, serialization_format: str = "yaml"
) -> dict[str, Path]:
    """Write each density case as YAML, JSON, or HDF5."""
    if serialization_format not in FORMAT_SUFFIXES:
        raise ValueError(f"Unsupported serialization format: {serialization_format!r}")
    output_dir = (output_dir or CFG_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = FORMAT_SUFFIXES[serialization_format]

    paths: dict[str, Path] = {}

    def write_profiles(profiles: dict[str, _DensityProfile]) -> None:
        for name, profile in profiles.items():
            path = output_dir / f"{name}{suffix}"
            _write_config(profile, path, serialization_format)
            paths[name] = path

    write_profiles(build_example_profiles())
    # Referenced configs must exist before constructing modified profiles.
    write_profiles(_build_modified_profiles(output_dir, serialization_format))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=tuple(FORMAT_SUFFIXES), default="yaml")
    args = parser.parse_args()
    paths = create_density_config_files(serialization_format=args.format)
    print(f"Wrote {len(paths)} {args.format.upper()} density files to {CFG_DIR}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
