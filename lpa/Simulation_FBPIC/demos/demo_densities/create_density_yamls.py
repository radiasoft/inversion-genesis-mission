"""Write example density-profile YAML configs for some concrete profile types."""

from __future__ import annotations

from pathlib import Path

from inversion_fbpic.lib.density_core import _DensityProfile
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

CFG_DIR = Path("cfg")

NOMINAL_DENSITY = 1e18 * 1e6  # m^-3
PARTICLE_COUNTS = dict(p_nz=2, p_nr=2, p_nt=2)
DEFAULT_R_MAX = 50e-6  # m


def _common_density_kwargs() -> dict:
    return dict(nominal_density=NOMINAL_DENSITY, **PARTICLE_COUNTS)


def build_example_profiles() -> dict[str, _DensityProfile]:
    """Instantiate some standalone concrete density profiles with representative parameters."""
    common = _common_density_kwargs()

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


def _build_modified_profiles(output_dir: Path) -> dict[str, _DensityProfile]:
    """Build modified density profiles that reference YAML files on disk."""
    modifier_path = output_dir / "matched_radial_modifier.yaml"

    MatchedRadialModifier(
        matched_density=NOMINAL_DENSITY,
        radial_extent=DEFAULT_R_MAX,
    ).to_yaml_file(modifier_path, include_nones=False)

    return {
        "modified_density_profile": ModifiedDensityProfile(
            base_density_profile=output_dir / "smooth_sine_flattop.yaml",
            modifiers=[output_dir / "matched_radial_modifier.yaml"],
        ),
    }


def create_density_yaml_files(output_dir: Path | None = None) -> dict[str, Path]:
    """Write a YAML file for each concrete density profile case."""
    output_dir = output_dir or CFG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, profile in build_example_profiles().items():
        path = output_dir / f"{name}.yaml"
        profile.to_yaml_file(path, include_nones=False)
        paths[name] = path

    for name, profile in _build_modified_profiles(output_dir).items():
        path = output_dir / f"{name}.yaml"
        profile.to_yaml_file(path, include_nones=False)
        paths[name] = path

    return paths


def main() -> None:
    paths = create_density_yaml_files()
    print(f"Wrote {len(paths)} YAML files to {CFG_DIR}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
