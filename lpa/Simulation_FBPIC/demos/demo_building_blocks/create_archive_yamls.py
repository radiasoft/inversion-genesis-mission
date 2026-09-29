"""Generate example YAML building blocks in ``cfg_archive``.

The archive holds density profiles, laser pulses, and simulation
hyperparameters. Copy or symlink the pieces you want into ``cfg_active`` to
assemble a simulation — no top-level ``Simulation`` YAML is written here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from inversion_fbpic.lib import density_modifiers, density_profiles, laser, simulation

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_ARCHIVE = SCRIPT_DIR / "cfg_archive"
CFG_ACTIVE = SCRIPT_DIR / "cfg_active"

NOMINAL_DENSITY = 1e18 * 1e6  # m^-3
DOPED_FRACTION = 0.1
Z0_DOPED = 1.0e-3  # m

# Coarse settings for quick demo runs.
P_NZ, P_NR, P_NT = 1, 1, 2
NZ, NR, NM = 256, 256, 2

DEFAULT_ACTIVE_TOP_LEVEL = (
    "simulation_hyperparameters.yaml",
    "doped_asymmetric_sine.yaml",
    "gaussian_linear.yaml",
)
DEFAULT_ACTIVE_DEPS = (
    "smooth_sine_flattop.yaml",
    "matched_radial_modifier.yaml",
)


def create_archive_yamls(output_dir: Path | None = None) -> dict[str, Path]:
    """Write example density, laser, and hyperparameter YAML files."""
    output_dir = (output_dir or CFG_ARCHIVE).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    smooth_sine_flattop = density_profiles.SmoothSineFlattop(
        nominal_density=NOMINAL_DENSITY,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        species="H",
        ionization=0,
        elec_name="electrons",
        elec_select={"uz": [10.0, None]},
        flattop_width=6e-3,
        upramp_length=0.5e-3,
        downramp_length=0.5e-3,
    )
    paths["smooth_sine_flattop"] = smooth_sine_flattop.to_yaml_file(
        output_dir / "smooth_sine_flattop.yaml", include_nones=False
    )

    matched_radial_modifier = density_modifiers.MatchedRadialModifier(
        matched_density=NOMINAL_DENSITY,
        radial_extent=100e-6,
    )
    paths["matched_radial_modifier"] = matched_radial_modifier.to_yaml_file(
        output_dir / "matched_radial_modifier.yaml", include_nones=False
    )

    modified_density_profile = density_modifiers.ModifiedDensityProfile(
        base_density_profile=output_dir / "smooth_sine_flattop.yaml",
        modifiers=[output_dir / "matched_radial_modifier.yaml"],
    )
    paths["modified_density_profile"] = modified_density_profile.to_yaml_file(
        output_dir / "modified_density_profile.yaml", include_nones=False
    )

    doped_asymmetric_sine = density_profiles.AsymmetricSine(
        nominal_density=NOMINAL_DENSITY * DOPED_FRACTION,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        species="N",
        ionization=0,
        elec_name="n2_electrons",
        elec_select={"uz": [10.0, None]},
        peak_z0=Z0_DOPED,
        upramp_length=250e-6,
        downramp_length=250e-6,
    )
    paths["doped_asymmetric_sine"] = doped_asymmetric_sine.to_yaml_file(
        output_dir / "doped_asymmetric_sine.yaml", include_nones=False
    )

    example_density_profile = density_profiles.ExampleDensityProfile(
        nominal_density=NOMINAL_DENSITY,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        length=5e-3,
        start_position=0.0,
    )
    paths["example_density_profile"] = example_density_profile.to_yaml_file(
        output_dir / "example_density_profile.yaml", include_nones=False
    )

    common_laser_kwargs = dict(
        energy=5.0,
        z0=-40e-6,
        wavelength=800e-9,
        tau_fwhm=30e-15,
        cep=0.0,
        waist=30e-6,
        focal_position=3e-3,
    )
    gaussian_linear = laser.GaussianLaserPulse(
        **common_laser_kwargs,
        polarization=0.0,
    )
    paths["gaussian_linear"] = gaussian_linear.to_yaml_file(
        output_dir / "gaussian_linear.yaml", include_nones=False
    )

    gaussian_left_circular = laser.GaussianLaserPulse(
        **common_laser_kwargs,
        polarization="left",
    )
    paths["gaussian_left_circular"] = gaussian_left_circular.to_yaml_file(
        output_dir / "gaussian_left_circular.yaml", include_nones=False
    )

    simulation_hyperparameters = simulation.SimulationHyperparameters(
        zmin=-100e-6,
        zmax=0.0,
        rmax=100e-6,
        nz=NZ,
        nr=NR,
        nm=NM,
        use_mpi=False,
        number_dumps=5,
        gamma_boost=4.0,
        field_diagnostics=["rho", "E", "B"],
        save_directory="diags",
        n_order_mpi=32,
        r_boundary="reflective",
    )
    paths["simulation_hyperparameters"] = simulation_hyperparameters.to_yaml_file(
        output_dir / "simulation_hyperparameters.yaml", include_nones=False
    )

    return paths


def populate_cfg_active(
    archive_dir: Path | None = None,
    active_dir: Path | None = None,
    top_level_files: tuple[str, ...] = DEFAULT_ACTIVE_TOP_LEVEL,
    deps_files: tuple[str, ...] = DEFAULT_ACTIVE_DEPS,
) -> list[Path]:
    """Copy a default selection of archive files into ``cfg_active``.

    Dependency YAML files used only by ``modified_density_profile`` are placed
    under ``cfg_active/deps/`` so they are not loaded as top-level elements.
    """
    archive_dir = (archive_dir or CFG_ARCHIVE).resolve()
    active_dir = (active_dir or CFG_ACTIVE).resolve()
    deps_dir = active_dir / "deps"
    active_dir.mkdir(parents=True, exist_ok=True)
    deps_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for name in top_level_files:
        source = archive_dir / name
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing archive file {source}. Run create_archive_yamls() first."
            )
        destination = active_dir / name
        shutil.copy2(source, destination)
        copied.append(destination)

    for name in deps_files:
        source = archive_dir / name
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing archive file {source}. Run create_archive_yamls() first."
            )
        destination = deps_dir / name
        shutil.copy2(source, destination)
        copied.append(destination)

    modified_density_profile = density_modifiers.ModifiedDensityProfile(
        base_density_profile=deps_dir / "smooth_sine_flattop.yaml",
        modifiers=[deps_dir / "matched_radial_modifier.yaml"],
    )
    modified_path = active_dir / "modified_density_profile.yaml"
    modified_density_profile.to_yaml_file(modified_path, include_nones=False)
    copied.append(modified_path)

    for path in active_dir.glob("*.yaml"):
        if path.name not in {*top_level_files, "modified_density_profile.yaml"}:
            path.unlink()

    for path in deps_dir.glob("*.yaml"):
        if path.name not in deps_files:
            path.unlink()

    return copied


def main() -> None:
    paths = create_archive_yamls()
    print(f"Wrote {len(paths)} YAML files to {CFG_ARCHIVE}")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    copied = populate_cfg_active()
    print(f"\nCopied {len(copied)} files into {CFG_ACTIVE}:")
    for path in copied:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
