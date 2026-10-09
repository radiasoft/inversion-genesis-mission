"""Run an FBPIC simulation from the YAML files in ``cfg_active``."""

from __future__ import annotations

from pathlib import Path

from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.lib.simulation import Simulation
from inversion_fbpic.utils.plotting import plot_from_hdf5_series
from inversion_fbpic.utils.make_movie import make_movie

DEMO_DIR = Path(__file__).resolve().parent
CFG_ACTIVE = DEMO_DIR / "cfg_active"
PLOTS_DIR = DEMO_DIR / "plots"
DIAGS_DIR = DEMO_DIR / "diags"


def load_sim(
    working_directory: Path | str = DEMO_DIR,
    cfg_active_dir: Path | str = CFG_ACTIVE,
    plot_path: Path | str = PLOTS_DIR,
) -> Simulation:
    """Build and run a simulation from every YAML file in ``cfg_active``."""
    working_directory = Path(working_directory).resolve()
    cfg_active_dir = Path(cfg_active_dir).resolve()
    plot_path = Path(plot_path)
    if not plot_path.is_absolute():
        plot_path = working_directory / plot_path
    plot_path = plot_path.resolve()
    plot_path.mkdir(parents=True, exist_ok=True)

    if not cfg_active_dir.is_dir():
        raise FileNotFoundError(
            f"Active config directory not found: {cfg_active_dir}. "
            "Run create_archive_yamls.py first."
        )

    yaml_files = sorted(cfg_active_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML files in {cfg_active_dir}. "
            "Copy building blocks from cfg_archive into cfg_active."
        )

    print(f"Working directory: {working_directory}")
    print(f"Active config directory: {cfg_active_dir}")
    print(f"Plot path: {plot_path}")
    print("Active YAML files:")
    for path in yaml_files:
        print(f"  {path.name}")

    with SerializableConfig.resolving_paths_relative_to(cfg_active_dir):
        sim = Simulation(elements=cfg_active_dir, verbosity=20)

    sim.hyparams.grid_parameters_yaml(file_name=cfg_active_dir / "grid_parameters.yaml")

    for density_profile in sim.densities:
        density_profile.plot(
            output_path=plot_path
            / f"{density_profile.SUBCLASS}_{density_profile.species}.png",
            show=False,
        )
    for laser_pulse in sim.lasers:
        laser_pulse.plot(
            output_path=plot_path / f"{laser_pulse.SUBCLASS}.png",
            show=False,
        )

    sim.setup_simulation(working_directory=working_directory)

    return sim


def create_movies() -> Path:
    """Process the simulation results."""
    for field, component in [
        ("rho", None),
        ("E", "x"),
        ("E", "y"),
        ("E", "z"),
        ("B", "x"),
        ("B", "y"),
        ("B", "z"),
        (None, "eme"),
    ]:
        save_dir, file_prefix = plot_from_hdf5_series(
            series_path=DIAGS_DIR / "hdf5",
            save_path=PLOTS_DIR / "stills",
            field_name=field,
            component=component,
            vminmax=(
                (1e14, None)
                if field == "rho"
                else (1e-15, 1e-5) if component == "eme" else "even"
            ),
            cmap="magma" if field == "rho" or component == "eme" else "bwr",
            scale="log",
            font_size=16,
        )
        movie_path = make_movie(
            images_dir=save_dir, image_prefix=file_prefix, filename=file_prefix
        )

        # move movie to the same dir as this script
        dest_path = PLOTS_DIR / f"{file_prefix}.mp4"
        movie_path.rename(dest_path)

        component_suffix = "_" + component if component is not None else ""
        print(
            f"Created movie for {field} {component} in {PLOTS_DIR / f'{field}{component_suffix}'}"
        )

    for file in (PLOTS_DIR / "stills").glob("*.png"):
        file.unlink()
    (PLOTS_DIR / "stills").rmdir()


if __name__ == "__main__":
    sim = load_sim()
    sim.run_simulation()
    create_movies()
