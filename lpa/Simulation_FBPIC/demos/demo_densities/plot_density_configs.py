"""Plot density configs with ``--format yaml|json|hdf5`` (default: YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.serializable_config import SerializableConfig

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR / "cfg"
PLOTS_DIR = SCRIPT_DIR / "plots"
FORMAT_SUFFIXES = {
    "yaml": {".yaml", ".yml"},
    "json": {".json", ".jsn"},
    "hdf5": {".h5", ".hdf5"},
}


def _load_density_profile(path: Path) -> _DensityProfile | None:
    try:
        config = SerializableConfig.from_file(path)
    except (ValueError, OSError):
        return None
    return config if isinstance(config, _DensityProfile) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=tuple(FORMAT_SUFFIXES), default="yaml")
    args = parser.parse_args()
    suffixes = FORMAT_SUFFIXES[args.format]
    if not CFG_DIR.is_dir():
        raise FileNotFoundError(f"Config directory not found: {CFG_DIR}")

    profiles = {}
    for path in sorted(CFG_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            profile = _load_density_profile(path)
            if profile is not None:
                profiles[path] = profile
    if not profiles:
        raise FileNotFoundError(
            f"No density-profile {args.format.upper()} files found in {CFG_DIR}"
        )

    plots_dir = PLOTS_DIR if args.format == "yaml" else PLOTS_DIR / args.format
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))

    for config_path, profile in profiles.items():
        name = config_path.stem
        print(f"Plotting {name} from {config_path}")
        profile.plot(
            output_path=plots_dir / f"{name}.png",
            show=False,
            ax=ax,
        )

    ax.legend()
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"Density ($\mathrm{cm}^{-3}$)")
    ax.set_title("Combined Density Profiles")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {len(profiles)} plots to {plots_dir}")


if __name__ == "__main__":
    main()
