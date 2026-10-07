"""Density and laser demos support YAML, JSON, and HDF5 generation and plotting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from inversion_fbpic.lib.serializable_config import SerializableConfig

DEMOS = Path(__file__).resolve().parents[2] / "demos"


def _load_demo(directory: str, script: str):
    spec = importlib.util.spec_from_file_location(
        script, DEMOS / directory / f"{script}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("domain,count", [("density", 6), ("laser", 5)])
@pytest.mark.parametrize("serialization_format", ["yaml", "json", "hdf5"])
def test_demo_generation_and_plotting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain: str,
    count: int,
    serialization_format: str,
) -> None:
    directory = "demo_densities" if domain == "density" else "demo_lasers"
    creator = _load_demo(directory, f"create_{domain}_configs")
    plotter = _load_demo(directory, f"plot_{domain}_configs")
    output = tmp_path / "cfg"
    monkeypatch.setattr(creator, "CFG_DIR", output)
    monkeypatch.setattr(plotter, "CFG_DIR", output)
    monkeypatch.setattr(plotter, "PLOTS_DIR", tmp_path / "plots")
    monkeypatch.setattr(sys, "argv", ["demo", "--format", serialization_format])
    creator.main()
    creator.main()  # HDF5 configs can safely be regenerated.
    create = getattr(creator, f"create_{domain}_config_files")
    paths = create(output, serialization_format=serialization_format)
    assert len(paths) == count
    suffix = {"yaml": ".yaml", "json": ".json", "hdf5": ".h5"}[serialization_format]
    assert all(path.suffix == suffix for path in paths.values())
    for path in paths.values():
        assert SerializableConfig.from_file(path).CONFIG_TYPE in (
            "density_profile",
            "laser_pulse",
        )
    # Plot after changing cwd to also verify linked density-file resolution.
    monkeypatch.chdir(tmp_path)
    plotter.main()
    plots = tmp_path / "plots"
    if serialization_format != "yaml":
        plots /= serialization_format
    assert (plots / "combined.png").is_file()
    assert len(list(plots.glob("*.png"))) == count + 1
