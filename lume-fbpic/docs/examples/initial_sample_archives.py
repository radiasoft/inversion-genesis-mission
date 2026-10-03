"""Build lume-fbpic archives from the `runs/initial_sample` sample dataset.

`sample_dataset.json` holds seven ionization-injection runs (`sim_0000` to `sim_0006`), each as
24 physical inputs and the 33 moment-descriptor outputs. For every run this writes
`<output dir>/<run>.h5`, a `LUMEFBPICModel` archive (`LUMEFBPICModel.archive()`) with

- the config rebuilt from the run's inputs, on the production hyperparameters of
  `ionization_injection_runscript_00.py`;
- the action inputs, read from that config;
- the 33 `descriptor_*` outputs exactly as recorded in the dataset;
- `metadata` marking it as reconstructed, with its source.

`LUMEFBPICModel.from_archive()` loads one and its `get()` returns the recorded descriptor values,
so it can be served with `lume-fbpic-serve`.

These are RECONSTRUCTED archives, not the runs themselves:

1. There are no final particles and no `stats`; the dataset keeps only the 33 descriptor scalars.
2. The config approximates the config that produced the numbers. `lume_fbpic` cannot match the
   original interaction length (`right_buffer` must be positive, see `ionization_injection.py`
   point 6), does not pass `p_zmin`/`p_rmax` to the particle loading (point 4), and the runs
   used 8 MPI ranks on GPUs, which this records only as `use_mpi=True`.
3. Only inputs that have an action (laser energy, pulse duration, focal position, the nitrogen
   dopant fraction and the 12 Zernike coefficients) may differ from the `ionization_injection.py`
   baseline, which equals `sim_0000`. Any other input that differs raises an error rather than
   being dropped silently; the dataset's runs vary only five of the 24 inputs.

Usage: `python initial_sample_archives.py [sample_dataset.json] [output dir]`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import attrs

import ionization_injection
from lume_fbpic.actions import LaserFieldAction, MomentDescriptorAction
from lume_fbpic.model import LUMEFBPICModel

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[3]
    / "lpa/Simulation_FBPIC/runs/initial_sample/sample_dataset/sample_dataset.json"
)

# Production settings of runs/initial_sample/ionization_injection_runscript_00.py (its
# HYPERPARAMETERS), which `ionization_injection.py` coarsens for a one-node CPU run.
PRODUCTION = {
    "nz": 1500,
    "nr": 300,
    "nm": 5,
    "number_dumps": 50,
    "write_period": 50,
    "use_mpi": True,  # submission_script_00.sh runs 8 ranks under srun
    "p_nt": 15,
    "laser_n_azimuthal_modes": 5,
    "laser_num_points": (600, 900),
}

# Dataset input -> action name, for the inputs that have an action.
_ACTION_FOR_INPUT = {
    "laser_energy_J": "laser_energy",
    "laser_focal_position_m": "laser_focal_position",
    "laser_pulse_duration_fwhm_s": "laser_temporal_width",
    "nitrogen_dopant_fraction": "nitrogen_dopant_fraction",
}


def build_archive(
    run: str, record: dict[str, Any], dataset_metadata: dict[str, Any], path: Path, source: Path
) -> Path:
    """Write the archive for one dataset run and return its path."""
    model = ionization_injection.build_model()
    model.dummy_run = True
    _use_production_settings(model)
    _check_descriptor_settings(model, dataset_metadata)
    _check_fixed_inputs(model, record["input"])
    _apply_inputs(model, record["input"])
    model.archive(
        path,
        metadata={
            "reconstructed": True,
            "source": str(source),
            "run": run,
            "dataset_metadata": dataset_metadata,
            "note": "Reconstructed from sample_dataset.json: no final particles or stats, and "
            "the config only approximates the original run (see initial_sample_archives.py).",
        },
        outputs={f"descriptor_{name}": value for name, value in record["output"].items()},
    )
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("initial_sample_archives"))
    args = parser.parse_args(argv)

    dataset = json.loads(args.dataset.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run, record in dataset["runs"].items():
        path = build_archive(
            run, record, dataset["metadata"], args.output_dir / f"{run}.h5", args.dataset
        )
        print(f"wrote {path}")


def _apply_inputs(model: LUMEFBPICModel, inputs: dict[str, float]) -> None:
    """Set the inputs that have an action (the rest were checked to equal the baseline)."""
    values = {}
    for name, value in inputs.items():
        if name in _ACTION_FOR_INPUT:
            values[_ACTION_FOR_INPUT[name]] = value
        elif name.startswith("zernike_"):
            values[name] = value
    model.set(values)


def _check_descriptor_settings(model: LUMEFBPICModel, dataset_metadata: dict[str, Any]) -> None:
    """The model's descriptor actions must select particles as the dataset's did."""
    wanted = {
        "uz_min": dataset_metadata["uz_min"],
        "central_fraction": dataset_metadata["central_fraction"],
        "longitudinal_bins": dataset_metadata["longitudinal_bins"],
    }
    for action in model._actions:
        if isinstance(action, MomentDescriptorAction):
            have = {key: getattr(action, key) for key in wanted}
            if have != wanted:
                raise ValueError(f"descriptor actions use {have}, the dataset used {wanted}")


def _check_fixed_inputs(model: LUMEFBPICModel, inputs: dict[str, float]) -> None:
    """Raise if an input without an action differs from what the model already has."""
    simulator = model.simulator
    he, nitrogen = simulator.densities
    baseline = {
        "density_alpha_m": he.alpha,
        "density_beta": he.beta,
        "density_center_location_m": he.z0,
        "density_peak": he.gauss_peak,
        "laser_spot_size_m": simulator.laser.waist,
        "laser_super_gaussian_order": simulator.laser.super_gaussian_order,
        "laser_wavelength_m": simulator.laser.wavelength,
        # n_plasma = 2 * n_gas, and n_gas is the He + N atom density (each profile's
        # nominal_density is its atom density times its ionization-level count).
        "peak_plasma_density_cm3": 2.0
        * (he.nominal_density / 2 + nitrogen.nominal_density / 7)
        / 1.0e6,
    }
    for name, value in inputs.items():
        if name in baseline and not math.isclose(value, baseline[name], rel_tol=1e-9):
            raise ValueError(
                f"input {name}={value} differs from the baseline {baseline[name]} and has no "
                "action to set it"
            )
    unknown = set(inputs) - set(baseline) - set(_ACTION_FOR_INPUT) - {
        k for k in inputs if k.startswith("zernike_")
    }
    if unknown:
        raise ValueError(f"inputs this script does not know how to apply: {sorted(unknown)}")


def _use_production_settings(model: LUMEFBPICModel) -> None:
    """Replace the example's coarsened grid and laser grid by the production ones."""
    simulator = model.simulator
    simulator.hyparams = attrs.evolve(
        simulator.hyparams,
        nz=PRODUCTION["nz"],
        nr=PRODUCTION["nr"],
        nm=PRODUCTION["nm"],
        number_dumps=PRODUCTION["number_dumps"],
        write_period=PRODUCTION["write_period"],
        use_mpi=PRODUCTION["use_mpi"],
    )
    simulator.densities = [
        attrs.evolve(density, p_nt=PRODUCTION["p_nt"]) for density in simulator.densities
    ]
    # LaserFieldAction._set also clears the derived energy/a0 sibling, which a plain
    # attrs.evolve of these fields would trip over.
    for field, value in (
        ("n_azimuthal_modes", PRODUCTION["laser_n_azimuthal_modes"]),
        ("num_points", PRODUCTION["laser_num_points"]),
    ):
        LaserFieldAction(name=field, field_name=field, unit=None)._set(simulator, value)


if __name__ == "__main__":
    main()
