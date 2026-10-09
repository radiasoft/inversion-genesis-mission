"""Make a `lume-fbpic` bunch the source of the HTU transport twin (`geecs-lume-twin`).

`build_chain(lpa_model)` is `StagedModel([lpa_model, TwinStage(twin_model)])`: it hands the LPA
stage's `final_particles` to the twin's `initial_particles` (see `docs/htu-twin/twin/twin.py`), and the
twin's own `Source_*` variables become read-only readbacks of that bunch. The bunch comes from either

- `--archive`: an archive written with `LUMEFBPICModel.archive(save_final_particles=True)`, or
- `--run-dir`: the working directory of a finished `ionization_injection.py` run, whose
  diagnostics are read with `load_results()`.

The script then reports what the twin received (its `Source_*` readbacks) and the charge each
screen sees along the line, as a fraction of the injected charge.

The bunch is injected where the snapshot has it. How the twin's source plane (its `PlasmaExit`
screen; the 5.2 cm to the first magnet is the twin's own `SrcToPMQ1` drift) relates to the
simulation's `z` is not known, so no drift is applied unless you give one:
`--plasma-exit-z <metres>` drifts the bunch from its mean `z` to that lab-frame plane first.

The twin needs `htu` (`geecs-lume-twin` in this repository) importable, for example
`PYTHONPATH=<repository>/geecs-lume-twin python lpa_to_twin.py --run-dir <dir>`. Its magnets keep
the settings they have for a 100 MeV beam, so a bunch of very different energy or large energy
spread is mostly lost; this shows the hand-off, not a matched optics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy
from lume_fbpic.model import LUMEFBPICModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import ionization_injection  # noqa: E402  (docs/examples/ionization_injection.py)
from twin import build_chain  # noqa: E402  (docs/htu-twin/twin/twin.py, this directory)

SCREENS = (
    "TCPhosphor",
    "ChicaneSlit",
    "DCPhosphor",
    "Phosphor1",
    "UC_ALineEbeam1",
    "UC_ALineEBeam2",
    "UC_ALineEBeam3",
)
SOURCE_READBACKS = (
    "Source_Energy_MeV",
    "Source_EnergySpread_pct",
    "Source_Charge_pC",
    "Source_NumParticles",
    "Source_BetaX_mm",
    "Source_BetaY_mm",
    "Source_NormEmitX_um",
    "Source_NormEmitY_um",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="archive with the final particles")
    source.add_argument(
        "--run-dir", type=Path, help="working directory of a finished run"
    )
    parser.add_argument("--uz-min", type=float, default=30.0)
    parser.add_argument("--central-fraction", type=float, default=0.95)
    parser.add_argument(
        "--plasma-exit-z",
        type=float,
        help="lab z [m] of the twin's source plane; the bunch is drifted there (default: no drift)",
    )
    args = parser.parse_args(argv)

    from htu.model import build_htu_model

    if args.archive:
        lpa = LUMEFBPICModel.from_archive(args.archive, dummy_run=True)
    else:
        lpa = ionization_injection.build_model(working_directory=args.run_dir)
        lpa.dummy_run = True
        lpa.simulator.load_results()
    chain = build_chain(
        lpa,
        build_htu_model(),
        central_fraction=args.central_fraction,
        uz_min=args.uz_min,
        plasma_exit_z=args.plasma_exit_z,
    )
    twin = chain.lume_model_instances[1]

    # Setting any twin variable passes the LPA bunch on first; re-setting a value is harmless.
    current = chain.get(["EMQ1H_Current"])["EMQ1H_Current"]
    chain.set({"EMQ1H_Current": float(current)})

    injected = twin.initial_particles
    print(f"injected {len(injected)} particles, {injected.charge * 1e12:.1f} pC")
    if twin.drift_length is None:
        print("not drifted: injected where the snapshot has it")
    else:
        print(
            f"drifted {twin.drift_length * 1e3:+.3f} mm to z = {args.plasma_exit_z * 1e3:.3f} mm"
        )
    print(
        f"Source_* are read-only: {chain.supported_variables['Source_Energy_MeV'].read_only}"
    )
    for name in SOURCE_READBACKS:
        print(f"  {name:26} {float(chain.get([name])[name]):.4g}")
    print("charge on each screen (pC, fraction of injected):")
    for screen in SCREENS:
        image = numpy.asarray(chain.get([f"{screen}_image"])[f"{screen}_image"])
        charge = float(image.sum()) * 1e12
        print(f"  {screen:18} {charge:8.1f}  {charge / (injected.charge * 1e12):6.1%}")


if __name__ == "__main__":
    main()
