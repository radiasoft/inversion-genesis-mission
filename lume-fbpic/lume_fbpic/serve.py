"""Serve an archived `LUMEFBPICModel` as EPICS PVs through `lume-pva`.

`lume-fbpic-serve <archive.h5 | directory> ... [--prefix LPA:SIM:]` loads the archives
(`LUMEFBPICModel.from_archive()`; a directory means its `*.h5` files) and serves every read-only
scalar action -- the stats and the moment-descriptor features -- over PVA and CA, each named
`<prefix><action name>`; `lume-pva` adds `<prefix>RESET` and `<prefix>SNAPSHOT`.

The archives are held by an `ArchiveSelector`: one writable enum PV, `<prefix>LPA_Archive`, lists
them by file name (without `.h5`) and putting one makes it the active run. The first is active at
start. The served values and, with `--twin`, the bunch the twin tracks follow the active run. They
must all have the same variables (be made with the same action list). The values are the archive's: the live results when it
holds the final particles, otherwise the recorded output values (as in the archives made by
`docs/examples/initial_sample_archives.py`). `final_particles` is not served, as `lume-pva` has no
handler for a particle group.

The model is always loaded with `dummy_run=True`, so a put can change a parameter but can never
start a simulation. Input actions are not served unless `--include-inputs` is given; putting one
makes the recorded outputs stale (they read NaN until `RESET`).

With `--twin` the archive's final particles are the source of the HTU transport twin
(`lume_fbpic.twin.build_chain`): the twin's own variables (magnets, steering, chicane, slit,
magspec, ...) are served read-write and re-track on every put, its `Source_*` variables read the
LPA bunch read-only, and the archive's outputs are served read-only next to them. The archive must
hold the final particles (`save_final_particles=True`). `htu` (geecs-lume-twin) must be importable,
and the default prefix becomes `HTU:SIM:`, the one the twin's Phoebus displays use.

A put is acknowledged at once and the served values update when the twin has re-tracked (about
3 s), because Phoebus waits one second for a put reply on its UI thread and would otherwise report
a write error and freeze. `--wait-for-puts` acknowledges only after the re-track
(`PutMode.Complete`), for scan clients that set and wait. The twin's camera images are made coarser
(`--screen-binning`, 4 by default): the bunch has far fewer macroparticles than the twin's
1024 x 1024 pixels.

An archive without final particles (one reconstructed from a dataset, as
`docs/examples/initial_sample_archives.py` writes) cannot feed the twin on its own. With
`--synthesize-bunch` a bunch is built from its recorded moment descriptor
(`LUMEFBPICModel.synthesize_bunch`): an APPROXIMATION of the real bunch, matching the descriptor's
means, variances and main correlations, not its tails. The served descriptor values stay the
recorded ones.

Needs `lume-pva` (not a dependency of this package); it is imported only when serving.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.selector import DEFAULT_SELECTOR_NAME, ArchiveSelector


def build_config(
    model,
    *,
    include_inputs: bool = False,
    prefix: str = "LPA:SIM:",
    protocol: list[str] | None = None,
    serve_always: set[str] | None = None,
    wait_for_puts: bool = False,
) -> dict[str, Any]:
    """The `lume_pva.runner.Runner` config for `model`: read-only variables only, unless
    `include_inputs`, in which case the writable ones are served read-write. Variables named in
    `serve_always` are served in their natural mode either way. `wait_for_puts` makes a put
    acknowledge only after the model has finished (`PutMode.Complete`)."""
    runner = _import_runner()
    config = runner.Runner.generate_config(
        model,
        prefix=prefix,
        put_mode=runner.PutMode.Complete if wait_for_puts else runner.PutMode.Immediate,
    )
    always = serve_always or set()
    config["variables"] = {
        name: entry
        for name, entry in config["variables"].items()
        if include_inputs or name in always or str(entry["mode"]) == "ro"
    }
    if protocol:
        config["protocol"] = list(protocol)
    return config


def expand_archives(paths: list[Path]) -> list[Path]:
    """The archive files named by `paths`: a file stays as it is, a directory becomes its `*.h5`
    files in name order.

    Raises:
        ValueError: If a path does not exist, or a directory holds no `.h5` file.
    """
    archives: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found = sorted(path.glob("*.h5"))
            if not found:
                raise ValueError(f"no .h5 archives in {path}")
            archives.extend(found)
        elif path.is_file():
            archives.append(path)
        else:
            raise ValueError(f"no such archive or directory: {path}")
    return archives


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "archives",
        nargs="+",
        type=Path,
        help="archives written by LUMEFBPICModel.archive(), or directories of them",
    )
    parser.add_argument(
        "--selector-name", default=DEFAULT_SELECTOR_NAME, help="name of the archive-selector PV"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="PV name prefix (default LPA:SIM:, or HTU:SIM: with --twin)",
    )
    parser.add_argument(
        "--include-inputs",
        action="store_true",
        help="also serve the writable actions (puts change the config but never run it)",
    )
    parser.add_argument(
        "--protocol", nargs="+", choices=["ca", "pva"], help="protocols to serve (default both)"
    )
    parser.add_argument(
        "--twin",
        action="store_true",
        help="feed the archive's final particles to the HTU twin and serve the twin as well",
    )
    parser.add_argument(
        "--synthesize-bunch",
        action="store_true",
        help="with --twin and no final particles in the archive, build an approximate bunch from "
        "the recorded moment descriptor",
    )
    parser.add_argument(
        "--bunch-particles", type=int, default=20_000, help="macroparticles of the synthetic bunch"
    )
    parser.add_argument(
        "--wait-for-puts",
        action="store_true",
        help="acknowledge a put only after the model has finished (PutMode.Complete), so a scan "
        "client can set-and-wait. The default acknowledges at once: a Phoebus write waits one "
        "second on its UI thread, shorter than a twin re-track",
    )
    parser.add_argument(
        "--screen-binning",
        type=int,
        default=4,
        help="with --twin, make the twin's camera images this many times coarser in each "
        "direction (default 4: 1024 x 1024 becomes 256 x 256, over the same field of view; 1 "
        "keeps the twin's own resolution)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    runner = _import_runner()
    if args.synthesize_bunch and not args.twin:
        parser.error("--synthesize-bunch only applies with --twin")
    try:
        paths = expand_archives(args.archives)
    except ValueError as error:
        parser.error(str(error))
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        parser.error(f"archive file names must be unique to name the runs: {sorted(stems)}")
    models = {
        path.stem: LUMEFBPICModel.from_archive(path, dummy_run=True) for path in paths
    }
    if args.twin:
        without = [name for name, m in models.items() if m.simulator.final_particles is None]
        if without and not args.synthesize_bunch:
            parser.error(
                f"--twin needs the final particles in every archive; without them: {without}. "
                "Write them with save_final_particles=True, or pass --synthesize-bunch to build "
                "an approximate bunch from each recorded descriptor"
            )
        for name in without:
            models[name].synthesize_bunch(n_particles=args.bunch_particles)
        if without:
            logging.getLogger(__name__).warning(
                "serving SYNTHETIC bunches built from the moment descriptor (%d macroparticles) "
                "for %s; they approximate the real bunches",
                args.bunch_particles,
                without,
            )
    try:
        selector = ArchiveSelector(models, selector_name=args.selector_name)
    except ValueError as error:
        parser.error(str(error))
    serve_always = {args.selector_name}
    if args.twin:
        from lume_fbpic.twin import build_chain

        try:
            served = build_chain(selector, screen_binning=args.screen_binning)
        except ValueError as error:
            parser.error(str(error))
        serve_always |= set(served.lume_model_instances[1].supported_variables)
        # Putting any twin variable passes the LPA bunch on first; do it once now so the twin
        # starts with the LPA bunch as its source.
        current = served.get(["EMQ1H_Current"])["EMQ1H_Current"]
        served.set({"EMQ1H_Current": float(current)})
    else:
        served = selector
    prefix = args.prefix or ("HTU:SIM:" if args.twin else "LPA:SIM:")
    config = build_config(
        served,
        include_inputs=args.include_inputs,
        prefix=prefix,
        protocol=args.protocol,
        serve_always=serve_always,
        wait_for_puts=args.wait_for_puts,
    )
    logging.getLogger(__name__).info("serving %d variables", len(config["variables"]))
    runner.Runner(model=served, config=config).run()


def _import_runner():
    """The `lume_pva.runner` module (it holds `Runner` and `PutMode`)."""
    try:
        import lume_pva.runner as runner
    except ImportError as error:
        raise ImportError(
            "lume-fbpic-serve needs the lume-pva package, which is not installed."
        ) from error
    return runner


if __name__ == "__main__":
    main()
