"""Serve the HTU transport twin as EPICS PVs (CA + PVA) under HTU:SIM:.

Entry point: ``htu-twin-serve`` (or ``python -m htu.serve``).

PutMode.Complete: puts acknowledge only after the model has re-tracked,
so scan clients can set-and-wait without a settle time.
"""

import argparse
import logging

from lume_pva.runner import PutMode, Runner

from htu.model import build_htu_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the HTU transport twin")
    parser.add_argument(
        "--config",
        help="YAML config (htu-twin-config save format) applied as boot "
        "defaults before serving",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model = build_htu_model()
    if args.config:
        from htu.config_tool import load_config_file

        values = load_config_file(args.config)
        model.set(values)  # PV suffixes == variable names
        logging.info("applied boot config %s (%d values)", args.config, len(values))
    config = Runner.generate_config(model, prefix="HTU:SIM:", put_mode=PutMode.Complete)
    config["description"] = (
        "HTU transport twin: source -> A-line + Dutch magspec (Cheetah, chromatic)"
    )
    Runner(model=model, config=config).run()


if __name__ == "__main__":
    main()
