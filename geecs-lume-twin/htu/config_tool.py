"""Save/load HTU twin machine configurations (client-side, YAML).

The server stays stateless: this CLI talks to a RUNNING server over PVA
and snapshots/restores every read-write PV. Doctrine (configs/README.md):
code defaults = pristine physics baseline; matched-to-experiment states
live in configs/ as named, git-tracked YAML files with notes.

    htu-twin-config save configs/mysession.yaml --note "matched to 08-25"
    htu-twin-config load configs/mysession.yaml
    htu-twin-serve --config configs/mysession.yaml   # boot with it
"""

import argparse
import datetime
import sys
from typing import Any, Callable

import yaml

from htu.model import boolean_pv_names, integer_pv_names, writable_pv_names

DEFAULT_PREFIX = "HTU:SIM:"


def coerce_value(name: str, value: Any, bools: set[str] | None = None,
                 ints: set[str] | None = None):
    """Coerce a raw PV/YAML value to the type the PV speaks."""
    bools = bools if bools is not None else boolean_pv_names()
    ints = ints if ints is not None else integer_pv_names()
    if name in bools:
        return bool(value)
    if name in ints:
        return int(value)
    return float(value)


def collect_config(get_value: Callable[[str], Any]) -> dict[str, Any]:
    """Snapshot every writable PV via get_value(pv_suffix) -> value.

    Factored over a getter so it works against a live server (p4p get)
    and headless against a model (model.get) — see tests/test_config_roundtrip.py.
    """
    bools = boolean_pv_names()
    return {
        name: coerce_value(name, get_value(name), bools)
        for name in writable_pv_names()
    }


def load_config_file(path: str) -> dict[str, Any]:
    """Read a config YAML and return the coerced {pv_suffix: value} map."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    bools = boolean_pv_names()
    ints = integer_pv_names()
    return {
        name: coerce_value(name, value, bools, ints)
        for name, value in doc["pvs"].items()
    }


def write_config_file(path: str, pvs: dict[str, Any], note: str = "",
                      prefix: str = DEFAULT_PREFIX) -> None:
    doc = {
        "saved": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": note,
        "prefix": prefix,
        "pvs": pvs,
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _save(args) -> int:
    from p4p.client.thread import Context

    ctx = Context("pva")
    pvs = collect_config(
        lambda name: ctx.get(args.prefix + name, timeout=args.timeout)
    )
    write_config_file(args.file, pvs, note=args.note, prefix=args.prefix)
    print(f"saved {len(pvs)} PVs -> {args.file}")
    return 0


def _load(args) -> int:
    from p4p.client.thread import Context

    pvs = load_config_file(args.file)
    ctx = Context("pva")
    failures = []
    for name, value in pvs.items():
        try:
            # PutMode.Complete: the put ACKs only after the re-track
            ctx.put(args.prefix + name, value, timeout=args.timeout)
            print(f"applied {args.prefix}{name} = {value}")
        except Exception as exc:  # noqa: BLE001 - report every failed put
            failures.append(name)
            print(f"FAILED  {args.prefix}{name} = {value}: {exc}")
    if failures:
        print(f"{len(failures)} puts failed: {', '.join(failures)}")
        return 1
    print(f"loaded {len(pvs)} PVs from {args.file}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    sub = parser.add_subparsers(dest="command", required=True)
    p_save = sub.add_parser("save", help="snapshot all rw PVs to YAML")
    p_save.add_argument("file")
    p_save.add_argument("--note", default="")
    p_save.add_argument("--timeout", type=float, default=10.0)
    p_save.set_defaults(func=_save)
    p_load = sub.add_parser("load", help="apply a YAML config to the server")
    p_load.add_argument("file")
    p_load.add_argument("--timeout", type=float, default=60.0)
    p_load.set_defaults(func=_load)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
