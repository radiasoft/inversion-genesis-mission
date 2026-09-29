"""Orchestration and CLI for generating MRO-merged ``.pyi`` config stubs.

Re-exports core types and helpers from :mod:`.index` and :mod:`.render` so
that existing callers that import from ``stub_sync`` continue to work.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from .index import (
    ClassKey as ClassKey,
    InitField as InitField,
    LibIndex as LibIndex,
    ParsedClass as ParsedClass,
    _field_default as _field_default,
    _init_fields as _init_fields,
    _parse_module as _parse_module,
    default_lib_dir as default_lib_dir,
    lib_module_names as lib_module_names,
    module_source_path as module_source_path,
)
from .render import (
    _quote_docstring as _quote_docstring,
    generate_module_pyi as generate_module_pyi,
)


def sync_config_stubs(
    *, lib_dir: Path | None = None, check: bool = False
) -> list[Path]:
    """Write or verify merged-docstring ``.pyi`` stubs for lib config modules."""
    lib_dir = lib_dir or default_lib_dir()
    index = LibIndex.from_lib_dir(lib_dir)
    changed: list[Path] = []

    for module in lib_module_names(lib_dir):
        pyi_path = module_source_path(lib_dir, module).with_suffix(".pyi")
        generated = generate_module_pyi(module, lib_dir=lib_dir, index=index)
        existing = pyi_path.read_text(encoding="utf-8") if pyi_path.exists() else ""

        if check:
            if existing != generated:
                changed.append(pyi_path)
            continue

        if existing != generated:
            pyi_path.write_text(generated, encoding="utf-8")
            changed.append(pyi_path)

    return changed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for stub generation.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: ``0`` on success, ``1`` when ``--check`` detects stale
        stubs.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate merged-docstring .pyi stubs for inversion_fbpic.lib."
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--lib-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    stale = sync_config_stubs(lib_dir=args.lib_dir, check=args.check)
    if args.check:
        if stale:
            paths = "\n".join(f"  - {path}" for path in stale)
            print(
                textwrap.dedent(
                    f"""\
                    Config docstring stubs are out of date. Regenerate with:

                      python lpa/Simulation_FBPIC/tools/sync_config_docstrings.py

                    Stale files:
                    {paths}
                    """
                )
            )
            return 1
        return 0

    if stale:
        print("Updated .pyi stubs (include them in your commit):")
        for path in stale:
            print(f"  {path}")
        return 1

    print("Config docstring stubs are up to date.")
    return 0
