"""Record HEAD after commits/checkouts for serialization without runtime Git."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def record_git_hash(repo_root: Path) -> str:
    """Use the same recording helper as package builds, without build imports."""
    source_dir = repo_root / "lpa" / "Simulation_FBPIC"
    helpers = runpy.run_path(str(source_dir / "_build_provenance.py"))
    revision = helpers["record_git_hash"](source_dir, strict=True)
    assert isinstance(revision, str)
    return revision


def main() -> int:
    """Refresh the recorded hash without importing simulation dependencies."""
    try:
        record_git_hash(REPO_ROOT)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"Could not record Git revision: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
