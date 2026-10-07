"""Build-only Git provenance helpers; safe to load without simulation packages."""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from pathlib import Path

HASH_FILE = Path("inversion_fbpic/git_hash.txt")


def _recorded_hash(source_dir: Path) -> str | None:
    try:
        return (source_dir / HASH_FILE).read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeError):
        return None


def _repository_root(source_dir: Path) -> Path | None:
    """Accept this project or its monorepo, never an unrelated enclosing repo."""
    if (source_dir / ".git").exists():
        return source_dir
    monorepo = source_dir.parent.parent
    if (
        source_dir == monorepo / "lpa" / "Simulation_FBPIC"
        and (monorepo / ".git").exists()
    ):
        return monorepo
    return None


def record_git_hash(source_dir: Path, *, strict: bool = False) -> str | None:
    """Refresh checkout provenance, or preserve bundled provenance without Git.

    *strict* is used by repository hooks so a failed recording is reported.
    Source-archive builds preserve their bundled hash instead of looking up an
    unrelated enclosing checkout. Unknown provenance remains absent (null at
    runtime), and never prevents a build merely because Git is unavailable.
    """
    source_dir = source_dir.resolve()
    repository = _repository_root(source_dir)
    if repository is None:
        if strict:
            raise ValueError(
                "The source directory is not in the project's Git checkout."
            )
        return _recorded_hash(source_dir)

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
        if not revision:
            raise ValueError("Git returned an empty HEAD revision.")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        if strict:
            raise
        warnings.warn(
            f"Could not refresh Git provenance; preserving bundled revision: {exc}",
            stacklevel=2,
        )
        return _recorded_hash(source_dir)

    destination = source_dir / HASH_FILE
    # Replacement in the same directory is atomic for readers during builds/hooks.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(revision + "\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return revision


def build_commands(source_dir: Path) -> dict[str, type]:
    """Setuptools hooks for sdists, wheels, and PEP 660 editable installations."""
    # Import build dependencies only when requested by setup.py, not by Git hooks.
    from setuptools.command.build_py import build_py
    from setuptools.command.editable_wheel import editable_wheel
    from setuptools.command.sdist import sdist

    class BuildPyWithGitHash(build_py):
        def run(self) -> None:
            record_git_hash(source_dir)
            super().run()

    class SdistWithGitHash(sdist):
        def run(self) -> None:
            record_git_hash(source_dir)
            super().run()

    class EditableWheelWithGitHash(editable_wheel):
        def run(self) -> None:
            record_git_hash(source_dir)
            super().run()

    return {
        "build_py": BuildPyWithGitHash,
        "sdist": SdistWithGitHash,
        "editable_wheel": EditableWheelWithGitHash,
    }
