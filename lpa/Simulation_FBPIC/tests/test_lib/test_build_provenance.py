"""Build-time provenance for fresh clones, portable archives, and editable pip."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

PROJECT = Path(__file__).resolve().parents[2]
HASH_FILE = Path("inversion_fbpic/git_hash.txt")


@pytest.fixture
def helpers():
    spec = importlib.util.spec_from_file_location(
        "build_provenance", PROJECT / "_build_provenance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_archive_preserves_bundled_revision_without_git(
    helpers, monkeypatch, tmp_path: Path, revision
) -> None:
    destination = tmp_path / HASH_FILE
    destination.parent.mkdir()
    if revision is not None:
        destination.write_text(revision + "\n", encoding="utf-8")
    run = Mock(side_effect=AssertionError("Archive builds must not query Git"))
    monkeypatch.setattr(helpers.subprocess, "run", run)
    assert helpers.record_git_hash(tmp_path) == revision
    run.assert_not_called()


def test_archive_ignores_unrelated_enclosing_repository(
    helpers, monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / ".git").mkdir()
    source = tmp_path / "archives" / "inversion_fbpic-0.2"
    (source / HASH_FILE).parent.mkdir(parents=True)
    (source / HASH_FILE).write_text("a" * 40, encoding="utf-8")
    run = Mock(side_effect=AssertionError("Do not record unrelated repository HEAD"))
    monkeypatch.setattr(helpers.subprocess, "run", run)
    assert helpers.record_git_hash(source) == "a" * 40
    run.assert_not_called()


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_build_without_git_preserves_existing_hash(
    helpers, monkeypatch, tmp_path: Path, revision
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / HASH_FILE).parent.mkdir()
    if revision is not None:
        (tmp_path / HASH_FILE).write_text(revision, encoding="utf-8")
    monkeypatch.setattr(
        helpers.subprocess, "run", Mock(side_effect=FileNotFoundError("git"))
    )
    with pytest.warns(UserWarning, match="preserving bundled revision"):
        assert helpers.record_git_hash(tmp_path) == revision


@pytest.mark.parametrize("command", ["sdist", "build_py", "editable_wheel"])
def test_build_commands_record_before_running(
    helpers, monkeypatch, tmp_path: Path, command: str
) -> None:
    from setuptools import Distribution

    command_cls = helpers.build_commands(tmp_path)[command]
    calls = []
    monkeypatch.setattr(
        helpers, "record_git_hash", lambda path: calls.append(("record", path))
    )
    monkeypatch.setattr(
        command_cls.__bases__[0], "run", lambda self: calls.append(("build", command))
    )
    command_cls(Distribution()).run()
    assert calls == [("record", tmp_path), ("build", command)]


def _run(*args: str, cwd: Path) -> str:
    # Nested build/commit tests should never inherit the caller's Git context.
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    result = subprocess.run(
        list(args), cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@pytest.fixture
def fresh_clone(tmp_path: Path) -> tuple[Path, str]:
    """Small disposable standalone package; no commits or hooks in the clone."""
    repository = tmp_path / "repository"
    repository.mkdir()
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "_build_provenance.py"):
        shutil.copy2(PROJECT / name, repository / name)
    (repository / "inversion_fbpic").mkdir()
    (repository / "inversion_fbpic" / "__init__.py").write_text("", encoding="utf-8")
    (repository / ".gitignore").write_text(
        "inversion_fbpic/git_hash.txt\nbuild/\n*.egg-info/\n", encoding="utf-8"
    )
    _run("git", "init", cwd=repository)
    _run("git", "config", "core.hooksPath", str(tmp_path / "no-hooks"), cwd=repository)
    _run("git", "config", "user.name", "Build provenance test", cwd=repository)
    _run("git", "config", "user.email", "build@example.invalid", cwd=repository)
    _run("git", "config", "commit.gpgsign", "false", cwd=repository)
    _run("git", "add", ".", cwd=repository)
    _run("git", "commit", "-m", "Fresh source", cwd=repository)
    clone = tmp_path / "clone"
    _run("git", "clone", str(repository), str(clone), cwd=tmp_path)
    assert not (clone / HASH_FILE).exists()
    return clone, _run("git", "rev-parse", "HEAD", cwd=clone)


def _wheel_revision(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        return archive.read(HASH_FILE.as_posix()).decode().strip()


def test_sdist_and_rebuilt_wheel_record_fresh_clone(
    fresh_clone: tuple[Path, str], tmp_path: Path
) -> None:
    source, revision = fresh_clone
    output = tmp_path / "dist"
    _run(
        sys.executable,
        "-m",
        "build",
        "--no-isolation",
        "--outdir",
        str(output),
        cwd=source,
    )
    assert (source / HASH_FILE).read_text().strip() == revision
    assert _wheel_revision(next(output.glob("*.whl"))) == revision
    archive_path = next(output.glob("*.tar.gz"))
    with tarfile.open(archive_path) as archive:
        assert any(
            name.endswith("/_build_provenance.py") for name in archive.getnames()
        )
        hash_name = next(
            name
            for name in archive.getnames()
            if name.endswith("/" + HASH_FILE.as_posix())
        )
        stream = archive.extractfile(hash_name)
        assert stream is not None and stream.read().decode().strip() == revision
        unpacked = tmp_path / "unpacked"
        archive.extractall(unpacked, filter="data")
    archive_source = next(unpacked.iterdir())
    assert not (archive_source / ".git").exists()
    # Put the source archive inside an unrelated Git repo: its HEAD must be ignored.
    _run("git", "init", cwd=unpacked)
    portable_output = tmp_path / "portable-dist"
    _run(
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(portable_output),
        cwd=archive_source,
    )
    assert _wheel_revision(next(portable_output.glob("*.whl"))) == revision


@pytest.mark.parametrize("editable", [False, True])
def test_pip_build_records_fresh_clone(
    fresh_clone: tuple[Path, str], tmp_path: Path, editable: bool
) -> None:
    source, revision = fresh_clone
    command = [sys.executable, "-m", "pip"]
    if editable:
        # Install in a temporary prefix, never replace the user's existing package.
        command += [
            "install",
            "--ignore-installed",
            "--prefix",
            str(tmp_path / "install"),
            "--editable",
        ]
    else:
        command += ["wheel", "--wheel-dir", str(tmp_path / "wheel")]
    command += [str(source), "--no-deps", "--no-build-isolation"]
    _run(*command, cwd=tmp_path)
    assert (source / HASH_FILE).read_text().strip() == revision
    if not editable:
        assert _wheel_revision(next((tmp_path / "wheel").glob("*.whl"))) == revision
