"""Tests for the lightweight Git revision recorder and post-commit hook."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "tools" / "record_git_hash.py"
BUILD_HELPER = REPO_ROOT / "lpa" / "Simulation_FBPIC" / "_build_provenance.py"
HASH_FILE = Path("lpa/Simulation_FBPIC/inversion_fbpic/git_hash.txt")


@pytest.fixture
def recorder():
    spec = importlib.util.spec_from_file_location("build_provenance", BUILD_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_git_hash_atomic_write(recorder, monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    destination = tmp_path / HASH_FILE
    destination.parent.mkdir(parents=True)
    destination.write_text("old revision\n", encoding="utf-8")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "a" * 40 + "\n"))
    monkeypatch.setattr(recorder.subprocess, "run", run)

    assert (
        recorder.record_git_hash(tmp_path / "lpa" / "Simulation_FBPIC", strict=True)
        == "a" * 40
    )
    assert destination.read_text() == "a" * 40 + "\n"
    assert list(destination.parent.iterdir()) == [destination]
    run.assert_called_once_with(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("git"), subprocess.CalledProcessError(128, ["git"])],
)
def test_recorder_failure_leaves_previous_file(
    recorder, monkeypatch, tmp_path: Path, error: Exception
) -> None:
    (tmp_path / ".git").mkdir()
    destination = tmp_path / HASH_FILE
    destination.parent.mkdir(parents=True)
    destination.write_text("previous\n", encoding="utf-8")
    monkeypatch.setattr(recorder.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(type(error)):
        recorder.record_git_hash(tmp_path / "lpa" / "Simulation_FBPIC", strict=True)
    assert destination.read_text() == "previous\n"


def test_recorder_cleans_up_failed_write(recorder, monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    destination = tmp_path / HASH_FILE
    destination.parent.mkdir(parents=True)
    destination.write_text("previous\n", encoding="utf-8")
    monkeypatch.setattr(
        recorder.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, "a" * 40)),
    )
    monkeypatch.setattr(recorder.os, "replace", Mock(side_effect=PermissionError))
    with pytest.raises(PermissionError):
        recorder.record_git_hash(tmp_path / "lpa" / "Simulation_FBPIC", strict=True)
    assert destination.read_text() == "previous\n"
    assert list(destination.parent.iterdir()) == [destination]


def test_recording_hook_configuration() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    recorder_hook = next(hook for hook in hooks if hook["id"] == "record-git-hash")
    assert recorder_hook["entry"] == "python tools/record_git_hash.py"
    assert recorder_hook["always_run"] is True
    assert recorder_hook["pass_filenames"] is False
    assert recorder_hook["stages"] == [
        "post-commit",
        "post-checkout",
        "post-merge",
        "post-rewrite",
    ]
    assert config["default_stages"] == ["pre-commit"]
    assert set(config["default_install_hook_types"]) == {
        "pre-commit",
        "post-commit",
        "post-checkout",
        "post-merge",
        "post-rewrite",
    }


def test_post_commit_records_new_head_in_linked_worktree(tmp_path: Path) -> None:
    """Use real commits in an isolated repository, never the user's checkout."""
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str, cwd: Path = repository) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Provenance test")
    git("config", "user.email", "provenance@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", str(repository / ".git" / "hooks"))
    script_copy = repository / "tools" / "record_git_hash.py"
    script_copy.parent.mkdir()
    script_copy.write_text(SCRIPT.read_text(), encoding="utf-8")
    destination = repository / HASH_FILE
    destination.parent.mkdir(parents=True)
    (destination.parent.parent / "_build_provenance.py").write_text(
        BUILD_HELPER.read_text(), encoding="utf-8"
    )
    (destination.parent / "__init__.py").write_text("", encoding="utf-8")
    (repository / ".gitignore").write_text(HASH_FILE.as_posix() + "\n")
    git("add", ".")
    git("commit", "-m", "Initial test repository")

    hook = repository / ".git" / "hooks" / "post-commit"
    hook.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" tools/record_git_hash.py\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)
    git("commit", "--allow-empty", "-m", "Record new commit")
    assert destination.read_text().strip() == git("rev-parse", "HEAD")
    assert git("status", "--porcelain") == ""

    worktree = tmp_path / "linked-worktree"
    git("worktree", "add", "-b", "linked", str(worktree))
    # An ignored file must have a parent directory present in the linked checkout.
    (worktree / HASH_FILE).parent.mkdir(parents=True, exist_ok=True)
    git("commit", "--allow-empty", "-m", "Record worktree commit", cwd=worktree)
    assert (worktree / HASH_FILE).read_text().strip() == git(
        "rev-parse", "HEAD", cwd=worktree
    )
    assert destination.read_text().strip() == git("rev-parse", "HEAD")
    assert git("status", "--porcelain", cwd=worktree) == ""


def test_post_rewrite_records_amend_and_rebase_but_reset_needs_refresh(
    tmp_path: Path,
) -> None:
    """Exercise only post-rewrite so post-commit cannot mask a missing refresh."""
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repository, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Provenance test")
    git("config", "user.email", "provenance@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", str(repository / ".git" / "hooks"))
    script = repository / "tools" / "record_git_hash.py"
    script.parent.mkdir()
    script.write_text(SCRIPT.read_text(), encoding="utf-8")
    destination = repository / HASH_FILE
    destination.parent.mkdir(parents=True)
    (destination.parent.parent / "_build_provenance.py").write_text(
        BUILD_HELPER.read_text(), encoding="utf-8"
    )
    (destination.parent / "__init__.py").write_text("", encoding="utf-8")
    (repository / ".gitignore").write_text(HASH_FILE.as_posix() + "\n")
    git("add", ".")
    git("commit", "-m", "Initial test repository")
    initial = git("rev-parse", "HEAD")
    base_branch = git("branch", "--show-current")

    hook = repository / ".git" / "hooks" / "post-rewrite"
    hook.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" tools/record_git_hash.py\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)
    git("switch", "-c", "topic")
    (repository / "topic").write_text("topic\n", encoding="utf-8")
    git("add", "topic")
    git("commit", "-m", "Topic commit")
    assert not destination.exists()
    git("commit", "--amend", "-m", "Amended topic commit")
    amended = git("rev-parse", "HEAD")
    assert destination.read_text().strip() == amended

    git("switch", base_branch)
    (repository / "upstream").write_text("upstream\n", encoding="utf-8")
    git("add", "upstream")
    git("commit", "-m", "Advance upstream")
    git("switch", "topic")
    git("rebase", base_branch)
    rebased = git("rev-parse", "HEAD")
    assert rebased != amended
    assert destination.read_text().strip() == rebased

    git("reset", "--hard", initial)
    assert destination.read_text().strip() == rebased
    subprocess.run([sys.executable, str(script)], cwd=repository, check=True)
    assert destination.read_text().strip() == initial
    assert git("status", "--porcelain") == ""
