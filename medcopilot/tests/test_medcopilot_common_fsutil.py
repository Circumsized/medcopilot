"""Contract for the durable atomic-replace helper (common/fsutil.py) and common/config.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from medcopilot.common import fsutil
from medcopilot.common.config import AppConfig
from medcopilot.common.fsutil import (
    atomic_write_json,
    durable_replace,
    durable_replace_bytes,
    git_metadata,
)


def test_replaces_existing_file_content(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("old", encoding="utf-8")
    durable_replace(target, lambda h: h.write("new"))
    assert target.read_text(encoding="utf-8") == "new"


def test_creates_file_and_missing_parent_dirs_with_same_dir_temp(tmp_path):
    target = tmp_path / "nested" / "deep" / "data.txt"
    seen_siblings: list[str] = []

    def write_and_observe(handle):
        seen_siblings.extend(p.name for p in target.parent.iterdir())
        handle.write("hello")

    durable_replace(target, write_and_observe)
    assert any(name.startswith(".data.txt-") and name.endswith(".tmp") for name in seen_siblings)
    assert target.read_text(encoding="utf-8") == "hello"
    assert list(target.parent.iterdir()) == [target]


def test_failed_write_keeps_previous_content_and_cleans_temp(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")

    def boom(handle):
        handle.write("partial")
        raise OSError("disk full")

    with pytest.raises(OSError):
        durable_replace(target, boom)

    assert target.read_text(encoding="utf-8") == "original"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "data.txt"]
    assert leftovers == [], f"temp files must be cleaned up, found {leftovers}"


def test_fsync_is_invoked_on_the_written_file(tmp_path, monkeypatch):
    """The durability step must actually call fsync on the temp file's data."""
    calls: list[int] = []
    real_fsync = os.fsync

    def spy_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    target = tmp_path / "data.txt"
    durable_replace(target, lambda h: h.write("durable"))
    assert calls, "durable_replace must fsync before publishing"


def test_failed_replace_keeps_previous_content_and_cleans_temp(tmp_path, monkeypatch):
    """``os.replace`` can fail on real platforms (a Windows sharing violation).
    The target must survive untouched and no temp residue may remain."""
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")

    def broken_replace(src, dst):
        raise OSError("target locked")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        durable_replace(target, lambda h: h.write("new"))

    assert target.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


def test_fsync_failure_propagates_and_keeps_previous_content(tmp_path, monkeypatch):
    """fsyncgate: a failing fsync must FAIL the write (never report success)."""
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")

    def broken_fsync(fd):
        raise OSError("EIO during fsync")

    monkeypatch.setattr(os, "fsync", broken_fsync)
    with pytest.raises(OSError):
        durable_replace(target, lambda h: h.write("new"))

    assert target.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


def test_fsync_precedes_replace_and_directory_fsync_follows(tmp_path, monkeypatch):
    """Order contract: the data fsync must happen BEFORE the rename and, on POSIX,
    the directory fsync must follow the rename."""
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    target = tmp_path / "data.txt"
    durable_replace(target, lambda h: h.write("ordered"))

    assert events[0] == "fsync" and events[1] == "replace", events
    if sys.platform != "win32":
        assert events == ["fsync", "replace", "fsync"], events
    else:
        assert events == ["fsync", "replace"], events


def test_text_mode_writes_lf_without_platform_translation(tmp_path):
    """``newline=""`` is load-bearing: default text mode would translate ``\\n``
    to ``os.linesep`` on Windows."""
    target = tmp_path / "lines.txt"
    durable_replace(target, lambda h: h.write("a\nb\n"))
    assert target.read_bytes() == b"a\nb\n"


def test_durable_replace_bytes_and_atomic_write_json(tmp_path, monkeypatch):
    bin_target = tmp_path / "blob.bin"
    durable_replace_bytes(bin_target, lambda f: f.write(b"\x00\x01\x02"))
    assert bin_target.read_bytes() == b"\x00\x01\x02"

    # Failure in durable_replace_bytes keeps old bytes and leaves no temp file.
    def boom(handle):
        handle.write(b"\xff")
        raise RuntimeError("binary write failed")

    with pytest.raises(RuntimeError, match="binary write failed"):
        durable_replace_bytes(bin_target, boom)
    assert bin_target.read_bytes() == b"\x00\x01\x02"
    assert list(tmp_path.iterdir()) == [bin_target]

    # Replace failure in durable_replace_bytes also keeps old bytes and cleans temp.
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("replace locked")))
    with pytest.raises(OSError, match="replace locked"):
        durable_replace_bytes(bin_target, lambda f: f.write(b"\x09"))
    monkeypatch.setattr(os, "replace", real_replace)
    assert bin_target.read_bytes() == b"\x00\x01\x02"
    assert list(tmp_path.iterdir()) == [bin_target]

    json_target = tmp_path / "config.json"
    atomic_write_json(json_target, {"drug": "布洛芬", "chunks": 3})
    raw_text = json_target.read_text(encoding="utf-8")
    assert "布洛芬" in raw_text, "ensure_ascii=False must preserve UTF-8 CJK characters"
    assert json.loads(raw_text) == {"drug": "布洛芬", "chunks": 3}


def test_git_metadata_success_and_graceful_degradation(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    status_output = " M file.py\n"

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef1234\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=status_output, stderr="")

    monkeypatch.setattr(fsutil.subprocess, "run", fake_run)
    meta = git_metadata(repo_root=tmp_path)
    assert meta == {"commit": "deadbeef1234", "dirty": True}
    assert any("--no-optional-locks" in cmd for cmd in calls)

    status_output = "   \n"
    assert git_metadata(repo_root=tmp_path) == {"commit": "deadbeef1234", "dirty": False}

    def failing_run(*args, **kwargs):
        raise subprocess.SubprocessError("git unavailable")

    monkeypatch.setattr(fsutil.subprocess, "run", failing_run)
    assert git_metadata(repo_root=tmp_path) is None


def test_log_level_validator_normalizes_and_rejects_invalid(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "warn")
    cfg = AppConfig(_env_file=None)
    assert cfg.log_level == "WARNING"
    cfg.log_level = "debug"
    assert cfg.log_level == "DEBUG"
    monkeypatch.setenv("LOG_LEVEL", "  ")
    assert AppConfig(_env_file=None).log_level == "INFO"
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")
    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        AppConfig(_env_file=None)
