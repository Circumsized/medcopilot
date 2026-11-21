"""W5.5 backup snapshot contract tests."""
from __future__ import annotations

import json

from medcopilot.scripts.backup_snapshot import snapshot, take_snapshot, verify


def test_snapshot_and_tamper_detection(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "corpus.jsonl").write_text('{"drug":"A"}\n', encoding="utf-8")
    destination = tmp_path / "backups"
    created = snapshot(source, destination)
    ok, errors = verify(created)
    assert ok is True and errors == []
    (created / "corpus.jsonl").write_text("tampered\n", encoding="utf-8")
    ok, errors = verify(created)
    assert ok is False and any(error.startswith("hash:") for error in errors)


def test_snapshot_manifest_is_machine_readable(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "index_manifest.json").write_text('{"collection":"blue"}', encoding="utf-8")
    created = snapshot(source, tmp_path / "backups")
    manifest = json.loads((created / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["path"] == "index_manifest.json"
    assert len(manifest["files"][0]["sha256"]) == 64


def test_create_snapshot_roundtrips_through_restore_cli(tmp_path):
    from medcopilot.scripts.backup_snapshot import main

    source = tmp_path / "corpus"
    source.mkdir()
    payload = '{"page_content":"A","metadata":{"chunk_id":"1","drug_name":"A"}}\n'
    (source / "corpus.jsonl").write_text(payload, encoding="utf-8")
    backups = tmp_path / "backups"
    assert main(["create", str(source), str(backups)]) == 0
    snap_dir = next(p for p in backups.iterdir() if p.is_dir())
    assert main(["verify", str(snap_dir)]) == 0
    restored = tmp_path / "restored" / "corpus.jsonl"
    assert main(["restore", str(snap_dir), str(restored)]) == 0
    assert restored.read_text(encoding="utf-8") == payload


def test_take_snapshot_with_chroma_dir_succeeds_and_verifies(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text('{"drug":"青霉素"}\n', encoding="utf-8")
    corpus_file.chmod(0o444)
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (chroma_dir / "index.bin").write_bytes(b"chroma-bytes")

    try:
        snap_info = take_snapshot(corpus_file, None, tmp_path / "backups_take", chroma_dir=chroma_dir)
        ok, errors = verify(tmp_path / "backups_take" / snap_info["snapshot_id"])
        assert ok is True and errors == []
        paths = {entry["path"] for entry in snap_info["files"]}
        assert "chroma.tar.gz" in paths
    finally:
        corpus_file.chmod(0o644)


def test_prune_only_deletes_tool_snapshots_with_manifest(tmp_path):
    from datetime import datetime, timezone

    from medcopilot.scripts.backup_snapshot import MANIFEST_NAME, prune

    root = tmp_path / "backups"
    root.mkdir()
    snap_old = root / "20200101T000000Z"
    snap_old.mkdir()
    (snap_old / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    today_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_new = root / today_stamp
    snap_new.mkdir()
    (snap_new / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    unmanaged_ts = root / "20200102T000000Z"
    unmanaged_ts.mkdir()
    (unmanaged_ts / "important.txt").write_text("keep", encoding="utf-8")
    user_dir = root / "live_corpus"
    user_dir.mkdir()
    (user_dir / "corpus.jsonl").write_text("keep", encoding="utf-8")

    removed = prune(root, keep={"daily": 1, "weekly": 0, "monthly": 0})
    assert removed == ["20200101T000000Z"]
    assert not snap_old.exists()
    assert snap_new.exists()
    assert unmanaged_ts.exists()
    assert user_dir.exists()

