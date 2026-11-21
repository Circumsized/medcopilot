"""权威语料/蓝绿 manifest 快照与恢复工具(W5.5 + 备份演练)。

快照包含 corpus JSONL、index_manifest.json(若存在)、可选 chroma 目录 tar,
以及每个文件的 SHA-256/字节数清单(``snapshot_manifest.json``)。恢复前重算
全部哈希,验证通过才写入目标;分阶段落盘 + ``os.replace`` 原子发布,绝不留
半恢复状态,绝不删除用户数据。

RPO/RTO 目标由 RELEASE_CHECKLIST 声明(RPO≤24h,RTO≤30min);恢复演练第七/八
场景(recovery_drill)调用本模块的纯函数直调,不走子进程。

安全边界(备份是完整性边界,不是机密性边界):
- 快照只含公开说明书语料与 manifest,绝不包含密钥;权限建议 0600。
- ``verify`` 拒绝:清单缺失/损坏、绝对路径、``..`` 穿越、反斜杠路径、
  符号链接、清单外多余文件、哈希或字节数不符。
- ``take_snapshot`` 先写 ``.staging-*`` 再原子换名;目标同名目录已存在且
  非空 ⇒ 拒绝(绝不覆盖既有用户数据)。
- ``restore_snapshot`` 验证失败 ⇒ 不触碰目标文件一个字节(调用方用返回
  值的 ``verified`` 判定,演练场景断言 mtime/内容不变)。
- ``prune`` 只删除本工具自己产出的、带时间戳命名且含清单的快照目录;
  其他任何目录/文件、实时语料一律不碰。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.common.fsutil import atomic_write_json, durable_replace_bytes  # noqa: E402
from medcopilot.common.logger import logger  # noqa: E402

MANIFEST_NAME = "snapshot_manifest.json"
#: Timestamp-named snapshot directories, e.g. 20261007T161300Z or ...-2.
_SNAPSHOT_DIR_RE = re.compile(r"(\d{8})T(\d{6})Z(?:-\d+)?")
DEFAULT_KEEP = {"daily": 7, "weekly": 4, "monthly": 12}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yield_manifest_files(snapshot_dir: Path):
    """Walk the snapshot WITHOUT following symlinks (os.walk semantics)."""
    for current, dirnames, filenames in os.walk(snapshot_dir, followlinks=False):
        base = Path(current)
        for name in sorted(filenames):
            yield base / name


def _safe_relative_path(raw: object) -> Path | None:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        return None
    if "\\" in raw or "\x00" in raw:
        return None
    candidate = Path(raw)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate


def _path_is_symlinked(snapshot_dir: Path, relative: Path) -> bool:
    """Any component between the snapshot root and the file being a symlink."""
    current = snapshot_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _read_manifest(snapshot_dir: Path) -> tuple[dict | None, list[str]]:
    manifest_path = snapshot_dir / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None, ["manifest:missing-or-symlink"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"manifest:invalid:{exc}"]
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return None, ["manifest:invalid-schema"]
    return manifest, []


def verify(snapshot_dir: Path) -> tuple[bool, list[str]]:
    """校验快照清单:路径包容性、无符号链接、无清单外文件、逐文件哈希+字节数。

    返回 (是否通过, 错误列表)。错误码形如 ``path:``/``symlink:``/``missing:``/
    ``hash:``/``bytes:``/``unexpected:``/``duplicate:``,供 drill 与运维断言。
    """
    snapshot_dir = Path(snapshot_dir)
    errors: list[str] = []
    manifest, manifest_errors = _read_manifest(snapshot_dir)
    if manifest is None:
        return False, manifest_errors

    listed: set[str] = set()
    verified = 0
    for index, entry in enumerate(manifest["files"]):
        if not isinstance(entry, dict):
            errors.append(f"entry:{index}")
            continue
        relative = _safe_relative_path(entry.get("path"))
        posix = relative.as_posix() if relative is not None else None
        if (
            relative is None
            or posix in listed
            or posix == MANIFEST_NAME
        ):
            errors.append(f"path:{entry.get('path')!r}")
            continue
        listed.add(posix)
        path = snapshot_dir / relative
        # Containment re-check (paranoia; _safe_relative_path already rejects ..).
        try:
            path.relative_to(snapshot_dir)
        except ValueError:
            errors.append(f"path:{posix}")
            continue
        if _path_is_symlinked(snapshot_dir, relative):
            errors.append(f"symlink:{posix}")
        elif not path.is_file():
            errors.append(f"missing:{posix}")
        elif not isinstance(entry.get("sha256"), str) or sha256(path) != entry["sha256"]:
            errors.append(f"hash:{posix}")
        elif entry.get("bytes") != path.stat().st_size:
            errors.append(f"bytes:{posix}")
        else:
            verified += 1

    actual: set[str] = set()
    for path in _yield_manifest_files(snapshot_dir):
        actual.add(path.relative_to(snapshot_dir).as_posix())
    actual.discard(MANIFEST_NAME)
    errors.extend(f"unexpected:{path}" for path in sorted(actual - listed))
    return not errors, errors


def _write_manifest(staging: Path, *, corpus: str, entries: list[dict]) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "files": entries,
    }
    atomic_write_json(staging / MANIFEST_NAME, payload)


def _add_chroma_tar(chroma_dir: Path, target: Path) -> None:
    """Pack the (rebuildable) dense-index directory; dereferences symlinks
    (tar default stores them as links, which verify would then reject)."""
    with tarfile.open(target, "w:gz", dereference=True) as tar:
        tar.add(chroma_dir, arcname=chroma_dir.name)


def _fsync_file(path: Path) -> None:
    original: int | None = None
    try:
        original = path.stat().st_mode
        if not (original & stat.S_IWUSR):
            os.chmod(path, original | stat.S_IWUSR)
        with path.open("rb+") as handle:
            os.fsync(handle.fileno())
    except OSError as exc:  # pragma: no cover - best-effort file data flush
        logger.warning("File fsync failed for {} ({}).", path, exc)
    finally:
        if original is not None and not (original & stat.S_IWUSR):
            try:
                os.chmod(path, original)
            except OSError as exc:  # pragma: no cover
                logger.warning("Could not restore mode {:o} on {} ({}).", original, path, exc)


def take_snapshot(
    corpus_path: str | Path,
    manifest_path: str | Path | None,
    dest_root: str | Path,
    chroma_dir: str | Path | None = None,
    keep: dict | None = None,
) -> dict:
    """备份语料(+manifest,+可选 chroma tar)到 ``<dest_root>/<UTC 时间戳>/``。

    先写 ``.staging-*`` 目录、后 ``os.replace`` 原子发布;发布点已存在且非空
    ⇒ :class:`FileExistsError`——绝不覆盖既有快照。返回快照摘要(含每个文件
    的 sha256 与保留统计);``retention_after`` 反映 prune 后的留存情况。
    """
    corpus = Path(corpus_path)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    root = Path(dest_root)
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = root / f".staging-{stamp}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)  # only ever our own same-pid leftover
    staging.mkdir(parents=True)
    try:
        files: list[tuple[Path, str]] = [(corpus, "corpus.jsonl")]
        manifest_source = (
            Path(manifest_path) if manifest_path is not None
            else corpus.parent / "index_manifest.json"
        )
        if manifest_source.is_file():
            files.append((manifest_source, "index_manifest.json"))
        if chroma_dir is not None:
            chroma = Path(chroma_dir)
            if chroma.is_dir():
                tar_target = staging / "chroma.tar.gz"
                _add_chroma_tar(chroma, tar_target)
                files.append((tar_target, "chroma.tar.gz"))
        entries: list[dict] = []
        for source, relative in files:
            if source.is_symlink():
                raise ValueError(f"refusing to snapshot symlink source: {source}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # tar_target is already created inside staging; copying a file onto
            # itself raises shutil.SameFileError (https://docs.python.org/3/library/shutil.html).
            if source.resolve() != target.resolve():
                shutil.copy2(source, target, follow_symlinks=False)
            _fsync_file(target)
            entries.append({
                "path": relative,
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
                "source": str(source),
            })
        _write_manifest(staging, corpus="corpus.jsonl", entries=entries)

        # A second snapshot inside the same second (a drill step, a retry, a cron
        # firing twice) must get its own directory: the stamp is a readable label,
        # and failing here would make the backup path flaky. Existing snapshots are
        # still never overwritten — the name is disambiguated instead.
        destination = root / stamp
        suffix = 1
        while destination.exists() and (
            destination.is_symlink() or not destination.is_dir() or any(destination.iterdir())
        ):
            suffix += 1
            destination = root / f"{stamp}-{suffix}"
        if destination.exists():
            destination.rmdir()  # empty leftover from a crashed run at this second
        os.replace(staging, destination)
        _fsync_dir(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        removed = prune(root, keep or DEFAULT_KEEP)
    except OSError as exc:  # pragma: no cover - post-publish retention must not fail backup
        logger.warning("Snapshot prune failed in {} ({}); published snapshot kept.", root, exc)
        removed = []
    return {
        "snapshot_id": destination.name,
        "snapshot_dir": str(destination),
        "corpus": str(corpus),
        "files": entries,
        "retention_after": {"kept": len(entries), "removed_snapshots": removed},
    }


def _fsync_dir(directory: Path) -> None:
    if sys.platform == "win32":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError as exc:  # pragma: no cover - platform without dir fds
        logger.warning("Cannot open directory {} for fsync ({}).", directory, exc)
        return
    try:
        os.fsync(fd)
    except OSError as exc:  # pragma: no cover - best-effort durability
        logger.warning("Directory fsync failed for {} ({}).", directory, exc)
    finally:
        try:
            os.close(fd)
        except OSError as exc:  # pragma: no cover
            logger.warning("Directory fd close failed for {} ({}).", directory, exc)


def restore_snapshot(
    snapshot_dir: str | Path,
    target_corpus: str | Path,
    dry_run: bool = False,
) -> dict:
    """从快照恢复语料:先验签(哈希/包容性/无多余文件),后原子替换目标。

    - 验证失败 ⇒ 目标文件一个字节都不碰(绝不恢复损坏数据,绝不半恢复)。
    - 成功 ⇒ 内容先写 ``.<name>.restore-*`` 临时文件并 fsync,再
      ``os.replace`` 原子发布(目标要么旧态要么新态,无中间态)。
    - ``dry_run`` 只验证并报告,不写目标。
    """
    snapshot_dir = Path(snapshot_dir)
    target = Path(target_corpus)
    ok, errors = verify(snapshot_dir)
    manifest, _ = _read_manifest(snapshot_dir)
    total = len(manifest["files"]) if manifest else 0
    result = {
        "verified": ok,
        "errors": list(errors),
        "verified_files": 0,
        "total_files": total,
        "restored": None,
        "dry_run": bool(dry_run),
    }
    if not ok or manifest is None:
        return result

    relative = manifest.get("corpus")
    if not isinstance(relative, str) or _safe_relative_path(relative) is None:
        result["verified"] = False
        result["errors"].append("manifest:corpus-missing-or-invalid")
        return result
    source = snapshot_dir / relative
    if not source.is_file() or source.is_symlink():  # verify() already proved this
        result["verified"] = False
        result["errors"].append("corpus:missing")
        return result
    result["verified_files"] = len(manifest["files"])
    if dry_run:
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        def _copy(out):
            with open(source, "rb") as inp:
                shutil.copyfileobj(inp, out, length=1024 * 1024)

        durable_replace_bytes(target, _copy)
    except BaseException:
        raise
    result["restored"] = str(target)
    return result


def _parse_snapshot_time(name: str) -> datetime | None:
    match = _SNAPSHOT_DIR_RE.fullmatch(name)
    if match is None:
        return None
    try:
        return datetime.strptime(
            f"{match.group(1)}T{match.group(2)}Z", "%Y%m%dT%H%M%SZ"
        ).replace(tzinfo=timezone.utc)
    except ValueError:  # pragma: no cover - regex already bounds the shape
        return None


def prune(dest_root: str | Path, keep: dict | None = None) -> list[str]:
    """按 daily/weekly/monthly 分层保留,返回被删除的快照 id 列表。

    只删除"本工具产出"的快照目录:时间戳命名、非符号链接、且目录内含
    ``snapshot_manifest.json``。其他目录/文件、实时语料一律不碰(绝不删除
    用户数据)。
    """
    root = Path(dest_root)
    if not root.is_dir():
        return []
    policy = {**DEFAULT_KEEP, **(keep or {})}
    daily = max(0, int(policy.get("daily", 0)))
    weekly = max(0, int(policy.get("weekly", 0)))
    monthly = max(0, int(policy.get("monthly", 0)))

    now = datetime.now(timezone.utc)
    snapshots: list[tuple[datetime, Path]] = []
    for entry in sorted(root.iterdir()):
        parsed = _parse_snapshot_time(entry.name)
        if parsed is None or entry.is_symlink() or not entry.is_dir():
            continue
        if not (entry / MANIFEST_NAME).is_file():
            continue  # 不是本工具的快照目录:绝不碰
        snapshots.append((parsed, entry))

    kept: set[str] = set()

    def _newest_of(bucket: dict) -> None:
        for entries in bucket.values():
            # Tie-break on the directory name: two snapshots taken inside the same
            # second parse to the same instant, and breaking that tie arbitrarily
            # deleted the snapshot that had just been created (its name is the
            # later one, so the lexicographic order decides correctly).
            kept.add(max(entries, key=lambda item: (item[0], item[1].name))[1].name)

    by_day: dict = {}
    by_week: dict = {}
    by_month: dict = {}
    for parsed, entry in snapshots:
        age_days = (now - parsed).days
        if age_days < daily:
            by_day.setdefault(parsed.date(), []).append((parsed, entry))
        if age_days < weekly * 7:
            by_week.setdefault(parsed.isocalendar()[:2], []).append((parsed, entry))
        month_age = (now.year - parsed.year) * 12 + (now.month - parsed.month)
        if month_age < monthly:
            by_month.setdefault((parsed.year, parsed.month), []).append((parsed, entry))
    for bucket in (by_day, by_week, by_month):
        _newest_of(bucket)

    removed: list[str] = []
    for parsed, entry in snapshots:
        if entry.name in kept:
            continue
        # 再次确认仍是本工具的快照目录(竞态/误放文件不进删除集)
        if not (entry / MANIFEST_NAME).is_file() or entry.is_symlink():
            continue
        shutil.rmtree(entry)
        removed.append(entry.name)
    return removed


def snapshot(source_dir: Path, destination_root: Path) -> Path:
    """整目录快照(运维备份与 systemd 定时任务共用):复制目录树 + SHA-256 清单,原子发布并裁剪旧快照。"""
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{stamp}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        entries: list[dict] = []
        for source_path in sorted(source.rglob("*")):
            if ".git" in source_path.parts:
                continue
            if source_path.is_symlink():
                raise ValueError(f"refusing to snapshot symlink in source tree: {source_path}")
            if not source_path.is_file():
                continue
            relative = source_path.relative_to(source)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target, follow_symlinks=False)
            _fsync_file(target)
            entries.append({
                "path": relative.as_posix(),
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
                "source": str(source_path),
            })
        corpus_rel = next(
            (
                item["path"]
                for item in entries
                if item["path"] in ("corpus.jsonl", "documents.jsonl")
                or item["path"].endswith(".jsonl")
            ),
            None,
        )
        manifest_payload: dict = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source),
            "files": entries,
        }
        if corpus_rel is not None:
            manifest_payload["corpus"] = corpus_rel
        atomic_write_json(staging / MANIFEST_NAME, manifest_payload)
        destination = root / stamp
        suffix = 1
        while destination.exists() and (
            destination.is_symlink() or not destination.is_dir() or any(destination.iterdir())
        ):
            suffix += 1
            destination = root / f"{stamp}-{suffix}"
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        _fsync_dir(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        prune(root, DEFAULT_KEEP)
    except OSError as exc:  # pragma: no cover - post-publish retention must not fail backup
        logger.warning("Snapshot prune failed in {} ({}); published snapshot kept.", root, exc)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="snapshot medical corpus and index manifest")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="snapshot a whole directory (manual ops)")
    create.add_argument("source", type=Path)
    create.add_argument("destination", type=Path)

    verify_cmd = sub.add_parser("verify", help="verify a snapshot's checksum manifest")
    verify_cmd.add_argument("snapshot_dir", type=Path)

    restore = sub.add_parser("restore", help="verify then atomically restore the corpus")
    restore.add_argument("snapshot_dir", type=Path)
    restore.add_argument("target_corpus", type=Path)
    restore.add_argument("--dry-run", action="store_true")

    backup = sub.add_parser("backup", help="snapshot the authoritative corpus (scheduled)")
    backup.add_argument("--corpus", type=Path, required=True)
    backup.add_argument("--dest", type=Path, required=True)
    backup.add_argument("--manifest", type=Path, default=None)
    backup.add_argument("--include-chroma", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "create":
        print(snapshot(args.source, args.destination))
        return 0
    if args.command == "verify":
        ok, errors = verify(args.snapshot_dir)
        print(json.dumps({"passed": ok, "errors": errors}, ensure_ascii=False))
        return 0 if ok else 1
    if args.command == "restore":
        result = restore_snapshot(args.snapshot_dir, args.target_corpus, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["verified"] else 1
    result = take_snapshot(
        args.corpus, args.manifest, args.dest, chroma_dir=args.include_chroma
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
