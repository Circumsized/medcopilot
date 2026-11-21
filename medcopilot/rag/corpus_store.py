"""权威语料外置存储（Slice 2）。

权威副本是进程外的 JSONL 文件（单行一个 chunk），进程内 dict 仅为服务
快照。写路径是 write-through：先经 ``common.fsutil.durable_replace``
持久落盘（fsync 数据 → ``os.replace`` 原子换名 → POSIX 下 fsync 目录），
成功后才提交内存快照——写盘失败时内存与磁盘都保持原状，摄取整体失败。

摄取是幂等且按药品原子替换的：重新摄取同一药品会删除其全部旧 chunk 再
插入新集合，其他药品不受影响；chunk_id 由 ClinicalChunker 内容寻址生成，
同内容重摄零增量。坏记录在加载时跳过并置 ``degraded``，由 /health 暴露。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from medcopilot.common.fsutil import durable_replace
from medcopilot.common.logger import logger
from medcopilot.rag.clinical_chunker import Document


class CorpusStore:
    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._records: dict[str, Document] = {}
        self._degraded = False

    # ------------------------------------------------------------------ state

    @property
    def persisted(self) -> bool:
        """True 当配置了权威文件路径（持久化已启用）。"""
        return self._path is not None

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def snapshot(self) -> list[Document]:
        with self._lock:
            return list(self._records.values())

    # ------------------------------------------------------------------ load

    def load(self) -> list[Document]:
        """从权威文件加载；坏记录跳过并标记 degraded，不阻断启动。"""
        with self._lock:
            self._records = {}
            self._degraded = False
            if self._path is None or not self._path.exists():
                return []
            try:
                raw_lines = self._path.read_bytes().splitlines()
            except OSError as exc:
                logger.warning("Corpus store {} could not be read ({}); serving degraded.", self._path, exc)
                self._degraded = True
                return []
            for line_no, raw_bytes in enumerate(raw_lines, 1):
                try:
                    raw = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("Corpus line {} in {} is not valid UTF-8; skipped.", line_no, self._path)
                    self._degraded = True
                    continue
                if line_no == 1:
                    raw = raw.lstrip("\ufeff")
                line = raw.strip()
                if not line:
                    continue
                record = self._parse_line(line)
                if record is None:
                    logger.warning("Corpus line {} in {} is invalid; skipped.", line_no, self._path)
                    self._degraded = True
                    continue
                chunk_id = record.metadata["chunk_id"]
                if chunk_id in self._records:
                    # Content-addressed ids make true collisions equal-content,
                    # but a duplicated line in the authoritative file still hides
                    # a record behind the later one: keep the later record
                    # (later-wins is unchanged) and flag the store degraded so
                    # /health does not report a healthy corpus over a lost one.
                    logger.warning(
                        "Corpus line {} in {} repeats chunk_id {}; keeping the later record.",
                        line_no, self._path, chunk_id,
                    )
                    self._degraded = True
                self._records[chunk_id] = record
            if self._degraded:
                logger.warning(
                    "Corpus store {} loaded with skipped records; serving degraded.",
                    self._path,
                )
            return list(self._records.values())

    @staticmethod
    def _parse_line(line: str) -> Document | None:
        try:
            payload = json.loads(line)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        content = payload.get("page_content")
        metadata = payload.get("metadata")
        if not isinstance(content, str) or not content:
            return None
        if not isinstance(metadata, dict):
            return None
        chunk_id = metadata.get("chunk_id")
        drug_name = metadata.get("drug_name")
        if not isinstance(chunk_id, str) or not chunk_id:
            return None
        if not isinstance(drug_name, str) or not drug_name:
            return None
        return Document(page_content=content, metadata=metadata)

    # ----------------------------------------------------------------- upsert

    def upsert_drug(self, drug_name: str, docs: list[Document]) -> tuple[int, int]:
        """按药品原子替换该药全部 chunk；返回 ``(added, replaced)``。

        盘为权威：先原子落盘新全集，成功后才提交内存快照。写盘失败抛异常，
        此时磁盘旧文件与内存快照均不变。
        """
        clean_name = (drug_name or "").strip()
        if not clean_name:
            raise ValueError("drug_name must not be blank")
        with self._lock:
            incoming: dict[str, Document] = {}
            for doc in docs:
                chunk_id = doc.metadata.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ValueError("every ingested chunk requires a chunk_id")
                incoming[chunk_id] = doc

            # ponytail: 下面是两次 O(n) 全量扫描（n = 语料总 chunk 数）。药品说明书
            # 语料为数百~数千 chunk 量级，单次摄取可忽略；若语料增长到 10^5 量级，
            # 按 drug_name 维护二级索引（drug -> set(chunk_id)）即可把替换与计数
            # 降到 O(该药品 chunk 数)。
            added = sum(1 for cid in incoming if cid not in self._records)
            replaced = sum(
                1
                for cid, doc in self._records.items()
                if doc.metadata.get("drug_name") == clean_name and cid not in incoming
            )
            new_records = {
                cid: doc
                for cid, doc in self._records.items()
                if doc.metadata.get("drug_name") != clean_name
            }
            new_records.update(incoming)

            if self._path is not None:
                self._atomic_write(new_records)
            self._records = new_records
            return added, replaced

    # --------------------------------------------------------------- persist

    def _atomic_write(self, records: dict[str, Document]) -> None:
        def _write(handle) -> None:
            for doc in records.values():
                payload = {"page_content": doc.page_content, "metadata": doc.metadata}
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")

        # Atomic visibility + (POSIX) durability: fsync the data before the swap
        # and the parent directory after, so a crash cannot leave a truncated
        # authoritative corpus under the target name.
        durable_replace(self._path, _write)
