"""医疗咨询审计日志(W1.2,safety-policy 合规面)。

设计依据:
- HIPAA §164.312(b):审计须"记录 + 审查";45 CFR §164.316(b)(2)(i):6 年保留;
- ONC/ASTM E2147-18:粒度支持事后法医重建(who/what/when),NTP 同步时钟;
- FDA Part 11 / 网络安全指南:记录防篡改(SHA-2 哈希链)、事件检测与日志。

硬边界:
- **免 PHI**:绝不记录 query/answer/OCR/历史正文;只记录意图、规则 id、
  chunk_id(内容寻址哈希,非内容)、计数、耗时、凭据派生 id。会话 id 取哈希。
- **fail-safe**:审计故障绝不影响请求成功路径;failures>0 ⇒ /health 如实报
  audit_ready=false。
- **哈希链防篡改不防删除**:链头由 /health 与 release_pack 外部锚定缓解。
- **启动校验**:既有文件在启动时整体重放校验;损坏 ⇒ sink 停用 + 响亮告警
  + /health 可见,**绝不静默从 GENESIS 重启**(那会让篡改者洗掉整段历史)。
- **链原子性**:prev_hash/record_hash/链头更新与落盘在同一把锁内完成;
  并发 record() 不可能读到同一个链头(否则链分叉)。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from medcopilot.common.logger import logger

AUDIT_SCHEMA_VERSION = 1
GENESIS_HASH = "GENESIS"
_KEY_ID_SALT = "audit-key-id"
_NTP_DELTA = 2208988800  # 1900-01-01 → 1970-01-01 seconds


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def hash_session(session_id: str) -> str:
    return hashlib.sha256((session_id or "").encode("utf-8")).hexdigest()[:16]


def derive_key_id(admin_key: str) -> str:
    """从密钥派生稳定 id:同一把密钥⇒同一 id;密钥本身不落盘。"""
    if not admin_key:
        return "anonymous"
    digest = hashlib.sha256(admin_key.encode("utf-8")).hexdigest()
    return hmac.new(digest.encode("utf-8"), _KEY_ID_SALT.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def sntp_offset_ms(server: str, timeout: float = 2.0) -> float | None:
    """最小 SNTP 客户端:返回本地时钟相对 NTP 服务器的偏移(毫秒);失败返回 None。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            request = b"\x1b" + 47 * b"\0"
            t1 = time.time()
            sock.sendto(request, (server, 123))
            data, _ = sock.recvfrom(48)
            t4 = time.time()
        if len(data) < 48:
            return None
        transmit_ts = struct.unpack("!12I", data)[10] - _NTP_DELTA
        server_time = transmit_ts + (t4 - t1) / 2
        return (server_time - (t4 + t1) / 2) * 1000.0
    except OSError:
        return None


@dataclass(frozen=True)
class AuditEvent:
    schema_version: int
    event_id: str
    ts_utc: str
    ts_mono: float
    event_type: str
    endpoint: str
    session_hash: str
    key_id: str
    intent: str
    is_emergency: bool
    red_rule_ids: tuple[str, ...]
    abstained: bool
    degradations: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]
    evidence_count: int
    policy_version: int
    latency_ms: float
    status_code: int
    prev_hash: str = GENESIS_HASH
    record_hash: str = ""

    def to_record(self) -> dict:
        payload = {key: value for key, value in self.__dict__.items()}
        payload["red_rule_ids"] = list(self.red_rule_ids)
        payload["degradations"] = list(self.degradations)
        payload["evidence_chunk_ids"] = list(self.evidence_chunk_ids)
        return payload


class AuditLogger:
    """追加式 JSONL 审计 sink + SHA-256 哈希链。线程安全。"""

    def __init__(
        self,
        path: str,
        *,
        retention_days: int = 2190,
        fsync: bool = False,
        ntp_server: str = "",
        clock=time.monotonic,
    ) -> None:
        self.path = path
        self.retention_days = retention_days
        self.fsync_enabled = fsync
        self.ntp_server = ntp_server
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._records_today = 0
        self._chain_head = GENESIS_HASH
        self._chain_verified = False
        self._chain_broken_at = 0
        self._startup_error: str | None = None
        self._ready = bool(path)
        self._ntp_probed = False
        self._time_offset_ms: float | None = None
        self._init_chain()

    # ------------------------------------------------------------ chain

    def _replay(self) -> tuple[bool, int, str]:
        """重放整条链;返回 (完整, 断裂行号(0=完整), 最后一条 record_hash)。

        空/缺失文件视为空链(完整);任何结构损坏、断链或不可读都返回
        not-intact——调用方必须失败可见,绝不静默重启 GENESIS。
        """
        if not self.path or not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            return True, 0, GENESIS_HASH
        prev = GENESIS_HASH
        try:
            with open(self.path, encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        return False, lineno, prev
                    if not isinstance(record, dict):
                        return False, lineno, prev
                    stored = record.get("record_hash", "")
                    # 与 record() 写入时的字典形态一致:record_hash 键存在但为空串
                    record["record_hash"] = ""
                    if record.get("prev_hash") != prev or AuditLogger._chain_digest(record, prev) != stored:
                        return False, lineno, prev
                    prev = stored
        except OSError:
            return False, 0, prev
        return True, 0, prev

    def _init_chain(self) -> None:
        """启动时整体重放校验:健康 ⇒ 接链头;损坏 ⇒ 停用 sink 并响亮告警。

        旧实现遇坏行把 last_hash 重置为 GENESIS 后继续——新记录会接到一个
        干净的 GENESIS 上,历史被悄悄孤立(静默新 genesis)。现在损坏即
        停用:/health 的 audit_ready 立刻变 false,故障可见、可审计。
        """
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        except OSError as exc:
            self._ready = False
            self._startup_error = f"audit sink directory unavailable: {exc}"
            return
        ok, broken, last_hash = self._replay()
        if ok:
            self._chain_head = last_hash
            self._chain_verified = True
            return
        self._chain_head = GENESIS_HASH
        self._chain_verified = False
        self._chain_broken_at = broken
        self._ready = False
        self._startup_error = (
            f"audit chain verification failed (broken at line {broken or 'unreadable'}); "
            "sink disabled rather than silently restarting a new genesis"
        )
        logger.error("Audit sink {}: {}", self.path, self._startup_error)

    @staticmethod
    def _chain_digest(payload: dict, prev_hash: str) -> str:
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False) + "|" + prev_hash
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def verify_chain(self) -> tuple[bool, int]:
        """重放文件校验哈希链;返回 (是否完整, 断裂行号)。行号从 1 开始,0 表示完整。"""
        ok, broken, _ = self._replay()
        return ok, broken

    # ------------------------------------------------------------ record

    def record(self, **fields) -> AuditEvent | None:
        """追加一条审计事件。任何 I/O 失败只累加 failures,绝不向调用方抛错。

        prev_hash/链摘要/链头更新/落盘在同一把锁内完成:两个并发 record()
        不可能基于同一个链头各自成链(断链分叉)。emergency 的"非抛错"契约
        保持不变——即使 sink 停用(损坏/不可写),急救回答照常返回。
        """
        if not self._ready:
            with self._lock:
                self._failures += 1
            return None
        with self._lock:
            try:
                event = AuditEvent(
                    schema_version=AUDIT_SCHEMA_VERSION,
                    event_id=uuid.uuid4().hex,
                    ts_utc=utc_now_iso(),
                    ts_mono=self._clock(),
                    red_rule_ids=tuple(fields.get("red_rule_ids") or ()),
                    degradations=tuple(fields.get("degradations") or ()),
                    evidence_chunk_ids=tuple(fields.get("evidence_chunk_ids") or ()),
                    prev_hash=self._chain_head,
                    record_hash="",
                    **{k: v for k, v in fields.items() if k not in {"red_rule_ids", "degradations", "evidence_chunk_ids"}},
                )
                event = AuditEvent(**{
                    **event.__dict__,
                    "record_hash": self._chain_digest(event.to_record(), self._chain_head),
                })
                serialized = json.dumps(event.to_record(), ensure_ascii=False) + "\n"
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    if self.fsync_enabled:
                        os.fsync(handle.fileno())
                self._chain_head = event.record_hash
                self._records_today += 1
                return event
            except (OSError, TypeError, ValueError) as exc:
                # A partial write may already be on disk; retrying with the old
                # chain head can fork the chain. Recovery requires revalidation.
                self._failures += 1
                if isinstance(exc, OSError):
                    self._ready = False
                logger.error("Audit sink record failed ({}); sink disabled until recovery.", type(exc).__name__)
                return None

    # ------------------------------------------------------------ status

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def records_today(self) -> int:
        return self._records_today

    @property
    def chain_verified(self) -> bool:
        """启动时整链重放是否通过(损坏 sink  ⇒ False,绝不静默新 genesis)。"""
        return self._chain_verified

    @property
    def chain_broken_at(self) -> int:
        """启动校验发现的断链行号(0 = 完整/未检出)。"""
        return self._chain_broken_at

    @property
    def startup_error(self) -> str | None:
        """启动校验失败原因(供 /health 与运维查询;None = 健康)。"""
        return self._startup_error

    def chain_head(self) -> str:
        return self._chain_head

    def time_offset_ms(self) -> float | None:
        if not self.ntp_server:
            return None
        if not self._ntp_probed:
            self._time_offset_ms = sntp_offset_ms(self.ntp_server)
            self._ntp_probed = True
        return self._time_offset_ms


@dataclass
class AuditContext:
    """端点级审计上下文:计时 + 派生 key_id。"""

    endpoint: str
    key_id: str
    status_code: int = 200
    _start: float = field(default_factory=time.monotonic, repr=False)

    def latency_ms(self, clock=time.monotonic) -> float:
        return round((clock() - self._start) * 1000.0, 3)
