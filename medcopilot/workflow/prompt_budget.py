"""W6.4 确定性 token 预算与医疗优先级截断(无 LLM 自评、无新增依赖)。

设计锚点(W6.4):

* 估算器:1 个 CJK 字符 = 1 token;非 CJK 4 字符 = 1 token(向上取整——宁可高估,
  预算判断只会更保守)。预算覆盖 **system + query + 所有围栏/章节头**,不只证据。
* 裁剪优先级:``禁忌 4 > 用法用量 3 > 不良反应 2 > 适应症 1 > 其它 0``,同级按
  rerank_score 升序先丢;其余安全章节(相互作用/注意事项/特殊人群…)与禁忌同档,最后丢。
* 降级阶梯:L0 正常 → L1 砍历史轮次 → L2 砍 OCR 行 → L3 按优先级丢/裁 chunk →
  L4 头保留截断 → L5 仍超 ⇒ 不调 LLM 弃答(调用方记 ``llm:budget-exceeded``)。
  任一阶梯触发 ⇒ 调用方记 ``llm:context-truncated`` + 结构化日志。

**硬不变量**:prompt 里的证据文本与调用方最终 ``state.evidence`` 同源同文本——
截断只发生一次(本模块),extractive gate / DosageGate 因此不会被"截断漂移"破坏。
裁剪一律头保留、句边界收尾,且绝不切开剂量片段(不留半句剂量)。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medcopilot.common.config import config
from medcopilot.workflow.safety_guard import DosageGate

# 每条 message 的信封开销(role 等),与公开计费口径一致的量级常数。
MESSAGE_OVERHEAD_TOKENS = 4

# 用户消息的章节头与空证据兜底(与 agent_graph 既有 prompt 逐字节一致)。
EVIDENCE_HEADER = "【权威医学参考资料】"
QUERY_HEADER = "【患者主诉/提问】"
EMPTY_EVIDENCE_NOTE = "无对应本地药品说明书。"

# 医疗优先级(设计稿 W6.4):禁忌 > 用法用量 > 不良反应 > 适应症 > 其它。
SECTION_PRIORITY = {"禁忌": 4, "用法用量": 3, "不良反应": 2, "适应症": 1}
OTHER_SECTION_PRIORITY = 0
# 其余安全章节与禁忌同档:同一个集合即检索层 ContraindicationTrigger.CONTRA_SECTIONS。
SAFETY_SECTION_PRIORITY = 4
SAFETY_SECTIONS = frozenset({
    "禁忌", "药物相互作用", "注意事项",
    "孕妇及哺乳期妇女用药", "儿童用药", "老年用药",
})

# CJK 表意文字/假名/全角:1 字符 = 1 token。
_CJK_RANGES = ((0x2E80, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))

# 句边界:裁剪只在这些字符之后收刀,避免半句(含半句剂量)。
_BOUNDARY_CHARS = "。！？；!?;\n"

# 剂量片段(与 clinical_chunker.DOSE_PROTECT_PATTERN 同源同哲学):数字+单位、
# 频次表达。裁剪点落在片段内部时回退到片段起点。
_DOSE_SPAN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|g|kg|ug|μg|µg|ng|ml|l|iu|u|克|毫克|毫升|微克|"
    r"单位|片|粒|袋|支|瓶|丸|滴|枚|贴|喷|%)|"
    r"(?:一日|每日|每天|一次|每次|一周|每周)\s*[一二三四五六七八九十\d]+\s*"
    r"(?:次|片|粒|袋|支|瓶|丸|滴|枚|贴|喷|毫升|克|毫克)|"
    + DosageGate.DOSAGE_SENTENCE_RE.pattern,
    re.IGNORECASE,
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """1 个 CJK 字符 = 1 token;其余 4 字符 = 1 token(向上取整)。"""
    if not text:
        return 0
    cjk = sum(1 for char in text if _is_cjk(char))
    return cjk + math.ceil((len(text) - cjk) / 4)


def estimate_messages_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    """system + user(围栏/章节头/证据/OCR/历史/query)+ 每条 message 的信封开销。"""
    return sum(
        estimate_tokens(str(message.get("content", ""))) + MESSAGE_OVERHEAD_TOKENS for message in messages
    )


def _safe_cut(text: str, cut: int) -> int:
    """把 cut 回退到句边界;若切在剂量片段中间则回退到片段起点(不留半句剂量)。"""
    if cut <= 0 or cut >= len(text):
        return cut
    boundary = max((text.rfind(char, 0, cut) for char in _BOUNDARY_CHARS), default=-1)
    if boundary >= 0:
        cut = boundary + 1
    for match in _DOSE_SPAN.finditer(text):
        if match.start() < cut < match.end():
            return match.start()
    return cut


def _prefix_within(text: str, max_tokens: int) -> int:
    """最长前缀长度 n,使 estimate_tokens(text[:n]) <= max_tokens(单调 ⇒ 二分)。"""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return low


def clip_head(text: str, max_tokens: int) -> str:
    """头保留裁剪:保留开头,句边界收尾,绝不切开剂量片段。"""
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    return text[: _safe_cut(text, _prefix_within(text, max_tokens))]


def clip_ocr_text(text: str, *, max_lines: int, max_line_chars: int) -> str:
    """OCR 收敛:只保留前 max_lines 个非空行,每行最多 max_line_chars 字符(句边界优先)。"""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines[:max_lines]:
        if len(line) > max_line_chars:
            boundary = max((line.rfind(char, 0, max_line_chars) for char in _BOUNDARY_CHARS), default=-1)
            cut = boundary + 1 if boundary >= max_line_chars - 32 else max_line_chars
            for match in _DOSE_SPAN.finditer(line):
                if match.start() < cut < match.end():
                    cut = match.start()
                    break
            line = line[:cut]
        kept.append(line)
    return "\n".join(kept)


def build_messages(
    *,
    system_text: str,
    query: str,
    evidence_texts: Sequence[str] = (),
    ocr_text: str = "",
    history_block: str = "",
    fence: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """组装 messages:system=策略原则,user=不可信历史/OCR/证据/query。

    历史与 OCR 单独标记为不可信上下文,不归属医学证据章节。
    ``fence`` 决定每份输入的围栏函数。
    """
    wrap = fence or (lambda text: text)
    context_str = "\n\n".join(wrap(text) for text in evidence_texts) if evidence_texts else EMPTY_EVIDENCE_NOTE
    untrusted_context = [wrap(text) for text in (history_block, ocr_text) if text]
    context_prefix = ""
    if untrusted_context:
        context_prefix = "【不可信上下文（历史/OCR，非医学证据）】\n" + "\n\n".join(untrusted_context) + "\n\n"
    user_text = f"{context_prefix}{EVIDENCE_HEADER}\n{context_str}\n\n{QUERY_HEADER}\n{wrap(query)}"
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


@dataclass(frozen=True)
class PromptBudget:
    """一套 prompt 预算(默认值来自 AppConfig,测试/运维可显式覆盖)。"""

    total: int = field(default_factory=lambda: config.llm_prompt_token_budget)
    evidence: int = field(default_factory=lambda: config.llm_evidence_token_budget)
    chunk_cap: int = field(default_factory=lambda: config.llm_chunk_token_cap)
    ocr_max_lines: int = field(default_factory=lambda: config.ocr_max_lines)
    ocr_max_line_chars: int = field(default_factory=lambda: config.ocr_max_line_chars)


@dataclass(frozen=True)
class EvidenceItem:
    """一份进入 prompt 的证据(``doc`` 是调用方的文档句柄,本模块只认 text)。"""

    text: str
    section: str = ""
    rerank_score: float = 0.0
    doc: Any = None


@dataclass(frozen=True)
class BudgetFit:
    """一次预算装配:messages + 最终证据(与 prompt 同文本)+ 降级足迹。"""

    messages: list[dict[str, str]]
    evidence: list[EvidenceItem]
    truncated: bool
    exceeded: bool
    original_tokens: int
    final_tokens: int
    steps: tuple[str, ...] = ()
    dropped_sections: tuple[str, ...] = ()


class PromptBudgetFitter:
    """按 L0→L5 阶梯把 prompt 收进预算;无状态,可并发复用。"""

    def __init__(
        self,
        budget: PromptBudget,
        *,
        protected_sections: Iterable[str] = (),
        fence: Callable[[str], str] | None = None,
    ) -> None:
        self.budget = budget
        self.protected_sections = frozenset(protected_sections)
        self._fence = fence or (lambda text: text)

    # --------------------------------- 装配 ---------------------------------

    def _build(
        self,
        *,
        system_text: str,
        query: str,
        evidence_texts: Sequence[str],
        history_block: str,
        ocr_text: str,
    ) -> list[dict[str, str]]:
        return build_messages(
            system_text=system_text,
            query=query,
            evidence_texts=evidence_texts,
            ocr_text=ocr_text,
            history_block=history_block,
            fence=self._fence,
        )

    def _tokens(
        self, *, system_text: str, query: str, chunks: Sequence[EvidenceItem], history_block: str, ocr_text: str
    ) -> int:
        return estimate_messages_tokens(self._build(
            system_text=system_text, query=query, evidence_texts=[chunk.text for chunk in chunks],
            history_block=history_block, ocr_text=ocr_text,
        ))

    # -------------------------------- 排序键 --------------------------------

    def _priority(self, section: str) -> int:
        if section in self.protected_sections or section in SAFETY_SECTIONS:
            return SAFETY_SECTION_PRIORITY
        return SECTION_PRIORITY.get(section, OTHER_SECTION_PRIORITY)

    def _drop_candidate(self, chunks: Sequence[EvidenceItem]) -> int:
        """优先级最低 ⇒ rerank_score 最低 ⇒ 位置最靠后;保护章节排在最后丢。"""
        return min(
            range(len(chunks)),
            key=lambda index: (self._priority(chunks[index].section), chunks[index].rerank_score, -index),
        )

    def _truncate_candidate(self, chunks: Sequence[EvidenceItem]) -> int:
        """优先级最低、体量最大的块先让(优先级相同时看 token 数)。"""
        return min(
            range(len(chunks)),
            key=lambda index: (self._priority(chunks[index].section), -estimate_tokens(chunks[index].text)),
        )

    def _cap_chunk(self, chunk: EvidenceItem) -> EvidenceItem:
        text = clip_head(chunk.text, self.budget.chunk_cap)
        if text == chunk.text:
            return chunk
        return EvidenceItem(text=text, section=chunk.section, rerank_score=chunk.rerank_score, doc=chunk.doc)

    # --------------------------------- 入口 ---------------------------------

    def fit(
        self,
        *,
        system_text: str,
        query: str,
        evidence: Sequence[EvidenceItem],
        ocr_text: str = "",
        history_blocks: Sequence[str] = (),
    ) -> BudgetFit:
        chunks = list(evidence)
        steps: list[str] = []
        dropped: list[str] = []
        history_block = history_blocks[0] if history_blocks else ""
        parts = dict(system_text=system_text, query=query, history_block=history_block, ocr_text=ocr_text)

        original_tokens = self._tokens(chunks=chunks, **parts)

        # 单 chunk 硬上限 + 证据总预算:与总预算无关,始终生效(医疗优先级裁剪)。
        capped = [self._cap_chunk(chunk) for chunk in chunks]
        if [chunk.text for chunk in capped] != [chunk.text for chunk in chunks]:
            chunks = capped
            steps.append("chunk-cap")
        evidence_tokens = sum(estimate_tokens(chunk.text) for chunk in chunks)
        while evidence_tokens > self.budget.evidence and len(chunks) > 1:
            victim = chunks.pop(self._drop_candidate(chunks))
            dropped.append(victim.section or "其它")
            steps.append("evidence-drop")
            evidence_tokens -= estimate_tokens(victim.text)

        if self._tokens(chunks=chunks, **parts) <= self.budget.total:
            return self._fit_result(chunks, steps, dropped, original_tokens, truncated=bool(steps), **parts)

        # ---- L1:砍历史轮次(逐级缩短,能收进预算就停)
        for alternative in history_blocks[1:]:
            if alternative == history_block:
                continue
            history_block = parts["history_block"] = alternative
            steps.append("history")
            if self._tokens(chunks=chunks, **parts) <= self.budget.total:
                return self._fit_result(chunks, steps, dropped, original_tokens, truncated=True, **parts)

        # ---- L2:砍 OCR 行
        if ocr_text:
            clipped_ocr = clip_ocr_text(
                ocr_text, max_lines=self.budget.ocr_max_lines, max_line_chars=self.budget.ocr_max_line_chars
            )
            if clipped_ocr != ocr_text:
                parts["ocr_text"] = clipped_ocr
                steps.append("ocr")
                if self._tokens(chunks=chunks, **parts) <= self.budget.total:
                    return self._fit_result(chunks, steps, dropped, original_tokens, truncated=True, **parts)

        # ---- L3:按医疗优先级丢 chunk(证据预算与总预算哪个先触顶都丢)
        while self._tokens(chunks=chunks, **parts) > self.budget.total and len(chunks) > 1:
            victim = chunks.pop(self._drop_candidate(chunks))
            dropped.append(victim.section or "其它")
            steps.append("evidence-drop")

        # ---- L4:头保留截断(优先级最低、体量最大的块先让;query 永不裁)
        while self._tokens(chunks=chunks, **parts) > self.budget.total and chunks:
            index = self._truncate_candidate(chunks)
            target = chunks[index]
            others = [chunk.text for chunk in chunks if chunk is not target]
            room = self.budget.total - self._tokens(
                chunks=[EvidenceItem(text="")] + [EvidenceItem(text=text) for text in others], **parts
            )
            clipped_text = clip_head(target.text, max(0, room))
            if not clipped_text or clipped_text == target.text:
                break
            chunks[index] = EvidenceItem(
                text=clipped_text, section=target.section, rerank_score=target.rerank_score, doc=target.doc,
            )
            steps.append("truncate")

        messages = self._build(evidence_texts=[chunk.text for chunk in chunks], **parts)
        return BudgetFit(
            messages=messages,
            evidence=list(chunks),
            truncated=True,
            exceeded=estimate_messages_tokens(messages) > self.budget.total,
            original_tokens=original_tokens,
            final_tokens=estimate_messages_tokens(messages),
            steps=tuple(dict.fromkeys(steps)),
            dropped_sections=tuple(dropped),
        )

    def _fit_result(
        self, chunks: Sequence[EvidenceItem], steps: Sequence[str], dropped: Sequence[str],
        original_tokens: int, *, truncated: bool, system_text: str, query: str, history_block: str, ocr_text: str,
    ) -> BudgetFit:
        messages = self._build(
            system_text=system_text, query=query, evidence_texts=[chunk.text for chunk in chunks],
            history_block=history_block, ocr_text=ocr_text,
        )
        return BudgetFit(
            messages=messages,
            evidence=list(chunks),
            truncated=truncated,
            exceeded=False,
            original_tokens=original_tokens,
            final_tokens=estimate_messages_tokens(messages),
            steps=tuple(dict.fromkeys(steps)),
            dropped_sections=tuple(dropped),
        )
