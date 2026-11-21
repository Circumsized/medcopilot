import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from medcopilot.common.logger import logger


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

class ClinicalChunker:
    """
    面向药品说明书与临床指南的层级化结构切分器 (Domain-Specific Chunker)

    原项目缺陷: 采用通用的 CharacterTextSplitter(chunk_size=100)，不仅将完整的【禁忌】、【用法用量】、【不良反应】切得支离破碎，
                而且没有任何 metadata 标记，导致检索出的上下文缺失药名和章节语境。
    专业改造:
    1. 基于医学说明书固定章节（适应症、禁忌、用法用量、不良反应、注意事项、药物相互作用）进行基于正则的正文感知解析。
    2. 为每一个分块强注入 metadata（药品通用名、所在章节、禁忌等级）。
    """
    VERSION = "chunker-v2-section-aware-dose-protect-spoken"
    SECTIONS = [
        "药品名称", "成份", "性状", "适应症", "功能主治",
        "规格", "用法用量", "不良反应", "禁忌", "注意事项",
        "孕妇及哺乳期妇女用药", "儿童用药", "老年用药",
        "药物相互作用", "药物过量", "药理毒理", "贮藏",
    ]
    # W3.4: align section chunks to bge-reranker-base max_length=512. The
    # enriched prefix is ~30-40 chars, so 480 chars keeps the reranker input
    # visible in the common Chinese case. Industry guidance recommends
    # semantic/paragraph boundaries, table-aware chunks, and overlap instead of
    # arbitrary hard cuts (Microsoft Azure RAG chunking guidance, 2025).
    CHUNK_MAX_CHARS = 480
    CHUNK_OVERLAP = 60
    HARD_MAX_CHARS = 960
    # W10 口语剂量形态:半、分数(1/2、1⁄2、二分之一)、Unicode 分数(½)、中文数字、
    # "一天两次"、"早晚各一粒"。与 workflow 的 DosageGate.DOSAGE_SENTENCE_RE 同源
    # 不同层:门是句级判定且匹配前做 NFKC+去零宽归一化;这里必须按**原文偏移**保护
    # 切点(归一化会移动 offset),因此各层保留本地 pattern,不跨层 import。
    _DOSE_QTY = (
        r"(?:\d+(?:\.\d+)?|半|[一二两三四五六七八九十百]+"
        r"|[0-9]+\s*[/⁄]\s*[0-9]+"
        r"|[一二两三四五六七八九十百0-9]+\s*分\s*之\s*[一二两三四五六七八九十百0-9]+"
        r"|[¼-¾⅐-⅟])"
    )
    _DOSE_UNIT = r"(?:mg|g|kg|ug|μg|µg|ng|ml|l|iu|u|克|毫克|毫升|微克|单位|片|粒|袋|支|瓶|丸|滴|枚|贴|喷|%)"
    _DOSE_FREQUENCY = (
        r"(?:一日|每日|每天|一天|一晚|每晚|一小时|每小时|一周|每周|一次|每次|早晚各|早中晚)"
        rf"\s*(?:服|吃|用|喝)?\s*(?:{_DOSE_QTY})\s*次"
    )
    # 频次副词 + 数量 + 单位(无"次"):早晚各一粒。
    _DOSE_SCHEDULE = (
        r"(?:早晚各|早中晚)"
        rf"\s*(?:{_DOSE_QTY})\s*(?:粒|片|袋|丸|支|滴|喷|贴|枚|瓶|包|管|单位)"
    )
    DOSE_PROTECT_PATTERN = re.compile(
        rf"(?:{_DOSE_QTY})\s*{_DOSE_UNIT}|{_DOSE_FREQUENCY}|{_DOSE_SCHEDULE}",
        re.IGNORECASE,
    )

    @classmethod
    def _split_long(cls, text: str) -> list[str]:
        """按结构/语义边界切分,绝不切开剂量单位或给药频次。"""
        if len(text) <= cls.CHUNK_MAX_CHARS:
            return [text]
        parts: list[str] = []
        start = 0
        while start < len(text):
            if len(text) - start <= cls.CHUNK_MAX_CHARS:
                parts.append(text[start:])
                break
            target = start + cls.CHUNK_MAX_CHARS
            hard_end = min(start + cls.HARD_MAX_CHARS, len(text))
            candidates: list[tuple[int, int]] = []
            for pos in range(start + 1, min(target, len(text)) + 1):
                ch = text[pos - 1]
                rank = 4 if ch == "\n" else 3 if ch in "。！？；;" else 2 if ch in "，,、" else 1 if ch.isspace() else 0
                if rank:
                    candidates.append((rank, pos))
            valid = [
                item for item in candidates
                if not any(m.start() < item[1] < m.end() for m in cls.DOSE_PROTECT_PATTERN.finditer(text, max(start, item[1] - 80), min(len(text), item[1] + 80)))
            ]
            if valid:
                _, end = max(valid, key=lambda item: (item[0], item[1]))
            else:
                forward = [
                    item for item in candidates if item[1] <= hard_end and not any(
                        m.start() < item[1] < m.end() for m in cls.DOSE_PROTECT_PATTERN.finditer(text, max(start, item[1] - 80), min(len(text), item[1] + 80))
                    )
                ]
                _, end = max(forward, key=lambda item: (item[0], -abs(item[1] - target))) if forward else (0, min(target, hard_end))
            # P1-5 修复:完全没有边界候选时,硬切点可能恰好落在剂量 span 中间
            # (实测分片尾出现"…一次0."这种被腰斩的给药指令,检索层照样会召回)。
            # 把切点推到 span 结束;若会越过 HARD_MAX 则退到 span 之前。相邻
            # span(如"0.5g"后紧跟"一日2次")用短循环收敛到安全切点。
            for _ in range(4):
                hit = next(
                    (
                        match
                        for match in cls.DOSE_PROTECT_PATTERN.finditer(
                            text, max(start, end - 80), min(len(text), end + 80)
                        )
                        if match.start() < end < match.end()
                    ),
                    None,
                )
                if hit is None:
                    break
                end = hit.end() if hit.end() <= hard_end else hit.start()
            if end <= start:
                end = min(start + cls.CHUNK_MAX_CHARS, len(text))
            parts.append(text[start:end].strip())
            block_start = start
            start = max(end - cls.CHUNK_OVERLAP, block_start + 1)
            # overlap 起点保护:切点保护只保证上一块完整,下一块的开头若落在剂量
            # span 中间仍会产出"次。。"式残缺给药片段(检索层照样召回)。优先后退到
            # span 起点——下一块以完整指令开头,且仍严格越过本块起点(有进展);
            # span 起点已在本块之前时只能前进到 span 结束,保证不重切同一区间、
            # 不漏文本(相邻块始终重叠或相接)。
            for _ in range(4):
                hit = next(
                    (
                        match
                        for match in cls.DOSE_PROTECT_PATTERN.finditer(
                            text, max(start - 80, 0), min(len(text), start + 80)
                        )
                        if match.start() < start < match.end()
                    ),
                    None,
                )
                if hit is None:
                    break
                start = hit.start() if hit.start() > block_start else hit.end()
        return [part for part in parts if part]

    @classmethod
    def chunk_drug_label(cls, raw_text: str, drug_name: str = "未知药品") -> list[Document]:
        # Sort sections by length descending so longer composite titles take precedence
        sorted_sections = sorted(cls.SECTIONS, key=len, reverse=True)
        # Line-anchored split so mentions of section words inside a body paragraph
        # (e.g. "如出现不良反应请停药") are not falsely treated as section starts.
        pattern = r"^[ \t]*(【?(?:" + "|".join(sorted_sections) + r")】?[:：]?)"
        splits = re.split(pattern, raw_text, flags=re.MULTILINE)
        docs: list[Document] = []

        if len(splits) <= 1:
            # 未识别出标准章节：整文按上限分块兜底保留（空白输入返回空列表，绝不落空 chunk）
            stripped = raw_text.strip()
            if not stripped:
                return []
            for idx, part in enumerate(cls._split_long(stripped)):
                if not part:
                    continue
                docs.append(Document(
                    page_content=part,
                    metadata={
                        "drug_name": drug_name,
                        "section": "综合说明",
                        "chunk_type": "fallback",
                        "chunk_part": idx,
                        # 包含 drug_name 参与哈希，防止不同药品相同文本片段跨药品冲突
                        "chunk_id": hashlib.sha256(f"{drug_name}:{idx}:{part}".encode()).hexdigest(),
                    }
                ))
            return docs

        # 保留首个章节标记前的前言/概述文本，防止静默丢弃通用说明
        preamble = splits[0].strip()
        if len(preamble) >= 10:
            preamble_parts = cls._split_long(preamble)
            for idx, part in enumerate(preamble_parts):
                preamble_text = f"【药品】{drug_name}\n【章节】概述前言\n【详细内容】\n{part}"
                docs.append(Document(
                    page_content=preamble_text,
                    metadata={
                        "chunk_id": hashlib.sha256(f"{drug_name}:preamble:{idx}:{part}".encode()).hexdigest(),
                        "drug_name": drug_name,
                        "section": "概述前言",
                        "chunk_part": idx,
                        "is_contraindication": False,
                    }
                ))

        i = 1
        while i < len(splits):
            sec_title = splits[i].strip("【】:： ")
            sec_content = splits[i + 1].strip() if (i + 1) < len(splits) else ""

            if not sec_content:
                i += 2
                continue
            sec_parts = cls._split_long(sec_content)
            for idx, part in enumerate(sec_parts):
                # 注入结构化上下文标题，极大提升密集向量嵌入质量
                numeric = f"\n【分段】{idx + 1}/{len(sec_parts)}" if len(sec_parts) > 1 else ""
                enriched_text = f"【药品】{drug_name}\n【章节】{sec_title}\n【详细内容】\n{part}{numeric}"
                chunk_id = hashlib.sha256(f"{drug_name}:{sec_title}:{idx}:{part}".encode()).hexdigest()
                docs.append(Document(
                    page_content=enriched_text,
                    metadata={
                        "chunk_id": chunk_id,
                        "drug_name": drug_name,
                        "section": sec_title,
                        "chunk_part": idx,
                        "is_contraindication": "禁忌" in sec_title,
                        "is_adverse_reaction": "不良反应" in sec_title,
                    }
                ))
            i += 2

        logger.info(f"Successfully chunked drug label for '{drug_name}' into {len(docs)} clinical sections.")
        return docs
