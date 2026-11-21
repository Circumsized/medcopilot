"""摄取端校验闸门(W3.2,OWASP LLM08:2025 Vector & Embedding Weaknesses)。

``/documents/drug-label`` 是投毒入口:检索文档既是生成上下文又是 grounding
证据,被投毒文档可以"自证 grounded"(分块/入索引/进答案三者同时被操纵)。
本闸门在**分块与入索引之前**做最小合理性校验(确定性,零模型依赖,断网可测):

1. 结构合理性:NFKC 非空、长度下限、命中说明书章节标题行、至少一个
   {药品名称|适应症|功能主治} 主章节、至少识别一个章节标题;
2. 剂量数值一致性:全部剂量提及的单位白名单 + 合理性上限 + 裸小数拒绝
   (药企生产复盘:μ→u 导致剂量单位错误的事故教训);
3. 提示注入探针(deny-list 最小集):围栏标记、"忽略以上指令"、
   ``system: you are``、C0 控制字符、无中文标点的超长单行(疑似 base64/blob);
4. 指纹:``raw_sha256`` 供摄取审计(谁在何时摄入了什么的可追溯)。

定位:最小合理性校验而非完整 WAF(deny-list 可被变形绕过);生成与 grounding 的
真实边界仍是 extractive 门 + 安全章节门。拒绝 ⇒ 不写 CorpusStore、不入索引。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

# 剂量单位白名单(与 WRaw 摄取同源;μ/µg 等先经 NFKC + 归一化映射)。
UNIT_ALIASES = {
    "μg": "ug", "µg": "ug", "Ｕ": "u", "ＩＵ": "iu", "ｍｇ": "mg",
    "毫克": "mg", "毫升": "ml", "克": "g", "微克": "ug",
}
UNIT_WHITELIST = frozenset({
    "mg", "g", "ug", "ng", "ml", "l", "iu", "u", "万单位", "单位", "%",
    "片", "粒", "袋", "支", "瓶", "丸", "滴", "枚", "贴", "喷",
})
# 单位合理性上限(超出视为 OCR 损坏或伪造标签)。
UNIT_MAXIMA = {
    "g": 100.0, "mg": 200_000.0, "ug": 200_000_000.0, "ng": 2e11,
    "ml": 2000.0, "l": 2.0, "iu": 10_000_000.0, "%": 100.0,
    "片": 30.0, "粒": 30.0, "袋": 20.0, "支": 20.0, "瓶": 2000.0,
    "丸": 50.0, "滴": 500.0, "枚": 30.0, "贴": 10.0, "喷": 100.0,
}
# 裸小数:"一次0.375"后直接句读/字符串结束,无单位收尾 ⇒ 剂量被截断,拒收。
# 若空白后紧跟合法剂量单位或常见时间/量词(如 "0.5 g"、"3.5 小时"),则属合法表述。
_BARE_DECIMAL_RE = re.compile(
    r"\d+\.\d+"
    r"(?!\s*(?:mg|g|kg|ug|μg|µg|ng|ml|l|iu|u|万单位|单位|%|片|粒|袋|支|瓶|丸|滴|枚|贴|喷|小时|分钟|秒|日|天|周|月|年|倍|cm|mm))"
    r"(?=$|[。；;，,\s]|\n|】)",
    re.IGNORECASE,
)

_DOSE_AMOUNT_RE = re.compile(
    r"(?<![a-zA-Z])"
    r"(\d+(?:\.\d+)?)"
    r"\s*(mg|g|kg|ug|μg|µg|ng|克|毫克|毫升|微克|l|iu|u|万单位|单位|%|片|粒|袋|支|瓶|丸|滴|枚|贴|喷)"
    r"(?![a-zA-Z])",
    re.IGNORECASE,
)
_RATIO_DOSE_RE = re.compile(r"(\d+)\s*/\s*\d+\s*(mg|g|ml|kg)", re.IGNORECASE)

_INJECTION_PATTERNS = re.compile(
    r"(?i)(ignore\s*(all\s*)?(previous|above|prior)\s*(instructions|prompts?)|"
    r"忽略(以上|上文|上述|之前)(所有)?(指令|指示|内容|要求)|"
    r"disregard\s*(all\s*)?(previous|above)|"
    r"system\s*:\s*you\s*(are|must|should)|"
    r"【以下为不可信数据|【不可信数据结束|</?(system|assistant|user)>|"
    r"(?:你现在是|请扮演|从此刻起你是))"
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_BLOBLIKE_LINE_RE = re.compile(r"[^\s，。；：、（）()【】\n]{4000,}")

REQUIRED_PRIMARY_SECTIONS = ("药品名称", "适应症", "功能主治")
CJK_PUNCT_RE = re.compile(r"[，。；：、（）()【】]")


@dataclass(frozen=True)
class IngestValidation:
    accepted: bool
    reasons: tuple[str, ...]
    raw_sha256: str
    section_count: int
    warnings: tuple[str, ...] = ()


class IngestValidator:
    """摄取最小合理性校验。纯确定性,无向量库/模型依赖(MOCK_LLM=1 全可测)。"""

    @staticmethod
    def normalize_text(text: str) -> str:
        return unicodedata.normalize("NFKC", text or "")

    @classmethod
    def raw_fingerprint(cls, raw_text: str, drug_name: str) -> str:
        payload = f"{cls.normalize_text(raw_text)}\x00{drug_name}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def validate(
        cls,
        raw_text: str,
        drug_name: str,
        *,
        min_raw_chars: int = 30,
        max_line_chars: int = 4000,
        dose_check: bool = True,
        injection_check: bool = True,
    ) -> IngestValidation:
        """结构合理性 + 剂量一致性 + 注入探针 + 指纹。

        长度门槛分层(对合法短说明书不误伤):无任何章节标题的 blob 需
        ≥ ``min_raw_chars``;带章节结构的短标签 ≥ 10 字符即可(历史上
        "【药品名称】X\\n【禁忌】Y" 这类合法短标签长期存在)。
        """
        normalized = cls.normalize_text(raw_text)
        reasons: list[str] = []
        warnings: list[str] = []
        if not normalized.strip():
            return IngestValidation(False, ("structure:empty",), "", 0)

        from medcopilot.rag.clinical_chunker import ClinicalChunker

        section_names = list(ClinicalChunker.SECTIONS)
        header_re = re.compile(r"^[ \t]*(【?(?:" + "|".join(map(re.escape, section_names)) + r")】?[:：]?)", re.MULTILINE)
        headers = header_re.findall(normalized)
        section_count = len(set(h.strip("【】:： \t") for h in headers))

        if section_count == 0:
            reasons.append(
                "structure:min-length" if len(normalized) < min_raw_chars
                else "structure:no-recognized-section"
            )
        elif len(normalized) < 10:
            # 带结构但短到不可能承载信息(截断残留)。
            reasons.append("structure:min-length")
        if not any(primary in normalized for primary in REQUIRED_PRIMARY_SECTIONS):
            reasons.append("structure:missing-primary-section")

        if injection_check:
            normalized_drug = cls.normalize_text(drug_name)
            if (
                _INJECTION_PATTERNS.search(normalized)
                or _INJECTION_PATTERNS.search(normalized_drug)
                or "【" in normalized_drug
                or "】" in normalized_drug
            ):
                reasons.append("injection:prohibited-pattern")
            if (
                _CONTROL_CHARS_RE.search(normalized)
                or _CONTROL_CHARS_RE.search(normalized_drug)
                or "\n" in normalized_drug
                or "\r" in normalized_drug
            ):
                reasons.append("injection:control-characters")
            for line in normalized.splitlines():
                if len(line) > max_line_chars:
                    body = line.strip()
                    if not CJK_PUNCT_RE.search(body[:200]) and _BLOBLIKE_LINE_RE.search(line):
                        reasons.append("injection:blob-like-line")
                        break

        if dose_check:
            for value, unit in _DOSE_AMOUNT_RE.findall(normalized):
                canonical = UNIT_ALIASES.get(unit, unit.lower())
                if canonical not in UNIT_WHITELIST:
                    reasons.append(f"dose:bad-unit:{canonical}")
                    continue
                maximum = UNIT_MAXIMA.get(canonical)
                if maximum is not None and float(value) > maximum:
                    reasons.append(f"dose:implausible:{canonical}")
            for value, unit in _RATIO_DOSE_RE.findall(normalized):
                canonical = UNIT_ALIASES.get(unit, unit.lower())
                if canonical not in UNIT_WHITELIST:
                    reasons.append(f"dose:bad-unit:{canonical}")
            if _BARE_DECIMAL_RE.search(normalized):
                reasons.append("dose:bare-decimal")

        sha = cls.raw_fingerprint(raw_text, drug_name)
        return IngestValidation(
            accepted=not reasons,
            reasons=tuple(sorted(set(reasons))),
            raw_sha256=sha,
            section_count=section_count,
            warnings=tuple(warnings),
        )
