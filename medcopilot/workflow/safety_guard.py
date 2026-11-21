from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from medcopilot.workflow.emergency_detector import (
    EmergencyDetector,
    EmergencyVerdict,
    compiled_patterns,
)

# 剂量门匹配前的归一化:NFKC(全角→半角、兼容字符)+ 去零宽/不可见格式字符。
# 刻意不去空白:英文剂量的词边界依赖空白,整体删除会把 "2 uses" 粘成 "2uses"
# 之类的假阳性;正则内的 \s* 已容忍正常间隔。归一化只用于"是否命中"的判定,
# 放行/替换的文本始终是原文逐字。
_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")

# 合规免责文本的单一事实源(W5.7):API 每次回答追加的那段文字与 UI 首屏常驻
# 横幅渲染的是同一个字符串,因此"屏幕上的告知"与"响应里的告知"不可能漂移。
# 注意:策略工件把这段文字作为 ``disclaimer_base`` 镜像(``current_code_content``
# 通过调用 ``post_process_response`` 取值),改动会让 ``policy_check`` 立刻变红
# —— 这是刻意的:改文本必须走策略四步闭环。
DISCLAIMER_TEXT = (
    "\n\n---\n"
    "📌 **医疗免责合规提示**: 本系统回答基于检索到的公开药品说明书及大模型生成，"
    "旨在提供医学科普与参考，严禁直接作为临床处方或自行调药依据。用药请务必遵循正规医疗机构专业医师指导。"
)
NO_EVIDENCE_PREFIX = "\n\n⚠️ **特别提醒**: 未在本地权威说明书库检索到充分依据，本次不提供具体用药或治疗指令。"
# 首屏常驻横幅:用户在提出任何问题之前就能看到"不能替代医师"与急救指引,
# 而不是只在某次回答的末尾。复用 DISCLAIMER_TEXT 以保持同源。
DISCLAIMER_BANNER = (
    "> ⚠️ **医疗免责声明**：本系统为技术演示用途，所有回答基于本地说明书检索与大模型生成，"
    "不能替代执业医师的诊断与处方；如出现胸痛、呼吸困难、意识不清等紧急情况，请立即拨打 120。"
    + DISCLAIMER_TEXT.replace("\n\n---\n", " ")
)


class DosageGate:
    """确定性剂量门(W6.1):指令性剂量句的统一裁决层。

    双模式(裁决自 2026-10-04 需求方:A+B 同时支持,模式可配,默认安全侧):

    - ``redact``(默认,B 方案):逐字有据的指令性剂量句替换为
      ``DOSAGE_REFERRAL_TEXT`` 转介话术;同义改写/编造剂量由 extractive 门
      弃答(本门双保险);风险披露句(过量/中毒/毒性等,逐字有据)保留。
    - ``quote``(A 方案):逐字有据的指令性剂量句放行 + 强制免责 —— 但仅在
      **证据中召回该药的禁忌/相互作用/注意事项安全章节**时(否则该句仍替换
      为转介,fail-closed:逐字剂量不得与缺失的禁忌并存);回答必须显式声明
      "以下为说明书原文引用"。

    两种模式下:编造/改写剂量 ⇒ 弃答(extractive 门);风险披露 ⇒ 保留;
    强制免责 ⇒ 恒追加(post_process_response)。
    """

    MODES = ("redact", "quote", "off")

    # 指令性剂量句判定(句级):剂量数值+单位、给药频次、或频次副词+数量+单位。
    # 西文单位带 (?![a-z]) 边界卫兵:"16GB" 不得读成 "16g"、"2 uses" 不得读成
    # "2u"(IGNORECASE 下的英文边界);中文单位无此问题。数量覆盖小数、半、
    # 中文数字与三种分数书写(1/2、½→NFKC 1⁄2、二分之一)。
    _FRACTION = (
        r"(?:[0-9]+\s*[/⁄]\s*[0-9]+"
        r"|[一二两三四五六七八九十百千万0-9]+\s*分\s*之\s*[一二两三四五六七八九十百千万0-9]+)"
    )
    _QTY_CJK = r"(?:\d+(?:\.\d+)?|半|[一二两三四五六七八九十百千万]+|" + _FRACTION + r")"
    _QTY_LATIN = r"(?:\d+(?:\.\d+)?|半|" + _FRACTION + r")"
    _AMOUNT_LATIN = rf"(?:{_QTY_LATIN})\s*(?:mg|μg|ug|mcg|ng|g|kg|ml|l|u|iu)(?![a-z])"
    _AMOUNT_CJK = (
        rf"(?:{_QTY_CJK})\s*"
        r"(?:微克|毫克|克|千克|公斤|毫升|升|万国际单位|国际单位|万单位|单位|片|粒|袋|丸|支|滴|喷|贴|枚|瓶|包|管)"
    )
    _FREQUENCY = (
        r"(?:一日|每日|每天|一天|一晚|每晚|一小时|每小时|一周|每周|一次|每次|早晚各|早中晚)"
        rf"\s*(?:服|吃|用|喝)?\s*(?:{_QTY_CJK})\s*次"
    )
    # 频次副词 + 数量 + 单位(无"次"):早晚各一粒 / 早中晚各一片。
    _SCHEDULE = (
        r"(?:早晚各|早中晚)"
        rf"\s*(?:{_QTY_CJK})\s*(?:粒|片|袋|丸|支|滴|喷|贴|枚|瓶|包|管|万国际单位|国际单位|万单位|单位)"
    )
    DOSAGE_SENTENCE_RE = re.compile(
        rf"{_AMOUNT_LATIN}|{_AMOUNT_CJK}|{_FREQUENCY}|{_SCHEDULE}", re.IGNORECASE
    )
    # 显式给药指令(风险豁免的否定条件):频次/排程副词 + 数量 + 单位,或
    # _FREQUENCY/_SCHEDULE。"一次1粒""一日2次""早晚各一粒"是指令;而
    # "超过4g""一日剂量不超过4g"是指标/限量,不是指令,仍可走风险豁免。
    _DIRECTIVE_ADVERB = (
        r"(?:一次|每次|一日|每日|每天|一天|一晚|每晚|一小时|每小时|一周|每周"
        r"|今早|今晚|早|晚|午|饭前|饭后|睡前|餐前|餐后|早晚各|早中晚)"
    )
    _DIRECTIVE_UNIT = (
        rf"{_DIRECTIVE_ADVERB}\s*(?:服用|食用|饮用|吸入|注射|外用|塞入|服|吃|用|喝)?\s*(?:{_QTY_CJK})\s*"
        r"(?:(?:mg|μg|ug|mcg|ng|g|kg|ml|l|u|iu)(?![a-z])"
        r"|微克|毫克|克|千克|公斤|毫升|升|万国际单位|国际单位|万单位|单位|粒|片|袋|丸|支|滴|喷|贴|枚|瓶|包|管)"
    )
    DOSAGE_DIRECTIVE_RE = re.compile(
        rf"{_DIRECTIVE_UNIT}|{_FREQUENCY}|{_SCHEDULE}", re.IGNORECASE
    )
    # 风险披露豁免:描述危害而非给出方案(对安全必要的信息,两模式均保留)。
    RISK_DISCLOSURE_RE = re.compile(r"(超过|超量|误服|误用|过量|中毒|致死|血药浓度|蓄积|毒性|肝损伤|肾损伤|禁用)")
    # quote 模式放行剂量时必须出现的显式引用声明(追加,不由模型自由生成)。
    QUOTE_ANNOUNCEMENT = "以下剂量信息为药品说明书原文逐字引用，非个性化用药建议：\n"

    # 固定转介话术(代码常量,进策略工件镜像;自身不得命中 DOSAGE_SENTENCE_RE)。
    DOSAGE_REFERRAL_TEXT = (
        "关于具体用法用量：该药品的剂量存在个体化给药风险（年龄、体重、肝肾功能、"
        "合并用药均影响剂量），本系统不直接提供剂量数字。请以药品说明书【用法用量】"
        "原文为准，并在医师或药师指导下使用。"
    )

    # quote 模式的安全底线:证据必须包含这些章节之一,逐字剂量才可放行。
    SAFETY_SECTIONS_FOR_QUOTE = frozenset({"禁忌", "药物相互作用", "注意事项", "孕妇及哺乳期妇女用药"})

    @classmethod
    def split_sentences(cls, text: str) -> list[str]:
        return [s for s in re.split(r"(?<=[。！？；;])|\n", text) if s.strip()]

    @classmethod
    def _normalize_for_match(cls, text: str) -> str:
        """匹配前归一化(NFKC + 去零宽);只用于判定,不改写任何输出文本。"""
        return _ZERO_WIDTH_RE.sub("", unicodedata.normalize("NFKC", text or ""))

    @classmethod
    def _split_for_gate(cls, sentence: str) -> list[str]:
        """风险披露与指令剂量同句(逗号不断句)时按逗号细分:风险片段保留,
        剂量片段照常裁决,不搭整句豁免的便车。纯风险句(无逗号)不受影响。
        """
        normalized = cls._normalize_for_match(sentence)
        if (
            cls.RISK_DISCLOSURE_RE.search(normalized)
            and cls.DOSAGE_SENTENCE_RE.search(normalized)
        ):
            return [piece for piece in re.split(r"(?<=[，,])", sentence) if piece.strip()]
        return [sentence]

    @classmethod
    def _is_verbatim(cls, sentence: str, evidence_texts: Sequence[str]) -> bool:
        stripped = re.sub(r"\s+", "", sentence)
        return any(stripped in re.sub(r"\s+", "", ev) for ev in evidence_texts)

    @classmethod
    def _has_safety_section(cls, evidence_sections: Sequence[str]) -> bool:
        return any(sec in cls.SAFETY_SECTIONS_FOR_QUOTE for sec in evidence_sections)

    @classmethod
    def _has_safety_section_for_piece(
        cls,
        piece: str,
        evidence_texts: Sequence[str],
        evidence_sections: Sequence[str],
        evidence_drugs: Sequence[str] | None,
    ) -> bool:
        if not cls._has_safety_section(evidence_sections):
            return False
        if evidence_drugs is None:
            return True
        if len(evidence_drugs) != len(evidence_texts) or len(evidence_sections) != len(evidence_texts):
            return False
        stripped = re.sub(r"\s+", "", piece)
        protected_drugs = {
            (drug or "").strip()
            for drug, sec in zip(evidence_drugs, evidence_sections)
            if drug and (drug or "").strip() and (drug or "").strip() != "未知药品" and sec in cls.SAFETY_SECTIONS_FOR_QUOTE
        }
        matched_drugs = [
            (drug or "").strip()
            for ev, drug in zip(evidence_texts, evidence_drugs)
            if stripped in re.sub(r"\s+", "", ev)
        ]
        named = [d for d in matched_drugs if d and d != "未知药品"]
        if not named:
            all_named = [
                (d or "").strip()
                for d in evidence_drugs
                if d and (d or "").strip() and (d or "").strip() != "未知药品"
            ]
            return not all_named
        return all(drug in protected_drugs for drug in named)

    @classmethod
    def apply(
        cls,
        answer: str,
        evidence_texts: Sequence[str],
        evidence_sections: Sequence[str],
        *,
        mode: str = "redact",
        evidence_drugs: Sequence[str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """裁决入口。返回 (可发布文本, 统计);整体不可发布时返回 ("", 统计)。

        统计键:instructional(命中指令性剂量句数)、redacted(替换数)、
        kept_quote(quote 模式放行数)、kept_risk(风险披露保留数)。
        mode="off" 时不做任何替换(仅限显式开发配置;生产链路必须传 redact/quote)。

        判定在归一化副本上进行(NFKC + 去零宽),kept/released 始终是原始文本
        逐字拼接——quote 模式放行的是原文,不是归一化改写文本。
        """
        stats = {"instructional": 0, "redacted": 0, "kept_quote": 0, "kept_risk": 0}
        if mode not in cls.MODES:
            raise ValueError(f"dosage_gate_mode must be one of {cls.MODES}, got {mode!r}")
        if mode == "off":
            return answer, stats

        kept: list[str] = []
        for sentence in cls.split_sentences(answer):
            for piece in cls._split_for_gate(sentence):
                normalized = cls._normalize_for_match(piece)
                if not cls.DOSAGE_SENTENCE_RE.search(normalized):
                    kept.append(piece)
                    continue
                stats["instructional"] += 1
                verbatim = cls._is_verbatim(piece, evidence_texts)
                # 风险警示不能携带给药指令绕过剂量门;纯风险限量在两模式均保留。
                if (
                    cls.RISK_DISCLOSURE_RE.search(normalized)
                    and verbatim
                    and not cls.DOSAGE_DIRECTIVE_RE.search(normalized)
                ):
                    kept.append(piece)  # 风险披露:两模式均保留
                    stats["kept_risk"] += 1
                    continue
                safety_ok = cls._has_safety_section_for_piece(
                    piece, evidence_texts, evidence_sections, evidence_drugs
                )
                if mode == "quote" and verbatim and safety_ok:
                    kept.append(piece)  # A 方案:逐字 + 该药安全章节在场
                    stats["kept_quote"] += 1
                    continue
                kept.append(cls.DOSAGE_REFERRAL_TEXT)  # B 方案 / quote 但安全章节缺失
                stats["redacted"] += 1

        if not any(s.strip() for s in kept):
            return "", stats
        released = "".join(kept)
        if stats["kept_quote"]:
            released = cls.QUOTE_ANNOUNCEMENT + released
        return released, stats


class MedicalSafetyGuard:
    """
    医疗问答三重防幻觉护栏 (Tri-Guard Architecture)

    1. 前置拦截 (Pre-Guard): 识别高危急重症关键词（如心梗、大出血、剧烈剧痛、昏迷、自残），强制要求就医。
    2. 循证引文比对 (Grounding Alignment): 检查模型生成的用药建议是否出现在召回的参考资料中，若无匹配则警告。
    3. 后置红线审查 (Post-Guard): 强制附加国家药监局/卫健委规范免责声明，剔除确定性误导言论。

    急救检测的执行源是 ``EmergencyDetector``(workflow/emergency_detector.py)的
    确定性代码常量;本类仅做薄封装,保持既有调用点签名不变。
    """

    # 运行时门控策略版本(代码常量 = 事实源;策略工件仅镜像,漂移由
    # safety.policy.check_code_equivalence 门禁捕获)。
    # v3: W6.1 剂量门双模式(redact/quote)进策略镜像。
    # v4: W3.1 安全章节召回开关 + W3.3 sufficiency 阈值进策略镜像。
    # v5: W10 评审波 —— 口语剂量形态(半片/一天两次/微克/mcg/U/早晚各/分数/
    #     中文数字/零宽)与混合风险+剂量句豁免收紧;急救规则新增裸胸痛与血压
    #     数值区间;信息疑问词移出叙事降级标记(当前急症的信息式追问不再被降级)。
    POLICY_VERSION = 5

    # Token pattern: match multi-character words/numbers or meaningful clinical entities,
    # avoiding trivial single-character function words (的/是/在/有/和) that artificially pass grounding.
    TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?")
    STOP_WORDS = frozenset({"可以", "如果", "但是", "因为", "所以", "这个", "那个", "建议", "患者", "出现", "引起", "注意"})
    # 急救正则的扁平化镜像(模式串与顺序由 EmergencyDetector.RULES 定义)。
    EMERGENCY_PATTERNS = compiled_patterns()

    @classmethod
    def check_emergency(cls, query: str) -> tuple[bool, str]:
        verdict = cls.check_emergency_details(query)
        if verdict.is_emergency:
            return True, (
                "⚠️【医疗急救最高警示】检测到您描述的情况可能属于急性急危重症！"
                "AI 系统不能代替急诊抢救，请立即拨打 120 急救电话或立即就近前往公立医院急诊科就医！"
            )
        return False, ""

    @classmethod
    def check_emergency_details(cls, query: str, *, source: str = "user_query") -> EmergencyVerdict:
        """急救判定明细(审计/评测用)。source 必须是当前用户 query;不可信输入驱动
        急救控制流会抛 ValueError(fail-closed),不静默放行。"""
        return EmergencyDetector.evaluate(query, source=source)

    @classmethod
    def check_groundedness(
        cls,
        answer: str,
        evidence_texts: Sequence[str],
        threshold: float = 0.15,
    ) -> tuple[float, bool]:
        raw_evidence_tokens = set(cls.TOKEN_PATTERN.findall("".join(evidence_texts)))
        evidence_tokens = raw_evidence_tokens - cls.STOP_WORDS
        sentences = [s.strip() for s in re.split(r"[。！？\n]", answer) if len(s.strip()) >= 4]
        if not sentences or not evidence_tokens:
            return 0.0, False
        hit = 0
        for sentence in sentences:
            sent_tokens = set(cls.TOKEN_PATTERN.findall(sentence)) - cls.STOP_WORDS
            if sent_tokens & evidence_tokens:
                hit += 1
        ratio = hit / len(sentences)
        return ratio, ratio >= threshold

    @staticmethod
    def check_extractive_support(answer: str, evidence_texts: Sequence[str]) -> bool:
        """Conservative release gate; copied spans are not proof of clinical validity."""
        def sentences(text: str) -> set[str]:
            return {
                re.sub(r"\s+", "", sentence).casefold()
                for sentence in re.split(r"[。！？\n]+|(?<!\d)\.|\.(?!\d)", text)
                if sentence.strip()
            }

        evidence_sentences = set().union(*(sentences(text) for text in evidence_texts))
        claims = sentences(answer)
        # Substring matches could drop a leading negation or a patient qualifier.
        return bool(claims) and claims.issubset(evidence_sentences)

    @classmethod
    def post_process_response(cls, response: str, has_retrieved_evidence: bool = True) -> str:
        # 强制追加合规临床法律免责声明（文本与 UI 常驻横幅同源，见 DISCLAIMER_TEXT）
        disclaimer = DISCLAIMER_TEXT

        if not has_retrieved_evidence:
            disclaimer = NO_EVIDENCE_PREFIX + disclaimer

        return response + disclaimer
