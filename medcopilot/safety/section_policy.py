"""检索层安全策略常量与确定性判据(W3.1 / W3.3)。

**执行源 = 本模块代码常量**(AppConfig 承载开关与阈值;不进策略工件镜像的
触发词族属于检索启发式,阈值默认值已镜像进工件)。

两个组件:
1. ``ContraindicationTrigger``:从 query(+OCR drug_context)识别"共病/过敏原/
   特殊人群/药物冲突/决策意图"语境。触发 ⇒ 检索层强制并入该药的安全章节
   (禁忌/相互作用/注意事项等),reviewer 层要求安全章节在场,否则弃答。
   对应临床危害最大单项发现:禁忌块被 top-k 挤出而 grounding gate 不设防。
2. ``SufficiencyGate``:证据充分性判据(MedRGB sufficiency 的确定性操作化),
   替代实测失效的 evidence_floor——锚点为"语料药名(整体) + 停用字过滤后的
   中文二元组 + 英文词 + 数字",query 锚点须被证据覆盖(单文档 ≥ min_doc,
   证据并集 ≥ min_union),否则不调 LLM 直接弃答。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyContext:
    """一次问询的安全语境判定结果(纯代码常量推导,可进审计)。"""

    conditions: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    decision_intent: bool = False

    @property
    def triggered(self) -> bool:
        return bool(self.conditions or self.conflicts or self.decision_intent)


class ContraindicationTrigger:
    """共病/决策意图触发器:命中 ⇒ 强制安全章节召回 + reviewer 在场检查。"""

    # 患者共病/特殊人群词族(来源:临床药理学禁忌人群 + 常见慢病;演进走 PR 评审)
    PATIENT_CONDITION_TERMS = frozenset({
        "过敏", "青霉素过敏", "磺胺过敏", "头孢过敏", "溃疡", "胃溃疡", "消化道溃疡",
        "出血", "肝功能不全", "肝肾不好", "肾功能不全", "肾衰", "尿毒症", "糖尿病",
        "酮症酸中毒", "高血压", "低血压", "血液病", "抗凝", "华法林", "阿司匹林",
        "哮喘", "青光眼", "痛风", "甲亢", "甲减", "甲状腺", "孕妇", "怀孕", "妊娠",
        "哺乳期", "备孕", "儿童", "婴儿", "新生儿", "老人", "老年", "手术", "透析",
    })
    # 药物联用/冲突意图词族
    DRUG_CONFLICT_TERMS = frozenset({
        "一起吃", "同服", "合用", "同时吃", "一起用", "混着吃", "配伍", "相互作用",
        "冲突", "感冒药", "退烧药", "止痛药", "安眠药", "镇静药", "头孢", "避孕药",
    })
    # 决策意图:正在决定"吃/用/买" ⇒ 必须见到安全章节
    DECISION_INTENT_PATTERNS = tuple(re.compile(p) for p in (
        r"(?:能|可以|可不可以|能不能|能否|应该|适合).{0,12}(?:吃|服用|喝|用|买|打)",
        r"我(?:有|患|得了).{1,24}(?:可以|能|能不能|能否|吃|用)",
        r"(?:怎么吃|怎么用|用量|剂量|吃多少|用多少|几粒|几片|一天几次|饭前|饭后|空腹)",
        r"(?:想吃|想用|想买|买点|买一|开点|来点|备点|囤点)",
        r"(?:哪个|哪种|哪款)(?:好|合适|更好|比较)",
        r"(?i)\b(?:contraindicated|interact\w*|take\s+with|safe\s+for)\b",
    ))
    # 安全章节:触发语境下,证据必须包含其中之一
    CONTRA_SECTIONS = frozenset({
        "禁忌", "药物相互作用", "注意事项", "孕妇及哺乳期妇女用药", "儿童用药", "老年用药",
    })

    @classmethod
    def assess(cls, query: str, drug_context: str | None = None) -> SafetyContext:
        text = unicodedata.normalize("NFKC", query or "")
        if drug_context:
            text += " " + unicodedata.normalize("NFKC", drug_context)
        conditions = tuple(sorted(t for t in cls.PATIENT_CONDITION_TERMS if t in text))
        conflicts = tuple(sorted(t for t in cls.DRUG_CONFLICT_TERMS if t in text))
        decision = any(p.search(text) for p in cls.DECISION_INTENT_PATTERNS)
        return SafetyContext(conditions=conditions, conflicts=conflicts, decision_intent=decision)


class SufficiencyGate:
    """证据充分性判据(确定性,禁 LLM 自评)。非语义相关性、非医疗置信度:
    只回答"query 的锚点是否被证据覆盖到可支撑作答的程度"。"""

    # 锚点提取时剔除的停用字(模态/助词/寒暄/疑问词),避免跨停用字二元组污染覆盖度
    STOP_CHARS = frozenset("的了吗呢吧啊呀哦是我你他她它咱请帮帮想看能在和与或要不要需该说让给被把这个那还又都就也很挺怎么什多少哪何")

    # a generic lexical overlap is not evidence that the requested clinical topic
    # was retrieved.
    INTENT_SECTION_PATTERNS = (
        ("禁忌", ("禁忌", "药物相互作用", "注意事项"), ("禁忌", "能不能", "可以吃", "过敏", "相互作用", "合用")),
        ("用法用量", ("用法用量",), ("怎么吃", "怎么用", "用量", "剂量", "几片", "几粒", "一天几次", "饭前", "饭后")),
        ("不良反应", ("不良反应",), ("副作用", "不良反应", "不舒服", "恶心", "皮疹")),
        ("特殊人群", ("孕妇及哺乳期妇女用药", "儿童用药", "老年用药"), ("孕妇", "妊娠", "哺乳", "儿童", "婴儿", "老人", "老年")),
    )

    @classmethod
    def required_sections(cls, query: str) -> tuple[str, ...]:
        """Return sections required by the query's explicit clinical intent."""
        text = unicodedata.normalize("NFKC", query or "")
        sections: list[str] = []
        for _name, candidates, terms in cls.INTENT_SECTION_PATTERNS:
            if any(term in text for term in terms):
                sections.extend(candidates)
        return tuple(dict.fromkeys(sections))


    @classmethod
    def anchors(cls, query: str, drug_vocabulary) -> tuple[frozenset[str], tuple[str, ...]]:
        """锚点 = 语料药名(整体,权重与单锚点相同) + 净化后中文二元组 + 英文词 + 数字。"""
        text = unicodedata.normalize("NFKC", query or "")
        found_drugs = tuple(sorted(d for d in drug_vocabulary if d and d in text))
        remaining = text
        for drug in found_drugs:
            remaining = remaining.replace(drug, " ")
        anchors: set[str] = set(found_drugs)
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", remaining):
            cleaned = "".join(ch for ch in run if ch not in cls.STOP_CHARS)
            if len(cleaned) < 2:
                continue
            anchors.update(cleaned[i:i + 2] for i in range(len(cleaned) - 1))
        for word in re.findall(r"[A-Za-z]{2,}", remaining):
            anchors.add(word.casefold())
        for number in re.findall(r"\d+(?:\.\d+)?", remaining):
            anchors.add(number)
        return frozenset(anchors), found_drugs

    @classmethod
    def evaluate(
        cls,
        query: str,
        evidence_texts,
        drug_vocabulary,
        *,
        min_doc_coverage: float = 0.10,
        min_union_coverage: float = 0.20,
        evidence_sections=None,
    ) -> tuple[bool, dict]:
        """passed ⇔ 存在文档覆盖 ≥ min_doc 且 证据并集覆盖 ≥ min_union。
        零锚点(如"帮我看看这个药"全部被停用字吸收)⇒ 不通过(不猜意图)。
        默认阈值只拦截近似零重叠(缺席药/纯噪声);口语化 query 与正确证据的
        词汇重叠天然低,硬门禁会误杀——故默认以影子模式(dry_run)运行。"""
        anchors, drugs = cls.anchors(query, drug_vocabulary)
        info: dict = {
            "anchors": sorted(anchors),
            "drugs": list(drugs),
            "anchor_count": len(anchors),
            "required_sections": list(cls.required_sections(query)),
            "served_sections": sorted({str(s) for s in (evidence_sections or [])}),
            "max_doc_coverage": 0.0,
            "union_coverage": 0.0,
            "passed": False,
            "reason": "",
        }
        if not anchors or not evidence_texts:
            info["reason"] = "no-anchors" if not anchors else "no-evidence"
            return False, info
        required = cls.required_sections(query)
        served = {str(s) for s in (evidence_sections or [])}
        for text in evidence_texts or []:
            for section in ("禁忌", "用法用量", "不良反应", "注意事项", "药物相互作用", "孕妇及哺乳期妇女用药", "儿童用药", "老年用药"):
                if f"【{section}】" in text or section in text[:80]:
                    served.add(section)
        if required and not (served & set(required)):
            info["reason"] = "required-section-missing"
            return False, info
        union: set[str] = set()
        max_doc = 0.0
        for text in evidence_texts:
            hits = {a for a in anchors if a in text}
            union |= hits
            max_doc = max(max_doc, len(hits) / len(anchors))
        union_cov = len(union) / len(anchors)
        info["max_doc_coverage"] = round(max_doc, 4)
        info["union_coverage"] = round(union_cov, 4)
        info["passed"] = max_doc >= min_doc_coverage and union_cov >= min_union_coverage
        if not info["passed"]:
            info["reason"] = "anchor-coverage-below-floor"
        return info["passed"], info
