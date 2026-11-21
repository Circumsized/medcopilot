"""Wave 10: 安全回归契约(全离线,无模型下载)。

覆盖 C1/C2/C3 与同批 P2 安全边界,全部先在探针里实证复现再固化为回归:

- C1 保护安全槽逐出:逐出目标是非保护章节中 rerank_score 最低者,受保护块
  (禁忌/相互作用/注意事项…)绝不弹出;容量不足走 reserve-capacity 弃答;
- C2 剂量门口语形态:半片/一天两次/微克/mcg/U/早晚各一粒/分数(1/2、½、
  二分之一)/中文数字/零宽变体;西文单位 (?![a-z]) 边界卫兵(16GB、2 uses
  不误伤);混合"风险披露+剂量"句按逗号细分(风险警示不丢、剂量不搭便车);
  quote 模式仍逐字放行原文(全角/零宽不被归一化改写);
- C3 信息疑问词(什么是/意思/介绍…)独立成类、不计入降级条件:当前急症的
  信息式追问必须触发;叙事标记(科普/影视/既往)仍降级;
- 裸胸痛与血压数值区间(≥180 收缩压 / ≥120 舒张压,症状联合而非数值诊断);
- dosage_gate_mode 入口归一化 + 启动校验(非法值构造期即失败);
- sufficiency 门与 OCR 检索同口径(硬门下 OCR 主流程不再 100% 弃答);
- 空 provider 响应有界重试(受 attempts 与 deadline 双封顶)且终局为真实
  错误状态,不伪称成功;
- 策略镜像 v5:与代码常量等价、评审覆盖新指纹、如实标注非临床。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import httpx
import pytest

from medcopilot.common.config import config
from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker import Reranker
from medcopilot.safety.section_policy import ContraindicationTrigger
from medcopilot.workflow.agent_graph import AgentGraph
from medcopilot.workflow.safety_guard import DosageGate, MedicalSafetyGuard

CONTRA = ContraindicationTrigger.CONTRA_SECTIONS
REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "medcopilot" / "datasets" / "policies" / "safety_policy_v1.json"

SAFETY_LABEL = (
    "【药品名称】布洛芬缓释胶囊\n"
    "【适应症】用于缓解轻至中度疼痛如头痛、关节痛、痛经。\n"
    "【规格】每粒含布洛芬0.3克。\n"
    "【成份】本品主要成份为布洛芬。\n"
    "【贮藏】密封保存。\n"
    "【禁忌】对阿司匹林过敏者禁用。\n"
    "【药物相互作用】避免与其他非甾体抗炎药同用。\n"
    "【注意事项】服药期间不得饮酒。"
)
ULCER_QUERY = "我有胃溃疡，最近头痛想买点止痛药，哪个好？"


class LabelledLLM:
    """可控 LLM:返回注入文本,记录调用。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        return self.reply

    def _should_mock(self):
        return True

    def _mock_reply(self, prompt, **kwargs):
        return self.invoke(prompt)


class _RecallWithoutSafety:
    """初始召回只返回非安全章节(最坏情形:安全块全靠保护位并入)。"""

    def __init__(self, documents):
        self.documents = documents

    def retrieve_with_scores(self, query, top_n):
        picked = [d for d in self.documents if d.metadata.get("section") not in CONTRA][:top_n]
        return [(d, 0.5) for d in picked]


class _ScoredReranker:
    """按给定分数表排序的 reranker 替身(mode 可配,避免降级信号噪声)。"""

    mode = "cross-encoder"

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def rerank(self, query, documents, top_n=3):
        keyed = {d.metadata.get("section"): d for d in documents}
        ordered = sorted(keyed, key=lambda section: -self.scores.get(section, 0.0))
        return [(keyed[s], self.scores.get(s, 0.0)) for s in ordered][:top_n]


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


def _safety_graph(**overrides) -> AgentGraph:
    kwargs = dict(
        retriever=_RecallWithoutSafety(
            ClinicalChunker.chunk_drug_label(SAFETY_LABEL, drug_name="布洛芬缓释胶囊")
        ),
        llm=LabelledLLM("请遵医嘱。"),
        reranker=_ScoredReranker({
            "药品名称": 0.7, "适应症": 0.9, "规格": 0.85, "成份": 0.8, "贮藏": 0.75,
            "禁忌": 0.6, "药物相互作用": 0.5, "注意事项": 0.4,
        }),
        retrieve_candidate_pool=5,
        evidence_top_n=5,
        safety_reserve_slots=3,
    )
    kwargs.update(overrides)
    return AgentGraph(**kwargs)


# ==================================================== C1: 保护安全槽逐出


def test_multi_safety_blocks_all_survive_force_merge():
    """三个受保护块全部缺席 top-5 时,必须全部存活(旧实现只剩最后一个)。"""
    graph = _safety_graph()
    result = graph.run(ULCER_QUERY)
    served = [s.get("section") for s in result["sources"]]
    assert set(served) >= {"禁忌", "药物相互作用", "注意事项"}, served
    assert len(result["sources"]) <= 5
    chunk_ids = [s["chunk_id"] for s in result["sources"]]
    assert len(chunk_ids) == len(set(chunk_ids)), "保护位不得重复追加同一 chunk"


def test_eviction_targets_lowest_scored_non_protected_chunk():
    """逐出顺序 = 非保护章节中 rerank_score 最低者;强证据与受保护块都在场。"""
    graph = _safety_graph()
    result = graph.run(ULCER_QUERY)
    served = [s.get("section") for s in result["sources"]]
    # 分数最低的三个非保护块(药品名称 0.7 / 贮藏 0.75 / 成份 0.8)被逐出,
    # 分数最高的两个非保护块(适应症 0.9 / 规格 0.85)保留,三个受保护块并入。
    assert served == ["适应症", "规格", "禁忌", "药物相互作用", "注意事项"], served


def test_single_missing_block_never_evicts_in_place_contraindication():
    """单块缺失时也不得弹出已在场的禁忌块(旧实现 pop 列表尾的第二种死法)。"""
    graph = _safety_graph(
        reranker=_ScoredReranker({
            "药品名称": 0.7, "适应症": 0.9, "规格": 0.85, "成份": 0.8, "贮藏": 0.75,
            "禁忌": 0.95, "药物相互作用": 0.5, "注意事项": 0.4,
        }),
    )
    result = graph.run(ULCER_QUERY)
    served = {s.get("section") for s in result["sources"]}
    assert {"禁忌", "药物相互作用", "注意事项"} <= served, served


def test_only_protected_evidence_abstains_instead_of_dropping():
    """在场证据全是受保护安全章节时:弃答并记 reserve-capacity,不丢任何一块。"""
    graph = _safety_graph(
        reranker=_ScoredReranker({
            "药品名称": 0.1, "适应症": 0.1, "规格": 0.1, "成份": 0.1, "贮藏": 0.1,
            "禁忌": 0.9, "药物相互作用": 0.8, "注意事项": 0.7,
        }),
        evidence_top_n=2,
        retrieve_candidate_pool=1,
    )
    result = graph.run(ULCER_QUERY)
    assert result["abstained"] is True
    assert "safety-section:reserve-capacity" in result["degradations"]


# ==================================================== C2: 口语剂量形态


COLLOQUIAL_DOSAGES = [
    "每次半片",
    "一天两次",
    "一天吃两次",
    "每次50微克",
    "每次50mcg",
    "每次10U",
    "早晚各一粒",
    "早晚各一次",
    "每晚1次",
    "一周一次",
    "每次二分之一片",
    "每次1/2片",
    "每次½片",
    "每次三分之一粒",
    "每次两片",
    "一次三粒",
    "每\u200b次\u200b1\u200b粒",       # 零宽插入
    "一日\u200b2\u200b次",             # 零宽插入
]


@pytest.mark.parametrize("sentence", COLLOQUIAL_DOSAGES)
def test_colloquial_dosage_forms_are_detected(sentence):
    released, stats = DosageGate.apply(sentence, [], [], mode="redact")
    assert stats["instructional"] == 1, sentence
    assert stats["redacted"] == 1, sentence
    assert DosageGate.DOSAGE_REFERRAL_TEXT in released, sentence


@pytest.mark.parametrize("sentence", ["内存16GB", "part 2 uses", "2 uses of it"])
def test_english_word_boundaries_are_not_dosage(sentence):
    """西文单位 (?![a-z]) 边界卫兵:16GB 不是 16g、2 uses 不是 2u
    (不删全部空白、不把英文词边界误读成剂量)。"""
    released, stats = DosageGate.apply(sentence, [], [], mode="redact")
    assert stats["instructional"] == 0, sentence
    assert released == sentence


def test_risk_and_dosage_in_one_comma_sentence_split():
    """混合句按逗号细分:逐字风险片段保留,剂量片段转介,不搭整句豁免便车。"""
    evidence = "【注意事项】一日剂量超过4g可致严重肝损伤,一次1粒一日2次。\n【禁忌】活动期消化道溃疡患者禁用。"
    answer = "一日剂量超过4g可致严重肝损伤,一次1粒一日2次"
    released, stats = DosageGate.apply(answer, [evidence], ["注意事项"], mode="redact")
    assert stats["kept_risk"] == 1 and stats["redacted"] == 1, stats
    assert "超过4g可致严重肝损伤" in released, released
    assert "一次1粒一日2次" not in released, released


def test_pure_risk_sentence_is_kept_in_both_modes():
    """纯风险句(无逗号)不受细分影响:两模式均逐字保留。"""
    evidence = "【注意事项】一日剂量超过4g可致严重肝损伤。"
    for mode in ("redact", "quote"):
        released, stats = DosageGate.apply(
            "一日剂量超过4g可致严重肝损伤", [evidence], ["注意事项"], mode=mode
        )
        assert stats["kept_risk"] == 1 and stats["redacted"] == 0, (mode, stats)
        assert "超过4g" in released, (mode, released)


@pytest.mark.parametrize(
    "sentence",
    [
        "严重肝损伤者一次1粒",      # 无逗号:一次 + 数量 + 单位
        "肝损伤患者一日2次",        # _FREQUENCY
        "肝损伤者早晚各一粒",       # _SCHEDULE
        "每次半片可致肝损伤",       # 每次 + 半 + 单位
        "一天两次加重肝损伤",       # 一天 + 两次
    ],
)
def test_risk_exemption_denied_when_same_piece_carries_dosing_directive(sentence):
    """独立 reviewer 实证的 P2:无逗号时"严重肝损伤者一次1粒"曾整句 kept_risk
    漏出剂量。同一 piece 含显式给药指令时风险豁免必须失效。证据逐字包含该句,
    排除 verbatim 判定的干扰——被否定的只有豁免本身。"""
    evidence = f"【注意事项】{sentence}。"
    released, stats = DosageGate.apply(sentence, [evidence], ["注意事项"], mode="redact")
    assert stats["kept_risk"] == 0, (sentence, stats)
    assert stats["redacted"] == 1, (sentence, stats)
    assert DosageGate.DOSAGE_REFERRAL_TEXT in released, (sentence, released)


@pytest.mark.parametrize("sections", [["注意事项"], ["适应症"]], ids=["with-safety", "no-safety"])
@pytest.mark.parametrize("mode", ["redact", "quote"])
@pytest.mark.parametrize(
    "sentence",
    ["一日剂量超过4g可致严重肝损伤", "一日剂量不超过4g", "每次用量不超过6g"],
)
def test_pure_risk_limits_are_kept_in_both_modes(sentence, mode, sections):
    """纯风险限量(超过4g/不超过4g/用量不超过)是指标不是指令:两模式均逐字保留,
    且不依赖证据里是否有安全章节(less evidence 也不把安全警示抹掉)。"""
    evidence = f"【注意事项】{sentence}。"
    released, stats = DosageGate.apply(sentence, [evidence], sections, mode=mode)
    assert stats["kept_risk"] == 1 and stats["redacted"] == 0, (sentence, mode, sections, stats)
    assert sentence in released, (sentence, mode, sections, released)


def test_pure_risk_kept_in_quote_mode_without_safety_section():
    """纯风险限量不是 instructional quote:quote 模式缺安全章节时仍保留
    (原契约——风险披露两模式均保留,不因缺禁忌把安全警示抹掉)。"""
    evidence = "【注意事项】一日剂量超过4g可致严重肝损伤。"
    released, stats = DosageGate.apply(
        "一日剂量超过4g可致严重肝损伤", [evidence], ["适应症"], mode="quote"
    )
    assert stats["kept_risk"] == 1 and stats["redacted"] == 0, stats
    assert "超过4g" in released, released


def test_directive_dose_with_risk_word_redacted_in_quote_mode_without_safety_section():
    """对照面:含显式给药指令的风险词句子("严重肝损伤者一次1粒")在 quote 无
    安全章节时不得放行——(a) 否定条件先否掉风险豁免,quote 底线再拒逐字剂量。"""
    evidence = "【注意事项】严重肝损伤者一次1粒。"
    released, stats = DosageGate.apply(
        "严重肝损伤者一次1粒", [evidence], ["适应症"], mode="quote"
    )
    assert stats["kept_risk"] == 0 and stats["kept_quote"] == 0, stats
    assert stats["redacted"] == 1, stats
    assert "一次1粒" not in released, released


def test_directive_regex_does_not_match_risk_limits():
    """否定条件自身不得误伤纯限量:指令正则不得命中"超过4g/不超过4g/用量不超过"。"""
    for sentence in (
        "一日剂量超过4g可致严重肝损伤",
        "一日剂量不超过4g",
        "每次用量不超过6g",
        "一次服用过量可致肝损伤",
    ):
        assert not DosageGate.DOSAGE_DIRECTIVE_RE.search(
            DosageGate._normalize_for_match(sentence)
        ), sentence


def test_quote_mode_releases_verbatim_never_normalized_text():
    """quote 放行的是原文逐字:全角与零宽形态不得被归一化改写后输出。"""
    full_width_evidence = "【用法用量】口服,一次１粒(０.３ｇ),一日2次。"
    full_width_sentence = "一次１粒(０.３ｇ)"
    released, stats = DosageGate.apply(
        full_width_sentence, [full_width_evidence], ["用法用量", "禁忌"], mode="quote"
    )
    assert stats["kept_quote"] == 1, stats
    assert full_width_sentence in released, released
    assert "一次1粒" not in released, released

    zero_width_sentence = "每次\u200b1\u200b粒"
    released, stats = DosageGate.apply(
        zero_width_sentence, [zero_width_sentence], ["用法用量", "禁忌"], mode="quote"
    )
    assert stats["kept_quote"] == 1, stats
    assert zero_width_sentence in released, released  # 原文(含零宽)逐字输出


def test_referral_and_announcement_do_not_self_match():
    assert not DosageGate.DOSAGE_SENTENCE_RE.search(
        DosageGate._normalize_for_match(DosageGate.DOSAGE_REFERRAL_TEXT)
    )
    assert not DosageGate.DOSAGE_SENTENCE_RE.search(
        DosageGate._normalize_for_match(DosageGate.QUOTE_ANNOUNCEMENT)
    )


def test_colloquial_dosage_redacted_end_to_end(monkeypatch):
    """链路级:FixedLLM 逐字复述口语剂量句 → redact 模式下最终回答不含剂量。"""
    label = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【适应症】用于缓解轻至中度疼痛。\n"
        "【用法用量】口服。每次半片，早晚各一次。\n"
        "【禁忌】活动期消化道溃疡患者禁用。"
    )
    from medcopilot.llm.unified_llm import UnifiedLLM

    reply = "口服。每次半片，早晚各一次。"

    class FixedLLM(UnifiedLLM):
        def invoke(self, prompt, **kwargs):
            return reply

        def _should_mock(self):
            return True

        def _mock_reply(self, prompt, **kwargs):
            return reply

    graph = AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(label, drug_name="布洛芬缓释胶囊"),
        ),
        llm=FixedLLM(provider="mock"),
        reranker=Reranker(enabled=False),
        dosage_gate_mode="redact",
    )
    result = graph.run("布洛芬缓释胶囊怎么吃")
    assert result["abstained"] is False
    assert "半片" not in result["answer"], result["answer"]
    assert DosageGate.DOSAGE_REFERRAL_TEXT in result["answer"]
    assert result["dosage_gate"]["redacted"] >= 1


# ==================================================== C3: 信息疑问词降级


@pytest.mark.parametrize(
    "query",
    [
        "我爸心梗了，帮我查一下心梗是什么意思",
        "我妈脑梗了，脑梗是什么意思",
        "什么是心肌梗死",
        "我想了解中风的原因",
    ],
)
def test_informational_mention_of_current_emergency_triggers(query):
    """信息疑问词描述提问方式而非病史:当前急症的信息式追问必须触发。"""
    assert MedicalSafetyGuard.check_emergency(query)[0] is True


def test_narrative_markers_still_demote_even_with_informational_words():
    """叙事标记(科普/影视/既往)在场时,信息疑问词不阻止降级。"""
    assert MedicalSafetyGuard.check_emergency("科普文章：什么是心肌梗死")[0] is False
    assert MedicalSafetyGuard.check_emergency("心肌梗死的急救措施有哪些，我在做科普资料")[0] is False
    assert MedicalSafetyGuard.check_emergency("电视剧里有人心梗发作的场景")[0] is False


def test_informational_markers_are_not_narrative_markers():
    from medcopilot.workflow.emergency_detector import (
        ACTIVE_MARKERS,
        INFORMATIONAL_MARKERS,
        NARRATIVE_MARKERS,
    )

    assert INFORMATIONAL_MARKERS and NARRATIVE_MARKERS and ACTIVE_MARKERS
    assert not (set(INFORMATIONAL_MARKERS) & set(NARRATIVE_MARKERS))
    assert not (set(INFORMATIONAL_MARKERS) & set(ACTIVE_MARKERS))
    assert not (set(NARRATIVE_MARKERS) & set(ACTIVE_MARKERS))


# ==================================================== 裸胸痛 / 血压区间


@pytest.mark.parametrize(
    "query",
    ["我胸痛", "突然胸痛", "胸口疼得厉害", "胸口痛", "胸部疼痛"],
)
def test_bare_chest_pain_triggers(query):
    assert MedicalSafetyGuard.check_emergency(query)[0] is True


def test_bare_chest_pain_narrative_context_is_a_known_conservative_boundary():
    """已知边界(保守方向):裸胸痛模式自身含急性标记"疼/痛",叙事语境实际
    不可降级——over-triage 仅损失一次 RAG 回答,under-triage 是临床危害。"""
    assert MedicalSafetyGuard.check_emergency("电视剧里有人胸痛的场景")[0] is True


@pytest.mark.parametrize(
    "query",
    [
        "血压250/180，头痛得厉害",   # 收缩压 ≥180
        "血压180，还恶心",           # 收缩压 ≥180(无斜杠)
        "血压200多，现在头晕得厉害",  # 口语区间
        "血压130/125，头晕得厉害",   # 舒张压 ≥120
    ],
)
def test_blood_pressure_crisis_ranges_with_symptoms_trigger(query):
    assert MedicalSafetyGuard.check_emergency(query)[0] is True


@pytest.mark.parametrize(
    "query",
    ["血压多少算正常范围", "血压120/80，精神还行", "高血压患者日常饮食怎么安排"],
)
def test_normal_blood_pressure_does_not_trigger(query):
    assert MedicalSafetyGuard.check_emergency(query)[0] is False


# ==================================================== dosage mode 归一化/启动校验


def _mode_graph(**overrides) -> AgentGraph:
    kwargs = dict(
        retriever=HybridRetriever(documents=[]),
        llm=LabelledLLM("x"),
        reranker=Reranker(enabled=False),
    )
    kwargs.update(overrides)
    return AgentGraph(**kwargs)


@pytest.mark.parametrize("raw,expected", [(" REDACT ", "redact"), ("Quote", "quote"), ("off", "off")])
def test_dosage_gate_mode_is_normalized_at_construction(raw, expected):
    assert _mode_graph(dosage_gate_mode=raw).dosage_gate_mode == expected


def test_invalid_dosage_gate_mode_fails_at_construction():
    with pytest.raises(ValueError, match="dosage_gate_mode must be one of"):
        _mode_graph(dosage_gate_mode="yolo")


def test_off_mode_with_whitespace_still_rejected_in_production(monkeypatch):
    monkeypatch.setattr(config, "project_env", "production")
    with pytest.raises(ValueError, match="not allowed when PROJECT_ENV is production"):
        _mode_graph(dosage_gate_mode=" off ")


# ==================================================== sufficiency 与 OCR 检索同口径


def test_ocr_only_query_survives_hard_sufficiency_gate():
    """OCR 主流程(原始 query 无药名)在硬门下不得弃答:门评的是检索同口径文本。"""
    label = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【适应症】用于缓解轻至中度疼痛如头痛、关节痛、痛经。\n"
        "【用法用量】口服。每次半片，早晚各一次。\n"
        "【禁忌】活动期消化道溃疡患者禁用。"
    )
    llm = LabelledLLM("口服。每次半片，早晚各一次。")
    graph = AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(label, drug_name="布洛芬缓释胶囊"),
        ),
        llm=llm,
        reranker=Reranker(enabled=False),
        sufficiency_dry_run=False,
    )
    result = graph.run(
        "这个药怎么吃",
        drug_context="识别药品: 布洛芬缓释胶囊（置信匹配）；OCR 原文: 布洛芬",
    )
    assert not any(d.startswith("sufficiency:") for d in result["degradations"])
    assert result["sources"], "OCR 主流程在硬门下必须有证据"
    assert result["abstained"] is False


def test_raw_query_without_ocr_drug_name_still_abstains_under_hard_gate():
    """对照面:同一提问无 OCR 药名时锚点确实不被覆盖,硬门照常弃答(不放松)。"""
    label = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【适应症】用于缓解轻至中度疼痛如头痛、关节痛、痛经。\n"
        "【用法用量】口服。每次半片，早晚各一次。\n"
        "【禁忌】活动期消化道溃疡患者禁用。"
    )
    llm = LabelledLLM("口服。每次半片，早晚各一次。")
    graph = AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(label, drug_name="布洛芬缓释胶囊"),
        ),
        llm=llm,
        reranker=Reranker(enabled=False),
        sufficiency_dry_run=False,
    )
    result = graph.run("这个药怎么吃")
    assert any(d.startswith("sufficiency:") for d in result["degradations"])
    assert llm.calls == 0


# ==================================================== 空 provider 响应有界重试


@pytest.fixture
def _clean_llm_environment(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(config, "mock_llm", False)
    monkeypatch.setattr(config, "project_env", "development")


def _mock_http_client(monkeypatch, outcomes):
    calls = Counter()
    remaining = list(outcomes)

    def handler(request):
        calls["count"] += 1
        result = remaining.pop(0) if remaining else outcomes[-1]
        if isinstance(result, Exception):
            raise result
        status, body = result
        return httpx.Response(status, json=body, request=request)

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr("medcopilot.llm.unified_llm.UnifiedLLM._should_mock", lambda self: False)
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.sleep", lambda _: None)
    return calls


def test_empty_provider_response_consumes_bounded_retry_then_reports_error(monkeypatch, _clean_llm_environment):
    """空响应消耗有界重试预算(不是一次抖动即终局),持续为空时是真实错误状态。"""
    from medcopilot.llm.unified_llm import UnifiedLLM

    calls = _mock_http_client(monkeypatch, [(200, {"choices": []})] * 3)
    result = UnifiedLLM(max_attempts=3, backoff_base=0).invoke_with_status("问题")
    assert calls["count"] == 3
    assert result.mode == "error"
    assert result.error_name == "EmptyLLMResponse"
    assert result.diagnostic == "empty_provider_response"
    assert result.mocked is False


def test_empty_provider_response_recovers_on_retry(monkeypatch, _clean_llm_environment):
    from medcopilot.llm.unified_llm import UnifiedLLM

    calls = _mock_http_client(
        monkeypatch,
        [(200, {"choices": []}), (200, {"choices": [{"message": {"content": "ready"}}]})],
    )
    result = UnifiedLLM(max_attempts=3, backoff_base=0).invoke_with_status("问题")
    assert calls["count"] == 2
    assert result.mode == "live" and result.text == "ready"


def test_empty_provider_response_respects_retry_deadline(monkeypatch, _clean_llm_environment):
    """重试仍受 retry_deadline 封顶:时钟越过预算即停,不无限重试空响应。"""
    from medcopilot.llm.unified_llm import UnifiedLLM

    calls = _mock_http_client(monkeypatch, [(200, {"choices": []})] * 5)
    ticks = iter([0.0, 0.1, 0.6, 0.7, 0.8, 0.9])
    monkeypatch.setattr(
        "medcopilot.llm.unified_llm.time.monotonic", lambda: next(ticks, 99.0)
    )
    result = UnifiedLLM(max_attempts=5, backoff_base=2, retry_deadline=0.5).invoke_with_status("问题")
    assert calls["count"] == 1
    assert result.error_name == "RetryBudgetExceeded"
    assert result.diagnostic == "retry_deadline_exceeded"


# ==================================================== 策略镜像 v5


def test_policy_mirror_v5_is_synced_with_honest_review():
    from medcopilot.safety.policy import check_code_equivalence, load_policy

    policy = load_policy(POLICY_PATH)
    assert policy.policy_version == MedicalSafetyGuard.POLICY_VERSION == 5
    assert check_code_equivalence(policy) == []
    wave10 = [r for r in policy.reviews if r["review_id"] == "eng-2026-10-10-wave10-safety"]
    assert len(wave10) == 1, "必须追加覆盖新指纹的 approved 评审记录"
    record = wave10[0]
    assert record["decision"] == "approved"
    assert record["covers_sha256"] == policy.content_sha256
    assert "非临床" in record["notes"], "评审记录必须如实标注非临床性质"
    assert policy.clinical_signoff_approved is False, "clinical_signoff 必须如实保持 none"
