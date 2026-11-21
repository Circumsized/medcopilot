"""W6.1 剂量门双模式契约(DosageGate)。

需求方裁决(2026-10-04):A+B 并存——
- ``redact``(默认):逐字有据的指令性剂量句替换为转介话术;
- ``quote``:逐字有据 + 证据含禁忌/相互作用/注意事项安全章节 ⇒ 放行并显式
  声明"说明书原文引用";安全章节缺失 ⇒ 该句仍替换为转介(fail-closed);
- 两模式共同:编造/改写剂量 ⇒ extractive 门弃答;风险披露句逐字 ⇒ 保留;
  mode="off" 仅限开发,production 下 AgentGraph 构造直接拒绝。
"""

from __future__ import annotations

import pytest

from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.workflow.agent_graph import AgentGraph
from medcopilot.workflow.safety_guard import DosageGate, MedicalSafetyGuard

LABEL_WITH_CONTRA = (
    "【药品名称】布洛芬缓释胶囊\n"
    "【适应症】用于缓解轻至中度疼痛。\n"
    "【用法用量】口服。成人及12岁以上儿童：一日2次，一次1粒（0.3g），早晚各一次。\n"
    "【注意事项】一日剂量超过4g可致严重肝损伤。\n"
    "【禁忌】活动期消化道溃疡患者禁用。"
)
LABEL_WITHOUT_CONTRA = (
    "【药品名称】某药片\n"
    "【适应症】用于缓解轻至中度疼痛。\n"
    "【用法用量】口服。成人及12岁以上儿童：一日2次，一次1粒（0.3g）。"
)
DOSAGE_SENTENCE = "成人及12岁以上儿童：一日2次，一次1粒（0.3g），早晚各一次"
RISK_SENTENCE = "一日剂量超过4g可致严重肝损伤"


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


# ------------------------------------------------------------ 单元:门本身


def test_redact_mode_replaces_instructional_dosage():
    evidence = ClinicalChunker.chunk_drug_label(LABEL_WITH_CONTRA, drug_name="布洛芬缓释胶囊")
    sections = [d.metadata["section"] for d in evidence]
    released, stats = DosageGate.apply(
        "用法：" + DOSAGE_SENTENCE,
        [d.page_content for d in evidence], sections, mode="redact",
    )
    assert stats["instructional"] == 1 and stats["redacted"] == 1
    assert "0.3g" not in released and "一次1粒" not in released
    assert DosageGate.DOSAGE_REFERRAL_TEXT in released


def test_quote_mode_releases_verbatim_when_safety_section_present():
    evidence = ClinicalChunker.chunk_drug_label(LABEL_WITH_CONTRA, drug_name="布洛芬缓释胶囊")
    sections = [d.metadata["section"] for d in evidence]
    released, stats = DosageGate.apply(
        DOSAGE_SENTENCE, [d.page_content for d in evidence], sections, mode="quote",
    )
    assert stats["kept_quote"] == 1 and stats["redacted"] == 0
    assert "0.3g" in released
    assert DosageGate.QUOTE_ANNOUNCEMENT in released


def test_quote_mode_fails_closed_without_safety_section():
    evidence = ClinicalChunker.chunk_drug_label(LABEL_WITHOUT_CONTRA, drug_name="某药片")
    sections = [d.metadata["section"] for d in evidence]
    assert not DosageGate._has_safety_section(sections)
    released, stats = DosageGate.apply(
        DOSAGE_SENTENCE, [d.page_content for d in evidence], sections, mode="quote",
    )
    # 逐字剂量不得与缺失的禁忌并存:该句仍替换为转介
    assert stats["redacted"] == 1 and stats["kept_quote"] == 0
    assert "0.3g" not in released


def test_risk_disclosure_kept_in_both_modes():
    evidence = ClinicalChunker.chunk_drug_label(LABEL_WITH_CONTRA, drug_name="布洛芬缓释胶囊")
    texts = [d.page_content for d in evidence]
    sections = [d.metadata["section"] for d in evidence]
    for mode in ("redact", "quote"):
        released, stats = DosageGate.apply(RISK_SENTENCE, texts, sections, mode=mode)
        assert stats["kept_risk"] == 1
        assert "4g" in released


def test_paraphrased_dosage_is_not_verbatim_so_extractive_gate_owns_it():
    """改写剂量(“每次吃1克”)不是证据子集 → reviewer 的 extractive 门整体弃答;
    本测试钉死 DosageGate 的 verbatim 判定不会把它误判为可放行。"""
    evidence = ClinicalChunker.chunk_drug_label(LABEL_WITH_CONTRA, drug_name="布洛芬缓释胶囊")
    texts = [d.page_content for d in evidence]
    sections = [d.metadata["section"] for d in evidence]
    answer = "每次吃1克，一天两次。"
    assert not MedicalSafetyGuard.check_extractive_support(answer, texts)
    released, stats = DosageGate.apply(answer, texts, sections, mode="quote")
    # 命中指令句模式但非逐字 ⇒ quote 模式也不放行(转介),安全底线不依赖 verbatim 判定漏洞
    assert stats["kept_quote"] == 0 and stats["redacted"] == 1
    assert "1克" not in released


def test_referral_text_does_not_match_its_own_pattern():
    """自反断言:转介话术自身不得命中指令性剂量正则(否则会被下一轮再替换)。"""
    assert not DosageGate.DOSAGE_SENTENCE_RE.search(DosageGate.DOSAGE_REFERRAL_TEXT)
    assert not DosageGate.DOSAGE_SENTENCE_RE.search(DosageGate.QUOTE_ANNOUNCEMENT)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        DosageGate.apply("x", [], [], mode="yolo")


# ------------------------------------------------------------ 集成:reviewer 链路


def _graph(label: str, drug: str, mode: str) -> AgentGraph:
    return AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(label, drug_name=drug),
        ),
        dosage_gate_mode=mode,
    )


def test_off_mode_rejected_in_production(monkeypatch):
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    with pytest.raises(ValueError):
        _graph(LABEL_WITH_CONTRA, "布洛芬缓释胶囊", "off")


def test_redact_mode_end_to_end(monkeypatch):
    """FixedLLM 逐字复述说明书剂量句 → redact 模式下最终回答不含剂量数字。"""
    from medcopilot.llm.unified_llm import UnifiedLLM

    class FixedLLM(UnifiedLLM):
        def invoke(self, prompt, **kwargs):
            return "口服。" + DOSAGE_SENTENCE + "。"

        def _should_mock(self):
            return True

        def _mock_reply(self, prompt, **kwargs):
            return "口服。" + DOSAGE_SENTENCE + "。"

    graph = AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(LABEL_WITH_CONTRA, drug_name="布洛芬缓释胶囊"),
        ),
        llm=FixedLLM(provider="mock"),
        dosage_gate_mode="redact",
    )
    result = graph.run("布洛芬缓释胶囊怎么吃")
    assert result["abstained"] is False
    assert "0.3g" not in result["answer"] and "一次1粒" not in result["answer"]
    assert DosageGate.DOSAGE_REFERRAL_TEXT in result["answer"]
    assert result["dosage_gate"]["redacted"] == 1


def test_quote_mode_end_to_end_with_safety_section(monkeypatch):
    from medcopilot.llm.unified_llm import UnifiedLLM

    class FixedLLM(UnifiedLLM):
        def invoke(self, prompt, **kwargs):
            return "口服。" + DOSAGE_SENTENCE + "。"

        def _should_mock(self):
            return True

        def _mock_reply(self, prompt, **kwargs):
            return "口服。" + DOSAGE_SENTENCE + "。"

    graph = AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(LABEL_WITH_CONTRA, drug_name="布洛芬缓释胶囊"),
        ),
        llm=FixedLLM(provider="mock"),
        dosage_gate_mode="quote",
    )
    result = graph.run("布洛芬缓释胶囊怎么吃")
    assert result["abstained"] is False
    assert "0.3g" in result["answer"]
    assert "说明书原文逐字引用" in result["answer"]
    assert result["dosage_gate"]["kept_quote"] == 1


def test_dosage_gate_catches_wan_units_and_enforces_per_drug_safety_section():
    penicillin_dose = "肌内注射，一日80万单位，分3次给药。"
    released, stats = DosageGate.apply(
        penicillin_dose,
        [penicillin_dose],
        ["用法用量"],
        mode="redact",
    )
    assert "80万单位" not in released
    assert stats["instructional"] == 1 and stats["redacted"] == 1

    # Multi-drug quote mode: Drug A has only 用法用量 (no safety section), while
    # Drug B has 注意事项. Drug A must NOT borrow Drug B's safety section.
    released_multi, stats_multi = DosageGate.apply(
        penicillin_dose,
        [penicillin_dose, "【注意事项】偶见胃肠道不适。"],
        ["用法用量", "注意事项"],
        mode="quote",
        evidence_drugs=["青霉素钠", "布洛芬"],
    )
    assert "80万单位" not in released_multi
    assert stats_multi["redacted"] == 1 and stats_multi["kept_quote"] == 0

    # Same-sentence match across two drugs where only one has a safety section:
    # must still fail closed (all matched named drugs must have safety sections).
    shared_dose = "口服，一次1片，一日2次。"
    released_shared, stats_shared = DosageGate.apply(
        shared_dose,
        [shared_dose, "【注意事项】慎用。", shared_dose],
        ["用法用量", "注意事项", "用法用量"],
        mode="quote",
        evidence_drugs=["阿莫西林", "阿莫西林", "布洛芬"],
    )
    assert "一次1片" not in released_shared
    assert stats_shared["redacted"] == 1 and stats_shared["kept_quote"] == 0
