import pytest

from medcopilot.agents.ocr_tool import OCRTool
from medcopilot.workflow.agent_graph import _wrap_untrusted
from medcopilot.workflow.prompt_budget import (
    EVIDENCE_HEADER,
    EvidenceItem,
    PromptBudget,
    PromptBudgetFitter,
    _safe_cut,
    build_messages,
)
from medcopilot.workflow.safety_guard import DosageGate


@pytest.mark.parametrize("verb", ["服用", "食用", "饮用", "吸入", "注射", "外用", "塞入"])
def test_multi_character_dosing_verbs_cannot_claim_risk_exemption(verb):
    sentence = f"严重肝损伤者每次{verb}1片。"
    answer, stats = DosageGate.apply(sentence, [sentence], ["注意事项"], mode="redact")
    assert stats["redacted"] == 1 and stats["kept_risk"] == 0
    assert "1片" not in answer


def test_chunk_cap_that_fits_preserves_full_history():
    fit = PromptBudgetFitter(PromptBudget(total=2000, evidence=1500, chunk_cap=100)).fit(
        system_text="policy", query="question", evidence=[EvidenceItem("说明" * 100)],
        history_blocks=["完整三轮历史", "缩短两轮历史", ""],
    )
    assert fit.truncated and not fit.exceeded
    assert fit.steps == ("chunk-cap",)
    assert "完整三轮历史" in fit.messages[-1]["content"]
    assert "缩短两轮历史" not in fit.messages[-1]["content"]


@pytest.mark.parametrize("dose", ["半片", "一天两次", "50mcg", "二分之一片", "早晚各一粒"])
def test_prompt_cuts_do_not_split_colloquial_dose_spans(dose):
    text = "前" + dose + "后"
    for cut in range(2, 1 + len(dose)):
        assert _safe_cut(text, cut) <= 1


def test_history_and_ocr_do_not_appear_under_medical_evidence_header():
    messages = build_messages(
        system_text="policy", query="question", evidence_texts=["approved label"],
        history_block="old query", ocr_text="box text", fence=_wrap_untrusted,
    )
    user = messages[-1]["content"]
    evidence_start = user.index(EVIDENCE_HEADER)
    assert user.index("old query") < evidence_start
    assert user.index("box text") < evidence_start
    assert user.index("approved label") > evidence_start
    assert _wrap_untrusted("box text") in user
    assert messages[0] == {"role": "system", "content": "policy"}


def test_drug_matching_cannot_overwrite_this_requests_ocr_status(monkeypatch):
    tool = OCRTool()
    monkeypatch.setattr(tool, "extract_text_with_status", lambda path: (["label"], "ok"))

    def match(lines, dictionary):
        tool.last_status = "inference_error"
        return "matched drug"

    monkeypatch.setattr(tool, "match_drug_name", match)
    assert tool.suggest_drug_from_image("unused", ["matched drug"]) == (["label"], "matched drug", "ok")
