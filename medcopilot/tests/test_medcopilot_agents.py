import pytest

from medcopilot.agents.ocr_tool import OCRTool
from medcopilot.agents.triage import TriageAgent

# ------------------------------- TriageAgent -------------------------------

def test_triage_routes_emergency_first():
    assert TriageAgent.classify("老人突发心肌梗死吐血了") == TriageAgent.EMERGENCY


def test_triage_routes_non_medical():
    assert TriageAgent.classify("帮我写一段python代码算斐波那契") == TriageAgent.NON_MEDICAL
    assert TriageAgent.classify("今天天气怎么样") == TriageAgent.NON_MEDICAL


def test_triage_routes_medical():
    assert TriageAgent.classify("头痛可以吃布洛芬吗") == TriageAgent.MEDICAL
    assert TriageAgent.classify("轻微感冒流涕如何处理？") == TriageAgent.MEDICAL


def test_triage_medical_hint_beats_non_medical_keyword():
    # 含"代码"但同时也是健康咨询，必须按医疗问题处理。
    assert TriageAgent.classify("吃头孢期间喝酒出问题怎么办") == TriageAgent.MEDICAL


# -------------------------------- OCRTool ----------------------------------

def test_ocr_match_prefers_exact_substring():
    lines = ["阿莫西林胶囊", "国药准字H12345678", "0.25g×24粒"]
    matched = OCRTool.match_drug_name(lines, ["阿莫西林胶囊", "布洛芬缓释胶囊"])
    assert matched == "阿莫西林胶囊"


def test_ocr_match_repairs_typical_ocr_noise():
    lines = ["布洛芬缓释胶囊", "用于缓解轻至中度疼痛"]
    matched = OCRTool.match_drug_name(lines, ["对乙酰氨基酚片", "布洛芬缓释胶囊"])
    assert matched == "布洛芬缓释胶囊"


def test_ocr_match_returns_none_below_cutoff():
    assert OCRTool.match_drug_name(["天气晴朗"], ["阿莫西林胶囊"]) is None
    assert OCRTool.match_drug_name([], ["阿莫西林胶囊"]) is None


def test_ocr_extract_degrades_without_engine_or_image():
    tool = OCRTool()
    assert tool.extract_text(None) == []
    assert tool.extract_text("Z:/definitely/not/exist.jpg") == []


# ------------------------------ FastAPI server ------------------------------

def test_api_health_and_emergency_guard():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    health = client.get("/health").json()
    assert health["status"] == "ok"

    resp = client.post("/consultations", json={"query": "突发心肌梗死吐血了"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_emergency"] is True
    assert "120" in body["answer"]


def test_api_ingest_and_consult():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    ingest = client.post(
        "/documents/drug-label",
        data={
            "raw_text": "【药品名称】阿莫西林胶囊\n【禁忌】青霉素过敏患者禁用。",
            "drug_name": "阿莫西林胶囊",
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_added"] >= 2

    resp = client.post("/consultations", json={"query": "青霉素过敏能吃阿莫西林吗"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "medical"
    assert "免责合规提示" in body["answer"]


def test_api_rejects_blank_query():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    resp = client.post("/consultations", json={"query": "   "})
    assert resp.status_code == 422
