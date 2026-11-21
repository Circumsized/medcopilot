"""W1.4 契约:每条响应与会话记录都携带运行时门控策略版本(policy_version)。

事实源是代码常量 ``MedicalSafetyGuard.POLICY_VERSION``;策略工件镜像它,
漂移由 ``safety.policy.check_code_equivalence`` 门禁捕获。工件被篡改时
API 仍报代码常量版本(fail-safe:版本声明不依赖工件可用性)。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.workflow.agent_graph import AgentGraph, Session
from medcopilot.workflow.safety_guard import MedicalSafetyGuard


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


def test_agent_graph_run_carries_policy_version():
    graph = AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(
                "【药品名称】阿莫西林胶囊\n【禁忌】青霉素过敏患者禁用。",
                drug_name="阿莫西林胶囊",
            )
        )
    )
    result = graph.run("阿莫西林胶囊的禁忌是什么")
    assert result["policy_version"] == MedicalSafetyGuard.POLICY_VERSION

    emergency = graph.run("心肌梗死怎么办")
    assert emergency["policy_version"] == MedicalSafetyGuard.POLICY_VERSION

    session = Session()
    graph.run("阿莫西林胶囊的用法用量是什么", session=session)
    assert session.turns[-1]["policy_version"] == str(MedicalSafetyGuard.POLICY_VERSION)
    graph.run("心梗了", session=session)
    assert session.turns[-1]["policy_version"] == str(MedicalSafetyGuard.POLICY_VERSION)


def test_api_responses_carry_policy_version():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())

    # 弃答路径(无充分证据)
    r = client.post("/consultations", json={"query": "布洛芬缓释胶囊的用法用量是什么"})
    assert r.status_code == 200
    assert r.json()["policy_version"] == MedicalSafetyGuard.POLICY_VERSION

    # 急救短路路径
    r = client.post("/consultations", json={"query": "家人突然心梗发作"})
    assert r.status_code == 200
    assert r.json()["is_emergency"] is True
    assert r.json()["policy_version"] == MedicalSafetyGuard.POLICY_VERSION

    # 非医疗拒答路径
    r = client.post("/consultations", json={"query": "今天天气怎么样"})
    assert r.status_code == 200
    assert r.json()["intent"] == "non_medical"
    assert r.json()["policy_version"] == MedicalSafetyGuard.POLICY_VERSION


def test_artifact_version_drift_detected_but_code_constant_wins(tmp_path):
    """工件版本被篡改 → 等价守护捕获漂移;API 版本声明仍取代码常量。"""
    from medcopilot.safety.policy import canonical_sha256, check_code_equivalence, load_policy

    repo_policy = Path(__file__).resolve().parents[2] / "medcopilot" / "datasets" / "policies" / "safety_policy_v1.json"
    tampered = copy.deepcopy(json.loads(repo_policy.read_text(encoding="utf-8")))
    tampered["policy_version"] = 999
    tampered["content"]["policy_version"] = 999
    # content 变了 → 重算指纹并换一条覆盖评审,单独暴露"版本漂移"检测
    tampered["content_sha256"] = canonical_sha256(tampered["content"])
    tampered["reviews"] = [dict(tampered["reviews"][-1], covers_sha256=tampered["content_sha256"])]
    path = tmp_path / "tampered_policy.json"
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    policy = load_policy(path)  # 指纹与评审门通过
    drift = check_code_equivalence(policy)
    assert drift, "工件 policy_version 偏离代码常量必须被等价守护捕获"
    assert MedicalSafetyGuard.POLICY_VERSION != 999
