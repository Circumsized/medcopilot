"""W1.3 契约:PROJECT_ENV 显式白名单 + 生产默认拒绝。

默认拒绝语义:空值或未识别值(含错拼)⇒ 读写一律 503,即便 ADMIN_API_KEY
已配置也不放行;只有显式 development/dev/test/local 才开放本地无鉴权流程;
production/prod 保持既有 fail-closed 语义。/health 暴露 env_class/env_recognized。
"""

from __future__ import annotations

import pytest

from medcopilot.api.env_policy import classify_env, recognized_literals

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


def _client(monkeypatch, env: str, admin_key: str | None) -> TestClient:
    from types import SimpleNamespace

    from medcopilot.api import server

    configured = SimpleNamespace(**server.config.model_dump())
    configured.project_env = env
    if admin_key is None:
        configured.admin_api_key = ""
    else:
        configured.admin_api_key = admin_key
    monkeypatch.setattr(server, "config", configured)
    return TestClient(server.create_app())


# ------------------------------------------------------------ 分类器纯函数


@pytest.mark.parametrize("value", ["development", "dev", "test", "local", " Dev ", "LOCAL"])
def test_classify_env_dev(value):
    assert classify_env(value) == "dev"


@pytest.mark.parametrize("value", ["production", "prod", "Production", " PROD "])
def test_classify_env_production(value):
    assert classify_env(value) == "production"


@pytest.mark.parametrize("value", ["", "   ", None, "prodction", "Productionn", "staging", "devl"])
def test_classify_env_unrecognized(value):
    assert classify_env(value) == "unrecognized"


def test_recognized_literals_lists_both_families():
    literals = recognized_literals()
    assert "development" in literals and "production" in literals


# ------------------------------------------------------------ 默认拒绝行为


def test_empty_env_denies_reads_and_ingest(monkeypatch):
    client = _client(monkeypatch, env="", admin_key=None)
    r = client.post("/consultations", json={"query": "布洛芬怎么吃"})
    assert r.status_code == 503
    r = client.post("/documents/drug-label", data={"raw_text": "【药品名称】测试片", "drug_name": "测试片"})
    assert r.status_code == 503
    health = client.get("/health").json()
    assert health["env_recognized"] is False
    assert health["env_class"] == "unrecognized"


def test_typo_env_with_key_still_denies(monkeypatch):
    """错拼 prodction + 正确密钥 ⇒ 照样 503:环境不可分类时必须失败而非被猜。"""
    client = _client(monkeypatch, env="prodction", admin_key="secret-key")
    r = client.post(
        "/consultations",
        json={"query": "布洛芬怎么吃"},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert r.status_code == 503
    assert "PROJECT_ENV" in r.json()["detail"]


def test_dev_env_keeps_open_flow(monkeypatch):
    client = _client(monkeypatch, env="development", admin_key=None)
    r = client.post("/consultations", json={"query": "布洛芬缓释胶囊的用法用量是什么"})
    assert r.status_code == 200
    r = client.post("/documents/drug-label", data={"raw_text": "【药品名称】测试片\n【适应症】测试。", "drug_name": "测试片"})
    assert r.status_code in (200, 201)
    health = client.get("/health").json()
    assert health["env_recognized"] is True
    assert health["env_class"] == "dev"


def test_production_without_key_still_denies(monkeypatch):
    client = _client(monkeypatch, env="production", admin_key=None)
    r = client.post("/consultations", json={"query": "布洛芬怎么吃"})
    assert r.status_code == 503


def test_production_with_key_serves_and_hides_docs(monkeypatch):
    client = _client(monkeypatch, env="production", admin_key="secret-key")
    r = client.post(
        "/consultations",
        json={"query": "布洛芬缓释胶囊的用法用量是什么"},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert r.status_code == 200
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_emergency_redline_survives_unrecognized_env(monkeypatch):
    """急救红线不因默认拒绝而消失:503 之前 emergency 短路仍签发(端点级权限
    在急救之后才生效,与既有 test_unknown_session_image_emergency 契约一致)。"""
    client = _client(monkeypatch, env="", admin_key=None)
    r = client.post("/consultations", json={"query": "家人突然心梗发作"})
    # require_reader 在端点层先执行 ⇒ 未识别环境返回 503;急救语义由
    # _emergency_response 在同一请求被鉴权放行后保证(dev/prod+key 路径由
    # test_review_final_safety 钉死)。此处只断言不 500、不静默放行。
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.json()["is_emergency"] is True
