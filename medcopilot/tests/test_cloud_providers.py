from types import SimpleNamespace

import httpx
import pytest

from medcopilot.agents.ocr_api import ApiOCRTool
from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.reranker import Reranker
from medcopilot.rag.reranker_api import ApiReranker, build_api_reranker


class JsonTransport(httpx.BaseTransport):
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if isinstance(self.payload, (bytes, bytearray)):
            return httpx.Response(self.status_code, content=self.payload, request=request)
        return httpx.Response(self.status_code, json=self.payload, request=request)


def test_cloud_reranker_validates_and_reorders_without_network():
    # Upstream returns ascending order; adapter must sort descending by score.
    transport = JsonTransport({"results": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.9}]})
    docs = [Document("a", {}), Document("b", {})]
    result = ApiReranker(api_base="https://rank.test", api_key="secret", transport=transport, sleep=lambda _: None).rerank("q", docs, 2)
    assert [doc.page_content for doc, _ in result] == ["b", "a"]
    assert b"secret" not in transport.requests[0].content


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"index": 9, "score": 1.0}]},
        {"results": [{"index": 0, "score": True}, {"index": 1, "score": 0.5}]},
        b'{"results": [{"index": 0, "score": NaN}, {"index": 1, "score": 0.5}]}',
        {"results": [{"index": 0, "score": 0.5}]},  # partial: 1 result for top_n=2 over 2 docs
    ],
)
def test_cloud_reranker_rejects_bad_or_partial_schema(payload):
    transport = JsonTransport(payload)
    docs = [Document("a", {}), Document("b", {})]
    with pytest.raises(RuntimeError):
        ApiReranker(api_base="https://rank.test", api_key="secret", transport=transport).rerank("q", docs, 2)


def test_reranker_cloud_mode_and_fallback_never_loads_local_cross_encoder(monkeypatch):
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "rerank_provider", "cloud")
    monkeypatch.setattr(config, "allow_text_egress", True)
    monkeypatch.setattr(config, "rerank_api_base", "https://rank.test")
    monkeypatch.setattr(config, "rerank_api_key", "secret")

    reranker = Reranker(enabled=True)
    assert reranker.mode == "cloud"
    assert reranker.weights_state == "cloud"

    load_calls = []
    monkeypatch.setattr(reranker, "_load_model", lambda: load_calls.append(1))
    reranker._api = ApiReranker(
        api_base="https://rank.test",
        api_key="secret",
        transport=JsonTransport({"error": "boom"}, status_code=500),
        attempts=1,
    )
    docs = [Document("消化道溃疡禁用", {}), Document("一次1片", {})]
    ranked = reranker.rerank("消化道溃疡", docs, top_n=2)
    assert len(ranked) == 2
    assert reranker.mode == "lexical"
    assert load_calls == []


def test_cloud_ocr_uses_openai_vision_shape(tmp_path, monkeypatch):
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "ocr_provider", "cloud")
    monkeypatch.setattr(config, "allow_image_egress", True)
    image = tmp_path / "box.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    transport = JsonTransport({"choices": [{"message": {"content": "阿莫西林胶囊\n500mg"}}]})
    tool = ApiOCRTool(api_base="https://ocr.test/v1", api_key="secret", model="vision", transport=transport)
    lines, matched, status = tool.suggest_drug_from_image(str(image), ["阿莫西林胶囊"])
    assert status == "ok"
    assert matched == "阿莫西林胶囊"
    assert lines[0] == "阿莫西林胶囊"
    assert b"secret" not in transport.requests[0].content


@pytest.mark.parametrize(
    "bad_body",
    [
        {"choices": []},
        {"choices": None},
        {"choices": {}},
        {"choices": "abc"},
        {"choices": [{"message": None}]},
    ],
)
def test_cloud_ocr_malformed_choices_returns_inference_error(tmp_path, monkeypatch, bad_body):
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "ocr_provider", "cloud")
    monkeypatch.setattr(config, "allow_image_egress", True)
    image = tmp_path / "box.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    tool = ApiOCRTool(api_base="https://ocr.test/v1", api_key="secret", model="vision", transport=JsonTransport(bad_body))
    lines, matched, status = tool.suggest_drug_from_image(str(image), ["阿莫西林胶囊"])
    assert lines == [] and matched is None and status == "inference_error"


def test_llm_local_mode_blocks_remote_bundle_when_enforced(monkeypatch):
    from medcopilot.common.config import config
    from medcopilot.llm.unified_llm import UnifiedLLM

    monkeypatch.setattr(config, "llm_mode", "local")
    monkeypatch.setattr(config, "llm_provider", "qianfan")
    llm = UnifiedLLM(provider="qianfan", api_key="real-key", enforce_mode=True)
    with pytest.raises(RuntimeError, match="LLM_MODE=local"):
        llm._bundle("头痛", stream=False)


def test_cloud_ocr_and_text_providers_require_explicit_egress(monkeypatch):
    from medcopilot.api import server
    from medcopilot.common.config import config

    configured = SimpleNamespace(**server.config.model_dump())
    configured.project_env = "production"
    configured.ocr_provider = "cloud"
    configured.allow_image_egress = False
    monkeypatch.setattr(server, "config", configured)
    with pytest.raises(RuntimeError, match="ALLOW_IMAGE_EGRESS"):
        server.create_app()

    monkeypatch.setattr(config, "rerank_provider", "cloud")
    monkeypatch.setattr(config, "allow_text_egress", False)
    with pytest.raises(RuntimeError, match="ALLOW_TEXT_EGRESS"):
        build_api_reranker()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "ocr_provider": "cloud",
                "allow_image_egress": True,
                "ocr_api_base": "http://ocr.test/v1",
                "ocr_api_key": "k",
                "ocr_api_model": "m",
            },
            "HTTPS OCR_API_BASE",
        ),
        (
            {
                "embedding_provider": "cloud",
                "allow_text_egress": True,
                "embedding_api_base": "http://emb.test/v1",
                "embedding_api_key": "k",
            },
            "HTTPS endpoint",
        ),
        (
            {
                "rerank_provider": "cloud",
                "allow_text_egress": True,
                "rerank_api_base": "http://rank.test/v1",
                "rerank_api_key": "k",
            },
            "HTTPS endpoint",
        ),
        (
            {
                "llm_mode": "cloud",
                "allow_text_egress": True,
                "llm_api_base": "http://llm.test/v1",
            },
            "HTTPS LLM_API_BASE",
        ),
        (
            {
                "llm_mode": "cloud",
                "allow_text_egress": False,
                "llm_api_base": "https://llm.test/v1",
            },
            "ALLOW_TEXT_EGRESS",
        ),
    ],
)
def test_production_cloud_providers_require_https(monkeypatch, overrides, match):
    from medcopilot.api import server

    configured = SimpleNamespace(**server.config.model_dump())
    configured.project_env = "production"
    for key, value in overrides.items():
        setattr(configured, key, value)
    monkeypatch.setattr(server, "config", configured)
    with pytest.raises(RuntimeError, match=match):
        server.create_app()


def test_llm_is_local_url_uses_strict_hostname_allowlist():
    from medcopilot.llm.unified_llm import UnifiedLLM

    for valid_local in ("http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "http://ollama:11434/v1"):
        assert UnifiedLLM._is_local_url(valid_local) is True
    for bypass in (
        "https://ollama.example.com/v1",
        "http://remote.host/ollama/v1",
        "https://api.example.com/v1?from=localhost",
        r"http://localhost\@evil.com/v1",
        "http://user:pass@localhost:11434/v1",
        "ftp://localhost:11434/v1",
        "http://[invalid-ipv6/v1",
    ):
        assert UnifiedLLM._is_local_url(bypass) is False


def test_build_ocr_tool_raises_when_cloud_without_image_egress(monkeypatch):
    from medcopilot.agents.ocr_api import build_ocr_tool
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "ocr_provider", "cloud")
    monkeypatch.setattr(config, "allow_image_egress", False)
    with pytest.raises(RuntimeError, match="ALLOW_IMAGE_EGRESS"):
        build_ocr_tool()


