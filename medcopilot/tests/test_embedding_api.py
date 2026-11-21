"""L0 契约层:API embedding 适配器(全离线,MockTransport,零真实调用)。

请求形态与批量上限(32)来自 SiliconFlow Create Embeddings 官方文档。
"""

from __future__ import annotations

import json

import httpx
import pytest

from medcopilot.rag.embedding_api import ApiEmbeddingFunction, build_api_encoder


def _handler(calls, *, responder):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append({
            "method": request.method,
            "url": str(request.url),
            "auth": request.headers.get("Authorization"),
            "content_type": request.headers.get("Content-Type"),
            "body": body,
        })
        return responder(request, body)

    return handle


def _make(monkeypatch, responder, **kwargs):
    calls: list[dict] = []
    transport = httpx.MockTransport(_handler(calls, responder=responder))
    monkeypatch.setattr(
        "medcopilot.common.config.config",
        kwargs.pop("config_obj", None) or __import__(
            "medcopilot.common.config", fromlist=["config"]
        ).config,
    )
    fn = ApiEmbeddingFunction(
        api_base="https://api.example.test/v1",
        api_key="sk-test-secret-DO-NOT-LEAK",
        model="BAAI/bge-m3",
        transport=transport,
        sleep=lambda seconds: None,
        **kwargs,
    )
    return fn, calls


def _ok_responder(request, body):
    inputs = body["input"]
    data = [
        {"object": "embedding", "index": position, "embedding": [float(len(text)), 1.0, 0.5]}
        for position, text in enumerate(inputs)
    ]
    return httpx.Response(200, json={"object": "list", "model": body["model"], "data": data})


# ------------------------------------------------------------------ L0-1


def test_request_shape_bearer_auth_and_payload(monkeypatch):
    fn, calls = _make(monkeypatch, _ok_responder)
    fn(["布洛芬禁忌"])
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api.example.test/v1/embeddings"
    assert call["auth"] == "Bearer sk-test-secret-DO-NOT-LEAK"
    assert call["content_type"] == "application/json"
    assert call["body"] == {"model": "BAAI/bge-m3", "input": ["布洛芬禁忌"]}


def test_string_input_returns_bare_vector(monkeypatch):
    fn, _ = _make(monkeypatch, _ok_responder)
    vector = fn("布洛芬禁忌")
    assert vector == [5.0, 1.0, 0.5]


# ------------------------------------------------------------------ L0-2


def test_batches_of_at_most_32_and_concatenates_in_order(monkeypatch):
    fn, calls = _make(monkeypatch, _ok_responder)
    texts = [f"文本{index:03d}" for index in range(40)]
    vectors = fn(texts)
    assert len(calls) == 2, "40 条输入必须按 32 上限分成 2 批"
    assert len(calls[0]["body"]["input"]) == 32
    assert len(calls[1]["body"]["input"]) == 8
    assert len(vectors) == 40
    # 每个向量的首元素是“该文本的长度”:验证跨批拼接顺序与输入一一对应。
    for text, vector in zip(texts, vectors):
        assert vector[0] == float(len(text))


# ------------------------------------------------------------------ L0-3


def _shuffled_responder(request, body):
    inputs = body["input"]
    data = [
        {"object": "embedding", "index": position, "embedding": [float(position), 1.0]}
        for position, _ in enumerate(inputs)
    ]
    data.reverse()  # 服务端乱序返回;顺序只能靠 index 对齐
    return httpx.Response(200, json={"object": "list", "model": body["model"], "data": data})


def test_response_reordered_by_index_field(monkeypatch):
    fn, _ = _make(monkeypatch, _shuffled_responder)
    vectors = fn(["一", "二", "三"])
    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]


# ------------------------------------------------------------------ L0-4


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[]], "empty"),
        ([[1.0, "oops"]], "non-numeric"),
        ([[1.0], [1.0, 2.0]], "inconsistent"),
    ],
)
def test_bad_vectors_rejected_and_dim_not_cached(monkeypatch, vectors, message):
    def responder(request, body):
        data = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]
        return httpx.Response(200, json={"data": data})

    fn, _ = _make(monkeypatch, responder)
    with pytest.raises(ValueError, match=message):
        fn([f"t{index}" for index in range(len(vectors))])
    assert fn.embedding_dim is None
    assert fn.dim_source == "unmeasured"


def test_validator_rejects_non_finite_values_directly():
    # JSON 无法携带 NaN/Inf(标准里没有),所以非有限值只能来自直连调用方;
    # 校验器单独钉死,作为纵深防御。
    from medcopilot.rag.embedding_api import _validate_vectors

    with pytest.raises(ValueError, match="finite"):
        _validate_vectors([[1.0, float("nan")]], expected=1)
    with pytest.raises(ValueError, match="finite"):
        _validate_vectors([[float("inf")]], expected=1)


def test_wrong_response_count_is_rejected(monkeypatch):
    def responder(request, body):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    fn, _ = _make(monkeypatch, responder)
    with pytest.raises(ValueError, match="count"):
        fn(["a", "b"])


# ------------------------------------------------------------------ L0-5


def _status_responder(status):
    def responder(request, body):
        return httpx.Response(status, json={"error": "boom"})

    return responder


def test_401_is_not_retried_and_names_the_credential(monkeypatch):
    fn, calls = _make(monkeypatch, _status_responder(401))
    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        fn(["文本"])
    assert len(calls) == 1


def test_400_is_not_retried(monkeypatch):
    fn, calls = _make(monkeypatch, _status_responder(400))
    with pytest.raises(RuntimeError, match="400"):
        fn(["文本"])
    assert len(calls) == 1


def test_500_retries_are_bounded_then_fail_loud(monkeypatch):
    fn, calls = _make(monkeypatch, _status_responder(500), attempts=3)
    with pytest.raises(RuntimeError, match="unavailable"):
        fn(["文本"])
    assert len(calls) == 3, "重试必须有界"


def test_429_retries_then_recovers(monkeypatch):
    calls: list[dict] = []

    def responder(request, body):
        calls.append(1)
        if len(calls) < 2:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
        )

    fn, _ = _make(monkeypatch, responder, attempts=3)
    assert fn(["文本"]) == [[1.0, 2.0]]
    assert len(calls) == 2


# ------------------------------------------------------------------ L0-6


def test_secrets_never_appear_in_errors_or_status(monkeypatch):
    secret = "sk-test-secret-DO-NOT-LEAK"

    def responder(request, body):
        # 恶意/故障服务端把密钥回显在错误体里:适配器也不得把它透传出去。
        return httpx.Response(500, json={"error": f"echo {secret}"})

    fn, _ = _make(monkeypatch, responder, attempts=1)
    with pytest.raises(RuntimeError) as exc_info:
        fn(["文本"])
    assert secret not in str(exc_info.value)
    assert secret not in json.dumps(fn.status(), ensure_ascii=False)


# ------------------------------------------------------------------ L0-7


def test_dim_is_measured_and_status_reports_configuration(monkeypatch):
    fn, _ = _make(monkeypatch, _ok_responder)
    assert fn.embedding_dim is None and fn.dim_source == "unmeasured"
    fn(["布洛芬"])
    assert fn.embedding_dim == 3
    assert fn.dim_source == "measured"
    status = fn.status()
    assert status["model"] == "BAAI/bge-m3"
    assert status["endpoint"] == "https://api.example.test/v1/embeddings"
    assert status["embedding_dim"] == 3
    assert status["dim_source"] == "measured"
    assert "sk-test" not in json.dumps(status)


def test_name_is_stable_for_future_manifest_identity():
    assert ApiEmbeddingFunction.name() == "api-embeddings"


# ------------------------------------------------------------------ 配置装配


def test_build_api_encoder_requires_base_and_key(monkeypatch):
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "embedding_api_base", "")
    monkeypatch.setattr(config, "embedding_api_key", "some-key")
    assert build_api_encoder() is None
    monkeypatch.setattr(config, "embedding_api_base", "https://api.example.test/v1")
    monkeypatch.setattr(config, "embedding_api_key", "")
    assert build_api_encoder() is None
    monkeypatch.setattr(config, "embedding_api_key", "some-key")
    encoder = build_api_encoder()
    assert encoder is not None and encoder.model_name == config.embedding_model
