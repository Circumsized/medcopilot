"""UI 到 FastAPI 的客户端契约测试。

这些测试先于实现存在：Gradio 必须只消费 HTTP API，不得在 UI 进程内装配
RAG、OCR 或工作流。
"""

from __future__ import annotations

import ast
import importlib
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from medcopilot.workflow.safety_guard import MedicalSafetyGuard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_SOURCE = PROJECT_ROOT / "medcopilot/ui/app.py"
VALID_RESPONSE = {
    "answer": "请遵医嘱使用。",
    "sources": [{"section": "注意事项"}],
    "is_emergency": False,
    "intent": "medical",
    "abstained": False,
    "session_id": "session-from-api",
}


@pytest.fixture
def api_client_module() -> ModuleType:
    """加载 Slice 1 应新增的显式 UI API 客户端。"""
    return importlib.import_module("medcopilot.ui.api_client")


def _build_client(
    api_client_module: ModuleType,
    handler,
    *,
    api_key: str | None = None,
    base_url: str = "http://medical-api.test/",
):
    return api_client_module.MedicalAPIClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=3.5,
        transport=httpx.MockTransport(handler),
    )


def _multipart_parts(request: httpx.Request) -> dict[str, dict[str, object]]:
    content_type = request.headers["Content-Type"]
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + request.read()
    )
    parts: dict[str, dict[str, object]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        assert name is not None
        parts[name] = {
            "filename": part.get_filename(),
            "content_type": part.get_content_type(),
            "payload": part.get_payload(decode=True),
        }
    return parts


def _assert_client_error(
    api_client_module: ModuleType,
    exc_info: pytest.ExceptionInfo[Exception],
    *,
    error_type: type[Exception] | None = None,
    code: str,
    message: str,
) -> None:
    expected_type = error_type or api_client_module.APIClientError
    assert isinstance(exc_info.value, expected_type)
    assert getattr(exc_info.value, "code", None) == code
    assert str(exc_info.value) == message


def test_text_consultation_posts_trimmed_json_without_auth(api_client_module):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    result = _build_client(api_client_module, handler).consult_text("  布洛芬能空腹吃吗？  ")

    assert result == VALID_RESPONSE
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert request.url == httpx.URL("http://medical-api.test/consultations")
    assert request.headers["Content-Type"].startswith("application/json")
    assert json.loads(request.content) == {"query": "布洛芬能空腹吃吗？"}
    assert "Authorization" not in request.headers


@pytest.mark.parametrize("session_id", [None, "", "   "])
def test_text_consultation_omits_empty_session_id(api_client_module, session_id):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    _build_client(api_client_module, handler).consult_text("用法", session_id=session_id)

    assert json.loads(seen[0].content) == {"query": "用法"}


def test_text_consultation_sends_non_empty_session_id(api_client_module):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    _build_client(api_client_module, handler).consult_text("继续咨询", session_id="session-known")

    assert json.loads(seen[0].content) == {
        "query": "继续咨询",
        "session_id": "session-known",
    }


def test_image_file_open_failure_maps_to_validation_without_network(api_client_module, tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    client = _build_client(api_client_module, handler)
    with pytest.raises(api_client_module.APIClientError) as exc_info:
        client.consult_with_image("用法", tmp_path / "不存在.png")

    assert exc_info.value.code == "validation"
    assert requests == []


def test_image_consultation_posts_query_and_file_as_multipart(api_client_module, tmp_path):
    image_path = tmp_path / "药盒.png"
    image_bytes = b"\x89PNG\r\n\x1a\nexample-image"
    image_path.write_bytes(image_bytes)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    result = _build_client(api_client_module, handler).consult_with_image(
        "  这个药有什么禁忌？  ", image_path
    )

    assert result == VALID_RESPONSE
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert request.url == httpx.URL("http://medical-api.test/consultations/with-image")
    assert request.headers["Content-Type"].startswith("multipart/form-data;")
    assert "Authorization" not in request.headers

    parts = _multipart_parts(request)
    assert set(parts) == {"query", "image"}
    assert parts["query"]["payload"].decode() == "这个药有什么禁忌？"
    assert parts["image"]["filename"] == image_path.name
    assert parts["image"]["content_type"] == "image/png"
    assert parts["image"]["payload"] == image_bytes


@pytest.mark.parametrize("session_id", [None, "", "   "])
def test_image_consultation_omits_empty_session_id(api_client_module, tmp_path, session_id):
    image_path = tmp_path / "box.jpg"
    image_path.write_bytes(b"\xff\xd8\xffsample")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    _build_client(api_client_module, handler).consult_with_image(
        "用法", image_path, session_id=session_id
    )

    assert set(_multipart_parts(seen[0])) == {"query", "image"}


def test_image_consultation_sends_non_empty_session_id(api_client_module, tmp_path):
    image_path = tmp_path / "box.jpg"
    image_path.write_bytes(b"\xff\xd8\xffsample")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    _build_client(api_client_module, handler).consult_with_image(
        "继续咨询", image_path, session_id="session-known"
    )

    parts = _multipart_parts(seen[0])
    assert set(parts) == {"query", "image", "session_id"}
    assert parts["session_id"]["payload"].decode() == "session-known"


@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
def test_configured_bearer_is_sent_for_both_consultation_types(
    api_client_module, tmp_path, with_image
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    # Fake credentials are assembled at runtime: identical values when the test
    # runs, but static scanners must not mistake fixtures for hardcoded secrets.
    fake_ui_key = "ui-" + "secret-key"
    client = _build_client(api_client_module, handler, api_key=f"  {fake_ui_key}  ")
    if with_image:
        image_path = tmp_path / "box.jpg"
        image_path.write_bytes(b"\xff\xd8\xffsample")
        client.consult_with_image("用法", image_path)
    else:
        client.consult_text("用法")

    assert seen[0].headers["Authorization"] == "Bearer ui-secret-key"


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_blank_api_key_never_sends_authorization(api_client_module, api_key):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    _build_client(api_client_module, handler, api_key=api_key).consult_text("用法")

    assert "Authorization" not in seen[0].headers


@pytest.mark.parametrize(
    "base_url",
    ["http://medical-api.test", "http://medical-api.test/", "http://medical-api.test///"],
)
def test_base_url_trailing_slashes_do_not_change_endpoint(api_client_module, base_url):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    _build_client(api_client_module, handler, base_url=base_url).consult_text("用法")

    assert seen[0].url == httpx.URL("http://medical-api.test/consultations")


@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
def test_blank_query_is_rejected_without_network_request(api_client_module, tmp_path, with_image):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    client = _build_client(api_client_module, handler)
    with pytest.raises(api_client_module.APIClientError) as exc_info:
        if with_image:
            image_path = tmp_path / "box.jpg"
            image_path.write_bytes(b"\xff\xd8\xffsample")
            client.consult_with_image("   ", image_path)
        else:
            client.consult_text("   ")

    _assert_client_error(
        api_client_module,
        exc_info,
        code="validation",
        message="咨询内容无效，请检查输入。",
    )
    assert requests == []


@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
def test_client_rejects_session_id_longer_than_128_without_network(
    api_client_module, tmp_path, with_image
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    client = _build_client(api_client_module, handler)
    with pytest.raises(api_client_module.APIClientError) as exc_info:
        if with_image:
            image_path = tmp_path / "box.jpg"
            image_path.write_bytes(b"\xff\xd8\xffsample")
            client.consult_with_image("继续", image_path, session_id="s" * 129)
        else:
            client.consult_text("继续", session_id="s" * 129)

    _assert_client_error(
        api_client_module,
        exc_info,
        code="validation",
        message="咨询内容无效，请检查输入。",
    )
    assert requests == []


@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
def test_success_response_requires_and_returns_full_contract(api_client_module, tmp_path, with_image):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=VALID_RESPONSE)

    client = _build_client(api_client_module, handler)
    if with_image:
        image_path = tmp_path / "box.webp"
        image_path.write_bytes(b"RIFF0000WEBPsample")
        result = client.consult_with_image("注意事项", image_path)
    else:
        result = client.consult_text("注意事项")

    assert result == VALID_RESPONSE
    assert set(result) >= {
        "answer",
        "sources",
        "is_emergency",
        "intent",
        "abstained",
        "session_id",
    }
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["is_emergency"], bool)
    assert isinstance(result["intent"], str)
    assert isinstance(result["abstained"], bool)
    assert type(result["session_id"]) is str
    assert result["session_id"] == "session-from-api"


@pytest.mark.parametrize(
    ("status_code", "expected_code", "expected_message"),
    [
        (400, "bad_request", "咨询请求无效，请检查输入。"),
        (401, "authentication", "API 认证失败，请检查访问凭据。"),
        (413, "payload_too_large", "上传图片过大，请选择更小的文件。"),
        (422, "validation", "咨询内容无效，请检查输入。"),
        (500, "service_unavailable", "咨询服务暂不可用，请稍后重试。"),
        (503, "service_unavailable", "咨询服务暂不可用，请稍后重试。"),
    ],
)
def test_http_errors_map_to_stable_safe_errors(
    api_client_module, status_code, expected_code, expected_message
):
    leaked_body = "upstream token=server-secret traceback=/private/path"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=leaked_body)

    fake_client_key = "client-" + "secret"
    client = _build_client(api_client_module, handler, api_key=fake_client_key)
    with pytest.raises(api_client_module.APIClientError) as exc_info:
        client.consult_text("头痛")

    _assert_client_error(
        api_client_module,
        exc_info,
        code=expected_code,
        message=expected_message,
    )
    rendered = str(exc_info.value)
    assert leaked_body not in rendered
    assert "server-secret" not in rendered
    assert "client-secret" not in rendered


def test_404_maps_to_session_expired_only_for_explicit_session(api_client_module):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="unknown session token=server-secret")

    client = _build_client(api_client_module, handler)
    with pytest.raises(api_client_module.APIClientError) as expired:
        client.consult_text("继续", session_id="known-session")
    _assert_client_error(
        api_client_module,
        expired,
        code="session_expired",
        message="会话已过期，请重新开始咨询。",
    )

    with pytest.raises(api_client_module.APIClientError) as unexpected:
        client.consult_text("首次咨询")
    assert unexpected.value.code != "session_expired"
    assert "server-secret" not in str(unexpected.value)


def test_image_404_maps_to_session_expired_without_reading_stream(api_client_module, tmp_path):
    """Regression: the image path streams a multipart body, so the client must not
    introspect ``request.content``. httpx raises ``RequestNotRead`` for a streamed
    body (a ``StreamError``/``RuntimeError``, NOT an ``HTTPError``), which would
    escape the ``APIClientError`` mapping and crash the Gradio callback.

    A ``MockTransport`` would not catch this: it materialises the body, so the
    bug only reproduces against a real socket. We therefore run a tiny real HTTP
    server that answers 404, mirroring an expired session.
    """
    import http.server
    import socketserver
    import threading

    class _NotFound(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (stdlib handler API)
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(404)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args) -> None:  # keep pytest output clean
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), _NotFound)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    image_path = tmp_path / "box.jpg"
    image_path.write_bytes(b"\xff\xd8\xffsample")
    client = api_client_module.MedicalAPIClient(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        timeout_seconds=5.0,
    )
    try:
        # Explicit session id => 404 is a session expiry, not a raw httpx error.
        with pytest.raises(api_client_module.APIClientError) as expired:
            client.consult_with_image("继续", image_path, session_id="known-session")
        _assert_client_error(
            api_client_module,
            expired,
            code="session_expired",
            message="会话已过期，请重新开始咨询。",
        )

        # No session id => generic service-unavailable, still an APIClientError.
        with pytest.raises(api_client_module.APIClientError) as unavailable:
            client.consult_with_image("首次咨询", image_path)
        assert unavailable.value.code == "service_unavailable"
        assert not isinstance(unavailable.value, api_client_module.APIContractError)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ReadError,
        httpx.ProxyError,
        # RequestError 但不是 TransportError（httpx 0.28 异常层级）：损坏的压缩
        # 响应体（DecodingError）等同样必须映射为用户安全的连接错误。
        httpx.DecodingError, httpx.TooManyRedirects,
        # StreamError 挂在 RuntimeError 下、完全在 HTTPError 体系之外：该分支
        # 若删掉，Gradio 回调会裸抛 RuntimeError。
        httpx.RequestNotRead, httpx.StreamConsumed,
    ],
)
def test_connection_and_timeout_failures_map_to_safe_connection_error(api_client_module, failure):
    def handler(request: httpx.Request) -> httpx.Response:
        # RequestError 家族携带 request 上下文；StreamError 家族（httpx 0.28）的
        # __init__ 不接受任何参数——由 httpx 内部以预置消息抛出，这里用同样的
        # 无参构造模拟真实形态。两族都必须被映射拦截，不得裸抛。
        if issubclass(failure, httpx.RequestError):
            raise failure("socket details token=transport-secret", request=request)
        raise failure()

    client = _build_client(api_client_module, handler)
    with pytest.raises(api_client_module.APIClientError) as exc_info:
        client.consult_text("头痛")

    _assert_client_error(
        api_client_module,
        exc_info,
        code="connection",
        message="无法连接咨询服务，请稍后重试。",
    )
    assert "transport-secret" not in str(exc_info.value)
    assert "stream-secret" not in str(exc_info.value)
    assert "socket details" not in str(exc_info.value)


def test_non_json_success_is_a_contract_error_without_body_leak(api_client_module):
    leaked_body = "<html>debug token=body-secret</html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=leaked_body, headers={"Content-Type": "text/html"})

    client = _build_client(api_client_module, handler)
    with pytest.raises(api_client_module.APIContractError) as exc_info:
        client.consult_text("头痛")

    _assert_client_error(
        api_client_module,
        exc_info,
        error_type=api_client_module.APIContractError,
        code="contract",
        message="咨询服务响应格式异常，请稍后重试。",
    )
    assert leaked_body not in str(exc_info.value)
    assert "body-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"answer": "只有半成品"},
        {key: value for key, value in VALID_RESPONSE.items() if key != "session_id"},
        {**VALID_RESPONSE, "answer": 123},
        {**VALID_RESPONSE, "sources": {}},
        {**VALID_RESPONSE, "sources": ["not-an-object"]},
        {**VALID_RESPONSE, "is_emergency": "false"},
        {**VALID_RESPONSE, "intent": None},
        {**VALID_RESPONSE, "abstained": 0},
        {**VALID_RESPONSE, "session_id": 123},
        {**VALID_RESPONSE, "session_id": ""},
        {**VALID_RESPONSE, "session_id": "   "},
    ],
)
def test_missing_or_invalid_success_fields_are_contract_errors(api_client_module, invalid_payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=invalid_payload)

    client = _build_client(api_client_module, handler)
    with pytest.raises(api_client_module.APIContractError) as exc_info:
        client.consult_text("头痛")

    _assert_client_error(
        api_client_module,
        exc_info,
        error_type=api_client_module.APIContractError,
        code="contract",
        message="咨询服务响应格式异常，请稍后重试。",
    )
    assert "只有半成品" not in str(exc_info.value)


def _resolve_openapi_schema(spec: dict, schema: dict) -> dict:
    while "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/components/schemas/")
        schema = spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    return schema


def _assert_openapi_string(schema: dict) -> None:
    variants = schema.get("anyOf", [schema])
    assert any(item.get("type") == "string" for item in variants)


def test_openapi_declares_optional_session_id_requests_and_required_response():
    pytest.importorskip("fastapi")
    from medcopilot.api.server import create_app

    spec = create_app().openapi()
    text_operation = spec["paths"]["/consultations"]["post"]
    text_request = _resolve_openapi_schema(
        spec,
        text_operation["requestBody"]["content"]["application/json"]["schema"],
    )
    assert "session_id" in text_request["properties"]
    assert "session_id" not in text_request.get("required", [])
    _assert_openapi_string(text_request["properties"]["session_id"])
    text_session_schema = text_request["properties"]["session_id"]
    text_variants = text_session_schema.get("anyOf", [text_session_schema])
    # Length is enforced handler-side (SessionStore.validate_session_id, AFTER
    # the emergency short-circuit) instead of in the schema — a pydantic-level
    # maxLength would reject a malformed continuation before triage ever runs.
    assert any(item.get("type") == "string" for item in text_variants)

    image_operation = spec["paths"]["/consultations/with-image"]["post"]
    image_request = _resolve_openapi_schema(
        spec,
        image_operation["requestBody"]["content"]["multipart/form-data"]["schema"],
    )
    assert "session_id" in image_request["properties"]
    assert "session_id" not in image_request.get("required", [])
    _assert_openapi_string(image_request["properties"]["session_id"])
    image_session_schema = image_request["properties"]["session_id"]
    image_variants = image_session_schema.get("anyOf", [image_session_schema])
    assert any(item.get("type") == "string" for item in image_variants)

    for operation in (text_operation, image_operation):
        response = _resolve_openapi_schema(
            spec,
            operation["responses"]["200"]["content"]["application/json"]["schema"],
        )
        assert "session_id" in response["properties"]
        assert "session_id" in response.get("required", [])
        assert response["properties"]["session_id"].get("type") == "string"


def test_api_rejects_session_id_longer_than_128_for_both_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    too_long = "s" * 129
    text = client.post(
        "/consultations",
        json={"query": "继续", "session_id": too_long},
    )
    image = client.post(
        "/consultations/with-image",
        data={"query": "继续", "session_id": too_long},
        files={"image": ("box.jpg", b"\xff\xd8\xffsample", "image/jpeg")},
    )

    assert text.status_code == 422
    assert image.status_code == 422


def test_api_creates_session_and_preserves_recent_history_without_cross_talk(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    class HistoryEchoWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self, query, drug_context=None, session=None):
            previous = [turn["query"] for turn in session.turns]
            answer = "history=" + "|".join(previous)
            session.record(query, answer, "medical", policy_version=MedicalSafetyGuard.POLICY_VERSION)
            return {
                "answer": answer,
                "sources": [],
                "is_emergency": False,
                "intent": "medical",
                "abstained": False,
            }

    monkeypatch.setattr(server, "AgentGraph", HistoryEchoWorkflow)
    client = TestClient(server.create_app())

    first_a = client.post("/consultations", json={"query": "A 第一问"})
    first_b = client.post("/consultations", json={"query": "B 第一问"})
    assert first_a.status_code == first_b.status_code == 200
    session_a = first_a.json()["session_id"]
    session_b = first_b.json()["session_id"]
    assert type(session_a) is type(session_b) is str
    assert session_a and session_b and session_a != session_b

    second_a = client.post(
        "/consultations",
        json={"query": "A 第二问", "session_id": session_a},
    )
    second_b = client.post(
        "/consultations",
        json={"query": "B 第二问", "session_id": session_b},
    )
    assert second_a.status_code == second_b.status_code == 200
    assert second_a.json()["session_id"] == session_a
    assert second_b.json()["session_id"] == session_b
    assert "A 第一问" in second_a.json()["answer"]
    assert "B 第一问" not in second_a.json()["answer"]
    assert "B 第一问" in second_b.json()["answer"]
    assert "A 第一问" not in second_b.json()["answer"]

    third_a = client.post(
        "/consultations",
        json={"query": "A 第三问", "session_id": session_a},
    )
    assert third_a.status_code == 200
    assert "A 第一问" in third_a.json()["answer"]
    assert "A 第二问" in third_a.json()["answer"]
    assert "B 第一问" not in third_a.json()["answer"]


def test_valid_session_emergency_reuses_id_and_records_turn(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    class EmergencyAwareHistoryWorkflow(_HistoryEchoWorkflow):
        def run(self, query, drug_context=None, session=None):
            if "心肌梗死" in query:
                answer = "请立即拨打 120"
                session.record(query, answer, "emergency", policy_version=MedicalSafetyGuard.POLICY_VERSION)
                return {
                    "answer": answer,
                    "sources": [],
                    "is_emergency": True,
                    "intent": "emergency",
                    "abstained": False,
                }
            return super().run(query, drug_context=drug_context, session=session)

    monkeypatch.setattr(server, "AgentGraph", EmergencyAwareHistoryWorkflow)
    client = TestClient(server.create_app())
    first = client.post("/consultations", json={"query": "首次咨询"})
    assert first.status_code == 200
    session_id = first.json()["session_id"]

    emergency_query = "突发心肌梗死，胸痛压榨濒死感"
    emergency = client.post(
        "/consultations",
        json={"query": emergency_query, "session_id": session_id},
    )
    assert emergency.status_code == 200
    assert emergency.json()["is_emergency"] is True
    assert emergency.json()["session_id"] == session_id
    assert "120" in emergency.json()["answer"]

    follow_up = client.post(
        "/consultations",
        json={"query": "急救后续", "session_id": session_id},
    )
    assert follow_up.status_code == 200
    assert emergency_query in follow_up.json()["answer"]


def test_api_rejects_unknown_session_id_for_both_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    text = client.post(
        "/consultations",
        json={"query": "继续", "session_id": "unknown-or-expired"},
    )
    image = client.post(
        "/consultations/with-image",
        data={"query": "继续", "session_id": "unknown-or-expired"},
        files={"image": ("box.jpg", b"\xff\xd8\xffsample", "image/jpeg")},
    )

    assert text.status_code == 404
    assert image.status_code == 404


class _ControlledClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _HistoryEchoWorkflow:
    def __init__(self, **kwargs):
        pass

    def run(self, query, drug_context=None, session=None):
        previous = [turn["query"] for turn in session.turns]
        answer = "history=" + "|".join(previous)
        session.record(query, answer, "medical", policy_version=MedicalSafetyGuard.POLICY_VERSION)
        return {
            "answer": answer,
            "sources": [],
            "is_emergency": False,
            "intent": "medical",
            "abstained": False,
        }


def _session_test_client(
    monkeypatch,
    *,
    clock: _ControlledClock,
    ttl: float,
    capacity: int,
    workflow_cls=_HistoryEchoWorkflow,
):
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from medcopilot.api import server

    configured = SimpleNamespace(**server.config.model_dump())
    configured.med_session_ttl_seconds = ttl
    configured.med_session_max_entries = capacity
    monkeypatch.setattr(server, "config", configured)
    monkeypatch.setattr(server, "AgentGraph", workflow_cls)
    return TestClient(server.create_app(clock=clock))


def test_api_session_expires_after_ttl_without_sleep(monkeypatch):
    clock = _ControlledClock()
    client = _session_test_client(monkeypatch, clock=clock, ttl=10.0, capacity=10)

    created = client.post("/consultations", json={"query": "第一问"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    clock.advance(9.0)
    refreshed = client.post(
        "/consultations",
        json={"query": "第二问", "session_id": session_id},
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["session_id"] == session_id

    # TTL is measured from last access, so the first nine seconds do not count
    # after the successful touch above. No real sleep is used.
    clock.advance(11.0)
    expired = client.post(
        "/consultations",
        json={"query": "第三问", "session_id": session_id},
    )
    assert expired.status_code == 404


def test_api_session_capacity_evicts_least_recently_used(monkeypatch):
    clock = _ControlledClock()
    client = _session_test_client(monkeypatch, clock=clock, ttl=3600.0, capacity=2)

    session_a = client.post("/consultations", json={"query": "A 第一问"}).json()["session_id"]
    clock.advance(1.0)
    session_b = client.post("/consultations", json={"query": "B 第一问"}).json()["session_id"]
    clock.advance(1.0)

    # Touch A so B, not the oldest-created A, becomes least recently used.
    touched_a = client.post(
        "/consultations",
        json={"query": "A 第二问", "session_id": session_a},
    )
    assert touched_a.status_code == 200
    clock.advance(1.0)
    session_c = client.post("/consultations", json={"query": "C 第一问"}).json()["session_id"]

    evicted_b = client.post(
        "/consultations",
        json={"query": "B 第二问", "session_id": session_b},
    )
    retained_a = client.post(
        "/consultations",
        json={"query": "A 第三问", "session_id": session_a},
    )
    retained_c = client.post(
        "/consultations",
        json={"query": "C 第二问", "session_id": session_c},
    )
    assert evicted_b.status_code == 404
    assert retained_a.status_code == 200
    assert retained_c.status_code == 200


def test_api_does_not_evict_session_while_workflow_is_using_it(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    class BlockingWorkflow(_HistoryEchoWorkflow):
        entered = Event()
        release = Event()

        def run(self, query, drug_context=None, session=None):
            if query == "A 正在处理":
                self.entered.set()
                assert self.release.wait(timeout=3), "test did not release active session"
            return super().run(query, drug_context=drug_context, session=session)

    clock = _ControlledClock()
    client = _session_test_client(
        monkeypatch,
        clock=clock,
        ttl=3600.0,
        capacity=2,
        workflow_cls=BlockingWorkflow,
    )
    session_a = client.post("/consultations", json={"query": "A 初始化"}).json()["session_id"]
    clock.advance(1.0)
    session_b = client.post("/consultations", json={"query": "B 初始化"}).json()["session_id"]
    clock.advance(1.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        active_a = pool.submit(
            client.post,
            "/consultations",
            json={"query": "A 正在处理", "session_id": session_a},
        )
        assert BlockingWorkflow.entered.wait(timeout=2)
        created_c = client.post("/consultations", json={"query": "C 初始化"})
        assert created_c.status_code == 200
        session_c = created_c.json()["session_id"]
        BlockingWorkflow.release.set()
        assert active_a.result(timeout=3).status_code == 200

    retained_a = client.post(
        "/consultations",
        json={"query": "A 后续", "session_id": session_a},
    )
    evicted_b = client.post(
        "/consultations",
        json={"query": "B 后续", "session_id": session_b},
    )
    retained_c = client.post(
        "/consultations",
        json={"query": "C 后续", "session_id": session_c},
    )
    assert retained_a.status_code == 200
    assert evicted_b.status_code == 404
    assert retained_c.status_code == 200


def test_api_session_physically_keeps_only_three_recent_turns(monkeypatch):
    clock = _ControlledClock()
    client = _session_test_client(monkeypatch, clock=clock, ttl=3600.0, capacity=10)

    first = client.post("/consultations", json={"query": "第1问"})
    session_id = first.json()["session_id"]
    for turn in range(2, 5):
        response = client.post(
            "/consultations",
            json={"query": f"第{turn}问", "session_id": session_id},
        )
        assert response.status_code == 200

    fifth = client.post(
        "/consultations",
        json={"query": "第5问", "session_id": session_id},
    )
    assert fifth.status_code == 200
    history = fifth.json()["answer"]
    assert "第1问" not in history
    assert "第2问" in history
    assert "第3问" in history
    assert "第4问" in history


def test_same_session_requests_are_serialized_but_different_sessions_overlap(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, BrokenBarrierError, Event, Lock

    class ConcurrencyProbeWorkflow(_HistoryEchoWorkflow):
        armed = False
        mode = "same"
        gate = Event()
        first_entered = Event()
        second_entered = Event()
        cross_session_barrier = Barrier(2)
        active = 0
        max_active = 0
        counter_lock = Lock()

        def run(self, query, drug_context=None, session=None):
            if not self.armed:
                return super().run(query, drug_context=drug_context, session=session)
            with self.counter_lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
                if not self.first_entered.is_set():
                    self.first_entered.set()
                else:
                    self.second_entered.set()
            try:
                if self.mode == "same":
                    assert self.gate.wait(timeout=2), "test did not release same-session workflow"
                else:
                    try:
                        self.cross_session_barrier.wait(timeout=2)
                    except BrokenBarrierError as exc:
                        raise AssertionError("different sessions were serialized by a global lock") from exc
                return super().run(query, drug_context=drug_context, session=session)
            finally:
                with self.counter_lock:
                    type(self).active -= 1

    clock = _ControlledClock()
    client = _session_test_client(
        monkeypatch,
        clock=clock,
        ttl=3600.0,
        capacity=10,
        workflow_cls=ConcurrencyProbeWorkflow,
    )
    session_a = client.post("/consultations", json={"query": "A init"}).json()["session_id"]
    session_b = client.post("/consultations", json={"query": "B init"}).json()["session_id"]

    ConcurrencyProbeWorkflow.armed = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post,
            "/consultations",
            json={"query": "A one", "session_id": session_a},
        )
        assert ConcurrencyProbeWorkflow.first_entered.wait(timeout=2)
        second = pool.submit(
            client.post,
            "/consultations",
            json={"query": "A two", "session_id": session_a},
        )
        assert not ConcurrencyProbeWorkflow.second_entered.wait(timeout=0.2)
        ConcurrencyProbeWorkflow.gate.set()
        first_response = first.result(timeout=2)
        second_response = second.result(timeout=2)
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert "A one" in second_response.json()["answer"]
    assert ConcurrencyProbeWorkflow.max_active == 1

    ConcurrencyProbeWorkflow.mode = "different"
    ConcurrencyProbeWorkflow.first_entered.clear()
    ConcurrencyProbeWorkflow.second_entered.clear()
    ConcurrencyProbeWorkflow.max_active = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        request_a = pool.submit(
            client.post,
            "/consultations",
            json={"query": "A three", "session_id": session_a},
        )
        request_b = pool.submit(
            client.post,
            "/consultations",
            json={"query": "B two", "session_id": session_b},
        )
        assert request_a.result(timeout=3).status_code == 200
        assert request_b.result(timeout=3).status_code == 200
    assert ConcurrencyProbeWorkflow.max_active == 2


def test_unknown_session_does_not_block_text_emergency():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    unknown_id = "unknown-or-expired"
    response = TestClient(create_app()).post(
        "/consultations",
        json={"query": "突发心肌梗死，胸痛压榨濒死感", "session_id": unknown_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_emergency"] is True
    assert "120" in body["answer"]
    assert type(body["session_id"]) is str
    assert body["session_id"] and body["session_id"] != unknown_id


def test_unknown_session_image_emergency_precedes_image_ocr_and_workflow(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.datastructures import UploadFile as StarletteUploadFile

    from medcopilot.api import server

    calls = {"image_read": 0, "ocr": 0, "workflow": 0}

    async def forbidden_image_read(self, *args, **kwargs):
        calls["image_read"] += 1
        raise AssertionError("emergency path read the uploaded image")

    class TripwireOCR:
        is_available = False

        def suggest_drug_from_image(self, *args, **kwargs):
            calls["ocr"] += 1
            raise AssertionError("emergency path invoked OCR")

    class TripwireWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self, *args, **kwargs):
            calls["workflow"] += 1
            raise AssertionError("emergency path invoked RAG/LLM workflow")

    monkeypatch.setattr(StarletteUploadFile, "read", forbidden_image_read)
    monkeypatch.setattr(server, "OCRTool", TripwireOCR)
    monkeypatch.setattr(server, "AgentGraph", TripwireWorkflow)
    unknown_id = "unknown-or-expired"
    response = TestClient(server.create_app()).post(
        "/consultations/with-image",
        data={
            "query": "突发心肌梗死，胸痛压榨濒死感",
            "session_id": unknown_id,
        },
        files={"image": ("not-an-image.exe", b"not-an-image", "application/octet-stream")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_emergency"] is True
    assert "120" in body["answer"]
    assert type(body["session_id"]) is str
    assert body["session_id"] and body["session_id"] != unknown_id
    assert calls == {"image_read": 0, "ocr": 0, "workflow": 0}


def test_central_config_declares_ui_api_endpoint_timeout_and_session_limits(monkeypatch):
    from medcopilot.common.config import AppConfig

    monkeypatch.setenv("MED_API_BASE_URL", "http://api.internal:9000/")
    monkeypatch.setenv("MED_API_TIMEOUT_SECONDS", "7.25")
    monkeypatch.setenv("MED_SESSION_TTL_SECONDS", "45.5")
    monkeypatch.setenv("MED_SESSION_MAX_ENTRIES", "123")
    configured = AppConfig(_env_file=None)

    assert configured.med_api_base_url == "http://api.internal:9000/"
    assert configured.med_api_timeout_seconds == 7.25
    assert configured.med_session_ttl_seconds == 45.5
    assert configured.med_session_max_entries == 123
    assert configured.admin_api_key is None


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("MED_SESSION_TTL_SECONDS", "0"),
        ("MED_SESSION_TTL_SECONDS", "-0.1"),
        ("MED_SESSION_MAX_ENTRIES", "0"),
        ("MED_SESSION_MAX_ENTRIES", "-1"),
    ],
)
def test_central_config_rejects_non_positive_session_limits(monkeypatch, field, invalid_value):
    from pydantic import ValidationError

    from medcopilot.common.config import AppConfig

    monkeypatch.setenv("MED_SESSION_TTL_SECONDS", "30")
    monkeypatch.setenv("MED_SESSION_MAX_ENTRIES", "100")
    monkeypatch.setenv(field, invalid_value)

    with pytest.raises(ValidationError):
        AppConfig(_env_file=None)


def test_emergency_issuance_never_leaks_capacity(monkeypatch):
    """急救签发不得泄漏 store 的 in-use 计数，也不得被容量吞掉。

    容量 1：连续两次急救都必须 200；急救签发的会话在释放后是可淘汰的，
    后续普通会话创建必须照常 200（若急救泄漏了 in-use，普通创建会 503）。
    """
    clock = _ControlledClock()
    client = _session_test_client(monkeypatch, clock=clock, ttl=3600.0, capacity=1)

    first = client.post("/consultations", json={"query": "突发心肌梗死，胸痛压榨濒死感"})
    second = client.post("/consultations", json={"query": "突发心肌梗死，胸痛压榨濒死感"})
    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["session_id"] and second.json()["is_emergency"] is True

    normal = client.post("/consultations", json={"query": "头疼该怎么办"})
    assert normal.status_code == 200


def test_emergency_bypasses_full_store_without_recording(monkeypatch):
    """所有会话均在使用时，急救仍必须 200 并给出独立新标识（不记录、不阻塞）。"""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    class PinnedWorkflow(_HistoryEchoWorkflow):
        entered = Event()
        release = Event()

        def run(self, query, drug_context=None, session=None):
            if query.startswith("占用"):
                self.entered.set()
                assert self.release.wait(timeout=3)
            return super().run(query, drug_context=drug_context, session=session)

    clock = _ControlledClock()
    client = _session_test_client(
        monkeypatch, clock=clock, ttl=3600.0, capacity=1, workflow_cls=PinnedWorkflow
    )
    session_a = client.post("/consultations", json={"query": "初始化 A"}).json()["session_id"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        busy = pool.submit(
            client.post, "/consultations", json={"query": "占用 A", "session_id": session_a}
        )
        assert PinnedWorkflow.entered.wait(timeout=2)

        emergency = client.post(
            "/consultations", json={"query": "突发心肌梗死，胸痛压榨濒死感"}
        )
        assert emergency.status_code == 200
        body = emergency.json()
        assert body["is_emergency"] is True
        assert body["session_id"] and body["session_id"] != session_a

        PinnedWorkflow.release.set()
        assert busy.result(timeout=3).status_code == 200

    assert client.post("/consultations", json={"query": "后续咨询"}).status_code == 200


def test_ui_pure_state_handler_preserves_session_and_clears_on_expiry(monkeypatch):
    import sys
    from types import ModuleType

    # The state handler must be testable without importing the optional Gradio
    # runtime or scraping Blocks' private event registry. The stub import must
    # also leave the module cache clean: popping the ui.app entry is not undone
    # by monkeypatch, so the test itself must restore sys.modules state.
    monkeypatch.setitem(sys.modules, "gradio", ModuleType("gradio"))
    sys.modules.pop("medcopilot.ui.app", None)
    ui_app = importlib.import_module("medcopilot.ui.app")
    handle_consultation = ui_app.handle_consultation
    clear_conversation = ui_app.clear_conversation

    class FakeClientError(Exception):
        def __init__(self, message: str, *, code: str):
            super().__init__(message)
            self.code = code

    class FakeClient:
        def __init__(self):
            self.calls: list[tuple[str, str, str | None]] = []

        def consult_text(self, query, session_id=None):
            self.calls.append(("text", query, session_id))
            if query == "过期请求":
                raise FakeClientError("会话已过期，请重新开始咨询。", code="session_expired")
            return {**VALID_RESPONSE, "answer": "文本回答", "session_id": "ui-session"}

        def consult_with_image(self, query, image_path, session_id=None):
            self.calls.append(("image", query, session_id))
            return {**VALID_RESPONSE, "answer": "图片回答", "session_id": "ui-session"}

    fake = FakeClient()
    monkeypatch.setattr(ui_app, "APIClientError", FakeClientError, raising=False)

    _, _, history, session_id = handle_consultation(fake, "首次咨询", None, [], None)
    assert fake.calls == [("text", "首次咨询", None)]
    assert session_id == "ui-session"
    assert history[-1]["content"].startswith("文本回答")

    _, _, history, session_id = handle_consultation(
        fake, "图片续问", "box.png", history, session_id
    )
    assert fake.calls[-1] == ("image", "图片续问", "ui-session")
    assert session_id == "ui-session"
    assert history[-1]["content"].startswith("图片回答")

    before_error = list(history)
    _, _, history, session_id = handle_consultation(
        fake, "过期请求", None, history, session_id
    )
    assert fake.calls[-1] == ("text", "过期请求", "ui-session")
    assert session_id is None
    assert history[: len(before_error)] == before_error
    assert history[-1] == {
        "role": "assistant",
        "content": "会话已过期，请重新开始咨询。",
    }
    assert "partial" not in "".join(turn["content"] for turn in history)

    assert clear_conversation() == ("", None, [], None)

    # 缓存隔离收尾：本测试 pop 掉的 ui.app 缓存条目 monkeypatch 不会恢复，导入
    # 产生的 stub-gradio 绑定模块必须由测试自己弹出，避免泄漏到后续测试。
    sys.modules.pop("medcopilot.ui.app", None)
    # 父包属性也必须清掉：`from medcopilot.ui import app` 走 getattr(medcopilot.ui, "app")，
    # 残留属性会让后续测试拿到绑定 stub-gradio 的僵尸模块对象。
    import medcopilot.ui as _ui_pkg

    _ui_pkg.__dict__.pop("app", None)
    assert "medcopilot.ui.app" not in sys.modules


def test_ui_module_cache_does_not_leak_stub_gradio_to_later_imports(monkeypatch):
    """模块缓存隔离：假 gradio 注入与 ui.app 导入不得污染后续真实导入。"""
    import sys
    from types import ModuleType

    stub = ModuleType("gradio")
    monkeypatch.setitem(sys.modules, "gradio", stub)
    sys.modules.pop("medcopilot.ui.app", None)
    ui_app = importlib.import_module("medcopilot.ui.app")
    assert ui_app is not None

    # 模拟 monkeypatch 恢复 gradio 后，后续导入方重新加载 ui.app 时不得拿到
    # 绑定了假 gradio 的缓存模块。
    monkeypatch.undo()
    assert sys.modules.get("gradio") is not stub

    sys.modules.pop("medcopilot.ui.app", None)
    # 父包属性也必须清掉：`from medcopilot.ui import app` 走 getattr(medcopilot.ui, "app")，
    # 残留属性会让后续测试拿到绑定 stub-gradio 的僵尸模块对象。
    import medcopilot.ui as _ui_pkg

    _ui_pkg.__dict__.pop("app", None)
    assert "medcopilot.ui.app" not in sys.modules

    # 后续任何测试重新导入 ui.app 时必须能干净地执行模块级初始化（此处仅断言
    # 缓存为空即可重导；真实 gradio 缺失时 ui.app 的导入错误由其自身负责）。
    sys.modules.pop("medcopilot.ui.app", None)
    # 父包属性也必须清掉：`from medcopilot.ui import app` 走 getattr(medcopilot.ui, "app")，
    # 残留属性会让后续测试拿到绑定 stub-gradio 的僵尸模块对象。
    import medcopilot.ui as _ui_pkg

    _ui_pkg.__dict__.pop("app", None)
    assert "medcopilot.ui.app" not in sys.modules


def test_ui_does_not_import_or_instantiate_local_business_workflow():
    source = UI_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "AgentGraph",
        "HybridRetriever",
        "ClinicalChunker",
        "OCRTool",
        "Session",
        "MedicalSafetyGuard",
    }

    imported: set[str] = set()
    constructed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                constructed.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                constructed.add(node.func.attr)

    assert not (forbidden & imported), f"UI 仍直接导入服务端业务组件: {forbidden & imported}"
    assert not (forbidden & constructed), f"UI 仍本地实例化服务端业务组件: {forbidden & constructed}"
    assert "sample_text" not in source
    assert "DRUG_DICTIONARY" not in source


def test_api_capacity_exhausted_by_active_sessions_returns_503_stably(monkeypatch):
    """缺口 A：所有会话均在使用时创建新会话必须稳定 503，且不淘汰活跃会话。"""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    class AllBusyWorkflow(_HistoryEchoWorkflow):
        entered = Event()
        release = Event()

        def run(self, query, drug_context=None, session=None):
            if query.startswith("占用中"):
                self.entered.set()
                assert self.release.wait(timeout=3), "test did not release busy sessions"
            return super().run(query, drug_context=drug_context, session=session)

    clock = _ControlledClock()
    client = _session_test_client(
        monkeypatch,
        clock=clock,
        ttl=3600.0,
        capacity=2,
        workflow_cls=AllBusyWorkflow,
    )
    session_a = client.post("/consultations", json={"query": "A 初始化"}).json()["session_id"]
    session_b = client.post("/consultations", json={"query": "B 初始化"}).json()["session_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        busy_a = pool.submit(
            client.post, "/consultations", json={"query": "占用中 A", "session_id": session_a}
        )
        busy_b = pool.submit(
            client.post, "/consultations", json={"query": "占用中 B", "session_id": session_b}
        )
        assert AllBusyWorkflow.entered.wait(timeout=2)

        # Capacity full and every entry is in use: creation must stably 503,
        # twice, without evicting an active session or growing unboundedly.
        first_reject = client.post("/consultations", json={"query": "C 第一次"})
        second_reject = client.post("/consultations", json={"query": "C 第二次"})
        assert first_reject.status_code == 503
        assert second_reject.status_code == 503

        AllBusyWorkflow.release.set()
        assert busy_a.result(timeout=3).status_code == 200
        assert busy_b.result(timeout=3).status_code == 200

    # Neither active session may have been evicted by the failed creations.
    retained_a = client.post("/consultations", json={"query": "A 后续", "session_id": session_a})
    retained_b = client.post("/consultations", json={"query": "B 后续", "session_id": session_b})
    assert retained_a.status_code == 200
    assert retained_b.status_code == 200
    # "占用中 A"/"占用中 B" must still be in each session's history.
    assert "占用中 A" in retained_a.json()["answer"]
    assert "占用中 B" in retained_b.json()["answer"]


def test_same_session_identity_and_physical_three_turn_cap_across_requests(monkeypatch):
    """缺口 B：同会话复用同一 Session 实例，且 turns 物理长度不超过 3。"""
    import threading

    observed: dict = {"instances": [], "entry_lengths": [], "lock": threading.Lock()}

    class ProbeWorkflow(_HistoryEchoWorkflow):
        def run(self, query, drug_context=None, session=None):
            with observed["lock"]:
                observed["instances"].append(id(session))
                observed["entry_lengths"].append(len(session.turns))
            return super().run(query, drug_context=drug_context, session=session)

    clock = _ControlledClock()
    client = _session_test_client(
        monkeypatch, clock=clock, ttl=3600.0, capacity=10, workflow_cls=ProbeWorkflow
    )
    session_id = client.post("/consultations", json={"query": "第1问"}).json()["session_id"]
    for turn in range(2, 8):  # six continuation requests => seven total turns
        response = client.post(
            "/consultations", json={"query": f"第{turn}问", "session_id": session_id}
        )
        assert response.status_code == 200

    assert len(observed["instances"]) == 7
    assert len(set(observed["instances"])) == 1, "同一会话的请求必须复用同一 Session 实例"
    # The first request sees 0 prior turns; every later request must observe at
    # most 3 physically recorded turns — not an unbounded prompt-side trim.
    assert observed["entry_lengths"] == [0, 1, 2, 3, 3, 3, 3]

    last = client.post("/consultations", json={"query": "第7问", "session_id": session_id})
    assert last.status_code == 200
    history = last.json()["answer"]
    for early in ("第1问", "第2问", "第3问", "第4问"):
        assert early not in history


def test_same_session_final_order_is_strictly_serial(monkeypatch):
    """缺口 C：并发提交的同一会话请求按会话锁串行，最终历史顺序无交错。"""
    from concurrent.futures import ThreadPoolExecutor

    order: dict = {"executed": [], "lock": __import__("threading").Lock()}

    class OrderProbeWorkflow(_HistoryEchoWorkflow):
        def run(self, query, drug_context=None, session=None):
            with order["lock"]:
                order["executed"].append(query)
            return super().run(query, drug_context=drug_context, session=session)

    clock = _ControlledClock()
    client = _session_test_client(
        monkeypatch, clock=clock, ttl=3600.0, capacity=10, workflow_cls=OrderProbeWorkflow
    )
    session_id = client.post("/consultations", json={"query": "第0问"}).json()["session_id"]
    order["executed"].clear()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post, "/consultations", json={"query": "第1问", "session_id": session_id}
        )
        second = pool.submit(
            client.post, "/consultations", json={"query": "第2问", "session_id": session_id}
        )
        assert first.result(timeout=3).status_code == 200
        assert second.result(timeout=3).status_code == 200

    assert sorted(order["executed"]) == ["第1问", "第2问"]
    # Snapshot before the third request: the probe records every run() call,
    # and the third turn is not part of its own echo history.
    executed_snapshot = list(order["executed"])

    # A third request must observe history strictly in serial-execution order:
    # no interleaving, no loss, no duplication. Which of the two concurrent
    # requests wins the session lock is scheduling-dependent — the contract is
    # that history order matches the executed order, not a fixed query order.
    third = client.post("/consultations", json={"query": "第3问", "session_id": session_id})
    assert third.status_code == 200
    history = third.json()["answer"]
    assert history.count("第1问") == 1 and history.count("第2问") == 1
    executed_positions = [history.index(q) for q in executed_snapshot]
    assert executed_positions == sorted(executed_positions)


def test_api_workflow_crash_releases_the_in_use_slot(monkeypatch):
    """A crashing workflow must release its in-use slot in the finally block.

    Leaked in-use entries are never evicted (the store skips them on purpose,
    they might be mid-request), so one leaked slot on a capacity-1 store
    permanently suffocates it: every later create() would 503. The first
    request crashes (500), the second must be able to create a session again.
    """
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from medcopilot.api import server

    class _CrashOnceWorkflow(_HistoryEchoWorkflow):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def run(self, query, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated workflow crash")
            return super().run(query, **kwargs)

    clock = _ControlledClock()
    configured = SimpleNamespace(**server.config.model_dump())
    configured.med_session_ttl_seconds = 3600.0
    configured.med_session_max_entries = 1
    monkeypatch.setattr(server, "config", configured)
    monkeypatch.setattr(server, "AgentGraph", _CrashOnceWorkflow)
    # raise_server_exceptions=False: the crash must surface as a 500 response
    # (after the finally released the slot), not as a test-side exception.
    client = TestClient(server.create_app(clock=clock), raise_server_exceptions=False)

    crashed = client.post("/consultations", json={"query": "第一问"})
    assert crashed.status_code == 500

    # With capacity=1, a leaked in-use slot would make this create() a 503;
    # the workflow itself succeeds this time, so the request must be 200.
    retry = client.post("/consultations", json={"query": "第二问"})
    assert retry.status_code == 200
