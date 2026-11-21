"""Explicit HTTP client used by the Gradio UI to reach the FastAPI service.

The UI must stay a thin consumer: it owns no RAG, OCR or workflow objects and
only keeps the opaque ``session_id`` returned by the API. All network errors
are mapped to stable, non-sensitive user-facing error types so the chat can
render a safe message without leaking response bodies, URLs or credentials.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from medcopilot.common.logger import logger


class APIClientError(Exception):
    """Stable, user-safe failure of a consultation request."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class APIContractError(APIClientError):
    """The server answered 200 but the payload violated the response contract."""


_CONTRACT_FIELDS = {
    "answer": str,
    "sources": list,
    "is_emergency": bool,
    "intent": str,
    "abstained": bool,
}

_HTTP_ERROR_MAP = {
    400: ("bad_request", "咨询请求无效，请检查输入。"),
    401: ("authentication", "API 认证失败，请检查访问凭据。"),
    413: ("payload_too_large", "上传图片过大，请选择更小的文件。"),
    422: ("validation", "咨询内容无效，请检查输入。"),
    429: (
        "rate_limited",
        "请求过于频繁，请稍后重试。如出现胸痛、呼吸困难、意识不清等紧急情况，请立即拨打 120 或前往急诊，不要等待本服务。",
    ),
}

_VALIDATION_MESSAGE = "咨询内容无效，请检查输入。"
_SESSION_EXPIRED_MESSAGE = "会话已过期，请重新开始咨询。"
_SERVICE_UNAVAILABLE_MESSAGE = "咨询服务暂不可用，请稍后重试。"
_CONNECTION_MESSAGE = "无法连接咨询服务，请稍后重试。"
_CONTRACT_MESSAGE = "咨询服务响应格式异常，请稍后重试。"
_SESSION_ID_MAX_CHARS = 128


def _validate_query(query: str) -> str:
    trimmed = (query or "").strip()
    if not trimmed:
        raise APIClientError(_VALIDATION_MESSAGE, code="validation")
    return trimmed


def _validate_session_id(session_id: str | None) -> str | None:
    if session_id is None:
        return None
    trimmed = session_id.strip()
    if not trimmed:
        return None
    if len(trimmed) > _SESSION_ID_MAX_CHARS:
        raise APIClientError(_VALIDATION_MESSAGE, code="validation")
    return trimmed


def _validate_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise APIContractError(_CONTRACT_MESSAGE, code="contract")
    for field, expected in _CONTRACT_FIELDS.items():
        value = payload.get(field)
        if expected is bool:
            if type(value) is not bool:
                raise APIContractError(_CONTRACT_MESSAGE, code="contract")
        elif not isinstance(value, expected):
            raise APIContractError(_CONTRACT_MESSAGE, code="contract")
    session_id = payload.get("session_id")
    if type(session_id) is not str or not session_id or not session_id.strip():
        raise APIContractError(_CONTRACT_MESSAGE, code="contract")
    sources = payload["sources"]
    if any(not isinstance(item, dict) for item in sources):
        raise APIContractError(_CONTRACT_MESSAGE, code="contract")
    return payload


class MedicalAPIClient:
    """Minimal httpx-based consultation client with stable error mapping."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        key = (api_key or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers=headers,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ calls

    def consult_text(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        clean_query = _validate_query(query)
        body: dict[str, Any] = {"query": clean_query}
        clean_session = _validate_session_id(session_id)
        if clean_session:
            body["session_id"] = clean_session
        request = self._client.build_request("POST", "/consultations", json=body)
        return self._send(request, had_session=clean_session is not None)

    def consult_with_image(
        self, query: str, image_path: str, session_id: str | None = None
    ) -> dict[str, Any]:
        clean_query = _validate_query(query)
        clean_session = _validate_session_id(session_id)
        data: dict[str, Any] = {"query": clean_query}
        if clean_session:
            data["session_id"] = clean_session
        try:
            file_handle = open(os.fspath(image_path), "rb")
        except OSError as exc:
            raise APIClientError("图片文件不可读，请重新上传。", code="validation") from exc
        # build_request runs inside the with-block: if it raises, the handle is
        # still closed (a prior version leaked it in that window).
        with file_handle:
            request = self._client.build_request(
                "POST",
                "/consultations/with-image",
                data=data,
                files={"image": (os.path.basename(os.fspath(image_path)), file_handle)},
            )
            try:
                return self._send(request, had_session=clean_session is not None)
            finally:
                # Multipart stream close does not reliably release the underlying
                # file handle on every httpx version; close it explicitly. A
                # stream-close failure must never skip the with-block's own
                # close, so it is contained separately.
                try:
                    request.stream.close()
                except Exception as exc:
                    logger.warning("Multipart stream close failed: {}", exc)

    # --------------------------------------------------------------- plumbing

    def _send(self, request: httpx.Request, *, had_session: bool) -> dict[str, Any]:
        try:
            response = self._client.send(request)
        except httpx.StreamError as exc:
            # Streaming-misuse (RequestNotRead/StreamConsumed/...). StreamError
            # subclasses RuntimeError, NOT HTTPError, so it must be caught
            # explicitly; it still maps to the same user-safe connection error
            # and must never surface a raw httpx traceback into the UI.
            # The cause goes to the operator log (never the UI): without it a
            # wrong base_url means endless identical retries with no trail.
            logger.warning("API request failed: {}: {}", type(exc).__name__, exc)
            raise APIClientError(_CONNECTION_MESSAGE, code="connection") from exc
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            # RequestError 统一覆盖请求侧全部失败：TransportError 家族
            # （ConnectTimeout/NetworkError/ProxyError/ProtocolError）以及同样挂在
            # RequestError 之下的 DecodingError/TooManyRedirects（已按 httpx 0.28
            # 异常层级核实，二者不在 TransportError 之下）——对用户而言都是同一个
            # "连接失败"，且不得把 socket/URL 等细节上行到 UI。
            # 运营日志保留异常类型与原因（含错误 base_url 产生的 InvalidURL），
            # 供诊断；UI 侧仍只看到稳定文案。
            logger.warning("API request failed: {}: {}", type(exc).__name__, exc)
            raise APIClientError(_CONNECTION_MESSAGE, code="connection") from exc

        if response.status_code == 404:
            # ``had_session`` is supplied by the caller rather than inferred from
            # the request body: a multipart request is streamed, so touching
            # ``request.content`` raises httpx.RequestNotRead and buffers the
            # whole upload. The caller already knows whether it sent a session.
            if had_session:
                raise APIClientError(_SESSION_EXPIRED_MESSAGE, code="session_expired")
            raise APIClientError(_SERVICE_UNAVAILABLE_MESSAGE, code="service_unavailable")
        if response.status_code in _HTTP_ERROR_MAP:
            code, message = _HTTP_ERROR_MAP[response.status_code]
            raise APIClientError(message, code=code)
        if response.status_code >= 500:
            raise APIClientError(_SERVICE_UNAVAILABLE_MESSAGE, code="service_unavailable")
        if response.status_code != 200:
            raise APIClientError("咨询请求失败，请稍后重试。", code="bad_request")

        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise APIContractError(_CONTRACT_MESSAGE, code="contract")
        try:
            payload = response.json()
        except ValueError as exc:
            raise APIContractError(_CONTRACT_MESSAGE, code="contract") from exc
        return _validate_response(payload)
