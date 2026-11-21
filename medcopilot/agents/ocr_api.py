"""OpenAI-compatible vision OCR adapter, opt-in only."""
from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx

from medcopilot.common.config import config
from medcopilot.common.logger import logger

_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})


class ApiOCRTool:
    def __init__(self, *, api_base: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float | None = None,
                 attempts: int = 3, transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep) -> None:
        self.api_base = (api_base if api_base is not None else config.ocr_api_base).rstrip("/")
        self.api_key = (api_key if api_key is not None else config.ocr_api_key).strip()
        if config.ocr_provider != "cloud" or not config.allow_image_egress:
            raise RuntimeError("cloud OCR requires OCR_PROVIDER=cloud and ALLOW_IMAGE_EGRESS=true")
        if not self.api_base.lower().startswith("https://"):
            raise RuntimeError("cloud OCR requires an HTTPS endpoint")
        self.model = model or config.ocr_api_model
        self.timeout = float(timeout if timeout is not None else config.ocr_api_timeout)
        self.attempts = max(1, int(attempts))
        self._transport = transport
        self._sleep = sleep
        self.last_status: str | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_base and self.api_key and self.model)

    def suggest_drug_from_image(self, image_path: str | None, drug_dictionary) -> tuple[list[str], str | None, str]:
        if not image_path or not Path(image_path).is_file():
            return [], None, "image_unreadable"
        try:
            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            payload = {"model": self.model, "temperature": 0,
                       "messages": [{"role": "user", "content": [
                           {"type": "text", "text": "Extract only visible drug/package text, one line per item."},
                           {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                       ]}]}
            response = None
            for attempt in range(1, self.attempts + 1):
                try:
                    with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                        response = client.post(
                            f"{self.api_base}/chat/completions",
                            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                            json=payload,
                        )
                except httpx.HTTPError as exc:
                    if attempt < self.attempts:
                        self._sleep(0.5 * attempt)
                        continue
                    raise RuntimeError(f"ocr API unavailable ({type(exc).__name__})") from exc
                if response.status_code == 200:
                    break
                if response.status_code in _RETRYABLE and attempt < self.attempts:
                    self._sleep(0.5 * attempt)
                    continue
                if response.status_code in (401, 403):
                    raise RuntimeError("ocr API rejected the credential")
                raise RuntimeError(f"ocr API failed with status {response.status_code}")
            if response is None:
                raise RuntimeError("ocr API unavailable")
            body: Any = response.json()
            choices = body.get("choices") if isinstance(body, dict) else None
            first = choices[0] if isinstance(choices, list) and choices else None
            message = first.get("message") if isinstance(first, dict) else None
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise RuntimeError("ocr API response has no text content")
            lines = [line.strip()[: config.ocr_max_line_chars] for line in text.splitlines() if line.strip()]
            from medcopilot.agents.ocr_tool import OCRTool
            matched = OCRTool.match_drug_name(lines[: config.ocr_max_lines], drug_dictionary)
            self.last_status = "ok"
            return lines[: config.ocr_max_lines], matched, "ok"
        except (OSError, ValueError, httpx.HTTPError, RuntimeError) as exc:
            logger.warning("Cloud OCR inference failed ({}): {}", type(exc).__name__, exc)
            self.last_status = "inference_error"
            return [], None, "inference_error"


def build_ocr_tool():
    if config.ocr_provider == "cloud":
        if not config.allow_image_egress:
            raise RuntimeError("cloud OCR requires ALLOW_IMAGE_EGRESS=true")
        return ApiOCRTool()
    from medcopilot.agents.ocr_tool import OCRTool
    return OCRTool()
