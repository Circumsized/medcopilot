"""OpenAI 兼容 `/embeddings` API 适配器(默认离线本地;显式 cloud + 文本出网授权时启用)。

定位与边界:
- 生产默认仍是**离线本地权重**(`EMBEDDING_PROVIDER=local`、`MODEL_WEIGHTS_OFFLINE=1`、
  镜像烘焙、manifest 身份固定)。仅当显式设置 ``EMBEDDING_PROVIDER=cloud`` 且
  ``ALLOW_TEXT_EGRESS=true`` 时才接入 `build_encoder` 服务路径。
- 请求/响应格式按 SiliconFlow Create Embeddings 官方文档实现(OpenAI 兼容):
  ``POST {base}/embeddings``、Bearer 头、``{"model", "input"}``;
  **单请求最多 32 条输入**(服务端硬限),响应以 ``data[].index`` 对齐顺序。
- 密钥卫生:任何异常/状态面都不携带密钥——服务端错误体可能回显凭据,本类不透传
  响应体,只报告状态码与稳定错误码。
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import httpx

from medcopilot.common.config import config
from medcopilot.common.logger import logger

# SiliconFlow Create Embeddings 文档:单请求 input 最多 32 条。
MAX_BATCH = 32
# 429/5xx/网络错误可重试;4xx 配置类错误重试无意义。
_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
FN_NAME = "api-embeddings"


class ApiEmbeddingFunction:
    """调用 OpenAI 兼容 embeddings 端点的稠密编码器。

    结构上满足 ``rag.encoder_protocol.DenseEmbedder``(str|Sequence[str] → 向量),
    因此可直接注入 ``HybridRetriever(dense_encoder=...)`` 与评测脚本——与本地
    ``MedBgeM3EmbeddingFunction`` 同一注入点,零工作流改动。
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        attempts: int = 3,
        retry_backoff: float = 0.5,
        max_batch: int = MAX_BATCH,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        self.api_base = (api_base if api_base is not None else config.embedding_api_base).rstrip("/")
        self.api_key = (api_key if api_key is not None else config.embedding_api_key).strip()
        self.model_name = (model if model is not None else config.embedding_model) or "BAAI/bge-m3"
        self.timeout = float(timeout if timeout is not None else config.embedding_api_timeout)
        self.attempts = max(1, int(attempts))
        self.retry_backoff = float(retry_backoff)
        self.max_batch = max(1, int(max_batch))
        self._transport = transport
        self._sleep = sleep
        self._embedding_dim: int | None = None

    # ------------------------------------------------------ 状态面(无密钥)

    @classmethod
    def name(cls) -> str:
        """Stable Chroma registry identity."""
        return FN_NAME

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict:
        return {"api_base": self.api_base, "model": self.model_name}

    @classmethod
    def build_from_config(cls, config_data: dict):
        return cls(api_base=config_data.get("api_base"), model=config_data.get("model"), api_key=config.embedding_api_key)

    @property
    def embedding_dim(self) -> int | None:
        """实测维度(首次成功编码后);None = 尚未实测。"""
        return self._embedding_dim

    @property
    def expected_dim(self) -> int:
        return int(config.embedding_dim)

    def measure_dim(self) -> int | None:
        if self._embedding_dim is not None:
            return self._embedding_dim
        try:
            self("维度探测")
        except Exception as exc:
            logger.warning("API embedding dimension probe failed ({}).", exc)
            return None
        return self._embedding_dim

    def embed_query(self, input):
        texts = [input] if isinstance(input, str) else [str(item) for item in input]
        return self(texts)

    @property
    def dim_source(self) -> str:
        return "measured" if self._embedding_dim is not None else "unmeasured"

    @property
    def endpoint(self) -> str:
        return f"{self.api_base}/embeddings"

    def status(self) -> dict:
        """诊断状态;密钥只以“是否配置”布尔值出现,值本身绝不进状态面。"""
        return {
            "embedding_fn": FN_NAME,
            "model": self.model_name,
            "endpoint": self.endpoint,
            "embedding_dim": self._embedding_dim,
            "dim_source": self.dim_source,
            "api_key_configured": bool(self.api_key),
        }

    # ------------------------------------------------------------- 请求

    def _post(self, inputs: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": inputs}
        last_status: int | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_status = None
                if attempt < self.attempts:
                    self._sleep(self.retry_backoff * attempt)
                    continue
                raise RuntimeError(
                    f"embedding API unavailable after {self.attempts} attempts (network: {type(exc).__name__})"
                ) from exc
            if response.status_code == 200:
                return self._parse(response, len(inputs))
            last_status = response.status_code
            if response.status_code in _RETRYABLE_STATUSES:
                if attempt < self.attempts:
                    self._sleep(self.retry_backoff * attempt)
                    continue
                # 重试耗尽的可重试错误 ≠ 请求被拒:调用方要能区分"服务端过载/限流"
                # 与"我们的请求本身有问题"。
                raise RuntimeError(
                    f"embedding API unavailable after {self.attempts} attempts (last status {last_status})"
                )
            if response.status_code in (401, 403):
                # 不透传响应体(服务端可能回显凭据);只给可操作的稳定错误。
                raise RuntimeError(
                    f"embedding API rejected the credential (status {response.status_code}); "
                    "check EMBEDDING_API_KEY"
                )
            raise RuntimeError(
                f"embedding API rejected the request (status {response.status_code})"
            )
        raise RuntimeError(
            f"embedding API unavailable after {self.attempts} attempts (last status {last_status})"
        )

    @staticmethod
    def _parse(response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("embedding API returned a non-JSON body") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("embedding API response has no 'data' array")
        # 服务端允许乱序:顺序只能靠 index 字段对齐。
        rows: list[list[float] | None] = [None] * len(data)
        for entry in data:
            if not isinstance(entry, dict):
                raise RuntimeError("embedding API response entry is not an object")
            index = entry.get("index")
            if type(index) is not int or not 0 <= index < len(data):
                raise RuntimeError("embedding API response entry has an out-of-range index")
            if rows[index] is not None:
                raise RuntimeError("embedding API response has duplicate indices")
            vector = entry.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError("embedding API response entry has no embedding array")
            rows[index] = vector
        if any(row is None for row in rows):
            raise RuntimeError("embedding API response is missing entries")
        vectors = [row for row in rows if row is not None]
        _validate_vectors(vectors, expected=expected)
        return vectors

    # ------------------------------------------------------------ 编码

    def __call__(self, input):
        single = isinstance(input, str)
        texts = [input] if single else [str(item) for item in input]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch):
            vectors.extend(self._post(texts[start : start + self.max_batch]))
        if vectors:
            _validate_vectors(vectors, expected=len(texts))
            self._embedding_dim = len(vectors[0])
        return vectors[0] if single else vectors


def _validate_vectors(vectors: list[list[float]], *, expected: int) -> None:
    """与本地 ``MedBgeM3EmbeddingFunction._validate`` 同一标准(有意复制:

    从 vector_store 复用会在导入期拉起 chromadb,评测通道不需要这个代价;
    坏向量在两种通道里都必须拦死——错误的向量空间没有任何下游组件能发现)。
    """
    if len(vectors) != expected:
        raise ValueError(f"embedding count mismatch: {len(vectors)} vectors for {expected} texts")
    dim: int | None = None
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise ValueError("embedding vector must be a sequence of numbers")
        if not vector:
            raise ValueError("embedding vector is empty")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("embedding vector contains a non-numeric value")
            if not math.isfinite(float(value)):
                raise ValueError("embedding vector must be finite")
        if dim is None:
            dim = len(vector)
        elif len(vector) != dim:
            raise ValueError(f"embedding dimension inconsistent: {len(vector)} != {dim}")


def build_api_encoder():
    """按配置装配 API 编码器;base 或 key 缺失 ⇒ None(调用方走降级/SKIP)。"""
    if not (config.embedding_api_base or "").strip():
        return None
    if not (config.embedding_api_key or "").strip():
        logger.warning("EMBEDDING_API_BASE is set but EMBEDDING_API_KEY is empty; API encoder disabled.")
        return None
    return ApiEmbeddingFunction()
