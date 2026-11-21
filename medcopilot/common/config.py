from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = frozenset({"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"})


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    # Global
    # Default-deny: an empty/unrecognized PROJECT_ENV must NEVER open
    # unauthenticated reads/writes. Only an explicit dev/test/local value
    # (see medcopilot/api/env_policy.classify_env) opens dev defaults;
    # "production"/"prod" is recognized-prod; anything else (including a typo
    # like "prodction") is "unrecognized" and fails closed at the API layer.
    project_env: str = Field(default="", alias="PROJECT_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["human", "json"] = Field(default="human", alias="LOG_FORMAT")
    app_version: str = Field(default="2.0.0", alias="APP_VERSION")
    app_git_sha: str = Field(default="unknown", alias="APP_GIT_SHA")
    # File sink for logs (rotated). Configurable so deployments can relocate it
    # or disable it entirely (empty string) in read-only/serverless environments;
    # avoids a hardcoded relative path created as an import side effect.
    log_file: str = Field(default="logs/app.log", alias="LOG_FILE")

    # 医疗审计 sink(W1.2):与 loguru 应用日志分离的追加式 JSONL + SHA-256 哈希链。
    # audit_log_path 留空 = 禁用审计(development 可用;生产必须设置)。
    # retention_days 默认 2190(6 年,按 45 CFR §164.316(b)(2)(i) 的保留要求)。
    audit_log_path: str = Field(default="logs/audit.jsonl", alias="AUDIT_LOG_PATH")
    audit_retention_days: int = Field(default=2190, alias="AUDIT_RETENTION_DAYS")
    audit_fsync: bool = Field(default=False, alias="AUDIT_FSYNC")
    # 可选:设置后 /health 暴露 audit_time_offset_ms(启动时做一次 SNTP 探测)。
    audit_ntp_server: str = Field(default="", alias="AUDIT_NTP_SERVER")

    # 剂量门模式(W6.1,需求方裁决 2026-10-04:A+B 双模式并存,默认安全侧):
    #   redact = 指令性剂量句替换为转介话术(B,默认)
    #   quote  = 逐字引用说明书剂量 + 强制免责,且仅当证据含禁忌/相互作用/
    #            注意事项安全章节时放行(A,含 fail-closed 底线)
    #   off    = 关闭(仅限开发环境;production 下 AgentGraph 构造直接拒绝)
    dosage_gate_mode: str = Field(default="redact", alias="DOSAGE_GATE_MODE")

    # W3.1 安全章节召回(禁忌保护位):触发语境下候选池扩容 + 保护位保底。
    retrieve_candidate_pool: int = Field(default=20, alias="RETRIEVE_CANDIDATE_POOL")
    evidence_top_n: int = Field(default=5, alias="EVIDENCE_TOP_N")
    safety_reserve_slots: int = Field(default=2, alias="SAFETY_RESERVE_SLOTS")
    safety_section_gate_enabled: bool = Field(default=True, alias="SAFETY_SECTION_GATE_ENABLED")

    # W3.3 sufficiency gate(替代实测失效的 evidence_floor——该字段保留仅作兼容,
    # 不再参与门控;显式设置 EVIDENCE_FLOOR 会在启动日志收到废弃提示)。
    # 阈值经影子数据校准:口语化 query 与正确证据的词汇重叠天然低,硬门禁会误杀;
    # 故 dry_run(影子模式,只记录不弃答)默认开启,影子数据足够后由部署方关闭。
    sufficiency_enabled: bool = Field(default=True, alias="SUFFICIENCY_ENABLED")
    sufficiency_min_doc_coverage: float = Field(default=0.10, alias="SUFFICIENCY_MIN_DOC_COVERAGE")
    sufficiency_min_union_coverage: float = Field(default=0.20, alias="SUFFICIENCY_MIN_UNION_COVERAGE")
    sufficiency_dry_run: bool = Field(default=True, alias="SUFFICIENCY_DRY_RUN")

    # W3.2 摄取端校验(OWASP LLM08 摄取闸门;拒绝 ⇒ 不写 CorpusStore、不入索引)。
    ingest_validation_enabled: bool = Field(default=True, alias="INGEST_VALIDATION_ENABLED")
    ingest_min_raw_chars: int = Field(default=30, alias="INGEST_MIN_RAW_CHARS")
    ingest_max_line_chars: int = Field(default=4000, alias="INGEST_MAX_LINE_CHARS")

    # W2.3 dependency-free process-local abuse controls (Docker workers=1).
    # Quotas are per API-key identity (per client IP only in no-key dev mode);
    # emergency triage runs before every one of them and is never throttled.
    # llm_concurrency caps in-flight retrieval+LLM+OCR requests; the overflow is
    # rejected immediately (503) so a slow upstream can never park every worker
    # thread. All values must be positive (misconfiguration fails at startup).
    rate_limit_enabled: bool = Field(default=True, alias="MED_RATE_LIMIT_ENABLED")
    rate_limit_consult_per_minute: int = Field(default=20, alias="MED_RATE_LIMIT_CONSULT_PER_MINUTE", gt=0)
    rate_limit_image_per_minute: int = Field(default=6, alias="MED_RATE_LIMIT_IMAGE_PER_MINUTE", gt=0)
    rate_limit_ingest_per_minute: int = Field(default=10, alias="MED_RATE_LIMIT_INGEST_PER_MINUTE", gt=0)
    rate_limit_rebuild_per_hour: int = Field(default=2, alias="MED_RATE_LIMIT_REBUILD_PER_HOUR", gt=0)
    llm_concurrency: int = Field(default=8, alias="MED_LLM_CONCURRENCY", gt=0)

    # API CORS: comma-separated explicit origins. Browsers forbid credentialed
    # requests against a "*" wildcard origin, so wildcard + credentials is both
    # invalid and unsafe (FastAPI CORS docs / MDN credentialed requests).
    # Set CORS_ALLOW_ALL=1 only for local development when you need arbitrary
    # origins (e.g. LAN testing) — never in production.
    cors_origins: str = Field(
        default="http://127.0.0.1:7860,http://localhost:7860,http://127.0.0.1:8000,http://localhost:8000",
        alias="CORS_ORIGINS",
    )
    cors_allow_all: bool = Field(default=False, alias="CORS_ALLOW_ALL")
    # Host 头白名单（逗号分隔，"*" = 不限）。starlette TrustedHostMiddleware
    # 用它拒绝伪造 Host 的请求（Host 头注入/缓存投毒缓解）；生产应固定为
    # 对外主机名。默认 "*" 保持既有零配置部署行为不变。
    trusted_hosts: str = Field(default="*", alias="MED_TRUSTED_HOSTS")
    # Bearer token guarding API access. Empty => open dev mode; production
    # deployments MUST set this (ingestion additionally requires it).
    admin_api_key: str | None = Field(default=None, alias="ADMIN_API_KEY")

    # UI -> API client. The Gradio app consumes the FastAPI service over HTTP
    # instead of assembling the clinical workflow locally; these settings make
    # the endpoint, timeout and optional Bearer key centrally configurable.
    med_api_base_url: str = Field(default="http://127.0.0.1:8000", alias="MED_API_BASE_URL")
    med_api_timeout_seconds: float = Field(default=30.0, alias="MED_API_TIMEOUT_SECONDS")

    # In-process consultation sessions (single-worker deployment). The store is
    # a bounded LRU with a sliding TTL; both limits must be positive.
    med_session_ttl_seconds: float = Field(default=1800.0, alias="MED_SESSION_TTL_SECONDS", gt=0)
    med_session_max_entries: int = Field(default=256, alias="MED_SESSION_MAX_ENTRIES", gt=0)

    # Authoritative corpus persistence (JSONL, one chunk per line). Empty =>
    # disabled and the API keeps the pure in-memory behavior of Slice 1; set it
    # (e.g. ./data/corpus/documents.jsonl) to make ingestion idempotent,
    # crash-safe and to survive restarts. Still process-local: single worker.
    med_corpus_path: str = Field(default="", alias="MED_CORPUS_PATH")

    # Versioned safety-policy artifact (audit mirror of the code-constant
    # gating; never the execution source). The /health fields and
    # medcopilot/scripts/policy_check.py read it; a broken/missing artifact only degrades
    # the report, never the actual gating (fail-safe).
    safety_policy_path: str = Field(
        default="medcopilot/datasets/policies/safety_policy_v1.json", alias="SAFETY_POLICY_PATH"
    )

    # LLM & RAG Settings
    llm_provider: str = Field(default="openai_compatible", alias="LLM_PROVIDER")
    llm_mode: Literal["local", "cloud"] = Field(default="local", alias="LLM_MODE")
    llm_api_base: str = Field(default="https://api.openai.com/v1", alias="LLM_API_BASE")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    # Offline structured-mock mode; honored from .env via AppConfig so copying
    # .env.example alone switches the demo to mock as documented.
    mock_llm: bool = Field(default=False, alias="MOCK_LLM")
    allow_mock_in_prod: bool = Field(default=False, alias="ALLOW_MOCK_IN_PRODUCTION")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS", gt=0)
    llm_timeout: float = Field(default=60.0, alias="LLM_TIMEOUT", gt=0)
    # max_attempts is the total number of provider calls; legacy max_retries
    # constructor inputs retain their historical total-attempt meaning.
    llm_max_attempts: int = Field(default=3, alias="LLM_MAX_ATTEMPTS", gt=0)
    llm_backoff_base: float = Field(default=1.0, alias="LLM_BACKOFF_BASE", ge=0)
    llm_backoff_cap: float = Field(default=20.0, alias="LLM_BACKOFF_CAP", gt=0)
    llm_retry_deadline: float = Field(default=45.0, alias="LLM_RETRY_DEADLINE", gt=0)
    llm_prompt_token_budget: int = Field(default=3500, alias="MED_LLM_PROMPT_TOKEN_BUDGET", gt=0)
    llm_evidence_token_budget: int = Field(default=2200, alias="MED_LLM_EVIDENCE_TOKEN_BUDGET", gt=0)
    llm_chunk_token_cap: int = Field(default=1200, alias="MED_LLM_CHUNK_TOKEN_CAP", gt=0)
    ocr_max_lines: int = Field(default=5, alias="MED_OCR_MAX_LINES", gt=0)
    ocr_max_line_chars: int = Field(default=120, alias="MED_OCR_MAX_LINE_CHARS", gt=0)
    ocr_provider: Literal["local", "cloud"] = Field(default="local", alias="OCR_PROVIDER")
    ocr_api_base: str = Field(default="", alias="OCR_API_BASE")
    ocr_api_key: str = Field(default="", alias="OCR_API_KEY")
    ocr_api_model: str = Field(default="", alias="OCR_API_MODEL")
    ocr_api_timeout: float = Field(default=30.0, alias="OCR_API_TIMEOUT", gt=0)
    allow_image_egress: bool = Field(default=False, alias="ALLOW_IMAGE_EGRESS")
    allow_text_egress: bool = Field(default=False, alias="ALLOW_TEXT_EGRESS")

    qianfan_ak: str | None = Field(default=None, alias="QIANFAN_AK")
    qianfan_sk: str | None = Field(default=None, alias="QIANFAN_SK")
    zhipuai_api_key: str | None = Field(default=None, alias="ZHIPUAI_API_KEY")

    chroma_persist_directory: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_provider: Literal["local", "cloud"] = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM", gt=0)
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE", gt=0)
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_cache_dir: str = Field(default="", alias="EMBEDDING_CACHE_DIR")
    # OpenAI 兼容 embedding API；只有 EMBEDDING_PROVIDER=cloud 才接入生产路径。
    embedding_api_base: str = Field(default="", alias="EMBEDDING_API_BASE")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_api_timeout: float = Field(default=30.0, alias="EMBEDDING_API_TIMEOUT", gt=0)
    rerank_cache_dir: str = Field(default="", alias="RERANK_CACHE_DIR")
    rerank_provider: Literal["local", "cloud"] = Field(default="local", alias="RERANK_PROVIDER")
    rerank_api_base: str = Field(default="", alias="RERANK_API_BASE")
    rerank_api_key: str = Field(default="", alias="RERANK_API_KEY")
    rerank_api_timeout: float = Field(default=30.0, alias="RERANK_API_TIMEOUT", gt=0)
    model_weights_offline: bool = Field(default=True, alias="MODEL_WEIGHTS_OFFLINE")
    reranker_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANKER_MODEL")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_top_n: int = Field(default=3, alias="RERANK_TOP_N")
    rerank_max_length: int = Field(default=512, alias="RERANK_MAX_LENGTH")
    rrf_k: int = Field(default=60, alias="RRF_K")
    evidence_floor: float = Field(default=0.05, alias="EVIDENCE_FLOOR")
    groundedness_threshold: float = Field(default=0.15, alias="GROUNDEDNESS_THRESHOLD")
    max_drug_label_chars: int = Field(default=50000, alias="MAX_DRUG_LABEL_CHARS")
    # Opt-in real dense retrieval: requires chromadb extra; when false the API
    # runs BM25+TF-IDF fallback and /health reports dense_degraded=true.
    enable_chroma_dense: bool = Field(default=False, alias="ENABLE_CHROMA_DENSE")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> str:
        if value is None:
            return "INFO"
        normalized = str(value).strip().upper()
        if not normalized:
            return "INFO"
        if normalized == "WARN":
            normalized = "WARNING"
        if normalized not in _VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}")
        return normalized


config = AppConfig()
