"""Public configuration contract for prompt budgets, offline weights and build metadata."""
import pytest
from pydantic import ValidationError

from medcopilot.common.config import AppConfig

DEFAULTS = {
    "llm_prompt_token_budget": 3500,
    "llm_evidence_token_budget": 2200,
    "llm_chunk_token_cap": 1200,
    "ocr_max_lines": 5,
    "ocr_max_line_chars": 120,
    "embedding_batch_size": 16,
    "embedding_normalize": True,
    "embedding_dim": 1024,
    "embedding_cache_dir": "",
    "rerank_cache_dir": "",
    "model_weights_offline": True,
    "log_format": "human",
    "app_version": "2.0.0",
    "app_git_sha": "unknown",
}


def test_defaults_are_safe_and_configured():
    defaults = {name: field.default for name, field in AppConfig.model_fields.items()}
    assert {name: defaults.get(name) for name in DEFAULTS} == DEFAULTS


@pytest.mark.parametrize("alias", ["MED_LLM_PROMPT_TOKEN_BUDGET", "MED_LLM_EVIDENCE_TOKEN_BUDGET",
                                    "MED_LLM_CHUNK_TOKEN_CAP", "MED_OCR_MAX_LINES", "MED_OCR_MAX_LINE_CHARS",
                                    "EMBEDDING_DIM", "EMBEDDING_BATCH_SIZE"])
def test_invalid_budget_or_dimension_is_rejected(alias):
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None, **{alias: 0})


def test_invalid_log_format_is_rejected():
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None, LOG_FORMAT="not-json-or-human")
