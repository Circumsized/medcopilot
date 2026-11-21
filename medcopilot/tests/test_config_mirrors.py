"""Guards against silently misleading reference config mirrors.

Each project's ``configs/`` directory is documented as a read-only mirror:
nothing loads it, so a stale value there cannot break runtime behavior, but it
CAN mislead a maintainer into "changing" a parameter that has no effect. These
tests pin two properties:

1. The mirrors stay non-authoritative (no source file imports/reads them).
2. Where a mirror states a default, it matches the code's real default.

The code is always the source of truth; when a default changes, update the
mirror (and its comment) rather than relaxing these assertions.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
P2_CONFIG_DIR = PROJECT_ROOT / "medcopilot" / "configs"
MIRROR_FILES = {
    "medcopilot.yaml": P2_CONFIG_DIR,
    "triage_keywords.txt": P2_CONFIG_DIR,
    "clinical_sections.txt": P2_CONFIG_DIR,
}


def _scalar(text: str, key: str) -> str:
    """Read a `key: value` scalar from the mirror, ignoring trailing comments."""
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*([^#\n]+)", text, re.MULTILINE)
    assert match, f"{key} missing from mirror"
    return match.group(1).strip()


@pytest.fixture(scope="module")
def medcopilot_mirror() -> str:
    return (P2_CONFIG_DIR / "medcopilot.yaml").read_text(encoding="utf-8")


# ------------------------- mirrors must stay non-authoritative ----------------


@pytest.mark.parametrize("filename", sorted(MIRROR_FILES))
def test_mirror_declares_it_is_not_loaded(filename):
    text = (MIRROR_FILES[filename] / filename).read_text(encoding="utf-8")
    assert "不被代码加载" in text, f"{filename} must warn that it is not loaded"


def test_no_source_file_reads_the_config_mirrors():
    """If someone wires these files up, these tests must be revisited."""
    searched = [
        path
        for path in (PROJECT_ROOT / "medcopilot").rglob("*.py")
        # 测试文件本就要点名镜像文件来守护一致性，只扫产品源码（含 CLI 脚本）。
        if "tests" not in path.parts
    ]
    assert searched, "no source files discovered"

    offenders = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in searched
        if re.search(r"configs/|medcopilot\.yaml|triage_keywords|clinical_sections|yaml\.safe_load",
                     path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "configs/ is documented as a non-loaded mirror but is referenced by: "
        f"{offenders}. Either revert the wiring or update the mirrors + README."
    )


# --------------------------- mirrors must match real defaults -----------------


def test_medcopilot_mirror_is_valid_yaml(medcopilot_mirror):
    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(medcopilot_mirror)
    assert isinstance(parsed, dict) and {"rag", "guard", "llm", "api", "triage"} <= set(parsed)


def test_medcopilot_mirror_matches_appconfig_defaults(medcopilot_mirror):
    from medcopilot.common.config import AppConfig

    # Build a config free of any ambient .env/environment overrides.
    defaults = {name: field.default for name, field in AppConfig.model_fields.items()}

    assert int(_scalar(medcopilot_mirror, "rrf_k")) == defaults["rrf_k"]
    assert int(_scalar(medcopilot_mirror, "rerank_top_n")) == defaults["rerank_top_n"]
    assert int(_scalar(medcopilot_mirror, "rerank_max_length")) == defaults["rerank_max_length"]
    assert float(_scalar(medcopilot_mirror, "evidence_floor")) == defaults["evidence_floor"]
    assert float(_scalar(medcopilot_mirror, "groundedness_threshold")) == defaults["groundedness_threshold"]
    assert int(_scalar(medcopilot_mirror, "max_drug_label_chars")) == defaults["max_drug_label_chars"]
    assert (_scalar(medcopilot_mirror, "rerank_enabled") == "true") is defaults["rerank_enabled"]
    assert (_scalar(medcopilot_mirror, "enable_chroma_dense") == "true") is defaults["enable_chroma_dense"]
    assert (_scalar(medcopilot_mirror, "cors_allow_all") == "true") is defaults["cors_allow_all"]
    assert _scalar(medcopilot_mirror, "trusted_hosts").strip('\"') == defaults["trusted_hosts"]
    assert (_scalar(medcopilot_mirror, "rate_limit_enabled") == "true") is defaults["rate_limit_enabled"]
    assert (_scalar(medcopilot_mirror, "allow_mock_in_prod") == "true") is defaults["allow_mock_in_prod"]
    assert int(_scalar(medcopilot_mirror, "llm_max_tokens")) == defaults["llm_max_tokens"]
    assert float(_scalar(medcopilot_mirror, "llm_timeout")) == defaults["llm_timeout"]
    assert int(_scalar(medcopilot_mirror, "llm_max_attempts")) == defaults["llm_max_attempts"]
    assert float(_scalar(medcopilot_mirror, "llm_backoff_base")) == defaults["llm_backoff_base"]
    assert float(_scalar(medcopilot_mirror, "llm_backoff_cap")) == defaults["llm_backoff_cap"]
    assert float(_scalar(medcopilot_mirror, "llm_retry_deadline")) == defaults["llm_retry_deadline"]
    for name in (
        "rate_limit_consult_per_minute",
        "rate_limit_image_per_minute",
        "rate_limit_ingest_per_minute",
        "rate_limit_rebuild_per_hour",
        "llm_concurrency",
        "llm_prompt_token_budget",
        "llm_evidence_token_budget",
        "llm_chunk_token_cap",
        "ocr_max_lines",
        "ocr_max_line_chars",
        "embedding_dim",
        "embedding_batch_size",
    ):
        assert int(_scalar(medcopilot_mirror, name)) == defaults[name], name
    for name in ("embedding_normalize", "model_weights_offline"):
        assert (_scalar(medcopilot_mirror, name) == "true") is defaults[name]
    for name in ("allow_image_egress", "allow_text_egress"):
        assert (_scalar(medcopilot_mirror, name) == "true") is defaults[name]
    for name in (
        "embedding_model",
        "embedding_provider",
        "rerank_provider",
        "ocr_provider",
        "embedding_cache_dir",
        "rerank_cache_dir",
        "log_format",
        "app_version",
        "app_git_sha",
    ):
        assert _scalar(medcopilot_mirror, name).strip('\"') == defaults[name]
    # 评测用/云端 API 通道:默认整体关闭,镜像与代码必须同时为空。
    for name in (
        "embedding_api_base",
        "embedding_api_key",
        "rerank_api_base",
        "rerank_api_key",
        "ocr_api_base",
        "ocr_api_key",
        "ocr_api_model",
    ):
        assert _scalar(medcopilot_mirror, name).strip('\"') == defaults[name] == ""
    assert float(_scalar(medcopilot_mirror, "embedding_api_timeout")) == defaults["embedding_api_timeout"]
    assert float(_scalar(medcopilot_mirror, "rerank_api_timeout")) == defaults["rerank_api_timeout"]
    assert float(_scalar(medcopilot_mirror, "ocr_api_timeout")) == defaults["ocr_api_timeout"]
    assert _scalar(medcopilot_mirror, "dosage_gate_mode").strip('\"') == defaults["dosage_gate_mode"]
    assert _scalar(medcopilot_mirror, "med_api_base_url").strip('\"') == defaults["med_api_base_url"]
    assert float(_scalar(medcopilot_mirror, "med_api_timeout_seconds")) == defaults["med_api_timeout_seconds"]
    assert float(_scalar(medcopilot_mirror, "med_session_ttl_seconds")) == defaults["med_session_ttl_seconds"]
    assert int(_scalar(medcopilot_mirror, "med_session_max_entries")) == defaults["med_session_max_entries"]
    assert _scalar(medcopilot_mirror, "med_corpus_path").strip('\"') == defaults["med_corpus_path"]
    for name in ("retrieve_candidate_pool", "evidence_top_n", "safety_reserve_slots"):
        assert int(_scalar(medcopilot_mirror, name)) == defaults[name], name
    for name in ("safety_section_gate_enabled", "sufficiency_enabled", "sufficiency_dry_run"):
        assert (_scalar(medcopilot_mirror, name) == "true") is defaults[name], name
    for name in ("sufficiency_min_doc_coverage", "sufficiency_min_union_coverage"):
        assert float(_scalar(medcopilot_mirror, name)) == defaults[name], name


def test_triage_mirror_has_no_overbroad_single_char_hints():
    """The code deliberately dropped single-char hints; the mirror must not re-add them."""
    from medcopilot.agents.triage import TriageAgent

    text = (P2_CONFIG_DIR / "triage_keywords.txt").read_text(encoding="utf-8")
    medical_section = text.split("# --- medical hints", 1)[1].split("# --- emergency", 1)[0]
    listed = {
        line.strip()
        for line in medical_section.splitlines()
        if line.strip() and not line.startswith("#") and not line.strip().startswith("（")
    }
    single_chars = {item for item in listed if len(item) == 1}
    assert not single_chars, f"mirror re-introduced over-broad single-char hints: {single_chars}"

    # Every plain medical hint in the mirror should exist in the authoritative tuple.
    known = set(TriageAgent.MEDICAL_HINTS)
    unknown = listed - known
    assert not unknown, f"mirror lists medical keywords absent from TriageAgent: {unknown}"


def test_clinical_sections_mirror_matches_chunker():
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    text = (P2_CONFIG_DIR / "clinical_sections.txt").read_text(encoding="utf-8")
    # Only parse the data region; the explanatory header intentionally uses brackets.
    data = text.split("\n\n", 1)[1]
    listed = set(re.findall(r"【([^】]+)】", data))
    assert listed == set(ClinicalChunker.SECTIONS), (
        "clinical_sections.txt drifted from ClinicalChunker.SECTIONS: "
        f"only-in-mirror={listed - set(ClinicalChunker.SECTIONS)}, "
        f"only-in-code={set(ClinicalChunker.SECTIONS) - listed}"
    )
