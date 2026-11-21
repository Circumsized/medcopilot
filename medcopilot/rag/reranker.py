from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Sequence

from medcopilot.common.config import config
from medcopilot.common.logger import logger
from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker_api import build_api_reranker

# Degradation code surfaced when the cross-encoder weights are not available
# locally (offline deployment without a baked cache): reranking still works via
# the lexical fallback, but the quality loss must be observable, not silent.
RERANK_WEIGHTS_UNAVAILABLE = "rerank:weights-unavailable"


class Reranker:
    """Cross-encoder reranker with graceful degradation.

    Primary path: BAAI/bge-reranker (via sentence-transformers CrossEncoder,
    optional dependency). Fallback: token overlap scoring so the retrieval
    pipeline still reorders sensibly without heavyweight models.

    Weights are loaded offline-first (``MODEL_WEIGHTS_OFFLINE``): a deployment
    that baked the weights into the image must never reach for the network, and
    one that did not gets an explicit ``missing`` state instead of a silent
    quality drop.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        enabled: bool = True,
        max_length: int | None = None,
        local_files_only: bool | None = None,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.max_length = int(max_length if max_length is not None else config.rerank_max_length)
        self.local_files_only = (
            bool(config.model_weights_offline) if local_files_only is None else bool(local_files_only)
        )
        self.cache_dir = cache_dir if cache_dir is not None else (config.rerank_cache_dir or "")
        self._model = None
        self._load_failed = False
        self._load_detail: str | None = None
        self._lock = threading.Lock()
        self._api = build_api_reranker()
        self.enabled = enabled
        # Mode of the most recent rerank() call: "cross-encoder" or "lexical".
        # None until the first call; `mode` then reports the anticipated
        # "cross-encoder" until the first call proves otherwise (or reveals a
        # permanent load failure via _load_failed → "lexical").
        self._last_mode: str | None = None

    @property
    def mode(self) -> str:
        """Process-level rerank mode for /health: disabled | lexical | cross-encoder | cloud."""
        if not self.enabled:
            return "disabled"
        if self._api is not None:
            return self._last_mode or "cloud"
        if self._load_failed:
            return "lexical"
        return self._last_mode or "cross-encoder"

    @property
    def weights_state(self) -> str:
        """Weights availability for /health: disabled | unloaded | local | missing | cloud."""
        if not self.enabled:
            return "disabled"
        if self._api is not None:
            return "cloud"
        if self._load_failed:
            return "missing"
        return "local" if self._model is not None else "unloaded"

    def weights_status(self) -> dict[str, str | None]:
        """Structured weights status; the degradation code is set only when degraded."""
        state = self.weights_state
        return {
            "state": state,
            "model_name": self.model_name,
            "offline": bool(self.local_files_only),
            "degradation_code": RERANK_WEIGHTS_UNAVAILABLE if state == "missing" else None,
            "detail": self._load_detail,
        }

    def _load_model(self):
        if self._model is None and not self._load_failed and self.enabled:
            # Double-checked lock: several first requests can arrive together and
            # each would otherwise load its own copy of the weights.
            with self._lock:
                if self._model is None and not self._load_failed:
                    try:
                        from sentence_transformers import CrossEncoder

                        kwargs = {"max_length": self.max_length, "local_files_only": self.local_files_only}
                        if self.cache_dir:
                            kwargs["cache_dir"] = self.cache_dir
                        self._model = CrossEncoder(self.model_name, **kwargs)
                        logger.info(
                            "Loaded cross-encoder reranker: {} (offline={}, cache_dir={}).",
                            self.model_name, self.local_files_only, self.cache_dir or "default",
                        )
                    except Exception as exc:
                        # Deliberately broader than "optional dependency missing":
                        # missing weights, OOM or a corrupt cache all degrade to the
                        # lexical fallback. Availability is traded for rerank quality,
                        # permanently for this process — surfaced via `mode` and
                        # `weights_state` in /health rather than only this log line.
                        logger.warning("CrossEncoder unavailable ({}); using lexical fallback rerank.", exc)
                        self._load_failed = True
                        self._load_detail = str(exc)
        return self._model

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        query_counts = Counter(HybridRetriever._tokenize(query))
        if not query_counts:
            return 0.0
        text_tokens = set(HybridRetriever._tokenize(text))
        hits = sum(weight for token, weight in query_counts.items() if token in text_tokens)
        return hits / sum(query_counts.values())

    def rerank(self, query: str, documents: Sequence[Document], top_n: int = 3) -> list[tuple[Document, float]]:
        """Return documents sorted by relevance score, best first."""
        ranked, _ = self.rerank_with_status(query, documents, top_n=top_n)
        return ranked

    def rerank_with_status(
        self, query: str, documents: Sequence[Document], top_n: int = 3
    ) -> tuple[list[tuple[Document, float]], str]:
        """Return ranked documents and the per-call mode (no shared-state race)."""
        docs = list(documents)
        if not docs or top_n <= 0:
            return [], self.mode

        if self._api is not None:
            if not self.enabled:
                scored = [(d, self._lexical_score(query, d.page_content)) for d in docs]
                call_mode = "disabled"
                self._last_mode = "lexical"
            else:
                try:
                    scored = self._api.rerank(query, docs, top_n)
                    call_mode = "cloud"
                    self._last_mode = call_mode
                    return scored, call_mode
                except Exception as exc:
                    logger.warning("Cloud reranker unavailable ({}); lexical fallback.", exc)
                    scored = [(d, self._lexical_score(query, d.page_content)) for d in docs]
                    call_mode = "lexical"
                    self._last_mode = call_mode
        elif (model := self._load_model()) is not None:
            try:
                pairs = [(query, d.page_content) for d in docs]
                scores = model.predict(pairs)
                scored = list(zip(docs, [float(s) for s in scores]))
                call_mode = "cross-encoder"
                self._last_mode = call_mode
            except Exception as exc:
                logger.error("Cross-encoder inference failed ({}); lexical fallback.", exc)
                scored = [(d, self._lexical_score(query, d.page_content)) for d in docs]
                call_mode = "lexical"
                self._last_mode = call_mode
        else:
            scored = [(d, self._lexical_score(query, d.page_content)) for d in docs]
            call_mode = "disabled" if not self.enabled else "lexical"
            self._last_mode = "lexical"

        # Enumerate positions once: ``docs.index(doc)`` is O(n) per comparison
        # AND compares by value (Document is a dataclass), so a value-equal
        # duplicate resolved to the FIRST occurrence's index and jumped ahead
        # of documents between them — breaking the stable-tie contract. An
        # id-keyed position map keeps ties in true input order at O(n log n).
        position = {id(doc): index for index, doc in enumerate(docs)}
        scored.sort(key=lambda pair: (pair[1], -position[id(pair[0])]), reverse=True)
        return scored[:top_n], call_mode
