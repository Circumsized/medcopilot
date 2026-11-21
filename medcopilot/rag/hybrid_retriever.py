from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Sequence

from medcopilot.common.logger import logger
from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.encoder_protocol import DenseEmbedder


class HybridRetriever:
    """BM25 + dense retrieval with deterministic reciprocal-rank fusion.

    ``dense_encoder`` is injectable so production deployments can use Chroma,
    BGE-M3 or MedCPT without changing the workflow. The dependency-free
    fallback uses cached TF-IDF vectors.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        k: int = 60,
        # Structural contract (PEP 544): single-string or batch embedding;
        # a batch attempt is tried first with a per-item fallback.
        dense_encoder: DenseEmbedder | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = list(documents)
        self.k = k
        self.k1 = float(k1)
        self.b = float(b)
        self._tokenized = [self._tokenize(d.page_content) for d in self.documents]
        self._idf = self._build_idf(self._tokenized)
        self._dense_encoder = dense_encoder
        self._dense_vectors: list[list[float]] | None = None
        self._dense_cooldown_until = 0.0
        self._fallback_vectors: list[dict[str, float]] | None = None
        if dense_encoder is not None and self.documents:
            # Graceful degradation: a failing dense backend (network, missing
            # weights, quota) must never break retrieval — fall back to TF-IDF.
            try:
                texts = [d.page_content for d in self.documents]
                try:
                    # Support batch encoders (e.g. SentenceTransformer, API clients)
                    raw_vectors = list(dense_encoder(texts))
                except Exception:
                    # Fallback to per-document encoding
                    raw_vectors = [dense_encoder(t) for t in texts]
                if not raw_vectors or len(raw_vectors) != len(self.documents):
                    raise ValueError("dense encoder returned wrong vector count")
                clean_vectors: list[list[float]] = []
                for v in raw_vectors:
                    cleaned = [float(x) for x in v]
                    if not cleaned:
                        raise ValueError("dense encoder returned empty vector")
                    if any(math.isnan(x) or math.isinf(x) for x in cleaned):
                        raise ValueError("dense vector contains NaN or Inf")
                    clean_vectors.append(cleaned)
                dim = len(clean_vectors[0])
                if dim == 0 or any(len(v) != dim for v in clean_vectors):
                    raise ValueError("dense encoder returned inconsistent vector dimensions")
                self._dense_vectors = clean_vectors
            except Exception as exc:
                logger.warning("Dense encoding failed ({}); falling back to TF-IDF vectors.", exc)
                self._dense_encoder = None
                self._dense_vectors = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+(?:\.\d+)?", text.lower())

    @staticmethod
    def _build_idf(tokenized: Sequence[Sequence[str]]) -> dict[str, float]:
        n_docs = max(len(tokenized), 1)
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        return {token: math.log((n_docs + 1) / (df + 1)) + 1.0 for token, df in document_frequency.items()}

    def _bm25_scores(self, query: str) -> list[float]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [0.0] * len(self.documents)
        avgdl = sum(len(tokens) for tokens in self._tokenized) / max(len(self._tokenized), 1)
        scores = []
        for tokens in self._tokenized:
            counts = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = self._idf.get(token, 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / max(avgdl, 1.0))
                score += idf * tf * (self.k1 + 1.0) / denominator
            scores.append(score)
        return scores

    def bm25_search(self, query: str, top_n: int = 5) -> list[Document]:
        return [doc for doc, score in self.bm25_search_with_scores(query, top_n) if score > 0]

    def bm25_search_with_scores(self, query: str, top_n: int = 5) -> list[tuple[Document, float]]:
        scores = self._bm25_scores(query)
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(self.documents[i], scores[i]) for i in ranking[: max(0, top_n)] if scores[i] > 0]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _fallback_dense_vector(self, text: str) -> dict[str, float]:
        counts = Counter(self._tokenize(text))
        return {token: count * self._idf.get(token, 1.0) for token, count in counts.items()}

    def _sparse_cosine(self, left: dict[str, float], right: dict[str, float]) -> float:
        vocabulary = set(left) | set(right)
        return self._cosine([left.get(token, 0.0) for token in vocabulary], [right.get(token, 0.0) for token in vocabulary])

    def _ensure_fallback_vectors(self) -> list[dict[str, float]]:
        if self._fallback_vectors is None:
            self._fallback_vectors = [self._fallback_dense_vector(d.page_content) for d in self.documents]
        return self._fallback_vectors

    @property
    def is_degraded(self) -> bool:
        """True when dense embeddings are unavailable or query retries are cooling down.

        This is a backend-health snapshot, not the status of any individual call.
        """
        return (
            self._dense_encoder is None
            or self._dense_vectors is None
            or time.monotonic() < self._dense_cooldown_until
        )

    def dense_search(self, query: str, top_n: int = 5) -> list[Document]:
        return [doc for doc, score in self.dense_search_with_scores(query, top_n) if score > 0]

    def dense_search_with_scores(self, query: str, top_n: int = 5) -> list[tuple[Document, float]]:
        results, _ = self._dense_search_with_status(query, top_n)
        return results

    def _dense_search_with_status(
        self, query: str, top_n: int = 5, *, allow_tfidf: bool = True,
    ) -> tuple[list[tuple[Document, float]], bool]:
        """Return hits and whether THIS call used genuine dense embeddings.

        Keep TF-IDF compatibility for direct dense-search callers, but let the
        fusion path skip it. Never infer a call's outcome from shared health state.
        """
        if not self.documents:
            return [], False
        scores: list[float] | None = None
        # 冷却时钟用单调钟:NTP 跳变会让 wall-clock 突进/回退,把 30s 冷却
        # 要么瞬间蒸发要么拖成无限长——降级窗口必须与时钟同步无关。
        now = time.monotonic()
        if self._dense_encoder is not None and self._dense_vectors is not None and now >= self._dense_cooldown_until:
            try:
                if hasattr(self._dense_encoder, "encode_query"):
                    raw_vec = self._dense_encoder.encode_query(query)
                elif hasattr(self._dense_encoder, "encode"):
                    raw_vec = self._dense_encoder.encode(query)
                else:
                    try:
                        # Try batch of single query
                        batch_res = self._dense_encoder([query])
                        raw_vec = batch_res[0] if isinstance(batch_res, (list, tuple)) and batch_res else batch_res
                    except Exception:
                        raw_vec = self._dense_encoder(query)
                raw_q = [float(x) for x in raw_vec]
                dim = len(self._dense_vectors[0])
                if len(raw_q) == dim and not any(math.isnan(x) or math.isinf(x) for x in raw_q):
                    scores = [self._cosine(raw_q, vector) for vector in self._dense_vectors]
                else:
                    logger.warning(
                        "Dense query dim {} != indexed dim {} or non-finite; cooldown 30s.",
                        len(raw_q), dim,
                    )
                    self._dense_cooldown_until = now + 30.0
            except Exception as exc:
                logger.warning("Dense query encoding failed ({}); cooldown 30s.", exc)
                self._dense_cooldown_until = now + 30.0

        used_dense = scores is not None
        if scores is None:
            if not allow_tfidf:
                return [], False
            fallback_docs = self._ensure_fallback_vectors()
            query_vector = self._fallback_dense_vector(query)
            scores = [self._sparse_cosine(query_vector, document_vector) for document_vector in fallback_docs]
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        hits = [(self.documents[i], scores[i]) for i in ranking[: max(0, top_n)] if scores[i] > 0]
        return hits, used_dense

    def rrf_fuse(self, list_a: Sequence[Document], list_b: Sequence[Document], top_n: int = 3) -> list[Document]:
        return [doc for doc, _ in self.rrf_fuse_with_scores(list_a, list_b, top_n)]

    def rrf_fuse_with_scores(
        self,
        list_a: Sequence[Document],
        list_b: Sequence[Document],
        top_n: int = 3,
    ) -> list[tuple[Document, float]]:
        scores: dict[str, tuple[float, Document]] = {}
        active_lists = [lst for lst in (list_a, list_b) if lst]
        if not active_lists:
            return []
        for ranked_list in active_lists:
            for rank, document in enumerate(ranked_list, start=1):
                key = document.metadata.get("chunk_id") or document.page_content
                current_score = scores.get(key, (0.0, document))[0]
                scores[key] = (current_score + 1.0 / (self.k + rank), document)
        ranked = sorted(scores.values(), key=lambda item: (-item[0], item[1].page_content))
        max_score = float(len(active_lists)) / (self.k + 1.0)
        return [(document, min(score / max_score, 1.0)) for score, document in ranked[: max(0, top_n)]]

    def retrieve(self, query: str, top_n: int = 3) -> list[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query, top_n)]

    def retrieve_with_scores(self, query: str, top_n: int = 3) -> list[tuple[Document, float]]:
        if not self.documents or top_n <= 0:
            return []
        lexical = self.bm25_search_with_scores(query, top_n=max(top_n * 2, 5))
        dense_hits, used_dense = self._dense_search_with_status(
            query, top_n=max(top_n * 2, 5), allow_tfidf=False,
        )
        if not used_dense:
            # TF-IDF and BM25 share the same lexical evidence. Fusing them as two
            # independent ranks double-counts token overlap. Neither the existing
            # BM25 normalization nor RRF ranks are calibrated medical probabilities.
            # Use score normalized to the strongest BM25 hit until genuine dense
            # embeddings are available, then mark copied results transparently.
            max_lexical = lexical[0][1] if lexical else 0.0
            result = [
                (
                    Document(doc.page_content, {**doc.metadata, "retrieval_degraded": True}),
                    score / max_lexical if max_lexical > 0 else 0.0,
                )
                for doc, score in lexical[:top_n]
            ]
            logger.info("Lexical fallback recalled {} documents (dense unavailable).", len(result))
            return result

        lexical_docs = [doc for doc, _ in lexical]
        dense = [doc for doc, _ in dense_hits]
        result = [
            (
                Document(doc.page_content, {k: v for k, v in doc.metadata.items() if k != "retrieval_degraded"})
                if "retrieval_degraded" in doc.metadata
                else doc,
                score,
            )
            for doc, score in self.rrf_fuse_with_scores(lexical_docs, dense, top_n=top_n)
        ]
        logger.info("Hybrid retrieval recalled {} documents after RRF fusion.", len(result))
        return result
