from __future__ import annotations

import difflib
import os
import threading
from collections.abc import Sequence

from medcopilot.common.logger import logger


class OCRTool:
    """Drug-box OCR wrapper with deterministic degradation.

    PaddleOCR is loaded lazily; when it (or the image) is unavailable the tool
    returns empty results instead of crashing the consultation workflow — the
    failure mode of the original notebook was a hard crash inside Gradio.
    """

    def __init__(self) -> None:
        self._engine = None
        self._engine_failed = False
        # PaddleOCR 构造重且慢(下载/初始化数秒):并发首请求必须只构造一次,
        # 否则每个线程各持一份模型副本直到进程结束(双检锁)。
        self._engine_lock = threading.Lock()
        # Status of the most recent extract attempt: "ok" | "image_unreadable"
        # | "engine_unavailable" | "inference_error". None before the first
        # call (e.g. when a test double replaces suggest_drug_from_image).
        # The three former silent `return []` paths are distinct failure modes
        # the API layer must be able to surface as ocr_status.
        # 仅供诊断/测试观测:请求路径以返回值为准,并发下最后写入者不定。
        self.last_status: str | None = None

    @property
    def is_available(self) -> bool:
        return self._engine is not None

    def _load_engine(self):
        if self._engine is None and not self._engine_failed:
            # 双检锁:构造在锁内且成功后重判,并发首请求只构造一次引擎。
            with self._engine_lock:
                if self._engine is None and not self._engine_failed:
                    try:
                        from paddleocr import PaddleOCR

                        try:
                            self._engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                        except TypeError:
                            # newer PaddleOCR releases removed ``show_log``
                            self._engine = PaddleOCR(use_angle_cls=True, lang="ch")
                    except Exception as exc:  # pragma: no cover - depends on optional dep
                        logger.warning("PaddleOCR unavailable ({}); OCR disabled for this session.", exc)
                        self._engine_failed = True
        return self._engine

    def extract_text_with_status(self, image_path: str | None) -> tuple[list[str], str]:
        """Extract text plus an honest status for the API layer.

        The three non-"ok" statuses were previously flattened into the same
        empty-list result as "engine ran but the box has no text", hiding real
        failures behind an indistinguishable empty answer.
        """
        # P2-12:状态是本次调用的局部值;last_status 只是并发下的诊断镜像,
        # 请求路径永远以返回值为准(API 层的 ocr_status 即来自这里)。
        if not image_path or not os.path.exists(str(image_path)):
            logger.warning("OCR skipped: image path is empty or nonexistent.")
            status = "image_unreadable"
            self.last_status = status
            return [], status
        engine = self._load_engine()
        if engine is None:
            status = "engine_unavailable"
            self.last_status = status
            return [], status
        try:
            try:
                result = engine.ocr(str(image_path), cls=True)
            except TypeError:
                result = engine.ocr(str(image_path))
        except Exception as exc:
            logger.error("OCR inference failed: {}", exc)
            status = "inference_error"
            self.last_status = status
            return [], status

        lines: list[str] = []
        for page in result or []:
            if isinstance(page, dict) and "rec_text" in page:
                lines.append(str(page["rec_text"]))
                continue
            for line in page or []:
                if isinstance(line, dict):
                    text = line.get("text") or line.get("rec_text")
                    if text:
                        lines.append(str(text))
                    continue
                try:
                    lines.append(str(line[1][0]))
                except (TypeError, IndexError):
                    continue
        logger.info("OCR extracted {} text lines.", len(lines))
        status = "ok"
        self.last_status = status
        return lines, status

    def extract_text(self, image_path: str | None) -> list[str]:
        lines, _ = self.extract_text_with_status(image_path)
        return lines

    @staticmethod
    def match_drug_name(
        text_lines: Sequence[str],
        drug_dictionary: Sequence[str],
        cutoff: float = 0.45,
    ) -> str | None:
        """Fuzzy-match OCR noise against a known drug dictionary.

        Exact substrings win immediately (longest match wins when multiple match,
        preventing generic sub-words like '头孢' from masking '头孢克肟'); otherwise
        difflib similarity repairs common OCR errors (e.g. 阿莫西杯 -> 阿莫西林).
        """
        if not text_lines or not drug_dictionary:
            return None
        corpus = "".join(text_lines)
        best_name: str | None = None
        best_score = 0.0
        best_len = 0
        second_score = 0.0
        for name in drug_dictionary:
            if not name:
                continue
            if name in corpus:
                score = 1.0
            else:
                # Compare against each individual text line and substrings of matching length
                line_scores = []
                n_len = len(name)
                for line in text_lines:
                    if not line:
                        continue
                    line_scores.append(difflib.SequenceMatcher(None, line, name).ratio())
                    if len(line) >= n_len:
                        for i in range(len(line) - n_len + 1):
                            window = line[i : i + n_len]
                            line_scores.append(difflib.SequenceMatcher(None, window, name).ratio())
                score = max(line_scores) if line_scores else difflib.SequenceMatcher(None, corpus, name).ratio()
            # When scores are tied (especially exact substring score 1.0), favor the longer/more specific drug name
            if score > best_score or (abs(score - best_score) < 1e-6 and len(name) > best_len):
                second_score = best_score
                best_name, best_score, best_len = name, score, len(name)
            elif score > second_score:
                second_score = score
        if best_name is None or best_score < cutoff:
            logger.info("No drug matched above cutoff {:.2f} (best={:.2f}).", cutoff, best_score)
            return None
        # Fuzzy (non-exact) match with an ambiguous runner-up within 0.15 is unsafe for
        # clinical context — treat as unrecognized instead of a confident lookup.
        if best_score < 1.0 and (best_score - second_score) < 0.15:
            logger.info(
                "Drug match ambiguous: '{}' {:.2f} vs runner-up {:.2f}; returning None.",
                best_name, best_score, second_score,
            )
            return None
        logger.info("Matched drug '{}' with score {:.2f}.", best_name, best_score)
        return best_name

    def suggest_drug_from_image(
        self,
        image_path: str | None,
        drug_dictionary: Sequence[str],
    ) -> tuple[list[str], str | None, str]:
        """Return ``(lines, matched_drug, status)``.

        Status travels with the return value rather than only on
        ``last_status``: the API layer reads it in the same request, so a
        concurrent with-image request can never overwrite another request's
        failure with its own "ok".
        """
        lines, status = self.extract_text_with_status(image_path)
        matched = self.match_drug_name(lines, drug_dictionary)
        return lines, matched, status
