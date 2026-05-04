"""Cross-Encoder Reranker implementation.

Scores each (query, candidate) pair using a cross-encoder model and sorts
by descending score. The actual scorer is injectable for testability.

In production, the scorer would wrap a sentence-transformers CrossEncoder
model. For now, the implementation accepts any callable scorer, and the
factory will attempt to load the model if the library is available.
"""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from src.libs.reranker.base_reranker import BaseReranker, RerankCandidate
from src.libs.reranker.reranker_factory import register_backend

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

# Type alias for the scorer function: (query, text) -> float
Scorer = Callable[[str, str], float]


class CrossEncoderReranker(BaseReranker):
    """Reranker that uses a cross-encoder model to score candidates."""

    def __init__(
        self,
        scorer: Scorer,
        timeout: float = 30.0,
    ):
        self._scorer = scorer
        self._timeout = timeout
        self._failed = False  # fallback signal for upstream

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: "TraceContext | None" = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        self._failed = False

        try:
            scored = []
            for c in candidates:
                score = self._scorer(query, c.text)
                scored.append(RerankCandidate(id=c.id, text=c.text, score=score))

            scored.sort(key=lambda c: c.score, reverse=True)
            return scored

        except Exception:
            # Signal failure so upstream can fall back to fusion ranking
            self._failed = True
            return list(candidates)

    @property
    def has_failed(self) -> bool:
        """True if the last rerank() call failed (timeout, model error, etc.).

        Upstream code (e.g. Core reranker module) can check this to decide
        whether to fall back to the fusion-stage ranking.
        """
        return self._failed

    @property
    def backend_name(self) -> str:
        return "cross_encoder"


def _create_cross_encoder(settings: Any) -> CrossEncoderReranker:
    """Factory function. Tries to load a real cross-encoder model.

    Falls back to a no-op scorer if sentence-transformers is not installed.
    """
    model_name = settings.rerank.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"

    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        model = CrossEncoder(model_name)

        def _scorer(query: str, text: str) -> float:
            return float(model.predict([(query, text)])[0])

    except ImportError:
        # sentence-transformers not installed — use a length-based stub
        def _scorer(query: str, text: str) -> float:
            # Deterministic fallback: longer overlap = higher score
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            overlap = len(q_words & t_words)
            return overlap / max(len(q_words), 1)

    return CrossEncoderReranker(scorer=_scorer)


register_backend("cross_encoder", _create_cross_encoder)
