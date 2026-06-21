"""MMR diversity re-ranking (#10) — optional, default-off.

Maximal Marginal Relevance trades relevance against redundancy::

    next = argmax_d [ λ · sim(d, query) − (1 − λ) · max_{s∈selected} sim(d, s) ]

Applied after reranking to reduce near-duplicate / same-entity clutter (e.g.
the 刘静/周静 same-name interference, repeated table rows). ``λ = 1`` reduces to
the input (relevance) order, i.e. identity. With MMR disabled the caller simply
does not invoke this module (no behaviour change).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_rerank(
    results: list[RetrievalResult],
    query_vector: list[float],
    vectors: dict[str, list[float]],
    lambda_: float = 0.5,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Greedy MMR re-ranking over *results* using precomputed *vectors*.

    Args:
        results: Input results (relevance order).
        query_vector: Embedding of the query.
        vectors: Map chunk_id -> embedding for every result.
        lambda_: Relevance/diversity trade-off in [0, 1]. ``>= 1`` => identity.
        top_k: Optional cap on returned results.

    Returns:
        Re-ordered results. If any result lacks a vector, MMR is skipped and the
        input order is returned (degrade, never raise).
    """
    if not results:
        return results
    limit = top_k if top_k is not None else len(results)
    if lambda_ >= 1.0:
        return results[:limit]
    if any(r.chunk_id not in vectors for r in results):
        logger.warning("MMR skipped: missing candidate vector(s); keeping order")
        return results[:limit]

    qv = np.asarray(query_vector, dtype=float)
    vmap = {cid: np.asarray(v, dtype=float) for cid, v in vectors.items()}

    pool = list(results)
    selected: list[RetrievalResult] = []
    while pool and len(selected) < limit:
        best, best_score = None, float("-inf")
        for d in pool:
            rel = _cosine(vmap[d.chunk_id], qv)
            red = max(
                (_cosine(vmap[d.chunk_id], vmap[s.chunk_id]) for s in selected),
                default=0.0,
            )
            score = lambda_ * rel - (1.0 - lambda_) * red
            if score > best_score:
                best, best_score = d, score
        selected.append(best)
        pool.remove(best)
    return selected


def apply_mmr(
    results: list[RetrievalResult],
    query: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    lambda_: float = 0.5,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Embed the query + result texts via *embed_fn* and apply MMR.

    Reuses ``metadata['dense_vector']`` when present; otherwise embeds the
    result text. On any failure returns *results* unchanged (degrade).
    """
    if not results:
        return results
    try:
        need_texts: list[str] = []
        need_ids: list[str] = []
        vectors: dict[str, list[float]] = {}
        for r in results:
            v = r.metadata.get("dense_vector")
            if v is not None:
                vectors[r.chunk_id] = v
            else:
                need_ids.append(r.chunk_id)
                need_texts.append(r.text or "")
        embedded = embed_fn([query, *need_texts]) if need_texts else embed_fn([query])
        query_vector = embedded[0]
        for cid, vec in zip(need_ids, embedded[1:]):
            vectors[cid] = vec
        return mmr_rerank(results, query_vector, vectors, lambda_=lambda_, top_k=top_k)
    except Exception as exc:
        logger.warning("MMR failed; returning original order: %s", exc)
        return results[:top_k] if top_k is not None else results
