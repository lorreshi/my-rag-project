"""Relevance threshold / abstain gate (#11) — generic baseline, default-off.

A final gate: if the top result's score is below ``min_score_threshold`` the
whole result set is dropped (return empty), so the upstream MCP/LLM gets an
explicit "not covered" signal instead of confidently-wrong context built from
low-relevance passages.

This is meaningful only once scores are on one comparable scale (see the
reranker score unification, #6). Calibrating a good threshold value is deferred
to the evaluation phase; here we only provide the mechanism. ``threshold <= 0``
disables the gate (unchanged behaviour).
"""
from __future__ import annotations

from src.core.types import RetrievalResult


def apply_threshold(
    results: list[RetrievalResult], threshold: float
) -> list[RetrievalResult]:
    """Return *results* unless the top score is below *threshold* (then [])."""
    if threshold <= 0 or not results:
        return results
    return results if results[0].score >= threshold else []
