"""Unit tests for query-side structured metadata filtering.

Covers Requirement 7: limiting retrieval to chunks matching a structured
metadata field (e.g. ``sheet_name``) via the generic ``filters`` channel, and
carrying those structured fields through results.

Strategy decision (see HybridSearch._apply_metadata_filters): structured fields
listed in ``_STRUCTURED_FILTER_KEYS`` use a STRICT missing-key policy
(missing -> exclude) so filtering by ``sheet_name`` returns only matching
chunks (Requirement 7.1), while generic keys keep the lenient policy.

Validates: Requirements 7.1, 7.2, 7.3
"""
from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.core.query_engine.fusion import ReciprocalRankFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor


class FakeRetriever:
    """Returns preset RetrievalResults (with metadata) regardless of input."""

    def __init__(self, results=None, kind="dense"):
        self._results = results or []
        self._kind = kind
        self.called_with = None

    def retrieve(self, query_or_keywords, top_k=20, filters=None, trace=None):
        self.called_with = {"arg": query_or_keywords, "top_k": top_k, "filters": filters}
        return self._results


def _r(cid, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=0.0, text=text, metadata=meta or {})


def _hybrid(dense_results, sparse_results=None):
    return HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=FakeRetriever(dense_results, kind="dense"),
        sparse_retriever=FakeRetriever(sparse_results or [], kind="sparse"),
        fusion=ReciprocalRankFusion(k=60),
    )


@pytest.mark.unit
class TestSheetNameFiltering:
    """Requirement 7.1 — filter by sheet_name returns only matching chunks."""

    def test_only_matching_sheet_returned(self):
        candidates = [
            _r("a", text="row a", meta={"sheet_name": "Sheet1", "is_table": True}),
            _r("b", text="row b", meta={"sheet_name": "Sheet2", "is_table": True}),
            _r("c", text="row c", meta={"sheet_name": "Sheet1", "is_table": True}),
        ]
        hs = _hybrid(candidates)
        results = hs.search("q", top_k=10, filters={"sheet_name": "Sheet1"})
        ids = {r.chunk_id for r in results}
        assert ids == {"a", "c"}
        assert all(r.metadata["sheet_name"] == "Sheet1" for r in results)

    def test_result_carries_structured_metadata(self):
        # Requirement 7.2 — sheet_name / row range carried through results.
        candidates = [
            _r(
                "a",
                meta={
                    "sheet_name": "Sheet1",
                    "row_start": 1,
                    "row_end": 10,
                    "is_table": True,
                },
            ),
        ]
        hs = _hybrid(candidates)
        results = hs.search("q", filters={"sheet_name": "Sheet1"})
        assert len(results) == 1
        meta = results[0].metadata
        assert meta["sheet_name"] == "Sheet1"
        assert meta["row_start"] == 1
        assert meta["row_end"] == 10

    def test_missing_sheet_name_excluded_strict(self):
        # Strict: a candidate lacking sheet_name is excluded when filtering by it.
        candidates = [
            _r("a", meta={"sheet_name": "Sheet1"}),
            _r("b", meta={}),  # no sheet_name -> excluded (strict)
            _r("c", meta={"doc_type": "pdf"}),  # unrelated key -> excluded
        ]
        hs = _hybrid(candidates)
        results = hs.search("q", filters={"sheet_name": "Sheet1"})
        ids = {r.chunk_id for r in results}
        assert ids == {"a"}


@pytest.mark.unit
class TestNoFilterUnchanged:
    """Requirement 7.3 — no filter means behavior is unchanged (all returned)."""

    def test_all_returned_without_filters(self):
        candidates = [
            _r("a", meta={"sheet_name": "Sheet1"}),
            _r("b", meta={"sheet_name": "Sheet2"}),
            _r("c", meta={}),
        ]
        hs = _hybrid(candidates)
        results = hs.search("q", top_k=10)
        ids = {r.chunk_id for r in results}
        assert ids == {"a", "b", "c"}


@pytest.mark.unit
class TestApplyMetadataFiltersDirect:
    """Direct unit tests for the filter helper (no orchestration)."""

    def test_structured_missing_excluded(self):
        candidates = [
            _r("a", meta={"sheet_name": "S1"}),
            _r("b", meta={}),
        ]
        out = HybridSearch._apply_metadata_filters(candidates, {"sheet_name": "S1"})
        assert {c.chunk_id for c in out} == {"a"}

    def test_generic_missing_included(self):
        candidates = [
            _r("a", meta={"doc_type": "pdf"}),
            _r("b", meta={}),  # missing generic key -> lenient include
        ]
        out = HybridSearch._apply_metadata_filters(candidates, {"doc_type": "pdf"})
        assert {c.chunk_id for c in out} == {"a", "b"}

    def test_present_but_different_excluded(self):
        candidates = [
            _r("a", meta={"sheet_name": "S1"}),
            _r("b", meta={"sheet_name": "S2"}),
        ]
        out = HybridSearch._apply_metadata_filters(candidates, {"sheet_name": "S1"})
        assert {c.chunk_id for c in out} == {"a"}

    def test_empty_filters_returns_all(self):
        candidates = [_r("a", meta={"sheet_name": "S1"}), _r("b", meta={})]
        out = HybridSearch._apply_metadata_filters(candidates, {})
        assert {c.chunk_id for c in out} == {"a", "b"}
