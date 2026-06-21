"""Tests for T6: shared match_filters + SparseRetriever pre-filter.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""
from __future__ import annotations

import pytest

from src.core.query_engine.metadata_filter import (
    STRUCTURED_FILTER_KEYS,
    match_filters,
)
from src.core.query_engine.sparse_retriever import SparseRetriever


# --- Shared match_filters predicate ----------------------------------------

@pytest.mark.unit
class TestMatchFilters:
    def test_no_filters_keeps(self):
        assert match_filters({"a": 1}, None) is True
        assert match_filters({"a": 1}, {}) is True

    def test_present_equal_keeps(self):
        assert match_filters({"collection": "x"}, {"collection": "x"}) is True

    def test_present_unequal_excludes(self):
        assert match_filters({"collection": "x"}, {"collection": "y"}) is False

    def test_generic_missing_is_lenient(self):
        # generic key missing -> include
        assert match_filters({}, {"collection": "x"}) is True

    def test_structured_missing_is_strict(self):
        # structured key missing -> exclude
        assert match_filters({}, {"sheet_name": "Employees"}) is False

    def test_structured_present_equal(self):
        assert match_filters({"sheet_name": "E"}, {"sheet_name": "E"}) is True
        assert "sheet_name" in STRUCTURED_FILTER_KEYS


# --- SparseRetriever pre-filter --------------------------------------------

class FakeBM25:
    def __init__(self, scored):
        self._scored = scored

    def query(self, keywords, top_k):
        return self._scored[:top_k]


class FakeStore:
    def __init__(self, records):
        self._records = {r["id"]: r for r in records}

    def get_by_ids(self, ids):
        return [self._records[i] for i in ids if i in self._records]


def _retriever(scored, records):
    return SparseRetriever(
        bm25_indexer=FakeBM25(scored),
        vector_store=FakeStore(records),
        auto_load=False,
    )


@pytest.mark.unit
class TestSparsePreFilter:
    def _data(self):
        scored = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
        records = [
            {"id": "a", "text": "A", "metadata": {"collection": "x", "sheet_name": "S1"}},
            {"id": "b", "text": "B", "metadata": {"collection": "y", "sheet_name": "S2"}},
            {"id": "c", "text": "C", "metadata": {"collection": "x"}},  # no sheet_name
        ]
        return scored, records

    def test_no_filters_returns_all(self):
        scored, records = self._data()
        out = _retriever(scored, records).retrieve(["q"], top_k=10)
        assert [r.chunk_id for r in out] == ["a", "b", "c"]

    def test_generic_filter_lenient_on_missing(self):
        scored, records = self._data()
        # collection=x matches a, c; b excluded (present, unequal)
        out = _retriever(scored, records).retrieve(["q"], top_k=10, filters={"collection": "x"})
        assert [r.chunk_id for r in out] == ["a", "c"]

    def test_structured_filter_strict_on_missing(self):
        scored, records = self._data()
        # sheet_name=S1 -> only a (c missing sheet_name -> excluded; b unequal)
        out = _retriever(scored, records).retrieve(["q"], top_k=10, filters={"sheet_name": "S1"})
        assert [r.chunk_id for r in out] == ["a"]

    def test_truncates_to_top_k_after_filter(self):
        scored = [(c, float(10 - i)) for i, c in enumerate("abcde")]
        records = [{"id": c, "text": c, "metadata": {"collection": "x"}} for c in "abcde"]
        out = _retriever(scored, records).retrieve(["q"], top_k=2, filters={"collection": "x"})
        assert [r.chunk_id for r in out] == ["a", "b"]

    def test_overfetch_used_only_when_filtering(self):
        captured = {}

        class SpyBM25(FakeBM25):
            def query(self, keywords, top_k):
                captured["top_k"] = top_k
                return super().query(keywords, top_k)

        scored = [(c, 1.0) for c in "abcdefghij"]
        records = [{"id": c, "text": c, "metadata": {"collection": "x"}} for c in "abcdefghij"]
        r = SparseRetriever(bm25_indexer=SpyBM25(scored), vector_store=FakeStore(records), auto_load=False)
        # without filters -> fetch exactly top_k
        r.retrieve(["q"], top_k=3)
        assert captured["top_k"] == 3
        # with filters -> over-fetch top_k * overfetch
        r.retrieve(["q"], top_k=3, filters={"collection": "x"}, overfetch=4)
        assert captured["top_k"] == 12

    def test_empty_keywords(self):
        scored, records = self._data()
        assert _retriever(scored, records).retrieve([], top_k=5) == []
