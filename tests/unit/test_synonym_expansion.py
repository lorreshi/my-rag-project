"""Tests for T8: synonym/alias OR-expansion into BM25.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""
from __future__ import annotations

import json
import types

import pytest

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import ProcessedQuery, QueryProcessor


@pytest.mark.unit
class TestExpandedKeywords:
    def test_field_exists_and_defaults(self):
        pq = ProcessedQuery(raw_query="q", normalized_query="q")
        assert pq.expanded_keywords == []

    def test_no_synonyms_equals_keywords(self):
        qp = QueryProcessor()  # no synonym map
        result = qp.process("hybrid search")
        assert result.expanded_keywords == result.keywords

    def test_expansion_appends_alias_tokens(self):
        qp = QueryProcessor(synonym_map={"app": ["application"]})
        result = qp.process("app")
        assert result.keywords == ["app"]
        assert result.expanded_keywords == ["app", "application"]

    def test_prefix_is_keywords_and_dedup(self):
        qp = QueryProcessor(synonym_map={"app": ["app", "application", "application"]})
        result = qp.process("app program")
        # prefix equals keywords; aliases deduped & order preserved
        assert result.expanded_keywords[: len(result.keywords)] == result.keywords
        assert len(result.expanded_keywords) == len(set(result.expanded_keywords))


class FakeQP:
    """Returns a fixed ProcessedQuery so we can observe which keyword list is used."""

    def __init__(self):
        self.pq = ProcessedQuery(
            raw_query="q",
            normalized_query="q",
            keywords=["base"],
            expanded_keywords=["base", "alias"],
            filters={},
        )

    def process(self, query, filters=None, trace=None):
        return self.pq

    def normalize_for_dense(self, query):
        return query


class RecordingSparse:
    def __init__(self):
        self.keywords = None

    def retrieve(self, keywords, top_k=20, filters=None, overfetch=4, trace=None):
        self.keywords = keywords
        return []


class NoopDense:
    def retrieve(self, *a, **k):
        return []


@pytest.mark.unit
class TestHybridUsesExpansionFlag:
    def _hs(self, enable):
        sparse = RecordingSparse()
        hs = HybridSearch(
            FakeQP(), NoopDense(), sparse,
            __import__("src.core.query_engine.fusion", fromlist=["ReciprocalRankFusion"]).ReciprocalRankFusion(),
            enable_synonym_expansion=enable,
        )
        return hs, sparse

    def test_expansion_off_uses_plain_keywords(self):
        hs, sparse = self._hs(enable=False)
        hs.search("q", top_k=5)
        assert sparse.keywords == ["base"]

    def test_expansion_on_uses_expanded_keywords(self):
        hs, sparse = self._hs(enable=True)
        hs.search("q", top_k=5)
        assert sparse.keywords == ["base", "alias"]


@pytest.mark.unit
class TestSynonymLoading:
    def test_missing_file_degrades_to_empty(self):
        assert HybridSearch._load_synonyms("/no/such/file.json") == {}

    def test_empty_path_returns_empty(self):
        assert HybridSearch._load_synonyms("") == {}

    def test_loads_valid_json(self, tmp_path):
        p = tmp_path / "syn.json"
        p.write_text(json.dumps({"app": ["application"]}), encoding="utf-8")
        assert HybridSearch._load_synonyms(str(p)) == {"app": ["application"]}

    def test_non_dict_json_ignored(self, tmp_path):
        p = tmp_path / "syn.json"
        p.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        assert HybridSearch._load_synonyms(str(p)) == {}
