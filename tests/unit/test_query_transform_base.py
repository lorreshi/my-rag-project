"""Tests for T10: QueryTransform base + NoOp + factory + HybridSearch wiring.

Validates: Requirements 8.1, 8.6, 12.1
"""
from __future__ import annotations

import types

import pytest

from src.core.query_engine.fusion import ReciprocalRankFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_transform import (
    BaseQueryTransform,
    NoOpTransform,
    TransformedQuery,
)
from src.core.query_engine.query_transform_factory import QueryTransformFactory
from src.core.settings import RetrievalConfig, Settings
from src.core.types import RetrievalResult


@pytest.mark.unit
class TestNoOpTransform:
    def test_returns_single_original_query(self):
        tq = NoOpTransform().transform("hello world")
        assert tq.dense_queries == ["hello world"]
        assert tq.used_llm is False
        assert tq.degraded is False

    def test_is_base(self):
        assert isinstance(NoOpTransform(), BaseQueryTransform)


@pytest.mark.unit
class TestFactory:
    def test_default_none_is_noop(self):
        s = Settings()
        s.retrieval = RetrievalConfig(query_transform="none")
        assert isinstance(QueryTransformFactory.create(s), NoOpTransform)

    def test_missing_retrieval_defaults_noop(self):
        class _S:
            pass

        assert isinstance(QueryTransformFactory.create(_S()), NoOpTransform)

    def test_unknown_raises(self):
        s = Settings()
        s.retrieval = RetrievalConfig(query_transform="bogus")
        with pytest.raises(ValueError, match="Unknown query_transform"):
            QueryTransformFactory.create(s)


# --- HybridSearch multi-list wiring ----------------------------------------

class FakeQP:
    def process(self, query, filters=None, trace=None):
        return types.SimpleNamespace(
            normalized_query=query, keywords=[query], expanded_keywords=[query],
            filters=filters or {},
        )

    def normalize_for_dense(self, query):
        return query


class RecordingDense:
    def __init__(self):
        self.calls = []

    def retrieve(self, text, top_k=20, filters=None, trace=None):
        self.calls.append(text)
        return [RetrievalResult(chunk_id=f"{text}:0", score=1.0, text=text, metadata={})]


class NoopSparse:
    def retrieve(self, keywords, top_k=20, filters=None, overfetch=4, trace=None):
        return []


class TwoVariantTransform(BaseQueryTransform):
    def transform(self, query, trace=None):
        return TransformedQuery(dense_queries=[query, query + " v2"], used_llm=True)


@pytest.mark.unit
class TestHybridMultiList:
    def test_noop_single_dense_call(self):
        dense = RecordingDense()
        hs = HybridSearch(FakeQP(), dense, NoopSparse(), ReciprocalRankFusion())
        hs.search("q", top_k=5)
        assert dense.calls == ["q"]  # one dense retrieval (baseline)

    def test_multi_variant_runs_dense_per_variant(self):
        dense = RecordingDense()
        hs = HybridSearch(
            FakeQP(), dense, NoopSparse(), ReciprocalRankFusion(),
            query_transform=TwoVariantTransform(),
        )
        results = hs.search("q", top_k=5)
        assert dense.calls == ["q", "q v2"]  # one retrieval per variant
        assert len(results) >= 1

    def test_n1_equivalent_to_baseline(self):
        # NoOp (N=1) must equal the plain two-list fusion of dense+sparse.
        dense = RecordingDense()
        hs = HybridSearch(FakeQP(), dense, NoopSparse(), ReciprocalRankFusion())
        out = hs.search("q", top_k=5)
        assert [r.chunk_id for r in out] == ["q:0"]
