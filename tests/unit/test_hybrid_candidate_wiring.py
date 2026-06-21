"""Tests for T5: HybridSearch candidate-pool wiring + FusionFactory assembly.

Validates: Requirements 3.1, 3.3, 4.3
"""
from __future__ import annotations

import types

import pytest

from src.core.query_engine.fusion import ReciprocalRankFusion, WeightedSumFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.settings import RetrievalConfig, Settings


class FakeQP:
    def process(self, query, filters=None, trace=None):
        return types.SimpleNamespace(
            normalized_query=query, keywords=[query], filters=filters or {},
            expanded_keywords=[query],
        )

    def normalize_for_dense(self, query):
        return query


class RecordingDense:
    def __init__(self):
        self.top_k = None

    def retrieve(self, query, top_k=20, filters=None, trace=None):
        self.top_k = top_k
        return []


class RecordingSparse:
    def __init__(self):
        self.top_k = None

    def retrieve(self, keywords, top_k=20, trace=None, **kwargs):
        self.top_k = top_k
        return []


def _settings(**retrieval_kwargs):
    s = Settings()
    s.retrieval = RetrievalConfig(**retrieval_kwargs)
    return s


@pytest.mark.unit
class TestCandidateWiring:
    def test_candidate_widths_from_config(self):
        dense, sparse = RecordingDense(), RecordingSparse()
        hs = HybridSearch(
            FakeQP(), dense, sparse, ReciprocalRankFusion(),
            candidate_multiplier=3, top_k_dense=20, top_k_sparse=40,
        )
        hs.search("q", top_k=5)
        assert dense.top_k == max(5, 20) * 3   # 60
        assert sparse.top_k == max(5, 40) * 3  # 120

    def test_top_k_larger_than_config_wins(self):
        dense, sparse = RecordingDense(), RecordingSparse()
        hs = HybridSearch(
            FakeQP(), dense, sparse, ReciprocalRankFusion(),
            candidate_multiplier=2, top_k_dense=20, top_k_sparse=20,
        )
        hs.search("q", top_k=50)
        assert dense.top_k == 50 * 2
        assert sparse.top_k == 50 * 2

    def test_config_width_dominates_when_top_k_small(self):
        # top_k(10) < top_k_dense(20): the config width now actually applies
        # (legacy ignored top_k_dense and used top_k*2=20). This is the fix.
        dense, sparse = RecordingDense(), RecordingSparse()
        hs = HybridSearch(FakeQP(), dense, sparse, ReciprocalRankFusion())
        hs.search("q", top_k=10)
        assert dense.top_k == max(10, 20) * 2   # 40
        assert sparse.top_k == max(10, 20) * 2  # 40


@pytest.mark.unit
class TestFromSettingsAssembly:
    def _overrides(self):
        return dict(
            query_processor=FakeQP(),
            dense_retriever=RecordingDense(),
            sparse_retriever=RecordingSparse(),
        )

    def test_uses_rrf_by_default(self):
        hs = HybridSearch.from_settings(_settings(fusion_algorithm="rrf"), **self._overrides())
        assert isinstance(hs._fusion, ReciprocalRankFusion)

    def test_uses_weighted_sum_when_configured(self):
        hs = HybridSearch.from_settings(
            _settings(fusion_algorithm="weighted_sum"), **self._overrides()
        )
        assert isinstance(hs._fusion, WeightedSumFusion)

    def test_candidate_config_propagated(self):
        hs = HybridSearch.from_settings(
            _settings(candidate_multiplier=4, top_k_dense=15, top_k_sparse=25),
            **self._overrides(),
        )
        assert hs._multiplier == 4
        assert hs._top_k_dense == 15
        assert hs._top_k_sparse == 25
