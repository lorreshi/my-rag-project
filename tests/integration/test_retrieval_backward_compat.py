"""T15 integration — backward compatibility: all enhancements OFF == baseline.

Validates Property 9 (进阶增强默认关 = 行为不变): with ``query_transform=none``,
``enable_synonym_expansion=False``, MMR disabled and ``min_score_threshold<=0``,
the pipeline output is item-for-item identical to the #1–#7 baseline.

Uses the real QueryProcessor + ReciprocalRankFusion with configurable fake
dense/sparse retrievers (same style as ``test_hybrid_search.py``).
"""
from __future__ import annotations

from src.core.types import RetrievalResult
from src.core.query_engine.fusion import ReciprocalRankFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.query_transform import NoOpTransform
from src.core.query_engine.diversity import apply_mmr, mmr_rerank
from src.core.query_engine.threshold import apply_threshold


class FakeRetriever:
    """Configurable fake for dense/sparse retrievers (accepts extra kwargs)."""

    def __init__(self, results=None, raises=False, kind="dense"):
        self._results = results or []
        self._raises = raises
        self._kind = kind
        self.called_with = None

    def retrieve(self, query_or_keywords, top_k=20, filters=None, **kwargs):
        self.called_with = {
            "arg": query_or_keywords,
            "top_k": top_k,
            "filters": filters,
            **kwargs,
        }
        if self._raises:
            raise RuntimeError(f"{self._kind} boom")
        return list(self._results)


def _r(cid, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=0.0, text=text, metadata=meta or {})


def _baseline_hybrid(dense, sparse, **kwargs):
    """A minimal baseline HybridSearch (no enhancements supplied)."""
    return HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=ReciprocalRankFusion(k=60),
        **kwargs,
    )


class TestAllOffEqualsBaseline:
    def test_noop_transform_single_dense_list(self):
        dense = FakeRetriever([_r("a"), _r("b")], kind="dense")
        sparse = FakeRetriever([_r("b"), _r("c")], kind="sparse")
        hs = _baseline_hybrid(dense, sparse, query_transform=NoOpTransform())
        from src.core.trace.trace_context import TraceContext

        trace = TraceContext(trace_type="query")
        hs.search("vector search query", top_k=10, trace=trace)
        stage = {s.name: s for s in trace.stages}["hybrid_search"]
        # query_transform=none -> exactly one dense list, no degradation.
        assert stage.details["dense_lists"] == 1
        assert stage.details["query_transform_degraded"] is False

    def test_explicit_noop_matches_implicit_default(self):
        """Supplying NoOpTransform == not supplying any transform (default)."""
        dense_a = FakeRetriever([_r("a"), _r("b"), _r("c")], kind="dense")
        sparse_a = FakeRetriever([_r("b"), _r("d")], kind="sparse")
        explicit = _baseline_hybrid(dense_a, sparse_a, query_transform=NoOpTransform())

        dense_b = FakeRetriever([_r("a"), _r("b"), _r("c")], kind="dense")
        sparse_b = FakeRetriever([_r("b"), _r("d")], kind="sparse")
        implicit = _baseline_hybrid(dense_b, sparse_b)  # default NoOp internally

        q = "configure azure endpoint"
        ids_explicit = [x.chunk_id for x in explicit.search(q, top_k=5)]
        ids_implicit = [x.chunk_id for x in implicit.search(q, top_k=5)]
        assert ids_explicit == ids_implicit

    def test_synonym_off_uses_raw_keywords(self):
        dense = FakeRetriever([], kind="dense")
        sparse = FakeRetriever([_r("a")], kind="sparse")
        hs = _baseline_hybrid(dense, sparse, enable_synonym_expansion=False)
        hs.search("configure azure endpoint")
        # Sparse route received the raw keyword list (== ProcessedQuery.keywords).
        processed = QueryProcessor().process("configure azure endpoint")
        assert sparse.called_with["arg"] == processed.keywords

    def test_default_candidate_width_equiv_topk_times_two(self):
        """Defaults (multiplier=2, top_k_dense/sparse=20) -> max(top_k,20)*2."""
        dense = FakeRetriever([_r("a")], kind="dense")
        sparse = FakeRetriever([_r("a")], kind="sparse")
        hs = _baseline_hybrid(dense, sparse)
        hs.search("q", top_k=10)
        assert dense.called_with["top_k"] == 40  # max(10,20)*2
        assert sparse.called_with["top_k"] == 40


class TestFilterPolicyUnchanged:
    def test_structured_field_strict_missing_excluded(self):
        dense = FakeRetriever(
            [
                _r("a", meta={"sheet_name": "Q1"}),
                _r("b", meta={}),  # missing structured key -> STRICT exclude
            ],
            kind="dense",
        )
        sparse = FakeRetriever([], kind="sparse")
        hs = _baseline_hybrid(dense, sparse)
        ids = {x.chunk_id for x in hs.search("q", filters={"sheet_name": "Q1"})}
        assert ids == {"a"}

    def test_generic_field_lenient_missing_included(self):
        dense = FakeRetriever(
            [
                _r("a", meta={"doc_type": "pdf"}),
                _r("b", meta={}),  # missing generic key -> LENIENT include
            ],
            kind="dense",
        )
        sparse = FakeRetriever([], kind="sparse")
        hs = _baseline_hybrid(dense, sparse)
        ids = {x.chunk_id for x in hs.search("q", filters={"doc_type": "pdf"})}
        assert ids == {"a", "b"}


class TestDisabledGatesAreIdentity:
    def test_threshold_zero_is_identity(self):
        results = [_r("a"), _r("b")]
        results[0].score, results[1].score = 0.9, 0.1
        assert apply_threshold(results, 0.0) == results

    def test_threshold_negative_is_identity(self):
        results = [_r("a")]
        results[0].score = 0.5
        assert apply_threshold(results, -1.0) == results

    def test_mmr_lambda_one_is_identity(self):
        results = [_r("a", text="x"), _r("b", text="y")]
        out = mmr_rerank(results, [1.0, 0.0], {"a": [1.0, 0.0], "b": [0.0, 1.0]}, lambda_=1.0)
        assert [r.chunk_id for r in out] == ["a", "b"]

    def test_mmr_empty_input_identity(self):
        assert apply_mmr([], "q", embed_fn=lambda t: [[0.0]]) == []
