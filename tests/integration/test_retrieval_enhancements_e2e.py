"""T16 integration — NFKC re-ingestion + multi-enhancement chaining (E2E).

Validates the most important real-link guarantees of this feature:

1. **词形对齐（symmetry）**: after re-ingesting with the normalization-enabled
   tokenizer, a query written in half-width / simplified form aligns with an
   index built from full-width / traditional text — BM25 hits do NOT regress.
2. **归一化前后命中率回归对比**: the same query MISSES against an index built
   *without* normalization, proving the normalization is what restores recall.
3. **多增强串联**: weighted RRF + sparse pre-filter + synonym expansion run
   end-to-end over the rebuilt index without error, and trace stages are sane.

The BM25 index is rebuilt in a temp dir from a small fixture corpus; the dense
route is faked (Chroma vectors are out of scope here), and chunk text/metadata
are resolved by an in-memory fake vector store.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.types import Chunk, RetrievalResult
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.tokenizer.jieba_tokenizer import JiebaTokenizer
from src.core.query_engine.fusion import ReciprocalRankFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseRetriever


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "normalization_samples.json"


def _load_chunks() -> list[Chunk]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [Chunk(id=c["id"], text=c["text"], metadata=c.get("metadata", {})) for c in data["chunks"]]


def _opencc_available() -> bool:
    try:
        import opencc  # noqa: F401

        return True
    except Exception:
        return False


class _FakeStore:
    """In-memory vector store: resolves chunk_ids -> {text, metadata}."""

    def __init__(self, chunks: list[Chunk]):
        self._by_id = {c.id: {"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks}

    def get_by_ids(self, ids):
        return [self._by_id[i] for i in ids if i in self._by_id]


class _FakeDense:
    """Dense route returning nothing (isolates the BM25 alignment assertions)."""

    def retrieve(self, text, top_k=20, filters=None, **kwargs):
        return []


def _build_index(tmp_path: Path, chunks: list[Chunk], *, nfkc: bool, to_simplified: bool = False) -> BM25Indexer:
    tokenizer = JiebaTokenizer(nfkc=nfkc, to_simplified=to_simplified)
    encoder = SparseEncoder(tokenizer=tokenizer)
    sparse_vectors = encoder.encode(chunks)
    indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    indexer.build(sparse_vectors)
    return indexer


def _hybrid_over_index(
    indexer: BM25Indexer,
    chunks: list[Chunk],
    *,
    nfkc: bool,
    to_simplified: bool = False,
    synonym_map: dict | None = None,
    enable_synonym: bool = False,
) -> HybridSearch:
    qp = QueryProcessor(
        tokenizer=JiebaTokenizer(nfkc=nfkc, to_simplified=to_simplified),
        nfkc=nfkc,
        to_simplified=to_simplified,
        synonym_map=synonym_map,
    )
    sparse = SparseRetriever(bm25_indexer=indexer, vector_store=_FakeStore(chunks))
    return HybridSearch(
        query_processor=qp,
        dense_retriever=_FakeDense(),
        sparse_retriever=sparse,
        fusion=ReciprocalRankFusion(k=60),
        enable_synonym_expansion=enable_synonym,
    )


class TestNFKCAlignment:
    def test_halfwidth_query_hits_fullwidth_index_with_nfkc(self, tmp_path):
        """Index has full-width 'ＡＰＩ'; half-width 'API' query hits via NFKC."""
        chunks = _load_chunks()
        indexer = _build_index(tmp_path, chunks, nfkc=True)
        hs = _hybrid_over_index(indexer, chunks, nfkc=True)
        results = hs.search("API", top_k=10)
        ids = {r.chunk_id for r in results}
        assert "c_fullwidth_api" in ids

    def test_halfwidth_digit_query_hits_fullwidth_index(self, tmp_path):
        chunks = _load_chunks()
        indexer = _build_index(tmp_path, chunks, nfkc=True)
        hs = _hybrid_over_index(indexer, chunks, nfkc=True)
        results = hs.search("2024 营收报告", top_k=10)
        ids = {r.chunk_id for r in results}
        assert "c_fullwidth_digits" in ids

    def test_without_nfkc_halfwidth_query_misses(self, tmp_path):
        """Regression contrast: no normalization -> form mismatch -> miss."""
        chunks = _load_chunks()
        indexer = _build_index(tmp_path, chunks, nfkc=False)
        hs = _hybrid_over_index(indexer, chunks, nfkc=False)
        results = hs.search("API", top_k=10)
        ids = {r.chunk_id for r in results}
        # Full-width 'ＡＰＩ' indexed as-is; half-width 'api' query cannot match.
        assert "c_fullwidth_api" not in ids

    def test_nfkc_recovers_hit_rate_vs_baseline(self, tmp_path):
        """归一化前后命中率回归对比: NFKC turns a miss into a hit."""
        chunks = _load_chunks()
        miss = _hybrid_over_index(
            _build_index(tmp_path / "raw", chunks, nfkc=False), chunks, nfkc=False
        ).search("API", top_k=10)
        hit = _hybrid_over_index(
            _build_index(tmp_path / "norm", chunks, nfkc=True), chunks, nfkc=True
        ).search("API", top_k=10)
        assert "c_fullwidth_api" not in {r.chunk_id for r in miss}
        assert "c_fullwidth_api" in {r.chunk_id for r in hit}


class TestTraditionalSimplified:
    @pytest.mark.skipif(not _opencc_available(), reason="OpenCC not installed")
    def test_simplified_query_hits_traditional_index_with_t2s(self, tmp_path):
        chunks = _load_chunks()
        indexer = _build_index(tmp_path, chunks, nfkc=True, to_simplified=True)
        hs = _hybrid_over_index(indexer, chunks, nfkc=True, to_simplified=True)
        # Query '数据库' (simplified) vs indexed '資料庫' (traditional).
        results = hs.search("数据库连线设定", top_k=10)
        ids = {r.chunk_id for r in results}
        assert "c_traditional_db" in ids


class TestEnhancementChaining:
    def test_synonym_expansion_chained_over_real_index(self, tmp_path):
        """Synonym OR-expansion lets an alias query reach the full-width chunk."""
        chunks = _load_chunks()
        indexer = _build_index(tmp_path, chunks, nfkc=True)
        # Map alias '接口' -> 'api' so an alias-only query still reaches the chunk.
        hs = _hybrid_over_index(
            indexer,
            chunks,
            nfkc=True,
            synonym_map={"接口": ["api"]},
            enable_synonym=True,
        )
        trace = TraceContext(trace_type="query")
        results = hs.search("接口 网关配置", top_k=10, trace=trace)
        ids = {r.chunk_id for r in results}
        assert "c_fullwidth_api" in ids
        # End-to-end trace is coherent.
        stage = {s.name: s for s in trace.stages}["hybrid_search"]
        assert stage.details["dense_lists"] == 1

    def test_sparse_prefilter_chained_over_real_index(self, tmp_path):
        """Sparse pre-filter (doc_type=markdown) keeps matching chunks only."""
        chunks = _load_chunks()
        indexer = _build_index(tmp_path, chunks, nfkc=True)
        hs = _hybrid_over_index(indexer, chunks, nfkc=True)
        results = hs.search("API 网关配置", top_k=10, filters={"doc_type": "markdown"})
        assert all(r.metadata.get("doc_type") == "markdown" for r in results)
        assert "c_fullwidth_api" in {r.chunk_id for r in results}

    def test_disabled_enhancements_regress_to_baseline(self, tmp_path):
        """All enhancements off over the real index == plain BM25 hit."""
        chunks = _load_chunks()
        indexer = _build_index(tmp_path, chunks, nfkc=True)
        hs = _hybrid_over_index(indexer, chunks, nfkc=True, enable_synonym=False)
        results = hs.search("API 网关配置", top_k=10)
        assert "c_fullwidth_api" in {r.chunk_id for r in results}
