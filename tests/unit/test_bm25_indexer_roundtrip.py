"""Unit tests for BM25Indexer — build/save/load roundtrip, IDF, query, updates."""
from __future__ import annotations

import math

import pytest

from src.ingestion.embedding.sparse_encoder import SparseEncoder, SparseVector
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.core.types import Chunk
from src.libs.tokenizer import JiebaTokenizer


def _sv(chunk_id: str, term_freqs: dict[str, int]) -> SparseVector:
    return SparseVector(
        chunk_id=chunk_id,
        term_freqs=term_freqs,
        doc_length=sum(term_freqs.values()),
    )


def _corpus() -> list[SparseVector]:
    return [
        _sv("c0", {"apple": 2, "banana": 1}),
        _sv("c1", {"apple": 1, "cherry": 3}),
        _sv("c2", {"banana": 1, "cherry": 1, "date": 5}),
    ]


class TestBuildAndQuery:
    def test_build_sets_counts(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        assert idx.num_documents == 3
        assert idx.num_terms == 4  # apple, banana, cherry, date

    def test_query_returns_relevant(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        results = idx.query(["date"])
        assert results[0][0] == "c2"  # only c2 has 'date'

    def test_query_empty_terms(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        assert idx.query([]) == []

    def test_query_unknown_term(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        assert idx.query(["nonexistent"]) == []

    def test_query_top_k_limits(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        results = idx.query(["apple", "banana", "cherry"], top_k=2)
        assert len(results) <= 2

    def test_query_stable_ordering(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        r1 = idx.query(["apple", "cherry"])
        r2 = idx.query(["apple", "cherry"])
        assert r1 == r2

    def test_empty_corpus_query(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build([])
        assert idx.query(["apple"]) == []


class TestIDF:
    def test_idf_formula(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        # 'apple' appears in 2 of 3 docs: df=2, N=3
        expected = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1.0)
        assert idx.get_idf("apple") == pytest.approx(expected)

    def test_rarer_term_higher_idf(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        # 'date' in 1 doc, 'apple' in 2 docs -> date rarer -> higher idf
        assert idx.get_idf("date") > idx.get_idf("apple")

    def test_unknown_term_idf_zero(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        assert idx.get_idf("zzz") == 0.0


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        idx.save()

        idx2 = BM25Indexer(index_dir=str(tmp_path))
        idx2.load()
        assert idx2.num_documents == 3
        assert idx2.num_terms == 4

    def test_query_stable_after_reload(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        before = idx.query(["apple", "cherry"])
        idx.save()

        idx2 = BM25Indexer(index_dir=str(tmp_path))
        idx2.load()
        after = idx2.query(["apple", "cherry"])
        assert before == after

    def test_load_missing_raises(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError):
            idx.load()


class TestUpdates:
    def test_rebuild_replaces_corpus(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        idx.build([_sv("x0", {"new": 1})])
        assert idx.num_documents == 1
        assert idx.get_idf("apple") == 0.0
        assert idx.get_idf("new") >= 0.0

    def test_incremental_add(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        idx.add_documents([_sv("c3", {"apple": 1, "elderberry": 2})])
        assert idx.num_documents == 4
        # apple now in 3 of 4 docs
        results = idx.query(["elderberry"])
        assert results[0][0] == "c3"

    def test_incremental_update_existing(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        # replace c0 content
        idx.add_documents([_sv("c0", {"fig": 10})])
        assert idx.num_documents == 3  # still 3 docs
        results = idx.query(["fig"])
        assert results[0][0] == "c0"
        # old term 'apple' from c0 gone, now only in c1
        assert idx.get_idf("apple") > 0

    def test_remove_document(self, tmp_path):
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(_corpus())
        idx.remove_document("c2")
        assert idx.num_documents == 2
        assert idx.query(["date"]) == []  # date was only in c2


class TestEndToEndWithEncoder:
    def test_encoder_to_indexer(self, tmp_path):
        enc = SparseEncoder(tokenizer=JiebaTokenizer(stopwords=set()))
        chunks = [
            Chunk(id="d0", text="machine learning models", metadata={}, source_ref="x"),
            Chunk(id="d1", text="deep learning networks", metadata={}, source_ref="x"),
        ]
        svs = enc.encode(chunks)
        idx = BM25Indexer(index_dir=str(tmp_path))
        idx.build(svs)
        results = idx.query(["learning"])
        # both docs contain 'learning'
        ids = {cid for cid, _ in results}
        assert ids == {"d0", "d1"}
