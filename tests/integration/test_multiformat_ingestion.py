"""Integration tests for multi-format ingestion (T17).

Exercises the new ingestion path end-to-end and guards the existing PDF flow
against regressions.

Assembly mirrors ``tests/integration/test_ingestion_pipeline.py``:
- Loaders are resolved per-file via ``LoaderFactory`` (loader=None on the
  pipeline), so ``.xlsx`` -> ``XlsxLoader`` and ``.pdf`` -> ``PdfLoader`` —
  exercising the real T15 dispatch.
- Embedding / vector store are fakes (no network); BM25 + integrity + image
  storage use temp dirs.
- The shared jieba tokenizer (``TokenizerFactory``) drives the BM25 SparseEncoder,
  matching ``IngestionPipeline.from_settings``.

The xlsx path uses the REAL ``XlsxLoader`` (MarkItDown converts each worksheet
to a Markdown table prefixed with a ``## {sheet_name}`` H2 marker) routed to the
REAL ``TableAwareSplitter`` via ``settings.splitter.by_doc_type={"xlsx":
"table_aware"}``. The sample.xlsx fixture is generated with openpyxl (installed).

Validates: Requirements 1.4, 5.4, 7.1
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Importing this module self-registers the "table_aware" splitter with the
# SplitterFactory so doc_type routing (xlsx -> table_aware) resolves.
import src.libs.splitter.table_aware_splitter  # noqa: F401

from src.core.settings import Settings, SplitterConfig
from src.core.query_engine.fusion import ReciprocalRankFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.types import RetrievalResult
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import SQLiteImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.tokenizer import TokenizerFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore

SAMPLES = Path("tests/fixtures/sample_documents")
SAMPLE_XLSX = SAMPLES / "sample.xlsx"
SAMPLE_MD = SAMPLES / "sample.md"
SIMPLE_PDF = SAMPLES / "simple.pdf"


# ---------------------------------------------------------------------------
# Offline fakes (no network) — mirror test_ingestion_pipeline.py
# ---------------------------------------------------------------------------

class FakeEmbedding:
    """Deterministic offline embedding (no network)."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts, trace=None):
        return [[float((len(t) + i) % 7) for i in range(self._dim)] for t in texts]

    @property
    def provider_name(self):
        return "fake"

    @property
    def dimension(self):
        return self._dim


class FakeVectorStore(BaseVectorStore):
    """In-memory vector store that records upserted chunks + metadata."""

    def __init__(self):
        self.records: dict[str, dict] = {}

    def upsert(self, records, trace=None):
        for r in records:
            self.records[r.id] = {"text": r.text, "metadata": r.metadata}
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None):
        return []

    def delete_by_metadata(self, filter, trace=None):
        return 0

    def get_by_ids(self, ids):
        return [
            {"id": i, "text": self.records[i]["text"], "metadata": self.records[i]["metadata"]}
            for i in ids if i in self.records
        ]

    @property
    def backend_name(self):
        return "fake"


class FakeRetriever:
    """Query-side fake returning preset RetrievalResults regardless of input."""

    def __init__(self, results=None, kind="dense"):
        self._results = results or []
        self._kind = kind
        self.called_with = None

    def retrieve(self, query_or_keywords, top_k=20, filters=None, trace=None):
        self.called_with = {"arg": query_or_keywords, "top_k": top_k, "filters": filters}
        return self._results


def _xlsx_settings() -> Settings:
    """Settings routing xlsx -> table_aware, char-unit sizing for determinism."""
    return Settings(
        splitter=SplitterConfig(
            type="recursive",
            size_unit="char",
            chunk_size=512,
            chunk_overlap=0,
            by_doc_type={"xlsx": "table_aware"},
        )
    )


def _build_pipeline(tmp_path, settings) -> tuple[IngestionPipeline, FakeVectorStore]:
    """Assemble a hermetic pipeline (loader resolved per-file via factory)."""
    chunker = DocumentChunker(settings)
    transforms = [
        ChunkRefiner(llm=None, use_llm=False),
        MetadataEnricher(llm=None, use_llm=False),
    ]
    dense = DenseEncoder(FakeEmbedding(), batch_size=4)
    sparse = SparseEncoder(tokenizer=TokenizerFactory.create(settings))
    batch = BatchProcessor(dense, sparse, batch_size=4)
    store = FakeVectorStore()
    upserter = VectorUpserter(store)
    bm25 = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    integrity = SQLiteIntegrityChecker(db_path=str(tmp_path / "ingest.db"))
    images = SQLiteImageStorage(
        images_base_dir=str(tmp_path / "images"),
        db_path=str(tmp_path / "image_index.db"),
    )
    pipeline = IngestionPipeline(
        loader=None,  # resolved per-file by LoaderFactory (extension-driven)
        chunker=chunker,
        transforms=transforms,
        batch_processor=batch,
        vector_upserter=upserter,
        bm25_indexer=bm25,
        integrity_checker=integrity,
        image_storage=images,
    )
    return pipeline, store


def _candidates_from_store(store: FakeVectorStore) -> list[RetrievalResult]:
    """Turn stored chunk records into query-side RetrievalResult candidates."""
    return [
        RetrievalResult(
            chunk_id=cid,
            score=0.0,
            text=rec["text"],
            metadata=rec["metadata"],
        )
        for cid, rec in store.records.items()
    ]


# ---------------------------------------------------------------------------
# xlsx end-to-end: ingest -> sheet metadata -> sheet filter
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestXlsxEndToEnd:
    def test_chunks_carry_sheet_metadata(self, tmp_path):
        """Full pipeline over xlsx yields table chunks with sheet metadata."""
        pipeline, store = _build_pipeline(tmp_path, _xlsx_settings())
        result = pipeline.run(str(SAMPLE_XLSX), collection="sheets")

        assert not result.skipped
        assert result.total_chunks >= 2  # at least one chunk per sheet

        table_chunks = [
            rec["metadata"]
            for rec in store.records.values()
            if rec["metadata"].get("is_table")
        ]
        assert table_chunks, "expected table_aware to produce table chunks"

        # Every table chunk must carry the structured sheet metadata.
        for meta in table_chunks:
            assert meta["is_table"] is True
            assert meta.get("sheet_name") in {"产品表", "订单表"}
            assert "row_start" in meta and "row_end" in meta
            assert meta["row_start"] >= 1 and meta["row_end"] >= meta["row_start"]

        # Both sheets are represented.
        sheets = {m["sheet_name"] for m in table_chunks}
        assert sheets == {"产品表", "订单表"}

    def test_sheet_filter_hits_only_target_sheet(self, tmp_path):
        """Query-side sheet_name filter returns only chunks from that sheet."""
        pipeline, store = _build_pipeline(tmp_path, _xlsx_settings())
        pipeline.run(str(SAMPLE_XLSX), collection="sheets")

        candidates = _candidates_from_store(store)
        # Sanity: candidates span both sheets before filtering.
        all_sheets = {
            c.metadata.get("sheet_name")
            for c in candidates
            if c.metadata.get("is_table")
        }
        assert {"产品表", "订单表"}.issubset(all_sheets)

        hs = HybridSearch(
            query_processor=QueryProcessor(
                tokenizer=TokenizerFactory.create(_xlsx_settings())
            ),
            dense_retriever=FakeRetriever(candidates, kind="dense"),
            sparse_retriever=FakeRetriever([], kind="sparse"),
            fusion=ReciprocalRankFusion(k=60),
        )

        results = hs.search("查询产品", top_k=50, filters={"sheet_name": "产品表"})

        assert results, "expected at least one chunk from the target sheet"
        # Only the target sheet is returned (strict structured filtering).
        assert all(r.metadata.get("sheet_name") == "产品表" for r in results)
        returned_sheets = {r.metadata.get("sheet_name") for r in results}
        assert returned_sheets == {"产品表"}

    def test_bm25_index_persisted(self, tmp_path):
        """jieba-tokenized sparse vectors are persisted to the BM25 index."""
        pipeline, _ = _build_pipeline(tmp_path, _xlsx_settings())
        pipeline.run(str(SAMPLE_XLSX), collection="sheets")
        assert (tmp_path / "bm25" / "bm25_index.json").exists()


# ---------------------------------------------------------------------------
# Markdown end-to-end (prose path, empty structured metadata)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMarkdownEndToEnd:
    def test_markdown_runs_and_has_no_table_metadata(self, tmp_path):
        pipeline, store = _build_pipeline(tmp_path, _xlsx_settings())
        result = pipeline.run(str(SAMPLE_MD), collection="docs")

        assert not result.skipped
        assert result.total_chunks >= 1
        for rec in store.records.values():
            # Prose (recursive) chunks carry no structured table metadata.
            assert "is_table" not in rec["metadata"]
            assert "sheet_name" not in rec["metadata"]
            assert rec["metadata"].get("doc_type") == "markdown"


# ---------------------------------------------------------------------------
# PDF regression: the existing PDF e2e path still works through this assembly
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPdfRegression:
    @pytest.mark.skipif(not SIMPLE_PDF.exists(), reason="sample PDF missing")
    def test_pdf_full_pipeline_still_passes(self, tmp_path):
        pipeline, store = _build_pipeline(tmp_path, _xlsx_settings())
        result = pipeline.run(str(SIMPLE_PDF), collection="pdf")

        assert not result.skipped
        assert result.total_chunks >= 1
        assert len(result.vector_ids) == result.total_chunks
        assert len(store.records) == result.total_chunks
        # PDF chunks are prose: no table structure metadata leaks in.
        for rec in store.records.values():
            assert "is_table" not in rec["metadata"]
            assert rec["metadata"].get("doc_type") == "pdf"
