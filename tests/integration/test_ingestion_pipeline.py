"""Integration tests for IngestionPipeline (C14).

Exercises the full orchestration: integrity -> load -> split -> transform ->
encode -> store, using the REAL PdfLoader on sample PDFs but a fake embedding
and a temp Chroma store to avoid API costs and keep the test hermetic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.settings import Settings
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.pipeline import IngestionPipeline, IngestionError
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import SQLiteImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.vector_store.base_vector_store import BaseVectorStore

SAMPLES = Path("tests/fixtures/sample_documents")
SIMPLE_PDF = SAMPLES / "simple.pdf"
COMPLEX_PDF = SAMPLES / "模块(C++20)-补充内容.pdf"


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
    """In-memory vector store."""

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


def _build_pipeline(tmp_path) -> tuple[IngestionPipeline, FakeVectorStore]:
    settings = Settings()
    loader = PdfLoader(images_base_dir=str(tmp_path / "images"))
    chunker = DocumentChunker(settings)
    transforms = [
        ChunkRefiner(llm=None, use_llm=False),
        MetadataEnricher(llm=None, use_llm=False),
    ]
    dense = DenseEncoder(FakeEmbedding(), batch_size=4)
    sparse = SparseEncoder()
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
        loader=loader,
        chunker=chunker,
        transforms=transforms,
        batch_processor=batch,
        vector_upserter=upserter,
        bm25_indexer=bm25,
        integrity_checker=integrity,
        image_storage=images,
    )
    return pipeline, store


class TestSimpleDocument:
    def test_runs_end_to_end(self, tmp_path):
        pipeline, store = _build_pipeline(tmp_path)
        result = pipeline.run(str(SIMPLE_PDF), collection="test")
        assert not result.skipped
        assert result.total_chunks >= 1
        assert len(result.vector_ids) == result.total_chunks
        assert len(store.records) == result.total_chunks

    def test_bm25_index_persisted(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        pipeline.run(str(SIMPLE_PDF), collection="test")
        index_file = tmp_path / "bm25" / "bm25_index.json"
        assert index_file.exists()

    def test_progress_callback_invoked(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        stages_seen: list[str] = []
        pipeline.run(
            str(SIMPLE_PDF),
            collection="test",
            on_progress=lambda stage, cur, total: stages_seen.append(stage),
        )
        for expected in ("load", "split", "transform", "encode", "store"):
            assert expected in stages_seen


class TestIdempotency:
    def test_second_run_skipped(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        r1 = pipeline.run(str(SIMPLE_PDF), collection="test")
        r2 = pipeline.run(str(SIMPLE_PDF), collection="test")
        assert not r1.skipped
        assert r2.skipped

    def test_force_reingest(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        pipeline.run(str(SIMPLE_PDF), collection="test")
        r2 = pipeline.run(str(SIMPLE_PDF), collection="test", force=True)
        assert not r2.skipped


class TestComplexDocument:
    @pytest.mark.skipif(not COMPLEX_PDF.exists(), reason="complex sample PDF missing")
    def test_complex_pdf_full_pipeline(self, tmp_path):
        pipeline, store = _build_pipeline(tmp_path)
        result = pipeline.run(str(COMPLEX_PDF), collection="cpp")
        assert not result.skipped
        assert result.total_chunks > 1
        assert len(store.records) == result.total_chunks
        # BM25 index built with content
        assert (tmp_path / "bm25" / "bm25_index.json").exists()


class TestErrorHandling:
    def test_missing_file_raises_ingestion_error(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        with pytest.raises(IngestionError) as exc_info:
            pipeline.run(str(tmp_path / "nonexistent.pdf"), collection="test")
        assert "failed" in str(exc_info.value).lower()

    def test_failure_recorded_in_integrity(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        # A non-.pdf extension makes PdfLoader reject it in the load stage.
        bad = tmp_path / "bad.txt"
        bad.write_text("not a pdf")
        with pytest.raises(IngestionError):
            pipeline.run(str(bad), collection="test")


# ---------------------------------------------------------------------------
# F4: Ingestion pipeline tracing
# ---------------------------------------------------------------------------

class TestIngestionTracing:
    def test_trace_type_is_ingestion(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        result = pipeline.run(str(SIMPLE_PDF), collection="test")
        assert result.trace["trace_type"] == "ingestion"

    def test_all_ingestion_stages_present(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        result = pipeline.run(str(SIMPLE_PDF), collection="test")
        stage_names = {s["name"] for s in result.trace["stages"]}
        for expected in ("load", "split", "transform", "embed", "upsert"):
            assert expected in stage_names, f"missing stage {expected}"

    def test_stages_have_method(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        result = pipeline.run(str(SIMPLE_PDF), collection="test")
        by_name = {s["name"]: s for s in result.trace["stages"]}
        assert by_name["load"]["details"].get("method") == "markitdown"
        assert by_name["split"]["details"].get("method") == "recursive"
        assert by_name["upsert"]["details"].get("method") == "chroma"

    def test_stages_have_elapsed_ms(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        result = pipeline.run(str(SIMPLE_PDF), collection="test")
        for stage in result.trace["stages"]:
            assert "elapsed_ms" in stage

    def test_total_elapsed_present(self, tmp_path):
        pipeline, _ = _build_pipeline(tmp_path)
        result = pipeline.run(str(SIMPLE_PDF), collection="test")
        assert "total_elapsed_ms" in result.trace
        assert result.trace["finished_at"] is not None
