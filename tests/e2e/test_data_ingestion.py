"""End-to-end tests for the ingest.py CLI (C15).

Drives the CLI via run_ingestion/main with the real PdfLoader on a sample PDF,
but injects a fake embedding + in-memory vector store (by patching
IngestionPipeline.from_settings) so the test is hermetic — no API calls, all
artifacts under a temp directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.settings import Settings
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
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.vector_store.base_vector_store import BaseVectorStore

import scripts.ingest as ingest_cli

SIMPLE_PDF = Path("tests/fixtures/sample_documents/simple.pdf")


class FakeEmbedding:
    def embed(self, texts, trace=None):
        return [[float(len(t) % 5)] * 4 for t in texts]

    @property
    def provider_name(self):
        return "fake"

    @property
    def dimension(self):
        return 4


class FakeVectorStore(BaseVectorStore):
    def __init__(self):
        self.records = {}

    def upsert(self, records, trace=None):
        for r in records:
            self.records[r.id] = r.text
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None):
        return []

    def delete_by_metadata(self, filter, trace=None):
        return 0

    def get_by_ids(self, ids):
        return [{"id": i, "text": self.records[i], "metadata": {}} for i in ids if i in self.records]

    @property
    def backend_name(self):
        return "fake"


@pytest.fixture
def patched_pipeline(tmp_path, monkeypatch):
    """Patch from_settings to build a hermetic pipeline rooted at tmp_path."""

    def _fake_from_settings(settings, **overrides):
        s = Settings()
        return IngestionPipeline(
            loader=PdfLoader(images_base_dir=str(tmp_path / "images")),
            chunker=DocumentChunker(s),
            transforms=[
                ChunkRefiner(llm=None, use_llm=False),
                MetadataEnricher(llm=None, use_llm=False),
            ],
            batch_processor=BatchProcessor(
                DenseEncoder(FakeEmbedding(), batch_size=4),
                SparseEncoder(),
                batch_size=4,
            ),
            vector_upserter=VectorUpserter(FakeVectorStore()),
            bm25_indexer=BM25Indexer(index_dir=str(tmp_path / "bm25")),
            integrity_checker=SQLiteIntegrityChecker(db_path=str(tmp_path / "ingest.db")),
            image_storage=SQLiteImageStorage(
                images_base_dir=str(tmp_path / "images"),
                db_path=str(tmp_path / "image_index.db"),
            ),
        )

    monkeypatch.setattr(
        IngestionPipeline, "from_settings", classmethod(
            lambda cls, settings, **kw: _fake_from_settings(settings, **kw)
        )
    )
    # Avoid requiring real config file content
    monkeypatch.setattr(ingest_cli, "load_settings", lambda *a, **k: Settings())
    return tmp_path


class TestCLIRun:
    def test_ingest_single_file(self, patched_pipeline):
        results = ingest_cli.run_ingestion(
            path=str(SIMPLE_PDF), collection="e2e", force=False, show_progress=False
        )
        assert len(results) == 1
        assert not results[0].skipped
        assert results[0].total_chunks >= 1

    def test_produces_bm25_artifact(self, patched_pipeline):
        ingest_cli.run_ingestion(
            path=str(SIMPLE_PDF), collection="e2e", force=False, show_progress=False
        )
        assert (patched_pipeline / "bm25" / "bm25_index.json").exists()

    def test_rerun_skips_unchanged(self, patched_pipeline):
        ingest_cli.run_ingestion(
            path=str(SIMPLE_PDF), collection="e2e", force=False, show_progress=False
        )
        results = ingest_cli.run_ingestion(
            path=str(SIMPLE_PDF), collection="e2e", force=False, show_progress=False
        )
        assert results[0].skipped is True

    def test_force_reingests(self, patched_pipeline):
        ingest_cli.run_ingestion(
            path=str(SIMPLE_PDF), collection="e2e", force=False, show_progress=False
        )
        results = ingest_cli.run_ingestion(
            path=str(SIMPLE_PDF), collection="e2e", force=True, show_progress=False
        )
        assert results[0].skipped is False

    def test_directory_ingestion(self, patched_pipeline):
        results = ingest_cli.run_ingestion(
            path="tests/fixtures/sample_documents",
            collection="e2e", force=False, show_progress=False,
        )
        # at least the two sample PDFs are picked up
        assert len(results) >= 2


class TestCLIMain:
    def test_main_success_exit_code(self, patched_pipeline):
        code = ingest_cli.main(
            ["--path", str(SIMPLE_PDF), "--collection", "e2e"]
        )
        assert code == 0

    def test_main_missing_path_exit_code(self, patched_pipeline):
        code = ingest_cli.main(["--path", "/nonexistent/file.pdf"])
        assert code == 1

    def test_arg_parsing(self):
        parser = ingest_cli._build_parser()
        args = parser.parse_args(["--path", "x.pdf", "--collection", "c", "--force"])
        assert args.path == "x.pdf"
        assert args.collection == "c"
        assert args.force is True

    def test_collection_defaults(self):
        parser = ingest_cli._build_parser()
        args = parser.parse_args(["--path", "x.pdf"])
        assert args.collection == "default"
        assert args.force is False


class TestFileCollection:
    def test_collect_single_file(self):
        files = ingest_cli._collect_files(SIMPLE_PDF)
        assert files == [SIMPLE_PDF]

    def test_collect_directory(self):
        files = ingest_cli._collect_files(Path("tests/fixtures/sample_documents"))
        assert all(f.suffix.lower() == ".pdf" for f in files)
        assert len(files) >= 2

    def test_collect_nonexistent(self, tmp_path):
        assert ingest_cli._collect_files(tmp_path / "nope") == []
