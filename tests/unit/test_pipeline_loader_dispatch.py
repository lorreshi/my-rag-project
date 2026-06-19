"""Tests for IngestionPipeline loader dispatch via LoaderFactory (T15).

The pipeline no longer hardcodes ``PdfLoader``. When no loader is injected the
loader is resolved per-file by extension through ``LoaderFactory.create``; an
explicitly injected loader always takes precedence.

Validates: Requirements 1.7
"""

import pytest

import src.libs.loader  # noqa: F401  (ensures built-in loaders are registered)
from src.core.types import Chunk, Document
from src.ingestion.pipeline import IngestionPipeline
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.docx_loader import DocxLoader
from src.libs.loader.loader_factory import LoaderFactory, _REGISTRY, register_loader
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.xlsx_loader import XlsxLoader


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLoader(BaseLoader):
    """Records the path it was asked to load."""

    def __init__(self) -> None:
        self.loaded_paths: list[str] = []

    def load(self, path: str) -> Document:
        self.loaded_paths.append(path)
        return Document(id="fake", text="hello", metadata={"source_path": path})

    @property
    def supported_extensions(self) -> list[str]:
        return [".fake"]


def _make_pipeline(loader: BaseLoader | None) -> IngestionPipeline:
    """Build a pipeline with only the loader supplied; other stages are unused
    by the dispatch tests (they exercise ``_resolve_loader`` directly)."""
    return IngestionPipeline(
        loader=loader,
        chunker=object(),
        transforms=[],
        batch_processor=object(),
        vector_upserter=object(),
        bm25_indexer=object(),
    )


# ---------------------------------------------------------------------------
# Extension -> Loader dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoaderDispatch:

    @pytest.mark.parametrize(
        "path,expected_type",
        [
            ("doc.pdf", PdfLoader),
            ("doc.md", MarkdownLoader),
            ("doc.markdown", MarkdownLoader),
            ("doc.docx", DocxLoader),
            ("doc.xlsx", XlsxLoader),
        ],
    )
    def test_resolves_correct_loader_by_extension(self, path, expected_type):
        pipeline = _make_pipeline(loader=None)
        resolved = pipeline._resolve_loader(path)
        assert isinstance(resolved, expected_type)

    def test_injected_loader_takes_precedence_over_factory(self):
        injected = FakeLoader()
        pipeline = _make_pipeline(loader=injected)
        # Even for a .pdf path the injected loader is used, not PdfLoader.
        assert pipeline._resolve_loader("doc.pdf") is injected
        assert pipeline._resolve_loader("doc.xlsx") is injected

    def test_unknown_extension_raises_clear_value_error(self):
        pipeline = _make_pipeline(loader=None)
        with pytest.raises(ValueError) as exc_info:
            pipeline._resolve_loader("doc.unknown")
        msg = str(exc_info.value)
        assert ".unknown" in msg
        # Error lists available extensions so the failure is actionable.
        assert "pdf" in msg

    def test_resolve_uses_factory_registration(self, monkeypatch):
        # Register a fake extension and confirm the pipeline dispatches to it.
        register_loader([".fake"], lambda: FakeLoader())
        pipeline = _make_pipeline(loader=None)
        resolved = pipeline._resolve_loader("doc.fake")
        assert isinstance(resolved, FakeLoader)


# ---------------------------------------------------------------------------
# End-to-end run() picks the resolved loader
# ---------------------------------------------------------------------------


class _FakeChunker:
    def split_document(self, document, collection):
        return [Chunk(id="c0", text=document.text, metadata={})]


class _Encoded:
    def __init__(self, chunk: Chunk) -> None:
        self.chunk = chunk
        self.dense_vector = [0.0]
        self.sparse_vector = None


class _FakeBatch:
    def process(self, chunks, trace=None):
        return [_Encoded(c) for c in chunks]


class _FakeUpserter:
    def upsert(self, chunks, dense_vectors, trace=None):
        return [c.id for c in chunks]


class _FakeBm25:
    def add_documents(self, docs):
        pass

    def save(self):
        pass


@pytest.mark.unit
class TestRunUsesResolvedLoader:

    def test_run_dispatches_to_factory_loader(self, monkeypatch):
        fake = FakeLoader()
        register_loader([".fake"], lambda: fake)

        pipeline = IngestionPipeline(
            loader=None,
            chunker=_FakeChunker(),
            transforms=[],
            batch_processor=_FakeBatch(),
            vector_upserter=_FakeUpserter(),
            bm25_indexer=_FakeBm25(),
            integrity_checker=None,
            image_storage=None,
        )

        result = pipeline.run("doc.fake", collection="default")

        assert result.error == ""
        assert fake.loaded_paths == ["doc.fake"]
        assert result.doc_id == "fake"
