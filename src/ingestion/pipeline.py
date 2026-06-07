"""IngestionPipeline — orchestrates the full ingestion flow.

Stages (serial):
    integrity -> load -> split -> transform -> encode -> store

- integrity: SHA256 early-exit; unchanged files are skipped (zero-cost).
- load:      PdfLoader -> Document (markdown text + metadata + images).
- split:     DocumentChunker -> List[Chunk] (ids, metadata, image refs).
- transform: ChunkRefiner -> MetadataEnricher -> ImageCaptioner (each optional).
- encode:    BatchProcessor drives DenseEncoder + SparseEncoder.
- store:     VectorUpserter (Chroma), BM25Indexer, ImageStorage.

Components are injectable for testing. ``from_settings`` builds a real pipeline
from a Settings object. Any stage failure raises a clear IngestionError.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, Document

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.ingestion.chunking.document_chunker import DocumentChunker
    from src.ingestion.embedding.batch_processor import BatchProcessor
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.ingestion.storage.image_storage import BaseImageStorage
    from src.ingestion.storage.vector_upserter import VectorUpserter
    from src.ingestion.transform.base_transform import BaseTransform
    from src.libs.loader.base_loader import BaseLoader
    from src.libs.loader.file_integrity import FileIntegrityChecker

logger = logging.getLogger(__name__)

# Progress callback signature: (stage_name, current, total)
ProgressCallback = Callable[[str, int, int], None]


class IngestionError(RuntimeError):
    """Raised when an ingestion stage fails."""


@dataclass
class IngestionResult:
    """Outcome of an ingestion run for a single file."""

    source_path: str
    collection: str
    trace_id: str
    skipped: bool = False
    doc_id: str = ""
    total_chunks: int = 0
    total_images: int = 0
    vector_ids: list[str] = field(default_factory=list)
    error: str = ""


class IngestionPipeline:
    """Serial ingestion orchestrator."""

    def __init__(
        self,
        loader: "BaseLoader",
        chunker: "DocumentChunker",
        transforms: "list[BaseTransform]",
        batch_processor: "BatchProcessor",
        vector_upserter: "VectorUpserter",
        bm25_indexer: "BM25Indexer",
        integrity_checker: "FileIntegrityChecker | None" = None,
        image_storage: "BaseImageStorage | None" = None,
    ):
        self._loader = loader
        self._chunker = chunker
        self._transforms = transforms
        self._batch = batch_processor
        self._upserter = vector_upserter
        self._bm25 = bm25_indexer
        self._integrity = integrity_checker
        self._image_storage = image_storage

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        source_path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Ingest a single file end-to-end.

        Args:
            source_path: Path to the source document.
            collection: Target collection name.
            force: If True, bypass the integrity early-exit.
            on_progress: Optional callback invoked per stage.

        Returns:
            IngestionResult describing the outcome.

        Raises:
            IngestionError: on any stage failure.
        """
        trace = TraceContext(trace_type="ingestion")
        result = IngestionResult(
            source_path=source_path, collection=collection, trace_id=trace.trace_id
        )

        def progress(stage: str, cur: int, total: int) -> None:
            if on_progress:
                on_progress(stage, cur, total)

        # Stage 0: integrity early-exit
        file_hash = ""
        if self._integrity is not None:
            try:
                file_hash = self._integrity.compute_sha256(source_path)
            except Exception as exc:
                result.error = str(exc)
                raise IngestionError(
                    f"Integrity stage failed for {source_path}: {exc}"
                ) from exc
            if not force and self._integrity.should_skip(file_hash):
                logger.info("Skipping unchanged file: %s", source_path)
                result.skipped = True
                progress("integrity", 1, 1)
                return result

        try:
            # Stage 1: load
            progress("load", 0, 1)
            document = self._load(source_path, trace)
            result.doc_id = document.id
            result.total_images = len(document.images)
            progress("load", 1, 1)

            # Stage 2: split
            progress("split", 0, 1)
            chunks = self._split(document, trace)
            result.total_chunks = len(chunks)
            progress("split", 1, 1)

            # Stage 3: transform
            chunks = self._transform(chunks, trace, progress)

            # Stage 4: encode
            progress("encode", 0, len(chunks))
            encoded = self._encode(chunks, trace)
            progress("encode", len(chunks), len(chunks))

            # Stage 5: store
            progress("store", 0, len(chunks))
            vector_ids = self._store(encoded, collection, document, trace)
            result.vector_ids = vector_ids
            result.total_chunks = len(encoded)
            progress("store", len(chunks), len(chunks))

        except IngestionError:
            raise
        except Exception as exc:
            if self._integrity is not None and file_hash:
                self._integrity.mark_failed(file_hash, source_path, str(exc))
            result.error = str(exc)
            raise IngestionError(
                f"Ingestion failed for {source_path}: {exc}"
            ) from exc

        # Record success in integrity history
        if self._integrity is not None and file_hash:
            self._integrity.mark_success(file_hash, source_path, result.total_chunks)

        logger.info(
            "Ingested %s: %d chunks, %d images (trace=%s)",
            source_path, result.total_chunks, result.total_images, trace.trace_id,
        )
        return result

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _load(self, source_path: str, trace: TraceContext) -> Document:
        try:
            return self._loader.load(source_path)
        except Exception as exc:
            raise IngestionError(f"Load stage failed: {exc}") from exc

    def _split(self, document: Document, trace: TraceContext) -> list[Chunk]:
        try:
            return self._chunker.split_document(document)
        except Exception as exc:
            raise IngestionError(f"Split stage failed: {exc}") from exc

    def _transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext,
        progress: ProgressCallback,
    ) -> list[Chunk]:
        total = len(self._transforms)
        for i, transform in enumerate(self._transforms):
            try:
                progress("transform", i, total)
                chunks = transform.transform(chunks, trace=trace)
            except Exception as exc:
                raise IngestionError(
                    f"Transform stage '{getattr(transform, 'name', transform)}' failed: {exc}"
                ) from exc
        progress("transform", total, total)
        return chunks

    def _encode(self, chunks: list[Chunk], trace: TraceContext):
        try:
            return self._batch.process(chunks, trace=trace)
        except Exception as exc:
            raise IngestionError(f"Encode stage failed: {exc}") from exc

    def _store(
        self,
        encoded: list,
        collection: str,
        document: Document,
        trace: TraceContext,
    ) -> list[str]:
        try:
            chunks = [e.chunk for e in encoded]
            dense_vectors = [e.dense_vector for e in encoded]

            # 1. Vector store (dense)
            vector_ids = self._upserter.upsert(chunks, dense_vectors, trace=trace)

            # 2. BM25 index (sparse)
            sparse_vectors = [e.sparse_vector for e in encoded if e.sparse_vector]
            if sparse_vectors:
                self._bm25.add_documents(sparse_vectors)
                self._bm25.save()

            # 3. Image storage
            if self._image_storage is not None:
                self._store_images(document, collection)

            return vector_ids
        except Exception as exc:
            raise IngestionError(f"Store stage failed: {exc}") from exc

    def _store_images(self, document: Document, collection: str) -> None:
        doc_hash = document.metadata.get("doc_hash", "")
        for img in document.images:
            if img.path:
                try:
                    self._image_storage.save_image(  # type: ignore[union-attr]
                        image_id=img.id,
                        source=img.path,
                        collection=collection,
                        doc_hash=doc_hash,
                        page_num=img.page,
                    )
                except FileNotFoundError:
                    logger.warning("Image file missing, skipping: %s", img.path)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides: Any) -> "IngestionPipeline":
        """Build a real pipeline from settings.

        Vision-LLM-dependent ImageCaptioner is enabled only if vision_llm is
        configured. Individual components can be overridden via kwargs (used in
        tests).
        """
        from src.ingestion.chunking.document_chunker import DocumentChunker
        from src.ingestion.embedding.batch_processor import BatchProcessor
        from src.ingestion.embedding.dense_encoder import DenseEncoder
        from src.ingestion.embedding.sparse_encoder import SparseEncoder
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.image_storage import SQLiteImageStorage
        from src.ingestion.storage.vector_upserter import VectorUpserter
        from src.ingestion.transform.chunk_refiner import ChunkRefiner
        from src.ingestion.transform.metadata_enricher import MetadataEnricher
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.llm.llm_factory import LLMFactory
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.libs.loader.pdf_loader import PdfLoader
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        loader = overrides.get("loader") or PdfLoader()
        chunker = overrides.get("chunker") or DocumentChunker(settings)

        llm = overrides.get("llm")
        if llm is None:
            try:
                llm = LLMFactory.create(settings)
            except Exception as exc:  # pragma: no cover - config dependent
                logger.warning("LLM unavailable, transforms run rule-only: %s", exc)
                llm = None

        transforms = overrides.get("transforms") or [
            ChunkRefiner(llm=llm, use_llm=llm is not None),
            MetadataEnricher(llm=llm, use_llm=llm is not None),
        ]

        embedding = overrides.get("embedding") or EmbeddingFactory.create(settings)
        dense = DenseEncoder(embedding)
        sparse = SparseEncoder()
        batch = overrides.get("batch_processor") or BatchProcessor(dense, sparse)

        vector_store = overrides.get("vector_store") or VectorStoreFactory.create(settings)
        upserter = VectorUpserter(vector_store)
        bm25 = overrides.get("bm25_indexer") or BM25Indexer()
        integrity = overrides.get("integrity_checker") or SQLiteIntegrityChecker()
        image_storage = overrides.get("image_storage") or SQLiteImageStorage()

        return cls(
            loader=loader,
            chunker=chunker,
            transforms=transforms,
            batch_processor=batch,
            vector_upserter=upserter,
            bm25_indexer=bm25,
            integrity_checker=integrity,
            image_storage=image_storage,
        )
