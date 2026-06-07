"""Transform module — enrichment processing.

Provides BaseTransform interface and concrete implementations:
- ChunkRefiner: rule-based denoising + optional LLM enhancement
- MetadataEnricher: (Phase C6)
- ImageCaptioner: (Phase C7)
"""
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.chunk_refiner import ChunkRefiner

__all__ = ["BaseTransform", "ChunkRefiner"]
