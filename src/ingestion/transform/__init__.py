"""Transform module — enrichment processing.

Provides BaseTransform interface and concrete implementations:
- ChunkRefiner: rule-based denoising + optional LLM enhancement
- MetadataEnricher: title/summary/tags generation (rule + optional LLM)
- ImageCaptioner: (Phase C7)
"""
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher

__all__ = ["BaseTransform", "ChunkRefiner", "MetadataEnricher"]
