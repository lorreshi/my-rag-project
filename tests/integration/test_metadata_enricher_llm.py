"""Integration tests for MetadataEnricher with real LLM.

These tests make actual API calls. They verify:
1. LLM connectivity and that structured metadata is produced.
2. Metadata quality (title/summary/tags are semantically relevant).
3. Graceful fallback when the model name is invalid.

Run with: pytest tests/integration/test_metadata_enricher_llm.py -v -s
WARNING: These tests incur real API costs.
"""
from __future__ import annotations

import pytest

from src.core.settings import load_settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.llm.llm_factory import LLMFactory

# Ensure OpenAI provider is registered
import src.libs.llm.openai_llm  # noqa: F401


def _make_chunk(text: str, chunk_id: str = "integ_meta_001") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "test.pdf", "doc_type": "pdf"},
        source_ref="doc_integ",
    )


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def llm(settings):
    return LLMFactory.create(settings)


class TestRealLLMEnrichment:
    @pytest.mark.integration
    def test_llm_generates_metadata(self, llm):
        enricher = MetadataEnricher(llm=llm, use_llm=True)

        text = (
            "## Reciprocal Rank Fusion\n\n"
            "Reciprocal Rank Fusion (RRF) combines multiple ranked result lists "
            "into a single ranking. Instead of relying on raw relevance scores, "
            "it sums the reciprocals of each item's rank across lists. This makes "
            "it robust when fusing dense semantic retrieval with sparse BM25 "
            "keyword retrieval in hybrid search systems."
        )

        chunk = _make_chunk(text)
        trace = TraceContext(trace_type="ingestion")
        result = enricher.transform([chunk], trace=trace)[0]

        print(f"\n--- Title ---\n{result.metadata['title']}")
        print(f"\n--- Summary ---\n{result.metadata['summary']}")
        print(f"\n--- Tags ---\n{result.metadata['tags']}")
        print(f"\n--- enriched_by ---\n{result.metadata['enriched_by']}")

        assert result.metadata["enriched_by"] == "llm"
        assert result.metadata["title"].strip()
        assert result.metadata["summary"].strip()
        assert isinstance(result.metadata["tags"], list)
        assert len(result.metadata["tags"]) >= 1

        # Semantic relevance: at least one expected concept appears somewhere
        blob = (
            result.metadata["title"]
            + " "
            + result.metadata["summary"]
            + " "
            + " ".join(result.metadata["tags"])
        ).lower()
        assert any(k in blob for k in ("rank", "fusion", "rrf", "retrieval", "hybrid"))

    @pytest.mark.integration
    def test_fallback_on_invalid_model(self, settings):
        from src.libs.llm.openai_llm import OpenAILLM

        bad_llm = OpenAILLM(
            api_key=settings.llm.api_key,
            model="nonexistent-model-xyz-999",
            base_url=settings.llm.base_url or "https://api.openai.com/v1",
        )

        enricher = MetadataEnricher(llm=bad_llm, use_llm=True)
        text = (
            "## Cross-Encoder Reranking\n\n"
            "Cross-encoders jointly encode the query and a candidate passage to "
            "produce a precise relevance score, used to rerank top candidates."
        )
        result = enricher.transform([_make_chunk(text, "integ_meta_fb")])[0]

        print(f"\n--- Fallback metadata ---\n{result.metadata}")

        assert result.metadata["enriched_by"] == "rule"
        assert result.metadata["enrich_fallback"] == "llm_failed"
        assert result.metadata["title"]  # rule baseline still works
        assert result.metadata["tags"]
