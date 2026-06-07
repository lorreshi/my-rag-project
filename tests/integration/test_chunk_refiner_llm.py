"""Integration tests for ChunkRefiner with real LLM.

These tests make actual API calls. They verify:
1. LLM configuration is correct and reachable.
2. Refinement produces meaningful output.
3. Fallback works with invalid model names.

Run with: pytest tests/integration/test_chunk_refiner_llm.py -v -s
WARNING: These tests incur real API costs.
"""
from __future__ import annotations

import pytest

from src.core.settings import load_settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.libs.llm.llm_factory import LLMFactory

# Ensure OpenAI provider is registered
import src.libs.llm.openai_llm  # noqa: F401


def _make_chunk(text: str, chunk_id: str = "integ_001") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "test.pdf", "doc_type": "pdf"},
        source_ref="doc_integ",
    )


@pytest.fixture(scope="module")
def settings():
    """Load real settings from config/settings.yaml."""
    return load_settings()


@pytest.fixture(scope="module")
def llm(settings):
    """Create real LLM instance from settings."""
    return LLMFactory.create(settings)


class TestRealLLMRefinement:
    """Integration tests with real LLM calls."""

    @pytest.mark.integration
    def test_llm_refines_noisy_text(self, llm):
        """Verify LLM can successfully refine noisy text."""
        refiner = ChunkRefiner(llm=llm, use_llm=True)

        noisy_text = (
            "Company Confidential\n\n"
            "Page 3 of 10\n\n"
            "The transformer architecture was introduced in the paper "
            "'Attention Is All You Need' (2017). It uses self-attention "
            "mechanisms to process sequences in parallel, replacing "
            "recurrent neural networks.\n\n"
            "<!-- footer -->\n"
            "- 3 -"
        )

        chunk = _make_chunk(noisy_text)
        trace = TraceContext(trace_type="ingestion")
        result = refiner.transform([chunk], trace=trace)

        refined = result[0]
        print(f"\n--- Original ---\n{noisy_text}")
        print(f"\n--- Refined ---\n{refined.text}")
        print(f"\n--- Metadata ---\n{refined.metadata}")

        # Core content preserved
        assert "transformer" in refined.text.lower()
        assert "attention" in refined.text.lower()

        # Noise removed (by either rule or LLM)
        assert "Company Confidential" not in refined.text
        assert "- 3 -" not in refined.text

        # Metadata indicates LLM refinement
        assert refined.metadata["refined_by"] == "llm"

    @pytest.mark.integration
    def test_llm_preserves_clean_text(self, llm):
        """Verify LLM doesn't destroy already clean text."""
        refiner = ChunkRefiner(llm=llm, use_llm=True)

        clean_text = (
            "## Vector Databases\n\n"
            "Vector databases are specialized storage systems designed for "
            "high-dimensional vector data. They enable efficient similarity "
            "search operations using algorithms like HNSW and IVF.\n\n"
            "Key features:\n"
            "- Approximate nearest neighbor (ANN) search\n"
            "- Metadata filtering\n"
            "- Scalable indexing"
        )

        chunk = _make_chunk(clean_text, "integ_clean")
        result = refiner.transform([chunk])

        refined = result[0]
        print(f"\n--- Original (clean) ---\n{clean_text}")
        print(f"\n--- Refined ---\n{refined.text}")

        # Key content must be preserved
        assert "vector database" in refined.text.lower()
        assert "similarity search" in refined.text.lower() or "nearest neighbor" in refined.text.lower()

    @pytest.mark.integration
    def test_fallback_on_invalid_model(self, settings):
        """Verify graceful fallback when model is invalid."""
        from src.libs.llm.openai_llm import OpenAILLM

        # Create LLM with invalid model name
        bad_llm = OpenAILLM(
            api_key=settings.llm.api_key,
            model="nonexistent-model-xyz-999",
            base_url=settings.llm.base_url or "https://api.openai.com/v1",
        )

        refiner = ChunkRefiner(llm=bad_llm, use_llm=True)
        chunk = _make_chunk(
            "This is valid content that should survive fallback processing intact.\n\n- 5 -",
            "integ_fallback",
        )
        result = refiner.transform([chunk])

        refined = result[0]
        print(f"\n--- Fallback result ---\n{refined.text}")
        print(f"\n--- Metadata ---\n{refined.metadata}")

        # Should fall back to rule-based
        assert refined.metadata["refined_by"] == "rule"
        assert "valid content" in refined.text
        assert "- 5 -" not in refined.text
