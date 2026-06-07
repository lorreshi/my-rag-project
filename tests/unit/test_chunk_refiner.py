"""Unit tests for ChunkRefiner — rule-based and LLM modes.

Tests use mock LLM to isolate logic. No real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_PATH = Path("tests/fixtures/noisy_chunks.json")


@pytest.fixture
def noisy_chunks() -> list[dict]:
    """Load noisy chunk test fixtures."""
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_llm():
    """Create a mock LLM that echoes cleaned text."""
    llm = MagicMock()
    llm.provider_name = "mock"

    def _chat(messages):
        # Simulate LLM: return the input text with a prefix removed
        text = messages[0].content
        # Extract the text after the prompt template
        if "{text}" not in text:
            # The prompt has been filled, extract content after last newline block
            lines = text.split("\n\n")
            refined = lines[-1] if lines else text
        else:
            refined = text
        response = MagicMock()
        response.content = refined
        return response

    llm.chat.side_effect = _chat
    return llm


@pytest.fixture
def failing_llm():
    """Create a mock LLM that always raises."""
    llm = MagicMock()
    llm.provider_name = "failing_mock"
    llm.chat.side_effect = RuntimeError("API connection failed")
    return llm


@pytest.fixture
def empty_llm():
    """Create a mock LLM that returns empty content."""
    llm = MagicMock()
    llm.provider_name = "empty_mock"
    response = MagicMock()
    response.content = ""
    llm.chat.return_value = response
    return llm


def _make_chunk(text: str, chunk_id: str = "test_0001_abc") -> Chunk:
    """Helper to create a test Chunk."""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "test.pdf", "doc_type": "pdf"},
        source_ref="doc_test",
    )


# ---------------------------------------------------------------------------
# Rule-based refinement tests
# ---------------------------------------------------------------------------

class TestRuleBasedRefine:
    """Tests for rule-based denoising (no LLM)."""

    def test_removes_page_numbers(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Content here.\n\n- 3 -\n\nMore content.")
        result = refiner.transform([chunk])
        assert "- 3 -" not in result[0].text
        assert "Content here" in result[0].text
        assert "More content" in result[0].text

    def test_removes_page_x_of_y(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Important text.\n\nPage 5 of 20")
        result = refiner.transform([chunk])
        assert "Page 5 of 20" not in result[0].text
        assert "Important text" in result[0].text

    def test_removes_chinese_page_numbers(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("第3页\n\n正文内容在这里。")
        result = refiner.transform([chunk])
        assert "第3页" not in result[0].text
        assert "正文内容在这里" in result[0].text

    def test_removes_html_comments(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("<!-- comment -->\nReal content here.")
        result = refiner.transform([chunk])
        assert "<!-- comment -->" not in result[0].text
        assert "Real content here" in result[0].text

    def test_removes_html_tags(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("<div>Hello</div> <br/> <p>World</p>")
        result = refiner.transform([chunk])
        assert "<div>" not in result[0].text
        assert "<br/>" not in result[0].text
        assert "Hello" in result[0].text
        assert "World" in result[0].text

    def test_removes_html_entities(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Text&nbsp;with&amp;entities")
        result = refiner.transform([chunk])
        assert "&nbsp;" not in result[0].text
        assert "&amp;" not in result[0].text
        assert "Text" in result[0].text

    def test_normalizes_excessive_whitespace(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Line one.\n\n\n\n\n\nLine two.")
        result = refiner.transform([chunk])
        # Should have at most 2 consecutive newlines
        assert "\n\n\n" not in result[0].text
        assert "Line one" in result[0].text
        assert "Line two" in result[0].text

    def test_removes_separator_lines(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Above.\n----------\nBelow.")
        result = refiner.transform([chunk])
        assert "----------" not in result[0].text
        assert "Above" in result[0].text
        assert "Below" in result[0].text

    def test_preserves_code_blocks(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        code_text = "Example:\n\n```python\ndef foo():\n    return 42\n```\n\nEnd."
        chunk = _make_chunk(code_text)
        result = refiner.transform([chunk])
        assert "```python" in result[0].text
        assert "def foo():" in result[0].text
        assert "return 42" in result[0].text

    def test_preserves_inline_code(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Use `pip install chromadb` to install.")
        result = refiner.transform([chunk])
        assert "`pip install chromadb`" in result[0].text

    def test_preserves_markdown_headings(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("## Section Title\n\nContent below heading.")
        result = refiner.transform([chunk])
        assert "## Section Title" in result[0].text

    def test_preserves_markdown_lists(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Items:\n- First\n- Second\n- Third")
        result = refiner.transform([chunk])
        assert "- First" in result[0].text
        assert "- Second" in result[0].text

    def test_empty_text_unchanged(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("")
        result = refiner.transform([chunk])
        assert result[0].text == ""

    def test_whitespace_only_text(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("   \n\n   ")
        result = refiner.transform([chunk])
        assert result[0].text == "   \n\n   "

    def test_metadata_marked_rule(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Some content.\n\n- 1 -")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"

    def test_fixture_scenarios(self, noisy_chunks):
        """Test all fixture scenarios with rule-based cleaning."""
        refiner = ChunkRefiner(llm=None, use_llm=False)

        for scenario in noisy_chunks:
            chunk = _make_chunk(scenario["input"], chunk_id=scenario["id"])
            result = refiner.transform([chunk])
            refined_text = result[0].text

            for expected in scenario["expected_contains"]:
                assert expected in refined_text, (
                    f"Scenario '{scenario['id']}': expected '{expected}' in result"
                )

            for not_expected in scenario["expected_not_contains"]:
                assert not_expected not in refined_text, (
                    f"Scenario '{scenario['id']}': did not expect '{not_expected}' in result"
                )


# ---------------------------------------------------------------------------
# LLM refinement tests (mocked)
# ---------------------------------------------------------------------------

class TestLLMRefine:
    """Tests for LLM-enhanced refinement (mocked)."""

    def test_llm_called_when_enabled(self, mock_llm):
        refiner = ChunkRefiner(llm=mock_llm, use_llm=True)
        chunk = _make_chunk("This is a reasonably long chunk of text that needs refinement.")
        refiner.transform([chunk])
        mock_llm.chat.assert_called_once()

    def test_metadata_marked_llm(self, mock_llm):
        refiner = ChunkRefiner(llm=mock_llm, use_llm=True)
        chunk = _make_chunk("This is a reasonably long chunk of text that needs refinement.")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "llm"

    def test_llm_not_called_when_disabled(self, mock_llm):
        refiner = ChunkRefiner(llm=mock_llm, use_llm=False)
        chunk = _make_chunk("Some content.\n\n- 1 -")
        result = refiner.transform([chunk])
        mock_llm.chat.assert_not_called()
        assert result[0].metadata["refined_by"] == "rule"

    def test_llm_not_called_for_short_text(self, mock_llm):
        refiner = ChunkRefiner(llm=mock_llm, use_llm=True)
        chunk = _make_chunk("Short.")
        refiner.transform([chunk])
        # Short text skips LLM, returns as-is
        mock_llm.chat.assert_not_called()

    def test_llm_failure_falls_back_to_rule(self, failing_llm):
        refiner = ChunkRefiner(llm=failing_llm, use_llm=True)
        chunk = _make_chunk("Content with noise.\n\n- 5 -\n\nPage 3 of 10")
        result = refiner.transform([chunk])
        # Should fall back to rule-based result
        assert result[0].metadata["refined_by"] == "rule"
        assert result[0].metadata.get("refine_fallback") == "llm_failed"
        assert "- 5 -" not in result[0].text
        assert "Content with noise" in result[0].text

    def test_llm_empty_response_falls_back(self, empty_llm):
        refiner = ChunkRefiner(llm=empty_llm, use_llm=True)
        chunk = _make_chunk("This is meaningful content that should be preserved after refinement.")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"
        assert result[0].metadata.get("refine_fallback") == "llm_failed"

    def test_multiple_chunks_independent(self, failing_llm):
        """One chunk failing LLM should not affect others."""
        refiner = ChunkRefiner(llm=failing_llm, use_llm=True)
        chunks = [
            _make_chunk("First chunk content is here and it is long enough.", "c1"),
            _make_chunk("Second chunk content is here and it is long enough.", "c2"),
            _make_chunk("Third chunk content is here and it is long enough.", "c3"),
        ]
        results = refiner.transform(chunks)
        assert len(results) == 3
        # All should have fallen back gracefully
        for r in results:
            assert r.metadata["refined_by"] == "rule"


# ---------------------------------------------------------------------------
# Trace integration tests
# ---------------------------------------------------------------------------

class TestTraceIntegration:
    """Tests for trace context recording."""

    def test_trace_records_stage(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        trace = TraceContext(trace_type="ingestion")
        chunk = _make_chunk("Some content.\n\n- 1 -")
        refiner.transform([chunk], trace=trace)

        assert len(trace.stages) == 1
        assert trace.stages[0].name == "chunk_refiner"
        assert "rule_refined" in trace.stages[0].details

    def test_trace_records_stats(self, mock_llm):
        refiner = ChunkRefiner(llm=mock_llm, use_llm=True)
        trace = TraceContext(trace_type="ingestion")
        chunks = [
            _make_chunk("Long enough content for LLM processing in this test.", "c1"),
            _make_chunk("Another long enough chunk for LLM processing here.", "c2"),
        ]
        refiner.transform(chunks, trace=trace)

        details = trace.stages[0].details
        assert details["llm_refined"] == 2

    def test_no_trace_no_error(self):
        """Transform works fine without trace context."""
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Content.")
        result = refiner.transform([chunk], trace=None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_chunk_list(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        result = refiner.transform([])
        assert result == []

    def test_chunk_id_preserved(self, mock_llm):
        refiner = ChunkRefiner(llm=mock_llm, use_llm=True)
        chunk = _make_chunk("Long enough text for processing by the LLM system.", "my_id_123")
        result = refiner.transform([chunk])
        assert result[0].id == "my_id_123"

    def test_source_ref_preserved(self, mock_llm):
        refiner = ChunkRefiner(llm=mock_llm, use_llm=True)
        chunk = _make_chunk("Long enough text for processing by the LLM system.", "c1")
        chunk.source_ref = "doc_abc"
        result = refiner.transform([chunk])
        assert result[0].source_ref == "doc_abc"

    def test_existing_metadata_preserved(self):
        refiner = ChunkRefiner(llm=None, use_llm=False)
        chunk = _make_chunk("Content.\n\n- 1 -")
        chunk.metadata["custom_field"] = "keep_me"
        result = refiner.transform([chunk])
        assert result[0].metadata["custom_field"] == "keep_me"

    def test_prompt_loading_fallback(self, tmp_path):
        """Uses fallback prompt when file doesn't exist."""
        refiner = ChunkRefiner(
            llm=None, use_llm=False,
            prompt_path=str(tmp_path / "nonexistent.txt"),
        )
        assert "{text}" in refiner._prompt_template

    def test_prompt_loading_from_file(self, tmp_path):
        """Loads prompt from custom file."""
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text("Refine this: {text}")
        refiner = ChunkRefiner(
            llm=None, use_llm=False,
            prompt_path=str(prompt_file),
        )
        assert refiner._prompt_template == "Refine this: {text}"
