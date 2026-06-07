"""Unit tests for ImageCaptioner — enabled mode and graceful degradation.

Tests use a mock Vision LLM and temp image files. No real API calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.image_captioner import ImageCaptioner


def _make_image_file(tmp_path, name: str = "img.png") -> str:
    """Create a tiny fake image file and return its path."""
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")
    return str(p)


def _make_chunk_with_image(image_path: str, image_id: str = "img_001") -> Chunk:
    return Chunk(
        id="chunk_0001",
        text="See the figure below.\n\n[IMAGE: %s]" % image_id,
        metadata={
            "source_path": "test.pdf",
            "images": [{"id": image_id, "path": image_path, "page": 0}],
            "image_refs": [image_id],
        },
        source_ref="doc_test",
    )


def _make_chunk_no_image() -> Chunk:
    return Chunk(
        id="chunk_0002",
        text="Just plain text, no images here.",
        metadata={"source_path": "test.pdf"},
        source_ref="doc_test",
    )


@pytest.fixture
def vision_llm():
    """Mock Vision LLM that returns a caption."""
    llm = MagicMock()
    llm.provider_name = "mock_vision"
    response = MagicMock()
    response.content = "A bar chart showing quarterly revenue growth."
    llm.chat_with_image.return_value = response
    return llm


@pytest.fixture
def failing_vision_llm():
    llm = MagicMock()
    llm.provider_name = "failing_vision"
    llm.chat_with_image.side_effect = RuntimeError("Vision API error")
    return llm


# ---------------------------------------------------------------------------
# Enabled mode
# ---------------------------------------------------------------------------

class TestEnabledMode:
    def test_caption_generated(self, vision_llm, tmp_path):
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        chunk = _make_chunk_with_image(img)
        result = captioner.transform([chunk])[0]

        vision_llm.chat_with_image.assert_called_once()
        assert "image_captions" in result.metadata
        assert result.metadata["image_captions"]["img_001"].startswith("A bar chart")
        assert result.metadata["has_unprocessed_images"] is False

    def test_caption_stitched_into_text(self, vision_llm, tmp_path):
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        chunk = _make_chunk_with_image(img)
        result = captioner.transform([chunk])[0]
        assert "IMAGE CAPTION img_001" in result.text
        assert "quarterly revenue growth" in result.text

    def test_chunk_without_images_untouched(self, vision_llm):
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        chunk = _make_chunk_no_image()
        original = chunk.text
        result = captioner.transform([chunk])[0]
        vision_llm.chat_with_image.assert_not_called()
        assert result.text == original
        assert "has_unprocessed_images" not in result.metadata

    def test_multiple_images_in_chunk(self, vision_llm, tmp_path):
        img1 = _make_image_file(tmp_path, "a.png")
        img2 = _make_image_file(tmp_path, "b.png")
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        chunk = Chunk(
            id="c1",
            text="Two figures.",
            metadata={
                "images": [
                    {"id": "i1", "path": img1, "page": 0},
                    {"id": "i2", "path": img2, "page": 1},
                ],
                "image_refs": ["i1", "i2"],
            },
            source_ref="doc",
        )
        result = captioner.transform([chunk])[0]
        assert vision_llm.chat_with_image.call_count == 2
        assert set(result.metadata["image_captions"].keys()) == {"i1", "i2"}


# ---------------------------------------------------------------------------
# Degradation / fallback
# ---------------------------------------------------------------------------

class TestDegradation:
    def test_disabled_marks_unprocessed(self, vision_llm, tmp_path):
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=False)
        chunk = _make_chunk_with_image(img)
        result = captioner.transform([chunk])[0]
        vision_llm.chat_with_image.assert_not_called()
        assert result.metadata["has_unprocessed_images"] is True
        # image_refs preserved
        assert result.metadata["image_refs"] == ["img_001"]
        assert "image_captions" not in result.metadata

    def test_no_vision_llm_marks_unprocessed(self, tmp_path):
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(vision_llm=None, use_vision=True)
        chunk = _make_chunk_with_image(img)
        result = captioner.transform([chunk])[0]
        assert result.metadata["has_unprocessed_images"] is True
        assert result.metadata["image_refs"] == ["img_001"]

    def test_vision_exception_marks_unprocessed(self, failing_vision_llm, tmp_path):
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(vision_llm=failing_vision_llm, use_vision=True)
        chunk = _make_chunk_with_image(img)
        result = captioner.transform([chunk])[0]
        # chat_with_image raised -> caption skipped -> unprocessed
        assert result.metadata["has_unprocessed_images"] is True
        assert result.metadata["image_refs"] == ["img_001"]

    def test_missing_image_file_marks_unprocessed(self, vision_llm):
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        chunk = _make_chunk_with_image("/nonexistent/path/img.png")
        result = captioner.transform([chunk])[0]
        vision_llm.chat_with_image.assert_not_called()
        assert result.metadata["has_unprocessed_images"] is True

    def test_empty_caption_marks_unprocessed(self, tmp_path):
        img = _make_image_file(tmp_path)
        llm = MagicMock()
        response = MagicMock()
        response.content = "   "  # whitespace only
        llm.chat_with_image.return_value = response
        captioner = ImageCaptioner(vision_llm=llm, use_vision=True)
        chunk = _make_chunk_with_image(img)
        result = captioner.transform([chunk])[0]
        assert result.metadata["has_unprocessed_images"] is True

    def test_partial_failure_flags_remainder(self, tmp_path):
        """One image succeeds, one fails -> captioned but flagged."""
        img1 = _make_image_file(tmp_path, "ok.png")
        llm = MagicMock()
        call_state = {"n": 0}

        def _side_effect(text, image, trace=None):
            call_state["n"] += 1
            if call_state["n"] == 1:
                resp = MagicMock()
                resp.content = "Good caption."
                return resp
            raise RuntimeError("second image fails")

        llm.chat_with_image.side_effect = _side_effect
        chunk = Chunk(
            id="c1",
            text="Two figures.",
            metadata={
                "images": [
                    {"id": "i1", "path": img1, "page": 0},
                    {"id": "i2", "path": img1, "page": 1},
                ],
                "image_refs": ["i1", "i2"],
            },
            source_ref="doc",
        )
        result = captioner_transform(llm, chunk)
        assert "i1" in result.metadata["image_captions"]
        assert result.metadata["has_unprocessed_images"] is True


def captioner_transform(llm, chunk):
    captioner = ImageCaptioner(vision_llm=llm, use_vision=True)
    return captioner.transform([chunk])[0]


# ---------------------------------------------------------------------------
# Trace + edges
# ---------------------------------------------------------------------------

class TestTraceAndEdges:
    def test_trace_records_stage(self, vision_llm, tmp_path):
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        trace = TraceContext(trace_type="ingestion")
        captioner.transform([_make_chunk_with_image(img)], trace=trace)
        assert len(trace.stages) == 1
        assert trace.stages[0].name == "image_captioner"
        assert trace.stages[0].details["captioned"] == 1

    def test_empty_chunk_list(self, vision_llm):
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        assert captioner.transform([]) == []

    def test_preserves_id_and_source_ref(self, vision_llm, tmp_path):
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(vision_llm=vision_llm, use_vision=True)
        chunk = _make_chunk_with_image(img)
        result = captioner.transform([chunk])[0]
        assert result.id == "chunk_0001"
        assert result.source_ref == "doc_test"

    def test_prompt_injection_via_path(self, vision_llm, tmp_path):
        prompt_file = tmp_path / "cap_prompt.txt"
        prompt_file.write_text("Describe precisely.")
        img = _make_image_file(tmp_path)
        captioner = ImageCaptioner(
            vision_llm=vision_llm, use_vision=True, prompt_path=str(prompt_file)
        )
        captioner.transform([_make_chunk_with_image(img)])
        # The custom prompt should have been passed to the vision LLM
        _, kwargs = vision_llm.chat_with_image.call_args
        assert kwargs.get("text") == "Describe precisely."
