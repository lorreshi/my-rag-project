"""Tests for Vision LLM abstract interface and factory (B8)."""
from __future__ import annotations

import pytest

from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.base_llm import ChatResponse
from src.libs.llm.llm_factory import (
    LLMFactory,
    register_vision_provider,
    _VISION_REGISTRY,
)
from src.core.settings import Settings, VisionLLMConfig


class FakeVisionLLM(BaseVisionLLM):
    """Deterministic stub for testing."""

    def chat_with_image(self, text, image, trace=None):
        b64 = self.image_to_base64(image)
        return ChatResponse(content=f"saw image ({len(b64)} chars): {text}")

    @property
    def provider_name(self) -> str:
        return "fake"


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_VISION_REGISTRY)
    _VISION_REGISTRY.clear()
    yield
    _VISION_REGISTRY.clear()
    _VISION_REGISTRY.update(saved)


def _settings(provider: str = "fake") -> Settings:
    return Settings(vision_llm=VisionLLMConfig(provider=provider))


@pytest.mark.unit
class TestBaseVisionLLMInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseVisionLLM()

    def test_fake_vision_llm(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG fake image data")
        vlm = FakeVisionLLM()
        resp = vlm.chat_with_image("describe this", str(img))
        assert "saw image" in resp.content
        assert "describe this" in resp.content

    def test_image_bytes_input(self):
        vlm = FakeVisionLLM()
        resp = vlm.chat_with_image("describe", b"\x89PNG raw bytes")
        assert "saw image" in resp.content

    def test_image_file_not_found(self):
        vlm = FakeVisionLLM()
        with pytest.raises(FileNotFoundError, match="not found"):
            vlm.chat_with_image("describe", "/nonexistent/image.png")

    def test_image_to_base64_roundtrip(self, tmp_path):
        data = b"hello image"
        img = tmp_path / "img.png"
        img.write_bytes(data)
        import base64
        b64 = BaseVisionLLM.image_to_base64(str(img))
        assert base64.b64decode(b64) == data

    def test_image_to_base64_bytes(self):
        import base64
        data = b"raw bytes"
        b64 = BaseVisionLLM.image_to_base64(data)
        assert base64.b64decode(b64) == data


@pytest.mark.unit
class TestVisionLLMFactory:

    def test_create_registered_provider(self):
        register_vision_provider("fake", lambda s: FakeVisionLLM())
        vlm = LLMFactory.create_vision_llm(_settings("fake"))
        assert isinstance(vlm, FakeVisionLLM)

    def test_create_case_insensitive(self):
        register_vision_provider("fake", lambda s: FakeVisionLLM())
        vlm = LLMFactory.create_vision_llm(_settings("FAKE"))
        assert isinstance(vlm, FakeVisionLLM)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown Vision LLM provider 'nope'"):
            LLMFactory.create_vision_llm(_settings("nope"))

    def test_unknown_lists_available(self):
        register_vision_provider("alpha", lambda s: FakeVisionLLM())
        with pytest.raises(ValueError, match="alpha"):
            LLMFactory.create_vision_llm(_settings("nope"))
