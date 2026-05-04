"""Tests for Azure Vision LLM (B9). All HTTP calls are mocked."""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from src.libs.llm.llm_factory import LLMFactory, _VISION_REGISTRY
from src.core.settings import Settings, VisionLLMConfig

# Trigger registration
import src.libs.llm.azure_vision_llm  # noqa: F401
from src.libs.llm.azure_vision_llm import AzureVisionLLM


def _mock_response(content: str = "I see a diagram"):
    data = {
        "choices": [{"message": {"content": content}}],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _settings(**overrides) -> Settings:
    defaults = dict(
        provider="azure",
        model="gpt-4o",
        api_key="sk-test",
        azure_endpoint="https://my.openai.azure.com",
    )
    defaults.update(overrides)
    return Settings(vision_llm=VisionLLMConfig(**defaults))


@pytest.mark.unit
class TestAzureVisionLLM:

    @patch("src.libs.llm.azure_vision_llm.urlopen")
    def test_chat_with_image_bytes(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("architecture diagram")
        vlm = LLMFactory.create_vision_llm(_settings())
        resp = vlm.chat_with_image("describe this", b"\x89PNG fake")
        assert resp.content == "architecture diagram"
        assert resp.model == "gpt-4o"

    @patch("src.libs.llm.azure_vision_llm.urlopen")
    def test_chat_with_image_file(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response("chart analysis")
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG test image")
        vlm = LLMFactory.create_vision_llm(_settings())
        resp = vlm.chat_with_image("analyze chart", str(img))
        assert resp.content == "chart analysis"

    @patch("src.libs.llm.azure_vision_llm.urlopen")
    def test_provider_name(self, mock_urlopen):
        vlm = LLMFactory.create_vision_llm(_settings())
        assert vlm.provider_name == "azure"

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            LLMFactory.create_vision_llm(_settings(api_key=""))

    def test_empty_endpoint_raises(self):
        with pytest.raises(ValueError, match="azure_endpoint"):
            LLMFactory.create_vision_llm(_settings(azure_endpoint=""))

    def test_empty_text_raises(self):
        vlm = LLMFactory.create_vision_llm(_settings())
        with pytest.raises(ValueError, match="text prompt"):
            vlm.chat_with_image("", b"\x89PNG")

    @patch("src.libs.llm.azure_vision_llm.urlopen")
    def test_url_contains_deployment(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("ok")
        vlm = LLMFactory.create_vision_llm(_settings(model="my-vision-deploy"))
        vlm.chat_with_image("describe", b"\x89PNG")
        req = mock_urlopen.call_args[0][0]
        assert "my-vision-deploy" in req.full_url
        assert "api-version=" in req.full_url

    @patch("src.libs.llm.azure_vision_llm.urlopen")
    def test_request_contains_base64_image(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("ok")
        vlm = LLMFactory.create_vision_llm(_settings())
        vlm.chat_with_image("describe", b"\x89PNG test")
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "base64," in content[1]["image_url"]["url"]

    @patch("src.libs.llm.azure_vision_llm.urlopen")
    def test_http_error_readable(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="http://x", code=401, msg="Unauthorized", hdrs=None, fp=None
        )
        vlm = LLMFactory.create_vision_llm(_settings())
        with pytest.raises(RuntimeError, match=r"\[azure-vision\] HTTP 401"):
            vlm.chat_with_image("describe", b"\x89PNG")

    @patch("src.libs.llm.azure_vision_llm.urlopen")
    def test_connection_error_readable(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        vlm = LLMFactory.create_vision_llm(_settings())
        with pytest.raises(RuntimeError, match=r"\[azure-vision\] Connection failed"):
            vlm.chat_with_image("describe", b"\x89PNG")
