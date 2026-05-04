"""Tests for Ollama LLM provider (B7.2). All HTTP calls are mocked."""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from src.libs.llm.base_llm import ChatMessage
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import Settings, LLMConfig

# Trigger registration
import src.libs.llm.ollama_llm  # noqa: F401


def _mock_response(content: str = "ollama reply", model: str = "llama3"):
    data = {
        "model": model,
        "message": {"role": "assistant", "content": content},
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _settings(**overrides) -> Settings:
    defaults = dict(provider="ollama", model="llama3")
    defaults.update(overrides)
    return Settings(llm=LLMConfig(**defaults))


@pytest.mark.unit
class TestOllamaLLM:

    @patch("src.libs.llm.ollama_llm.urlopen")
    def test_chat_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("hi from ollama")
        llm = LLMFactory.create(_settings())
        resp = llm.chat([ChatMessage(role="user", content="hello")])
        assert resp.content == "hi from ollama"
        assert resp.model == "llama3"

    @patch("src.libs.llm.ollama_llm.urlopen")
    def test_provider_name(self, mock_urlopen):
        llm = LLMFactory.create(_settings())
        assert llm.provider_name == "ollama"

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            LLMFactory.create(_settings(model=""))

    def test_empty_messages_raises(self):
        llm = LLMFactory.create(_settings())
        with pytest.raises(ValueError, match="messages must not be empty"):
            llm.chat([])

    @patch("src.libs.llm.ollama_llm.urlopen")
    def test_connection_error_readable(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        llm = LLMFactory.create(_settings())
        with pytest.raises(RuntimeError, match=r"\[ollama\] Connection failed"):
            llm.chat([ChatMessage(role="user", content="hi")])

    @patch("src.libs.llm.ollama_llm.urlopen")
    def test_error_does_not_leak_secrets(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        llm = LLMFactory.create(_settings())
        with pytest.raises(RuntimeError) as exc_info:
            llm.chat([ChatMessage(role="user", content="hi")])
        # Should not contain any api_key (ollama doesn't use one, but verify)
        assert "sk-" not in str(exc_info.value)

    @patch("src.libs.llm.ollama_llm.urlopen")
    def test_http_error_readable(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="http://x", code=500, msg="Internal Server Error", hdrs=None, fp=None
        )
        llm = LLMFactory.create(_settings())
        with pytest.raises(RuntimeError, match=r"\[ollama\] HTTP 500"):
            llm.chat([ChatMessage(role="user", content="hi")])

    @patch("src.libs.llm.ollama_llm.urlopen")
    def test_custom_base_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("ok")
        llm = LLMFactory.create(_settings(base_url="http://myhost:9999"))
        llm.chat([ChatMessage(role="user", content="hi")])
        req = mock_urlopen.call_args[0][0]
        assert "myhost:9999" in req.full_url
