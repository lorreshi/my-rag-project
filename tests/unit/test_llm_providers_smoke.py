"""Smoke tests for OpenAI / Azure / DeepSeek LLM providers (B7.1).

All HTTP calls are mocked — no real network access needed.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from src.libs.llm.base_llm import ChatMessage
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import Settings, LLMConfig

# Import providers to trigger register_provider() side effects
import src.libs.llm.openai_llm  # noqa: F401
import src.libs.llm.azure_llm   # noqa: F401
import src.libs.llm.deepseek_llm  # noqa: F401


def _mock_response(content: str = "hello", model: str = "test-model"):
    """Build a fake HTTP response that looks like OpenAI's JSON."""
    data = {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _settings(provider: str, **overrides) -> Settings:
    defaults = dict(
        provider=provider,
        model="test-model",
        api_key="sk-test-key",
        azure_endpoint="https://my.openai.azure.com",
    )
    defaults.update(overrides)
    return Settings(llm=LLMConfig(**defaults))


# ---------------------------------------------------------------------------
# OpenAI tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAILLM:

    @patch("src.libs.llm.openai_llm.urlopen")
    def test_chat_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("hi there", "gpt-4o")
        llm = LLMFactory.create(_settings("openai"))
        resp = llm.chat([ChatMessage(role="user", content="hello")])
        assert resp.content == "hi there"
        assert resp.model == "gpt-4o"
        assert resp.usage["total_tokens"] == 8

    @patch("src.libs.llm.openai_llm.urlopen")
    def test_provider_name(self, mock_urlopen):
        llm = LLMFactory.create(_settings("openai"))
        assert llm.provider_name == "openai"

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            LLMFactory.create(_settings("openai", api_key=""))

    def test_empty_messages_raises(self):
        llm = LLMFactory.create(_settings("openai"))
        with pytest.raises(ValueError, match="messages must not be empty"):
            llm.chat([])

    @patch("src.libs.llm.openai_llm.urlopen")
    def test_http_error_readable(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="http://x", code=429, msg="Rate limited", hdrs=None, fp=None
        )
        llm = LLMFactory.create(_settings("openai"))
        with pytest.raises(RuntimeError, match=r"\[openai\] HTTP 429"):
            llm.chat([ChatMessage(role="user", content="hi")])


# ---------------------------------------------------------------------------
# Azure tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAzureLLM:

    @patch("src.libs.llm.azure_llm.urlopen")
    def test_chat_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("azure reply")
        llm = LLMFactory.create(_settings("azure"))
        resp = llm.chat([ChatMessage(role="user", content="hello")])
        assert resp.content == "azure reply"

    @patch("src.libs.llm.azure_llm.urlopen")
    def test_provider_name(self, mock_urlopen):
        llm = LLMFactory.create(_settings("azure"))
        assert llm.provider_name == "azure"

    def test_empty_endpoint_raises(self):
        with pytest.raises(ValueError, match="azure_endpoint"):
            LLMFactory.create(_settings("azure", azure_endpoint=""))

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            LLMFactory.create(_settings("azure", api_key=""))

    @patch("src.libs.llm.azure_llm.urlopen")
    def test_url_contains_deployment(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("ok")
        llm = LLMFactory.create(_settings("azure", model="my-deploy"))
        llm.chat([ChatMessage(role="user", content="hi")])
        # Check the URL passed to urlopen
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "my-deploy" in req.full_url
        assert "api-version=" in req.full_url


# ---------------------------------------------------------------------------
# DeepSeek tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeepSeekLLM:

    @patch("src.libs.llm.openai_llm.urlopen")
    def test_chat_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("deepseek reply")
        llm = LLMFactory.create(_settings("deepseek"))
        resp = llm.chat([ChatMessage(role="user", content="hello")])
        assert resp.content == "deepseek reply"

    @patch("src.libs.llm.openai_llm.urlopen")
    def test_provider_name(self, mock_urlopen):
        llm = LLMFactory.create(_settings("deepseek"))
        assert llm.provider_name == "deepseek"

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            LLMFactory.create(_settings("deepseek", api_key=""))


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFactoryRouting:

    def test_all_three_providers_registered(self):
        for p in ["openai", "azure", "deepseek"]:
            llm = LLMFactory.create(_settings(p))
            assert llm.provider_name == p
