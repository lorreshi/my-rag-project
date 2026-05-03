"""Tests for LLM abstract interface and factory (B1)."""

import pytest

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import LLMFactory, register_provider, _REGISTRY
from src.core.settings import Settings, LLMConfig


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------

class FakeLLM(BaseLLM):
    """Deterministic stub that echoes the last user message."""

    def __init__(self, model: str = "fake-model"):
        self._model = model

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        last = messages[-1].content if messages else ""
        return ChatResponse(content=f"echo: {last}", model=self._model)

    @property
    def provider_name(self) -> str:
        return "fake"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean registry and restores it."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _make_settings(provider: str = "fake", model: str = "fake-model") -> Settings:
    return Settings(llm=LLMConfig(provider=provider, model=model))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseLLMInterface:
    """BaseLLM cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_fake_llm_chat(self):
        llm = FakeLLM()
        resp = llm.chat([ChatMessage(role="user", content="hi")])
        assert resp.content == "echo: hi"
        assert resp.model == "fake-model"

    def test_fake_llm_provider_name(self):
        assert FakeLLM().provider_name == "fake"


@pytest.mark.unit
class TestLLMFactory:
    """Factory routes to the correct provider based on settings."""

    def test_create_registered_provider(self):
        register_provider("fake", lambda s: FakeLLM(s.llm.model))
        llm = LLMFactory.create(_make_settings("fake", "test-model"))
        assert isinstance(llm, FakeLLM)
        assert llm.provider_name == "fake"

    def test_create_case_insensitive(self):
        register_provider("fake", lambda s: FakeLLM(s.llm.model))
        llm = LLMFactory.create(_make_settings("FAKE"))
        assert isinstance(llm, FakeLLM)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider 'nope'"):
            LLMFactory.create(_make_settings("nope"))

    def test_unknown_provider_lists_available(self):
        register_provider("alpha", lambda s: FakeLLM())
        register_provider("beta", lambda s: FakeLLM())
        with pytest.raises(ValueError, match="alpha, beta"):
            LLMFactory.create(_make_settings("nope"))

    def test_factory_passes_settings(self):
        register_provider("fake", lambda s: FakeLLM(s.llm.model))
        llm = LLMFactory.create(_make_settings("fake", "my-model"))
        resp = llm.chat([ChatMessage(role="user", content="test")])
        assert resp.model == "my-model"


@pytest.mark.unit
class TestChatMessageAndResponse:
    """Data classes behave correctly."""

    def test_chat_message_fields(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_chat_response_defaults(self):
        resp = ChatResponse(content="ok")
        assert resp.model == ""
        assert resp.usage == {}
        assert resp.raw is None
