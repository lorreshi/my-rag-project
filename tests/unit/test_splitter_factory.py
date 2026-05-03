"""Tests for Splitter abstract interface and factory (B3)."""

import pytest

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import (
    SplitterFactory,
    register_splitter,
    _REGISTRY,
)
from src.core.settings import Settings


class FakeSplitter(BaseSplitter):
    """Splits text by newlines."""

    def split_text(self, text, trace=None):
        return [chunk for chunk in text.split("\n") if chunk.strip()]

    @property
    def splitter_type(self) -> str:
        return "fake"


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


@pytest.mark.unit
class TestBaseSplitterInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseSplitter()

    def test_fake_splitter_splits(self):
        s = FakeSplitter()
        chunks = s.split_text("hello\nworld")
        assert chunks == ["hello", "world"]

    def test_fake_splitter_empty(self):
        assert FakeSplitter().split_text("") == []


@pytest.mark.unit
class TestSplitterFactory:

    def test_create_registered(self):
        register_splitter("fake", lambda s: FakeSplitter())
        sp = SplitterFactory.create(Settings(), "fake")
        assert isinstance(sp, FakeSplitter)

    def test_create_case_insensitive(self):
        register_splitter("fake", lambda s: FakeSplitter())
        sp = SplitterFactory.create(Settings(), "FAKE")
        assert isinstance(sp, FakeSplitter)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown splitter type 'nope'"):
            SplitterFactory.create(Settings(), "nope")

    def test_unknown_lists_available(self):
        register_splitter("alpha", lambda s: FakeSplitter())
        with pytest.raises(ValueError, match="alpha"):
            SplitterFactory.create(Settings(), "nope")
