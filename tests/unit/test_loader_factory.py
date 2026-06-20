"""Tests for LoaderFactory + register_loader (T11, Property 9).

Validates: Requirements 1.1, 1.5
"""

import pytest

import src.libs.loader  # noqa: F401  (ensures built-in loaders are registered)
from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.loader_factory import (
    LoaderFactory,
    register_loader,
    _REGISTRY,
)
from src.libs.loader.pdf_loader import PdfLoader


class FakeLoader(BaseLoader):
    """A trivial loader supporting multiple extensions."""

    def load(self, path: str) -> Document:
        return Document(id="fake", text="", metadata={"source_path": path})

    @property
    def supported_extensions(self) -> list[str]:
        return [".fake", ".fk"]


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


@pytest.mark.unit
class TestLoaderFactory:

    def test_pdf_routes_to_pdf_loader(self):
        loader = LoaderFactory.create("x.pdf")
        assert isinstance(loader, PdfLoader)

    def test_uppercase_extension_routes(self):
        loader = LoaderFactory.create("x.PDF")
        assert isinstance(loader, PdfLoader)

    def test_unknown_extension_raises_with_available_list(self):
        with pytest.raises(ValueError) as exc_info:
            LoaderFactory.create("x.unknown")
        msg = str(exc_info.value)
        assert ".unknown" in msg
        assert "pdf" in msg

    def test_register_loader_multiple_extensions(self):
        register_loader([".fake", ".fk"], lambda: FakeLoader())
        assert isinstance(LoaderFactory.create("a.fake"), FakeLoader)
        assert isinstance(LoaderFactory.create("b.fk"), FakeLoader)

    def test_register_loader_normalizes_extension(self):
        # Without leading dot and uppercase -> normalized to ".fake"
        register_loader(["FAKE"], lambda: FakeLoader())
        assert isinstance(LoaderFactory.create("c.fake"), FakeLoader)
