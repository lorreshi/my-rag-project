"""Tests for MarkdownLoader (T12).

Validates: Requirements 1.2, 1.4, 1.6
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src.libs.loader  # noqa: F401  (ensures built-in loaders are registered)
from src.core.types import Document
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.markdown_loader import MarkdownLoader


def _write_md(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _write_image(tmp_path: Path, name: str) -> Path:
    # A minimal valid-enough file; content doesn't matter for existence checks.
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


@pytest.mark.unit
class TestMarkdownLoader:

    def test_plain_markdown_no_images(self, tmp_path: Path):
        md = _write_md(
            tmp_path,
            "doc.md",
            "# Title\n\nSome plain text without any images.\n",
        )
        loader = MarkdownLoader()
        doc = loader.load(str(md))

        assert isinstance(doc, Document)
        assert doc.metadata["source_path"] == str(md)
        assert doc.metadata["doc_type"] == "markdown"
        assert "doc_hash" in doc.metadata and doc.metadata["doc_hash"]
        assert doc.metadata["images"] == []
        assert doc.images == []
        assert doc.id.startswith("md_")
        assert doc.id == f"md_{doc.metadata['doc_hash'][:12]}"
        assert "plain text" in doc.text

    def test_markdown_with_existing_image(self, tmp_path: Path):
        img = _write_image(tmp_path, "pic.png")
        md = _write_md(
            tmp_path,
            "doc.md",
            f"# Title\n\n![a picture]({img.name})\n\nmore text\n",
        )
        loader = MarkdownLoader()
        doc = loader.load(str(md))

        images = doc.images
        assert len(images) == 1
        ref = images[0]
        assert ref.path == str(img)
        assert ref.text_length > 0
        assert doc.text[ref.text_offset:ref.text_offset + ref.text_length].startswith(
            "!["
        )

    def test_broken_image_link_is_skipped(self, tmp_path: Path, caplog):
        md = _write_md(
            tmp_path,
            "doc.md",
            "# Title\n\n![missing](does_not_exist.png)\n",
        )
        loader = MarkdownLoader()
        with caplog.at_level("WARNING"):
            doc = loader.load(str(md))

        # load succeeds, broken image skipped, warning logged.
        assert doc.metadata["images"] == []
        assert any("not found" in rec.message.lower() or "not found" in rec.getMessage().lower()
                   for rec in caplog.records)

    def test_mixed_existing_and_broken_images(self, tmp_path: Path):
        img = _write_image(tmp_path, "good.png")
        md = _write_md(
            tmp_path,
            "doc.md",
            f"![ok]({img.name})\n\n![bad](missing.png)\n",
        )
        loader = MarkdownLoader()
        doc = loader.load(str(md))

        images = doc.images
        assert len(images) == 1
        assert images[0].path == str(img)

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path):
        txt = tmp_path / "doc.txt"
        txt.write_text("hello", encoding="utf-8")
        loader = MarkdownLoader()
        with pytest.raises(ValueError):
            loader.load(str(txt))

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        loader = MarkdownLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "nope.md"))

    def test_supported_extensions(self):
        loader = MarkdownLoader()
        assert loader.supported_extensions == [".md", ".markdown"]

    def test_factory_routes_md_to_markdown_loader(self):
        assert isinstance(LoaderFactory.create("x.md"), MarkdownLoader)

    def test_factory_routes_markdown_to_markdown_loader(self):
        assert isinstance(LoaderFactory.create("x.markdown"), MarkdownLoader)
