"""Loader abstractions.

Importing this package registers the built-in loader implementations with the
``LoaderFactory`` (self-registration on import).
"""
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.docx_loader import DocxLoader
from src.libs.loader.loader_factory import LoaderFactory, register_loader
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader

# Register built-in loaders with the factory so it works out of the box.
register_loader([".pdf"], lambda: PdfLoader())
register_loader([".md", ".markdown"], lambda: MarkdownLoader())
register_loader([".docx"], lambda: DocxLoader())

__all__ = [
    "BaseLoader",
    "LoaderFactory",
    "register_loader",
    "PdfLoader",
    "MarkdownLoader",
    "DocxLoader",
]
