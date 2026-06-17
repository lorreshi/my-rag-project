"""Loader abstractions.

Importing this package registers the built-in loader implementations with the
``LoaderFactory`` (self-registration on import).
"""
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.loader_factory import LoaderFactory, register_loader
from src.libs.loader.pdf_loader import PdfLoader

# Register built-in loaders with the factory so it works out of the box.
register_loader([".pdf"], lambda: PdfLoader())

__all__ = ["BaseLoader", "LoaderFactory", "register_loader", "PdfLoader"]
