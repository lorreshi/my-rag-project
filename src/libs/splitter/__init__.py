"""Splitter abstractions.

Importing this package registers the built-in splitter implementations with
the SplitterFactory (self-registration on import).
"""
# Import concrete splitters so they self-register with SplitterFactory.
from src.libs.splitter import recursive_splitter  # noqa: F401
from src.libs.splitter import table_aware_splitter  # noqa: F401
