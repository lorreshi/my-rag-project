"""Pluggable length functions for splitters.

A splitter measures chunk size via a ``length_fn(text) -> int``. Two strategies
are provided:

* :func:`char_length` — character count (``len``), the legacy behaviour.
* :func:`token_length` — a factory returning a tiktoken-based token counter,
  aligned with the embedding model's encoding (default ``cl100k_base``).

The token counter caches its ``Encoding`` object so it is fetched once, not on
every call. If ``tiktoken`` is not installed, building a token counter raises a
readable :class:`RuntimeError` explaining how to install it (design §2.1b:
"tiktoken 缺失时给出可读错误或回退 char").
"""
from __future__ import annotations

from typing import Callable


def char_length(text: str) -> int:
    """Return the character count of *text* (legacy size metric)."""
    return len(text)


def token_length(encoding: str = "cl100k_base") -> Callable[[str], int]:
    """Build a token-counting callable backed by tiktoken.

    The returned callable counts tokens of its input string using the given
    tiktoken *encoding*. The ``Encoding`` object is created once and reused on
    every call (avoids repeated ``get_encoding`` lookups).

    Args:
        encoding: tiktoken encoding name (default ``cl100k_base``, aligned with
            common embedding models).

    Returns:
        A callable ``(text: str) -> int`` returning the token count.

    Raises:
        RuntimeError: if ``tiktoken`` is not installed, with an install hint.
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise RuntimeError(
            "tiktoken is required for token-based length measurement "
            "(size_unit='token'). Install it with `pip install tiktoken`, "
            "or use size_unit='char' to measure by character count instead."
        ) from exc

    enc = tiktoken.get_encoding(encoding)

    def _count(text: str) -> int:
        return len(enc.encode(text))

    return _count
