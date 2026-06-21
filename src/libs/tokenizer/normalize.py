"""Shared deterministic text normalization (Phase D enhancement #1).

A single normalization pipeline used by BOTH the ingestion side
(``SparseEncoder`` via the shared tokenizer) and the query side
(``QueryProcessor._normalize`` for dense + the tokenizer for sparse keywords),
so the BM25 vocabulary stays symmetric and the dense query text uses the same
canonical form.

Pipeline (fixed order, idempotent):
    1. NFKC      — full-width -> half-width, compatibility-char folding (stdlib)
    2. casefold  — case folding (stdlib; more thorough than ``str.lower``)
    3. t2s       — traditional -> simplified Chinese (OPTIONAL, needs OpenCC;
                   degrades to a no-op with a warning when OpenCC is missing)

Putting these in one shared place is what makes "benefits both dense and
sparse paths, symmetrically" actually true: a query-only normalization would
desync the index-side BM25 vocabulary.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# Lazily-initialized OpenCC converter; cached across calls.
_opencc_converter: Any = None
_opencc_unavailable: bool = False


def _get_opencc_converter() -> Any:
    """Return a cached OpenCC t2s converter, or ``None`` if unavailable."""
    global _opencc_converter, _opencc_unavailable
    if _opencc_converter is not None:
        return _opencc_converter
    if _opencc_unavailable:
        return None
    try:
        import opencc  # type: ignore

        _opencc_converter = opencc.OpenCC("t2s")
        return _opencc_converter
    except Exception as exc:  # ImportError or config error
        _opencc_unavailable = True
        logger.warning(
            "OpenCC unavailable; traditional->simplified normalization will be "
            "skipped (NFKC/casefold still applied): %s",
            exc,
        )
        return None


def _to_simplified(text: str) -> str:
    """Convert traditional Chinese to simplified; no-op if OpenCC is missing."""
    converter = _get_opencc_converter()
    if converter is None:
        return text
    try:
        return converter.convert(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OpenCC conversion failed; skipping t2s: %s", exc)
        return text


def normalize_text(
    text: str,
    *,
    nfkc: bool = True,
    casefold: bool = True,
    to_simplified: bool = False,
) -> str:
    """Apply the deterministic normalization pipeline to *text*.

    Args:
        text: Raw input text.
        nfkc: Apply Unicode NFKC normalization (full/half-width, compat chars).
        casefold: Apply case folding.
        to_simplified: Convert traditional Chinese to simplified (needs OpenCC;
            silently skipped with a warning when OpenCC is not installed).

    Returns:
        The normalized text. Empty input yields an empty string. The function
        is idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    if not text:
        return ""
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if casefold:
        text = text.casefold()
    if to_simplified:
        text = _to_simplified(text)
    return text
