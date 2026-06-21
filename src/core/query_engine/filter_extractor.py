"""Filter extraction — parse structured constraints from query text.

DEV_SPEC 3.1.2 asks the query-processing stage to parse structured constraints
into the generic ``filters`` dict. This is a pluggable, opt-in step: the
default ``QueryProcessor`` does NOT extract (behaviour unchanged) unless an
extractor is injected.

The baseline ``RuleBasedFilterExtractor`` recognizes explicit ``key:value``
(or full-width ``key：value``) pairs for a small allow-list of known keys. It
is intentionally conservative and MUST NEVER raise — any internal error yields
an empty result so a query is never blocked.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseFilterExtractor(ABC):
    """Parse structured constraints from a raw query into a filters dict."""

    @abstractmethod
    def extract(self, query: str) -> dict[str, Any]:
        """Return extracted filters ({} if none). Must not raise."""
        ...


# Allow-list of keys the rule extractor will populate, mapped to a coercer.
def _as_bool(v: str) -> bool:
    return v.strip().lower() in {"true", "1", "yes", "y", "是", "true."}


def _as_int(v: str) -> int:
    return int(v.strip())


_ALLOWED_KEYS: dict[str, Any] = {
    "collection": str,
    "doc_type": str,
    "sheet_name": str,
    "language": str,
    "is_table": _as_bool,
    "row_start": _as_int,
    "row_end": _as_int,
}

# Matches  key:value  or  key：value  (ASCII or full-width colon). Value is a
# run of non-space chars, allowing quotes which are stripped afterwards.
_KV_RE = re.compile(r"(?P<key>[A-Za-z_]+)\s*[:：]\s*(?P<val>\"[^\"]+\"|'[^']+'|\S+)")


class RuleBasedFilterExtractor(BaseFilterExtractor):
    """Conservative explicit ``key:value`` extractor (Phase D baseline)."""

    def extract(self, query: str) -> dict[str, Any]:
        if not query:
            return {}
        out: dict[str, Any] = {}
        try:
            for m in _KV_RE.finditer(query):
                key = m.group("key").lower()
                if key not in _ALLOWED_KEYS:
                    continue
                raw = m.group("val").strip().strip("\"'")
                coerce = _ALLOWED_KEYS[key]
                try:
                    out[key] = coerce(raw)
                except (ValueError, TypeError):
                    continue  # skip malformed value, keep going
        except Exception as exc:  # pragma: no cover - defensive, never raise
            logger.warning("Filter extraction failed; ignoring: %s", exc)
            return {}
        return out
