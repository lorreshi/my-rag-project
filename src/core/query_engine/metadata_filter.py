"""Shared metadata-filter matching (pre-filter & post-filter use one rule).

A single ``match_filters`` predicate is used by both the sparse-route
pre-filter (``SparseRetriever``) and the fused post-filter
(``HybridSearch._apply_metadata_filters``), so the two stages can never drift
apart in their missing-key policy.

Missing-key policy (per key):
- Structured fields (``STRUCTURED_FILTER_KEYS``, e.g. ``sheet_name``): STRICT —
  a candidate missing the key is excluded. Filtering by ``sheet_name`` thus
  returns only chunks belonging to that sheet.
- Generic fields (collection / doc_type / ...): LENIENT — a candidate missing
  the key is included, avoiding wrongly dropping recall on incomplete metadata.

A key that is present but unequal is always excluded.
"""
from __future__ import annotations

from typing import Any, Mapping

# Structured metadata fields produced at split time (e.g. by TableAwareSplitter).
STRUCTURED_FILTER_KEYS: frozenset[str] = frozenset(
    {"sheet_name", "row_start", "row_end", "is_table"}
)


def match_filters(
    metadata: Mapping[str, Any],
    filters: Mapping[str, Any] | None,
    structured_keys: "frozenset[str] | set[str]" = STRUCTURED_FILTER_KEYS,
) -> bool:
    """Return True if *metadata* satisfies *filters* under the missing policy."""
    if not filters:
        return True
    for key, value in filters.items():
        if key not in metadata:
            if key in structured_keys:
                return False  # structured: missing -> exclude (strict)
            continue  # generic: missing -> include (lenient)
        if metadata[key] != value:
            return False
    return True
