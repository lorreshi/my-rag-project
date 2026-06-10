"""TraceCollector — collect finished traces and persist them.

Bridges TraceContext (F1) and the JSON Lines trace log (F2). ``collect()``
finishes the trace if needed and appends its serialized form to the trace log.

Persistence is delegated to a writer callable (defaults to
``observability.logger.write_trace``) so the collector stays decoupled and
testable. Failures to persist are logged but never raised — observability must
not break the main flow.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class TraceCollector:
    """Collect traces and persist them via a pluggable writer."""

    def __init__(self, writer: Callable[[dict[str, Any]], None] | None = None):
        """Initialize.

        Args:
            writer: Callable that persists a trace dict. Defaults to
                ``observability.logger.write_trace`` (lazy import).
        """
        self._writer = writer
        self.collected: list[dict[str, Any]] = []

    def collect(self, trace: "TraceContext") -> dict[str, Any]:
        """Finish, serialize, persist, and return the trace dict."""
        trace.finish()
        payload = trace.to_dict()
        self.collected.append(payload)
        self._persist(payload)
        return payload

    def _persist(self, payload: dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            try:
                from src.observability.logger import write_trace
                writer = write_trace
            except Exception as exc:  # pragma: no cover - import guard
                logger.warning("No trace writer available: %s", exc)
                return
        try:
            writer(payload)
        except Exception as exc:
            logger.warning("Failed to persist trace %s: %s", payload.get("trace_id"), exc)
