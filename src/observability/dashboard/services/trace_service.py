"""TraceService — parse logs/traces.jsonl into trace records (G5/G6).

Reads the JSON Lines trace log written by ``observability.logger.write_trace``
and exposes filtered, time-ordered trace dicts for the dashboard tracing pages.
Each log line looks like ``{"timestamp":..., "trace": {<trace dict>}}``.
Pure data layer — no Streamlit imports, unit-testable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TRACE_FILE = "logs/traces.jsonl"


class TraceService:
    """Read and filter traces from a JSON Lines file."""

    def __init__(self, trace_file: str = _DEFAULT_TRACE_FILE):
        self._path = Path(trace_file)

    def _read_all(self) -> list[dict[str, Any]]:
        """Parse all trace records from the log file (skips bad lines)."""
        if not self._path.exists():
            return []
        traces: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed trace line")
                continue
            # The trace payload is nested under "trace" (see logger.write_trace).
            trace = entry.get("trace", entry)
            if isinstance(trace, dict) and trace.get("trace_id"):
                traces.append(trace)
        return traces

    def list_traces(
        self,
        trace_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return traces, optionally filtered by type, newest first."""
        traces = self._read_all()
        if trace_type is not None:
            traces = [t for t in traces if t.get("trace_type") == trace_type]
        # Newest first by started_at (fallback: keep file order reversed).
        traces.sort(key=lambda t: t.get("started_at", 0), reverse=True)
        if limit is not None:
            traces = traces[:limit]
        return traces

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Return a single trace by id, or None."""
        for t in self._read_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    @staticmethod
    def stage_durations(trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Return [{name, elapsed_ms}] for a trace's stages (waterfall data)."""
        return [
            {"name": s.get("name", "?"), "elapsed_ms": s.get("elapsed_ms", 0.0)}
            for s in trace.get("stages", [])
        ]

    def search(self, keyword: str, trace_type: str | None = None) -> list[dict[str, Any]]:
        """Search traces whose metadata values contain *keyword* (case-insensitive)."""
        kw = keyword.lower()
        results = []
        for t in self.list_traces(trace_type=trace_type):
            blob = json.dumps(t.get("metadata", {}), ensure_ascii=False).lower()
            if kw in blob:
                results.append(t)
        return results
