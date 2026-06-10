"""Trace context — full implementation (Phase F).

Records pipeline stage timings + details for a single query or ingestion run,
and serializes to a JSON-ready dict for persistence / dashboard display.

Usage::

    trace = TraceContext(trace_type="ingestion")
    trace.start_stage("transform")
    # ... do work ...
    trace.end_stage(details={"chunks_refined": 5})
    trace.finish()
    payload = trace.to_dict()
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class StageRecord:
    """Record of a single pipeline stage execution."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class TraceContext:
    """Trace context for recording pipeline stages and total elapsed time."""

    def __init__(self, trace_type: str = "query", trace_id: str | None = None):
        """Initialize a trace.

        Args:
            trace_type: ``"query"`` or ``"ingestion"``.
            trace_id: Optional explicit id; a random 16-hex id otherwise.
        """
        self.trace_id: str = trace_id or uuid.uuid4().hex[:16]
        self.trace_type: str = trace_type
        self.started_at: float = time()
        self.finished_at: float | None = None
        self.stages: list[StageRecord] = []
        self.metadata: dict[str, Any] = {}
        # Stack of in-progress stages, supports nested start/end calls.
        self._stage_stack: list[StageRecord] = []

    # ------------------------------------------------------------------
    # Stage recording
    # ------------------------------------------------------------------

    def start_stage(self, name: str) -> None:
        """Begin recording a new stage (supports nesting)."""
        self._stage_stack.append(StageRecord(name=name, start_time=time()))

    def end_stage(self, details: dict[str, Any] | None = None) -> None:
        """Finish the most recently started stage and store its record."""
        if not self._stage_stack:
            return
        stage = self._stage_stack.pop()
        stage.end_time = time()
        if details:
            stage.details.update(details)
        self.stages.append(stage)

    def record_stage(
        self, name: str, details: dict[str, Any] | None = None
    ) -> None:
        """Record a completed stage in one call (zero-duration timestamp)."""
        now = time()
        record = StageRecord(name=name, start_time=now, end_time=now)
        if details:
            record.details = dict(details)
        self.stages.append(record)

    def set_metadata(self, **kwargs: Any) -> None:
        """Attach top-level metadata (e.g. query text, source_path)."""
        self.metadata.update(kwargs)

    # ------------------------------------------------------------------
    # Lifecycle / timing
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Mark the trace as finished and freeze the total elapsed time."""
        if self.finished_at is None:
            self.finished_at = time()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return elapsed time in milliseconds.

        Args:
            stage_name: If given, return that stage's duration (sum if multiple
                stages share the name). Otherwise return the total trace time.
        """
        if stage_name is not None:
            total = sum(
                s.duration_ms for s in self.stages if s.name == stage_name
            )
            return round(total, 2)
        end = self.finished_at if self.finished_at is not None else time()
        return round((end - self.started_at) * 1000, 2)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a JSON-ready dict."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.elapsed_ms(),
            "metadata": self.metadata,
            "stages": [
                {
                    "name": s.name,
                    "elapsed_ms": round(s.duration_ms, 2),
                    "details": s.details,
                }
                for s in self.stages
            ],
        }
