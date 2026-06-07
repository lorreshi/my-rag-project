"""Trace context — minimal implementation for Phase C.

Provides TraceContext for recording pipeline stage data.
Full implementation (JSON export, collector integration) deferred to Phase F.
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
    """Lightweight trace context for recording pipeline stages.

    Usage::

        trace = TraceContext(trace_type="ingestion")
        trace.start_stage("transform")
        # ... do work ...
        trace.end_stage(details={"chunks_refined": 5})
    """

    def __init__(self, trace_type: str = "query", trace_id: str | None = None):
        self.trace_id: str = trace_id or uuid.uuid4().hex[:16]
        self.trace_type: str = trace_type
        self.stages: list[StageRecord] = []
        self._current_stage: StageRecord | None = None

    def start_stage(self, name: str) -> None:
        """Begin recording a new stage."""
        self._current_stage = StageRecord(name=name, start_time=time())

    def end_stage(self, details: dict[str, Any] | None = None) -> None:
        """Finish the current stage and store its record."""
        if self._current_stage is None:
            return
        self._current_stage.end_time = time()
        if details:
            self._current_stage.details.update(details)
        self.stages.append(self._current_stage)
        self._current_stage = None

    def record_stage(
        self, name: str, details: dict[str, Any] | None = None
    ) -> None:
        """Record a completed stage in one call (no timing)."""
        record = StageRecord(name=name, start_time=time(), end_time=time())
        if details:
            record.details = details
        self.stages.append(record)

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace to dict (for future JSON export)."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "details": s.details,
                }
                for s in self.stages
            ],
        }
