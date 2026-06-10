"""Structured logging utilities.

Provides:
- ``get_logger`` — human-readable stderr logger (unchanged contract).
- ``JSONFormatter`` — emits log records as single-line JSON.
- ``get_trace_logger`` — a JSON Lines logger writing to a trace file.
- ``write_trace`` — append a trace dict as one JSON line to logs/traces.jsonl.

stdout is never used here — MCP stdio transport requires a clean stdout.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_DEFAULT_TRACE_FILE = "logs/traces.jsonl"


def get_logger(name: str = "smart-knowledge-hub") -> logging.Logger:
    """Return a logger that writes human-readable lines to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach a structured trace payload if provided via extra={"trace": ...}.
        trace = getattr(record, "trace", None)
        if trace is not None:
            payload["trace"] = trace
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_trace_logger(
    name: str | None = None,
    trace_file: str = _DEFAULT_TRACE_FILE,
) -> logging.Logger:
    """Return a logger that writes JSON Lines to *trace_file*.

    The logger is keyed by *trace_file* (unless an explicit *name* is given) so
    that distinct files get distinct handlers. Configured once per logger.
    """
    if name is None:
        name = f"skh-trace:{trace_file}"
    logger = logging.getLogger(name)
    if not getattr(logger, "_skh_trace_configured", False):
        path = Path(trace_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._skh_trace_configured = True  # type: ignore[attr-defined]
        logger._skh_trace_file = str(path)  # type: ignore[attr-defined]
    return logger


def write_trace(
    trace_dict: dict[str, Any],
    trace_file: str = _DEFAULT_TRACE_FILE,
) -> None:
    """Append a trace dict as one JSON line to *trace_file*.

    Used by TraceCollector (F1) to persist query/ingestion traces.
    """
    logger = get_trace_logger(trace_file=trace_file)
    logger.info("trace", extra={"trace": trace_dict})
