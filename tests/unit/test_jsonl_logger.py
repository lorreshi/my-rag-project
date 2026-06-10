"""Unit tests for JSON Lines logging + trace persistence (F2)."""
from __future__ import annotations

import json
import logging

import pytest

from src.observability.logger import (
    JSONFormatter,
    get_logger,
    get_trace_logger,
    write_trace,
)


class TestJSONFormatter:
    def test_outputs_valid_json(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "f.py", 1, "hello", None, None
        )
        out = fmt.format(record)
        parsed = json.loads(out)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"

    def test_includes_trace_payload(self):
        fmt = JSONFormatter()
        record = logging.LogRecord("t", logging.INFO, "f.py", 1, "m", None, None)
        record.trace = {"trace_id": "abc", "trace_type": "query"}
        parsed = json.loads(fmt.format(record))
        assert parsed["trace"]["trace_id"] == "abc"

    def test_includes_exception(self):
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = logging.LogRecord(
                "t", logging.ERROR, "f.py", 1, "err", None, sys.exc_info()
            )
        parsed = json.loads(fmt.format(record))
        assert "exc_info" in parsed


class TestWriteTrace:
    def test_writes_one_line(self, tmp_path):
        trace_file = str(tmp_path / "traces.jsonl")
        write_trace({"trace_id": "t1", "trace_type": "query"}, trace_file=trace_file)
        lines = (tmp_path / "traces.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_line_is_valid_json_with_trace_type(self, tmp_path):
        trace_file = str(tmp_path / "traces.jsonl")
        write_trace({"trace_id": "t1", "trace_type": "ingestion"}, trace_file=trace_file)
        line = (tmp_path / "traces.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["trace"]["trace_type"] == "ingestion"

    def test_appends_multiple(self, tmp_path):
        trace_file = str(tmp_path / "traces.jsonl")
        write_trace({"trace_id": "t1", "trace_type": "query"}, trace_file=trace_file)
        write_trace({"trace_id": "t2", "trace_type": "query"}, trace_file=trace_file)
        lines = (tmp_path / "traces.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        ids = {json.loads(l)["trace"]["trace_id"] for l in lines}
        assert ids == {"t1", "t2"}

    def test_creates_parent_dir(self, tmp_path):
        trace_file = str(tmp_path / "nested" / "dir" / "traces.jsonl")
        write_trace({"trace_id": "t1", "trace_type": "query"}, trace_file=trace_file)
        assert (tmp_path / "nested" / "dir" / "traces.jsonl").exists()


class TestGetTraceLogger:
    def test_writes_jsonl(self, tmp_path):
        trace_file = str(tmp_path / "t.jsonl")
        logger = get_trace_logger(name="test-trace-unique", trace_file=trace_file)
        logger.info("trace", extra={"trace": {"k": "v"}})
        for h in logger.handlers:
            h.flush()
        content = (tmp_path / "t.jsonl").read_text().strip()
        assert json.loads(content)["trace"]["k"] == "v"

    def test_does_not_propagate(self, tmp_path):
        logger = get_trace_logger(name="test-noprop", trace_file=str(tmp_path / "t.jsonl"))
        assert logger.propagate is False


class TestGetLoggerUnchanged:
    def test_get_logger_returns_logger(self):
        logger = get_logger("test-plain")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_has_handler(self):
        logger = get_logger("test-plain-2")
        assert len(logger.handlers) >= 1
