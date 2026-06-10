"""Unit tests for TraceContext (F1) and TraceCollector."""
from __future__ import annotations

import json
import time

from src.core.trace.trace_context import TraceContext, StageRecord
from src.core.trace.trace_collector import TraceCollector


class TestStageRecording:
    def test_start_end_stage(self):
        t = TraceContext(trace_type="query")
        t.start_stage("dense")
        t.end_stage(details={"count": 3})
        assert len(t.stages) == 1
        assert t.stages[0].name == "dense"
        assert t.stages[0].details["count"] == 3

    def test_record_stage_oneshot(self):
        t = TraceContext()
        t.record_stage("fusion", {"algorithm": "rrf"})
        assert t.stages[0].name == "fusion"
        assert t.stages[0].details["algorithm"] == "rrf"

    def test_nested_stages(self):
        t = TraceContext()
        t.start_stage("outer")
        t.start_stage("inner")
        t.end_stage(details={"x": 1})
        t.end_stage(details={"y": 2})
        names = [s.name for s in t.stages]
        assert names == ["inner", "outer"]

    def test_end_without_start_noop(self):
        t = TraceContext()
        t.end_stage()  # should not raise
        assert t.stages == []


class TestTraceType:
    def test_default_query(self):
        assert TraceContext().trace_type == "query"

    def test_ingestion(self):
        assert TraceContext(trace_type="ingestion").trace_type == "ingestion"

    def test_explicit_trace_id(self):
        assert TraceContext(trace_id="abc123").trace_id == "abc123"

    def test_auto_trace_id(self):
        assert len(TraceContext().trace_id) == 16


class TestFinishAndTiming:
    def test_finish_sets_finished_at(self):
        t = TraceContext()
        assert t.finished_at is None
        t.finish()
        assert t.finished_at is not None

    def test_finish_idempotent(self):
        t = TraceContext()
        t.finish()
        first = t.finished_at
        t.finish()
        assert t.finished_at == first

    def test_total_elapsed_positive(self):
        t = TraceContext()
        time.sleep(0.01)
        t.finish()
        assert t.elapsed_ms() >= 10

    def test_stage_elapsed(self):
        t = TraceContext()
        t.start_stage("s")
        time.sleep(0.01)
        t.end_stage()
        assert t.elapsed_ms("s") >= 10

    def test_stage_elapsed_unknown_zero(self):
        t = TraceContext()
        assert t.elapsed_ms("nonexistent") == 0.0


class TestToDict:
    def test_required_fields(self):
        t = TraceContext(trace_type="ingestion")
        t.start_stage("load")
        t.end_stage(details={"method": "markitdown"})
        t.finish()
        d = t.to_dict()
        for key in ("trace_id", "trace_type", "started_at", "finished_at",
                    "total_elapsed_ms", "stages"):
            assert key in d

    def test_trace_type_in_dict(self):
        t = TraceContext(trace_type="ingestion")
        t.finish()
        assert t.to_dict()["trace_type"] == "ingestion"

    def test_json_serializable(self):
        t = TraceContext()
        t.start_stage("dense")
        t.end_stage(details={"count": 5, "provider": "openai"})
        t.finish()
        # Must not raise
        s = json.dumps(t.to_dict())
        assert "dense" in s

    def test_stages_have_elapsed_ms(self):
        t = TraceContext()
        t.start_stage("rerank")
        t.end_stage()
        t.finish()
        assert "elapsed_ms" in t.to_dict()["stages"][0]

    def test_metadata_included(self):
        t = TraceContext()
        t.set_metadata(query="hello", collection="docs")
        t.finish()
        assert t.to_dict()["metadata"]["query"] == "hello"


class TestTraceCollector:
    def test_collect_returns_payload(self):
        captured = []
        c = TraceCollector(writer=captured.append)
        t = TraceContext(trace_type="query")
        t.record_stage("dense")
        payload = c.collect(t)
        assert payload["trace_type"] == "query"

    def test_collect_finishes_trace(self):
        c = TraceCollector(writer=lambda d: None)
        t = TraceContext()
        c.collect(t)
        assert t.finished_at is not None

    def test_collect_persists_via_writer(self):
        captured = []
        c = TraceCollector(writer=captured.append)
        c.collect(TraceContext())
        assert len(captured) == 1

    def test_writer_failure_not_raised(self):
        def _bad(d):
            raise IOError("disk full")

        c = TraceCollector(writer=_bad)
        # Should not raise
        c.collect(TraceContext())

    def test_collected_history(self):
        c = TraceCollector(writer=lambda d: None)
        c.collect(TraceContext())
        c.collect(TraceContext())
        assert len(c.collected) == 2
