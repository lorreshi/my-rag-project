"""Unit tests for TraceService (G5/G6)."""
from __future__ import annotations

import json

from src.observability.dashboard.services.trace_service import TraceService


def _write_log(tmp_path, traces):
    """Write trace dicts as JSON-Lines log entries (logger format)."""
    path = tmp_path / "traces.jsonl"
    lines = []
    for t in traces:
        lines.append(json.dumps({"timestamp": "t", "level": "INFO", "trace": t}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _trace(tid, ttype, started, source=None, stages=None):
    meta = {}
    if source:
        meta["source_path"] = source
    return {
        "trace_id": tid,
        "trace_type": ttype,
        "started_at": started,
        "total_elapsed_ms": 100.0,
        "metadata": meta,
        "stages": stages or [],
    }


class TestListTraces:
    def test_reads_all(self, tmp_path):
        f = _write_log(tmp_path, [_trace("a", "query", 1), _trace("b", "ingestion", 2)])
        svc = TraceService(f)
        assert len(svc.list_traces()) == 2

    def test_filter_by_type(self, tmp_path):
        f = _write_log(tmp_path, [_trace("a", "query", 1), _trace("b", "ingestion", 2)])
        svc = TraceService(f)
        ing = svc.list_traces(trace_type="ingestion")
        assert len(ing) == 1
        assert ing[0]["trace_id"] == "b"

    def test_newest_first(self, tmp_path):
        f = _write_log(tmp_path, [_trace("a", "query", 1), _trace("b", "query", 5), _trace("c", "query", 3)])
        svc = TraceService(f)
        ids = [t["trace_id"] for t in svc.list_traces()]
        assert ids == ["b", "c", "a"]

    def test_limit(self, tmp_path):
        f = _write_log(tmp_path, [_trace(str(i), "query", i) for i in range(5)])
        svc = TraceService(f)
        assert len(svc.list_traces(limit=2)) == 2

    def test_missing_file(self, tmp_path):
        svc = TraceService(str(tmp_path / "nope.jsonl"))
        assert svc.list_traces() == []

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        path.write_text(
            json.dumps({"trace": _trace("a", "query", 1)}) + "\n"
            + "{ not json\n"
            + json.dumps({"trace": _trace("b", "query", 2)}) + "\n"
        )
        svc = TraceService(str(path))
        assert len(svc.list_traces()) == 2


class TestGetTrace:
    def test_get_by_id(self, tmp_path):
        f = _write_log(tmp_path, [_trace("a", "query", 1), _trace("b", "query", 2)])
        svc = TraceService(f)
        assert svc.get_trace("b")["trace_id"] == "b"

    def test_get_missing(self, tmp_path):
        f = _write_log(tmp_path, [_trace("a", "query", 1)])
        assert TraceService(f).get_trace("zzz") is None


class TestStageDurations:
    def test_extracts_durations(self):
        trace = _trace("a", "ingestion", 1, stages=[
            {"name": "load", "elapsed_ms": 10.0, "details": {}},
            {"name": "split", "elapsed_ms": 5.0, "details": {}},
        ])
        durations = TraceService.stage_durations(trace)
        assert durations == [
            {"name": "load", "elapsed_ms": 10.0},
            {"name": "split", "elapsed_ms": 5.0},
        ]

    def test_empty_stages(self):
        assert TraceService.stage_durations(_trace("a", "query", 1)) == []


class TestSearch:
    def test_search_metadata(self, tmp_path):
        f = _write_log(tmp_path, [
            _trace("a", "query", 1, source="azure_guide.pdf"),
            _trace("b", "query", 2, source="other.pdf"),
        ])
        svc = TraceService(f)
        results = svc.search("azure")
        assert len(results) == 1
        assert results[0]["trace_id"] == "a"

    def test_search_case_insensitive(self, tmp_path):
        f = _write_log(tmp_path, [_trace("a", "query", 1, source="AZURE.pdf")])
        assert len(TraceService(f).search("azure")) == 1


class TestPageImport:
    def test_render_callable(self):
        from src.observability.dashboard.pages import ingestion_traces
        assert callable(ingestion_traces.render)
