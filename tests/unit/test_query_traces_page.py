"""Unit tests for the query traces page helpers (G6)."""
from __future__ import annotations

import json

from src.observability.dashboard.pages import query_traces
from src.observability.dashboard.services.trace_service import TraceService


def _write_log(tmp_path, traces):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join(json.dumps({"trace": t}) for t in traces) + "\n", encoding="utf-8"
    )
    return str(path)


def _query_trace(tid, query, started):
    return {
        "trace_id": tid,
        "trace_type": "query",
        "started_at": started,
        "total_elapsed_ms": 50.0,
        "metadata": {"query": query},
        "stages": [
            {"name": "query_processing", "elapsed_ms": 1.0, "details": {"method": "keyword_extraction"}},
            {"name": "dense_retrieval", "elapsed_ms": 10.0, "details": {"method": "vector_search", "count": 5}},
            {"name": "sparse_retrieval", "elapsed_ms": 8.0, "details": {"method": "bm25", "count": 4}},
            {"name": "fusion", "elapsed_ms": 1.0, "details": {"method": "rrf"}},
            {"name": "rerank", "elapsed_ms": 20.0, "details": {"backend": "cross_encoder", "fallback": False}},
        ],
    }


class TestStageHelper:
    def test_finds_stage(self):
        trace = _query_trace("a", "q", 1)
        assert query_traces._stage(trace, "dense_retrieval")["details"]["count"] == 5

    def test_missing_stage_returns_none(self):
        trace = _query_trace("a", "q", 1)
        assert query_traces._stage(trace, "nonexistent") is None


class TestTraceServiceWithQueries:
    def test_lists_only_query_traces(self, tmp_path):
        f = _write_log(tmp_path, [
            _query_trace("a", "azure config", 1),
            {"trace_id": "b", "trace_type": "ingestion", "started_at": 2, "stages": [], "metadata": {}},
        ])
        svc = TraceService(f)
        q = svc.list_traces(trace_type="query")
        assert len(q) == 1
        assert q[0]["trace_id"] == "a"

    def test_search_by_query_keyword(self, tmp_path):
        f = _write_log(tmp_path, [
            _query_trace("a", "how to configure azure", 1),
            _query_trace("b", "vector search basics", 2),
        ])
        svc = TraceService(f)
        results = svc.search("azure", trace_type="query")
        assert len(results) == 1
        assert results[0]["trace_id"] == "a"

    def test_stage_durations_for_query(self, tmp_path):
        trace = _query_trace("a", "q", 1)
        durations = TraceService.stage_durations(trace)
        names = [d["name"] for d in durations]
        assert names == [
            "query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank"
        ]


class TestPageImport:
    def test_render_callable(self):
        assert callable(query_traces.render)
