"""End-to-end tests for the query.py CLI (D7).

Hermetic: patches HybridSearch.from_settings and load_settings so no real
embedding/vector store/network is needed.
"""
from __future__ import annotations

import pytest

from src.core.settings import Settings
from src.core.types import RetrievalResult

import scripts.query as query_cli


class FakeHybrid:
    def __init__(self, results):
        self._results = results
        self.search_calls = []

    def search(self, query, top_k=10, filters=None, trace=None):
        self.search_calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self._results)


def _r(cid, score, text="", meta=None):
    return RetrievalResult(chunk_id=cid, score=score, text=text or cid, metadata=meta or {})


@pytest.fixture
def patch_pipeline(monkeypatch):
    """Patch settings + HybridSearch.from_settings with a configurable fake."""
    state = {"results": []}

    def _fake_from_settings(settings, **kw):
        return FakeHybrid(state["results"])

    monkeypatch.setattr(query_cli, "load_settings", lambda *a, **k: Settings())
    from src.core.query_engine.hybrid_search import HybridSearch
    monkeypatch.setattr(
        HybridSearch, "from_settings",
        classmethod(lambda cls, settings, **kw: _fake_from_settings(settings, **kw)),
    )
    return state


class TestRunQuery:
    def test_returns_results_no_rerank(self, patch_pipeline):
        patch_pipeline["results"] = [_r("a", 0.9), _r("b", 0.8)]
        results = query_cli.run_query("test query", top_k=10, use_rerank=False)
        assert [r.chunk_id for r in results] == ["a", "b"]

    def test_top_k_applied_no_rerank(self, patch_pipeline):
        patch_pipeline["results"] = [_r(c, 1.0) for c in "abcde"]
        results = query_cli.run_query("q", top_k=2, use_rerank=False)
        assert len(results) == 2

    def test_empty_results(self, patch_pipeline):
        patch_pipeline["results"] = []
        assert query_cli.run_query("q", use_rerank=False) == []

    def test_rerank_path_runs(self, patch_pipeline):
        patch_pipeline["results"] = [_r("a", 0.5), _r("b", 0.9)]
        # default reranker backend is built from Settings() -> 'cross_encoder';
        # with no model installed it uses the stub scorer, still deterministic.
        results = query_cli.run_query("learning models", top_k=2, use_rerank=True)
        assert len(results) <= 2
        # rerank metadata tagged
        assert "rerank_backend" in results[0].metadata


class TestMain:
    def test_main_prints_results(self, patch_pipeline, capsys):
        patch_pipeline["results"] = [_r("a", 0.9, text="hello world", meta={"source_path": "a.pdf"})]
        code = query_cli.main(["--query", "hello", "--no-rerank"])
        assert code == 0
        out = capsys.readouterr().out
        assert "hello world" in out
        assert "a.pdf" in out

    def test_main_no_results_message(self, patch_pipeline, capsys):
        patch_pipeline["results"] = []
        code = query_cli.main(["--query", "nothing", "--no-rerank"])
        assert code == 0
        err = capsys.readouterr().err
        assert "未找到相关文档" in err

    def test_collection_forwarded(self, patch_pipeline, monkeypatch):
        from src.core.query_engine.hybrid_search import HybridSearch

        fake = FakeHybrid([_r("a", 0.9)])
        monkeypatch.setattr(
            HybridSearch, "from_settings",
            classmethod(lambda cls, settings, **kw: fake),
        )
        query_cli.run_query("q", collection="mycoll", use_rerank=False)
        assert fake.search_calls[0]["filters"] == {"collection": "mycoll"}


class TestArgParsing:
    def test_defaults(self):
        args = query_cli._build_parser().parse_args(["--query", "x"])
        assert args.top_k == 10
        assert args.collection is None
        assert args.verbose is False
        assert args.no_rerank is False

    def test_all_flags(self):
        args = query_cli._build_parser().parse_args([
            "--query", "x", "--top-k", "5", "--collection", "c",
            "--verbose", "--no-rerank",
        ])
        assert args.top_k == 5
        assert args.collection == "c"
        assert args.verbose is True
        assert args.no_rerank is True

    def test_query_required(self):
        with pytest.raises(SystemExit):
            query_cli._build_parser().parse_args([])


class TestFormatting:
    def test_format_result(self):
        r = _r("a", 0.1234, text="some text", meta={"source_path": "doc.pdf", "page": 3})
        line = query_cli._format_result(1, r)
        assert "[1]" in line
        assert "0.1234" in line
        assert "doc.pdf" in line
        assert "p.3" in line
