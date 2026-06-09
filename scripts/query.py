"""Online query CLI entry point.

Runs the full retrieval pipeline (HybridSearch + Reranker) against an ingested
collection and prints the Top-K results.

Usage::

    python scripts/query.py --query "How do I configure Azure?"
    python scripts/query.py --query "RRF fusion" --top-k 5 --verbose
    python scripts/query.py --query "vector search" --no-rerank
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger

logger = get_logger()


def _format_result(rank: int, result: Any) -> str:
    """Format a single RetrievalResult for display."""
    meta = result.metadata or {}
    source = meta.get("source_path", meta.get("source", "?"))
    page = meta.get("page", meta.get("page_num", ""))
    page_str = f" p.{page}" if page != "" else ""
    snippet = (result.text or "").strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:160] + "…"
    return (
        f"[{rank}] score={result.score:.4f}  ({source}{page_str})\n"
        f"    {snippet}"
    )


def run_query(
    query: str,
    top_k: int = 10,
    collection: str | None = None,
    use_rerank: bool = True,
    verbose: bool = False,
    config_path: str = "config/settings.yaml",
) -> list[Any]:
    """Execute a query and return the final ranked results."""
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import Reranker

    settings = load_settings(config_path)
    trace = TraceContext(trace_type="query")

    hybrid = HybridSearch.from_settings(settings)
    filters = {"collection": collection} if collection else None

    # Fetch a larger candidate pool when reranking will refine it.
    candidate_k = max(top_k, getattr(settings.rerank, "top_m", top_k)) if use_rerank else top_k
    candidates = hybrid.search(query, top_k=candidate_k, filters=filters, trace=trace)

    if verbose:
        _print_stage_summary(trace)

    if not candidates:
        return []

    if use_rerank:
        reranker = Reranker(settings=settings)
        results = reranker.rerank(query, candidates, top_k=top_k, trace=trace)
        if verbose:
            fb = results[0].metadata.get("rerank_fallback") if results else None
            print(f"  [rerank] backend used, fallback={fb}", file=sys.stderr)
    else:
        results = candidates[:top_k]

    return results


def _print_stage_summary(trace: TraceContext) -> None:
    """Print per-stage trace details to stderr (verbose mode)."""
    print("--- Stage summary ---", file=sys.stderr)
    for stage in trace.stages:
        print(f"  {stage.name}: {stage.details}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query the Smart Knowledge Hub."
    )
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results.")
    parser.add_argument("--collection", default=None, help="Limit to a collection.")
    parser.add_argument(
        "--verbose", action="store_true", help="Show intermediate stage results."
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip the rerank stage."
    )
    parser.add_argument(
        "--config", default="config/settings.yaml", help="Settings YAML path."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    try:
        results = run_query(
            query=args.query,
            top_k=args.top_k,
            collection=args.collection,
            use_rerank=not args.no_rerank,
            verbose=args.verbose,
            config_path=args.config,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - surfaced to user
        logger.error("Query failed: %s", exc)
        return 1

    if not results:
        print(
            "未找到相关文档，请先运行 scripts/ingest.py 摄取数据。",
            file=sys.stderr,
        )
        return 0

    print(f"\nTop {len(results)} results for: {args.query}\n")
    for i, result in enumerate(results, start=1):
        print(_format_result(i, result))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
