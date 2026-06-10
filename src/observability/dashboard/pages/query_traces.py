"""Query Traces page (G6) — history, waterfall, dense/sparse + rerank views."""
from __future__ import annotations

import streamlit as st

from src.core.settings import load_settings
from src.observability.dashboard.services.trace_service import TraceService


def _trace_file() -> str:
    try:
        settings = load_settings()
        return settings.observability.log_file or "logs/traces.jsonl"
    except Exception:
        return "logs/traces.jsonl"


def _stage(trace: dict, name: str) -> dict | None:
    for s in trace.get("stages", []):
        if s.get("name") == name:
            return s
    return None


def render() -> None:
    st.title("🔎 Query 追踪")
    st.caption("查询历史、各阶段耗时、Dense/Sparse 对比与 Rerank 变化")

    service = TraceService(_trace_file())

    keyword = st.text_input("按 Query 关键词搜索", value="")
    if keyword.strip():
        traces = service.search(keyword, trace_type="query")
    else:
        traces = service.list_traces(trace_type="query")

    if not traces:
        st.info("暂无 Query 追踪记录。请先运行 query.py 或调用 MCP 工具。")
        return

    labels = [
        f"{t.get('metadata', {}).get('query', t['trace_id'])} "
        f"({t.get('total_elapsed_ms', 0):.0f} ms)"
        for t in traces
    ]
    idx = st.selectbox("选择一次查询", options=range(len(traces)), format_func=lambda i: labels[i])
    trace = traces[idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Trace ID", trace["trace_id"])
    c2.metric("总耗时 (ms)", f"{trace.get('total_elapsed_ms', 0):.1f}")
    c3.metric("阶段数", len(trace.get("stages", [])))

    # Waterfall of stage durations
    st.subheader("阶段耗时瀑布图")
    durations = service.stage_durations(trace)
    if durations:
        import pandas as pd
        st.bar_chart(pd.DataFrame(durations).set_index("name"), horizontal=True)

    # Dense vs Sparse side-by-side
    st.subheader("Dense vs Sparse 召回对比")
    dense = _stage(trace, "dense_retrieval")
    sparse = _stage(trace, "sparse_retrieval")
    col_d, col_s = st.columns(2)
    with col_d:
        st.markdown("**Dense**")
        st.json(dense.get("details", {}) if dense else {"(无)": True})
    with col_s:
        st.markdown("**Sparse**")
        st.json(sparse.get("details", {}) if sparse else {"(无)": True})

    # Rerank change
    st.subheader("Rerank")
    rerank = _stage(trace, "rerank")
    if rerank:
        details = rerank.get("details", {})
        st.write(f"后端: `{details.get('backend', '—')}` · fallback: `{details.get('fallback')}`")
        st.json(details)
    else:
        st.caption("本次查询未执行 Rerank。")
