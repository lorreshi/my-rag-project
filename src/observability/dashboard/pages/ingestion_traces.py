"""Ingestion Traces page (G5) — history + stage-duration waterfall."""
from __future__ import annotations

import streamlit as st

from src.core.settings import load_settings
from src.observability.dashboard.services.trace_service import TraceService


def _trace_file() -> str:
    try:
        settings = load_settings()
        log_file = settings.observability.log_file
        return log_file or "logs/traces.jsonl"
    except Exception:
        return "logs/traces.jsonl"


def render() -> None:
    st.title("🔍 Ingestion 追踪")
    st.caption("摄取历史与各阶段耗时分布")

    service = TraceService(_trace_file())
    traces = service.list_traces(trace_type="ingestion")

    if not traces:
        st.info("暂无 Ingestion 追踪记录。请先运行 ingest.py。")
        return

    labels = [
        f"{t.get('metadata', {}).get('source_path', t['trace_id'])} "
        f"({t.get('total_elapsed_ms', 0):.0f} ms)"
        for t in traces
    ]
    idx = st.selectbox("选择一次摄取", options=range(len(traces)), format_func=lambda i: labels[i])
    trace = traces[idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Trace ID", trace["trace_id"])
    c2.metric("总耗时 (ms)", f"{trace.get('total_elapsed_ms', 0):.1f}")
    c3.metric("阶段数", len(trace.get("stages", [])))

    st.subheader("阶段耗时瀑布图")
    durations = service.stage_durations(trace)
    if durations:
        import pandas as pd
        df = pd.DataFrame(durations).set_index("name")
        st.bar_chart(df, horizontal=True)

    st.subheader("阶段详情")
    for s in trace.get("stages", []):
        with st.expander(f"{s.get('name')} · {s.get('elapsed_ms', 0):.1f} ms"):
            st.json(s.get("details", {}))
