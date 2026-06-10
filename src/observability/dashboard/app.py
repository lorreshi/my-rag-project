"""Streamlit dashboard entry point — multi-page navigation.

Registers the six dashboard pages via st.navigation. Pages that are not yet
implemented render a placeholder notice.

Run with:
    streamlit run src/observability/dashboard/app.py
or:
    python scripts/start_dashboard.py
"""
from __future__ import annotations

import streamlit as st

from src.observability.dashboard.pages import (
    data_browser,
    ingestion_manager,
    ingestion_traces,
    overview,
    query_traces,
)

try:
    from src.observability.dashboard.pages import evaluation_panel
    _HAS_EVAL = True
except Exception:
    _HAS_EVAL = False


def _page(render_fn, title: str):
    """Wrap a render function as a Streamlit page callable."""
    def _runner():
        render_fn()
    _runner.__name__ = title.replace(" ", "_")
    return _runner


def main() -> None:
    st.set_page_config(page_title="Smart Knowledge Hub", layout="wide")

    pages = [
        st.Page(_page(overview.render, "overview"), title="系统总览", icon="📊"),
        st.Page(_page(data_browser.render, "data_browser"), title="数据浏览器", icon="📁"),
        st.Page(_page(ingestion_manager.render, "ingestion_manager"), title="Ingestion 管理", icon="📥"),
        st.Page(_page(ingestion_traces.render, "ingestion_traces"), title="Ingestion 追踪", icon="🔍"),
        st.Page(_page(query_traces.render, "query_traces"), title="Query 追踪", icon="🔎"),
    ]
    if _HAS_EVAL:
        pages.append(
            st.Page(_page(evaluation_panel.render, "evaluation_panel"), title="评估面板", icon="📈")
        )

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
