"""System Overview page — component config + data statistics."""
from __future__ import annotations

import streamlit as st

from src.core.settings import load_settings
from src.observability.dashboard.services.config_service import ConfigService


def _load_vector_store(settings):
    """Best-effort vector store construction for stats (may fail offline)."""
    try:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        import src.libs.vector_store.chroma_store  # noqa: F401  (register backend)
        return VectorStoreFactory.create(settings)
    except Exception:
        return None


def render() -> None:
    """Render the system overview page."""
    st.title("📊 系统总览")
    st.caption("当前可插拔组件配置与数据资产统计")

    try:
        settings = load_settings()
    except Exception as exc:
        st.error(f"无法加载配置: {exc}")
        return

    service = ConfigService(settings)

    # Component configuration cards
    st.subheader("组件配置")
    cards = service.component_cards()
    cols = st.columns(3)
    for i, card in enumerate(cards):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{card['name']}**")
                st.write(f"Provider: `{card['provider'] or '—'}`")
                if card["model"]:
                    st.write(f"Model: `{card['model']}`")
                for k, v in card["details"].items():
                    st.caption(f"{k}: {v}")

    # Data statistics
    st.subheader("数据统计")
    vector_store = _load_vector_store(settings)
    stats = service.data_stats(vector_store)
    c1, c2 = st.columns(2)
    c1.metric("向量后端", stats.get("backend", "—"))
    c2.metric("Chunk 数量", stats.get("chunk_count", 0))
