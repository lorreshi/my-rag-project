"""Data Browser page (G3) — browse documents, chunks, and images."""
from __future__ import annotations

import streamlit as st

from src.core.settings import load_settings


def _build_data_service():
    """Construct a DataService from settings (best effort)."""
    from src.ingestion.document_manager import DocumentManager
    from src.ingestion.storage.image_storage import SQLiteImageStorage
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    import src.libs.vector_store.chroma_store  # noqa: F401
    from src.observability.dashboard.services.data_service import DataService

    settings = load_settings()
    store = VectorStoreFactory.create(settings)
    images = SQLiteImageStorage()
    dm = DocumentManager(store, image_storage=images)
    return DataService(dm, image_storage=images)


def render() -> None:
    st.title("📁 数据浏览器")
    st.caption("浏览已索引的文档、Chunk 详情与关联图片")

    try:
        service = _build_data_service()
    except Exception as exc:
        st.error(f"无法连接数据存储: {exc}")
        return

    collections = service.list_collections()
    selected = st.selectbox(
        "集合筛选", options=["(全部)"] + collections, index=0
    )
    collection = None if selected == "(全部)" else selected

    documents = service.list_documents(collection)
    if not documents:
        st.info("暂无已摄入的文档。请先运行 ingest.py。")
        return

    st.subheader(f"文档列表（{len(documents)}）")
    for doc in documents:
        label = f"{doc['source_path']} · {doc['chunk_count']} chunks · {doc['image_count']} images"
        with st.expander(label):
            chunks = service.get_chunks(doc["source_path"])
            for chunk in chunks:
                meta = chunk.get("metadata", {}) or {}
                idx = meta.get("chunk_index", "?")
                st.markdown(f"**Chunk #{idx}** · `{chunk.get('id', '')}`")
                st.text(chunk.get("text", ""))
                if meta:
                    st.json(meta, expanded=False)
                for img in service.chunk_images(chunk):
                    if img["path"]:
                        try:
                            st.image(img["path"], caption=img["id"], width=240)
                        except Exception:
                            st.caption(f"图片: {img['id']} ({img['path']})")
                st.divider()
