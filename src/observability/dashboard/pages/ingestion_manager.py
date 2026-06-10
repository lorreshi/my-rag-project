"""Ingestion Manager page (G4) — upload, ingest with progress, delete."""
from __future__ import annotations

import streamlit as st

from src.core.settings import load_settings

_STAGES = ["load", "split", "transform", "embed", "store"]


def _build_services():
    from src.ingestion.document_manager import DocumentManager
    from src.ingestion.pipeline import IngestionPipeline
    from src.ingestion.storage.image_storage import SQLiteImageStorage
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    import src.libs.vector_store.chroma_store  # noqa: F401
    from src.observability.dashboard.services.ingestion_service import IngestionService

    settings = load_settings()
    pipeline = IngestionPipeline.from_settings(settings)
    store = VectorStoreFactory.create(settings)
    images = SQLiteImageStorage()
    dm = DocumentManager(store, image_storage=images)
    return IngestionService(pipeline), dm


def render() -> None:
    st.title("📥 Ingestion 管理")
    st.caption("上传文件触发摄取、查看实时进度、删除已摄入文档")

    try:
        ingestion, dm = _build_services()
    except Exception as exc:
        st.error(f"初始化失败: {exc}")
        return

    # --- Upload + ingest ---
    st.subheader("上传与摄取")
    collection = st.text_input("集合名称", value="default")
    uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"])
    force = st.checkbox("强制重新摄取", value=False)

    if uploaded is not None and st.button("开始摄取"):
        path = ingestion.save_upload(uploaded.name, uploaded.getvalue(), collection)
        progress_bar = st.progress(0.0, text="准备中…")
        stage_index = {name: i for i, name in enumerate(_STAGES)}

        def _on_progress(stage, current, total):
            base = stage_index.get(stage, 0)
            frac = (base + (current / total if total else 1)) / len(_STAGES)
            progress_bar.progress(min(frac, 1.0), text=f"{stage} ({current}/{total})")

        try:
            result = ingestion.ingest(path, collection=collection, force=force, on_progress=_on_progress)
            progress_bar.progress(1.0, text="完成")
            if result.skipped:
                st.warning("文件未变更，已跳过。")
            else:
                st.success(f"摄取完成：{result.total_chunks} chunks, {result.total_images} images")
        except Exception as exc:
            st.error(f"摄取失败: {exc}")

    # --- Existing documents + delete ---
    st.subheader("已摄入文档")
    documents = dm.list_documents()
    if not documents:
        st.info("暂无已摄入文档。")
        return

    for doc in documents:
        col1, col2 = st.columns([4, 1])
        col1.write(f"`{doc.source_path}` · {doc.chunk_count} chunks · {doc.collection}")
        if col2.button("删除", key=f"del_{doc.source_path}"):
            result = dm.delete_document(doc.source_path, doc.collection)
            st.success(f"已删除 {result.chunks_deleted} chunks")
            st.rerun()
