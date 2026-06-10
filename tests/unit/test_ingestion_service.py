"""Unit tests for IngestionService (G4) + page import."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.observability.dashboard.services.ingestion_service import IngestionService


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, path, collection="default", force=False, on_progress=None):
        self.calls.append({"path": path, "collection": collection, "force": force})
        if on_progress:
            on_progress("load", 1, 1)
            on_progress("store", 1, 1)
        return {"path": path, "collection": collection}


class TestSaveUpload:
    def test_saves_file(self, tmp_path):
        svc = IngestionService(FakePipeline(), documents_base_dir=str(tmp_path))
        path = svc.save_upload("doc.pdf", b"%PDF data", collection="mycoll")
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"%PDF data"

    def test_saved_under_collection(self, tmp_path):
        svc = IngestionService(FakePipeline(), documents_base_dir=str(tmp_path))
        path = svc.save_upload("doc.pdf", b"x", collection="mycoll")
        assert "mycoll" in path
        assert path.endswith("doc.pdf")

    def test_strips_path_components(self, tmp_path):
        svc = IngestionService(FakePipeline(), documents_base_dir=str(tmp_path))
        path = svc.save_upload("../../etc/evil.pdf", b"x", collection="c")
        # only the basename is used
        assert path.endswith("evil.pdf")
        assert ".." not in Path(path).parts

    def test_empty_filename_raises(self, tmp_path):
        svc = IngestionService(FakePipeline(), documents_base_dir=str(tmp_path))
        with pytest.raises(ValueError):
            svc.save_upload("", b"x")


class TestIngest:
    def test_runs_pipeline(self, tmp_path):
        pipeline = FakePipeline()
        svc = IngestionService(pipeline, documents_base_dir=str(tmp_path))
        svc.ingest("doc.pdf", collection="c", force=True)
        assert pipeline.calls[0]["collection"] == "c"
        assert pipeline.calls[0]["force"] is True

    def test_progress_forwarded(self, tmp_path):
        pipeline = FakePipeline()
        svc = IngestionService(pipeline, documents_base_dir=str(tmp_path))
        seen = []
        svc.ingest("doc.pdf", on_progress=lambda s, c, t: seen.append(s))
        assert "load" in seen and "store" in seen


class TestPageImport:
    def test_render_callable(self):
        from src.observability.dashboard.pages import ingestion_manager
        assert callable(ingestion_manager.render)
