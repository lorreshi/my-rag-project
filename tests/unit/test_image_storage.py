"""Unit tests for SQLiteImageStorage — file persistence + id->path mapping."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.storage.image_storage import SQLiteImageStorage


@pytest.fixture
def storage(tmp_path):
    return SQLiteImageStorage(
        images_base_dir=str(tmp_path / "images"),
        db_path=str(tmp_path / "db" / "image_index.db"),
    )


_FAKE_PNG = b"\x89PNG\r\n\x1a\n fake image content"


class TestSaveFromBytes:
    def test_file_created(self, storage):
        path = storage.save_image("img_001", _FAKE_PNG, collection="docs")
        assert Path(path).exists()
        assert Path(path).read_bytes() == _FAKE_PNG

    def test_stored_under_collection_dir(self, storage):
        path = storage.save_image("img_001", _FAKE_PNG, collection="mycoll")
        assert "mycoll" in path
        assert path.endswith("img_001.png")

    def test_mapping_persisted(self, storage):
        storage.save_image("img_001", _FAKE_PNG, collection="docs")
        assert storage.get_path("img_001") is not None

    def test_get_path_returns_correct(self, storage):
        path = storage.save_image("img_xyz", _FAKE_PNG)
        assert storage.get_path("img_xyz") == path


class TestSaveFromFile:
    def test_copy_from_path(self, storage, tmp_path):
        src = tmp_path / "source.png"
        src.write_bytes(_FAKE_PNG)
        path = storage.save_image("img_002", str(src), collection="docs")
        assert Path(path).exists()
        assert Path(path).read_bytes() == _FAKE_PNG

    def test_missing_source_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.save_image("img_003", "/nonexistent/file.png")


class TestLookup:
    def test_unknown_id_returns_none(self, storage):
        assert storage.get_path("does_not_exist") is None

    def test_list_by_collection(self, storage):
        storage.save_image("a", _FAKE_PNG, collection="c1")
        storage.save_image("b", _FAKE_PNG, collection="c1")
        storage.save_image("c", _FAKE_PNG, collection="c2")
        c1 = storage.list_by_collection("c1")
        assert len(c1) == 2
        assert {r["image_id"] for r in c1} == {"a", "b"}

    def test_list_empty_collection(self, storage):
        assert storage.list_by_collection("empty") == []

    def test_metadata_fields_recorded(self, storage):
        storage.save_image(
            "img_meta", _FAKE_PNG, collection="docs",
            doc_hash="abc123", page_num=4,
        )
        rec = storage.list_by_collection("docs")[0]
        assert rec["doc_hash"] == "abc123"
        assert rec["page_num"] == 4


class TestPersistenceAcrossInstances:
    def test_mapping_survives_reopen(self, tmp_path):
        db = str(tmp_path / "db" / "image_index.db")
        imgs = str(tmp_path / "images")
        s1 = SQLiteImageStorage(images_base_dir=imgs, db_path=db)
        path = s1.save_image("persist_01", _FAKE_PNG)

        s2 = SQLiteImageStorage(images_base_dir=imgs, db_path=db)
        assert s2.get_path("persist_01") == path

    def test_db_file_location(self, tmp_path):
        db = str(tmp_path / "db" / "image_index.db")
        SQLiteImageStorage(
            images_base_dir=str(tmp_path / "images"), db_path=db
        )
        assert Path(db).exists()


class TestDeletion:
    def test_delete_by_doc_hash(self, storage):
        storage.save_image("d1", _FAKE_PNG, collection="docs", doc_hash="H1")
        storage.save_image("d2", _FAKE_PNG, collection="docs", doc_hash="H1")
        storage.save_image("d3", _FAKE_PNG, collection="docs", doc_hash="H2")

        deleted = storage.delete_by_doc_hash("H1")
        assert deleted == 2
        assert storage.get_path("d1") is None
        assert storage.get_path("d2") is None
        assert storage.get_path("d3") is not None

    def test_delete_removes_files(self, storage):
        path = storage.save_image("d1", _FAKE_PNG, collection="docs", doc_hash="H1")
        assert Path(path).exists()
        storage.delete_by_doc_hash("H1")
        assert not Path(path).exists()

    def test_idempotent_save_overwrites(self, storage):
        storage.save_image("same_id", _FAKE_PNG, collection="docs")
        storage.save_image("same_id", b"new content", collection="docs")
        # still a single record
        recs = storage.list_by_collection("docs")
        assert len(recs) == 1
        assert Path(storage.get_path("same_id")).read_bytes() == b"new content"
