"""Tests for file integrity checker (C2)."""
from __future__ import annotations

import pytest
from pathlib import Path

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


@pytest.fixture
def checker(tmp_path: Path) -> SQLiteIntegrityChecker:
    db_path = str(tmp_path / "test_history.db")
    return SQLiteIntegrityChecker(db_path=db_path)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("hello world")
    return f


@pytest.mark.unit
class TestSQLiteIntegrityChecker:

    def test_compute_sha256_deterministic(self, checker, sample_file):
        h1 = checker.compute_sha256(str(sample_file))
        h2 = checker.compute_sha256(str(sample_file))
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex length

    def test_compute_sha256_different_content(self, checker, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert checker.compute_sha256(str(f1)) != checker.compute_sha256(str(f2))

    def test_should_skip_false_initially(self, checker):
        assert checker.should_skip("nonexistent_hash") is False

    def test_mark_success_then_skip(self, checker, sample_file):
        h = checker.compute_sha256(str(sample_file))
        assert checker.should_skip(h) is False
        checker.mark_success(h, str(sample_file), chunk_count=5)
        assert checker.should_skip(h) is True

    def test_mark_failed_does_not_skip(self, checker, sample_file):
        h = checker.compute_sha256(str(sample_file))
        checker.mark_failed(h, str(sample_file), "parse error")
        assert checker.should_skip(h) is False

    def test_remove_record_allows_reingestion(self, checker, sample_file):
        h = checker.compute_sha256(str(sample_file))
        checker.mark_success(h, str(sample_file))
        assert checker.should_skip(h) is True
        checker.remove_record(h)
        assert checker.should_skip(h) is False

    def test_list_processed(self, checker, sample_file):
        h = checker.compute_sha256(str(sample_file))
        checker.mark_success(h, str(sample_file), chunk_count=3)
        records = checker.list_processed()
        assert len(records) == 1
        assert records[0]["file_hash"] == h
        assert records[0]["status"] == "success"
        assert records[0]["chunk_count"] == 3

    def test_list_processed_empty(self, checker):
        assert checker.list_processed() == []

    def test_mark_success_updates_existing(self, checker, sample_file):
        h = checker.compute_sha256(str(sample_file))
        checker.mark_failed(h, str(sample_file), "first attempt failed")
        checker.mark_success(h, str(sample_file), chunk_count=10)
        assert checker.should_skip(h) is True
        records = checker.list_processed()
        assert len(records) == 1
        assert records[0]["status"] == "success"

    def test_db_file_created(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "history.db")
        SQLiteIntegrityChecker(db_path=db_path)
        assert Path(db_path).exists()

    def test_concurrent_safe_wal_mode(self, tmp_path):
        """Verify WAL mode is enabled (basic check)."""
        db_path = str(tmp_path / "wal_test.db")
        checker = SQLiteIntegrityChecker(db_path=db_path)
        import sqlite3
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        conn.close()
        assert mode == "wal"


@pytest.mark.unit
class TestFileIntegrityCheckerABC:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            FileIntegrityChecker()
