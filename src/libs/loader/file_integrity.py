"""File integrity checker — SHA256 hash-based deduplication.

Provides an abstract interface and a default SQLite implementation for
tracking which files have been successfully ingested, enabling incremental
(zero-cost) updates for unchanged files.
"""
from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class FileIntegrityChecker(ABC):
    """Abstract interface for file integrity checking."""

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """Compute SHA256 hash of a file."""
        ...

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return True if this file hash was already successfully processed."""
        ...

    @abstractmethod
    def mark_success(
        self, file_hash: str, file_path: str, chunk_count: int = 0
    ) -> None:
        """Record a successful ingestion."""
        ...

    @abstractmethod
    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        """Record a failed ingestion attempt."""
        ...

    @abstractmethod
    def remove_record(self, file_hash: str) -> None:
        """Remove a record (allows re-ingestion)."""
        ...

    @abstractmethod
    def list_processed(self) -> list[dict[str, Any]]:
        """List all processed file records."""
        ...


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_history (
    file_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_msg TEXT,
    chunk_count INTEGER
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status);",
    "CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at);",
]


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed file integrity checker.

    Stores ingestion history in a local SQLite database with WAL mode
    for safe concurrent access.
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDEXES_SQL:
                conn.execute(idx_sql)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def compute_sha256(self, path: str) -> str:
        """Compute SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """Return True if this hash has status='success'."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            ).fetchone()
        return row is not None

    def mark_success(
        self, file_hash: str, file_path: str, chunk_count: int = 0
    ) -> None:
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, file_size, status, chunk_count)
                   VALUES (?, ?, ?, 'success', ?)""",
                (file_hash, file_path, file_size, chunk_count),
            )

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, file_size, status, error_msg)
                   VALUES (?, ?, ?, 'failed', ?)""",
                (file_hash, file_path, file_size, error_msg),
            )

    def remove_record(self, file_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )

    def list_processed(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_history ORDER BY processed_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
