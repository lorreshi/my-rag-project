"""ImageStorage — image file persistence + SQLite id→path index.

Saves image files under ``data/images/{collection}/`` and records the
``image_id -> file_path`` mapping in a SQLite database
(``data/db/image_index.db``), following the same SQLite pattern used by
``file_integrity.py`` (WAL mode, row factory, indexed columns).

This lets the retrieval / response layers resolve an image_id (referenced in a
chunk's metadata) back to a concrete file path for Base64 encoding in MCP
responses.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseImageStorage(ABC):
    """Abstract interface for image storage backends."""

    @abstractmethod
    def save_image(
        self,
        image_id: str,
        source: str | bytes,
        collection: str = "default",
        doc_hash: str = "",
        page_num: int = 0,
    ) -> str:
        """Persist an image and record its mapping. Returns the stored path."""
        ...

    @abstractmethod
    def get_path(self, image_id: str) -> str | None:
        """Return the file path for an image_id, or None if unknown."""
        ...

    @abstractmethod
    def list_by_collection(self, collection: str) -> list[dict[str, Any]]:
        """Return all image records for a collection."""
        ...

    @abstractmethod
    def delete_by_doc_hash(self, doc_hash: str) -> int:
        """Delete all images for a document hash (files + records)."""
        ...


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS image_index (
    image_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    collection TEXT,
    doc_hash TEXT,
    page_num INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection);",
    "CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash);",
]


class SQLiteImageStorage(BaseImageStorage):
    """SQLite-backed image storage with a file-system blob store."""

    def __init__(
        self,
        images_base_dir: str = "data/images",
        db_path: str = "data/db/image_index.db",
    ):
        self._images_base_dir = Path(images_base_dir)
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

    def save_image(
        self,
        image_id: str,
        source: str | bytes,
        collection: str = "default",
        doc_hash: str = "",
        page_num: int = 0,
    ) -> str:
        """Save image bytes/file into the collection dir and record mapping.

        Args:
            image_id: Unique image identifier.
            source: Either raw image bytes or a path to an existing image file.
            collection: Collection name (becomes a subdirectory).
            doc_hash: Source document hash (for grouped deletion).
            page_num: Page number in the source document.

        Returns:
            The absolute-ish stored file path (as a string).
        """
        collection_dir = self._images_base_dir / collection
        collection_dir.mkdir(parents=True, exist_ok=True)
        dest = collection_dir / f"{image_id}.png"

        if isinstance(source, bytes):
            dest.write_bytes(source)
        else:
            src_path = Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"Source image not found: {source}")
            if src_path.resolve() != dest.resolve():
                shutil.copyfile(src_path, dest)

        file_path = str(dest)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO image_index
                   (image_id, file_path, collection, doc_hash, page_num)
                   VALUES (?, ?, ?, ?, ?)""",
                (image_id, file_path, collection, doc_hash, page_num),
            )
        return file_path

    def get_path(self, image_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        return row["file_path"] if row else None

    def list_by_collection(self, collection: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM image_index WHERE collection = ? ORDER BY image_id",
                (collection,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_by_doc_hash(self, doc_hash: str) -> int:
        """Delete all images (files + records) for a document hash."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT image_id, file_path FROM image_index WHERE doc_hash = ?",
                (doc_hash,),
            ).fetchall()
            for r in rows:
                try:
                    Path(r["file_path"]).unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Failed to delete image file %s: %s", r["file_path"], exc)
            conn.execute(
                "DELETE FROM image_index WHERE doc_hash = ?", (doc_hash,)
            )
        return len(rows)
