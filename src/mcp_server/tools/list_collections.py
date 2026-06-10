"""MCP tool: list_collections.

Lists available knowledge-base collections. A collection corresponds to a
subdirectory under the documents base dir (default ``data/documents/``).
Optionally augments each entry with a document count.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("mcp-server")

TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = "List the available document collections in the knowledge base."
INPUT_SCHEMA = {"type": "object", "properties": {}}

_DOC_EXTENSIONS = {".pdf"}


class ListCollectionsTool:
    """List collections by scanning the documents base directory."""

    def __init__(self, documents_base_dir: str = "data/documents"):
        self._base = Path(documents_base_dir)

    def __call__(self, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the list of collections with basic statistics."""
        collections = self._scan()
        if not collections:
            text = "暂无可用集合。请先运行 ingest.py 摄取数据。"
        else:
            lines = ["### 可用集合", ""]
            for c in collections:
                lines.append(f"- **{c['name']}** ({c['document_count']} 个文档)")
            text = "\n".join(lines)

        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": {"collections": collections},
            "isError": False,
        }

    def _scan(self) -> list[dict[str, Any]]:
        """Scan the base dir for collection subdirectories."""
        if not self._base.exists() or not self._base.is_dir():
            return []
        collections: list[dict[str, Any]] = []
        for entry in sorted(self._base.iterdir()):
            if not entry.is_dir():
                continue
            doc_count = sum(
                1 for p in entry.rglob("*")
                if p.is_file() and p.suffix.lower() in _DOC_EXTENSIONS
            )
            collections.append({"name": entry.name, "document_count": doc_count})
        return collections

    def register(self, handler) -> None:
        """Register this tool with a ProtocolHandler."""
        handler.register_tool(
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            input_schema=INPUT_SCHEMA,
            handler=self.__call__,
        )
