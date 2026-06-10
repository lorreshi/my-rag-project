"""Unit tests for list_collections tool (E4)."""
from __future__ import annotations

from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.tools.list_collections import ListCollectionsTool


def _make_docs(tmp_path, layout: dict[str, int]):
    """Create collection dirs with N pdf files each. Returns base dir path."""
    base = tmp_path / "documents"
    for name, n in layout.items():
        d = base / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake")
    return str(base)


class TestListCollections:
    def test_lists_collection_names(self, tmp_path):
        base = _make_docs(tmp_path, {"alpha": 2, "beta": 1})
        tool = ListCollectionsTool(documents_base_dir=base)
        result = tool({})
        names = {c["name"] for c in result["structuredContent"]["collections"]}
        assert names == {"alpha", "beta"}

    def test_document_counts(self, tmp_path):
        base = _make_docs(tmp_path, {"alpha": 3, "beta": 1})
        tool = ListCollectionsTool(documents_base_dir=base)
        result = tool({})
        by_name = {c["name"]: c["document_count"] for c in result["structuredContent"]["collections"]}
        assert by_name["alpha"] == 3
        assert by_name["beta"] == 1

    def test_sorted_order(self, tmp_path):
        base = _make_docs(tmp_path, {"zeta": 1, "alpha": 1, "mu": 1})
        tool = ListCollectionsTool(documents_base_dir=base)
        names = [c["name"] for c in tool({})["structuredContent"]["collections"]]
        assert names == ["alpha", "mu", "zeta"]

    def test_ignores_non_pdf_files(self, tmp_path):
        base = tmp_path / "documents"
        (base / "c1").mkdir(parents=True)
        (base / "c1" / "a.pdf").write_bytes(b"x")
        (base / "c1" / "note.txt").write_text("ignore me")
        tool = ListCollectionsTool(documents_base_dir=str(base))
        result = tool({})
        assert result["structuredContent"]["collections"][0]["document_count"] == 1

    def test_ignores_loose_files_in_base(self, tmp_path):
        base = tmp_path / "documents"
        base.mkdir()
        (base / "loose.pdf").write_bytes(b"x")  # not in a collection dir
        (base / "real_coll").mkdir()
        tool = ListCollectionsTool(documents_base_dir=str(base))
        names = [c["name"] for c in tool({})["structuredContent"]["collections"]]
        assert names == ["real_coll"]

    def test_empty_when_no_base(self, tmp_path):
        tool = ListCollectionsTool(documents_base_dir=str(tmp_path / "nope"))
        result = tool({})
        assert result["structuredContent"]["collections"] == []
        assert "暂无" in result["content"][0]["text"]

    def test_content_lists_names(self, tmp_path):
        base = _make_docs(tmp_path, {"alpha": 1})
        tool = ListCollectionsTool(documents_base_dir=base)
        assert "alpha" in tool({})["content"][0]["text"]

    def test_no_args_ok(self, tmp_path):
        base = _make_docs(tmp_path, {"alpha": 1})
        tool = ListCollectionsTool(documents_base_dir=base)
        assert tool() is not None


class TestRegistration:
    def test_registers_and_calls(self, tmp_path):
        base = _make_docs(tmp_path, {"alpha": 1})
        handler = ProtocolHandler()
        ListCollectionsTool(documents_base_dir=base).register(handler)
        resp = handler.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "list_collections", "arguments": {}}}
        )
        assert "content" in resp["result"]
        names = {c["name"] for c in resp["result"]["structuredContent"]["collections"]}
        assert "alpha" in names
