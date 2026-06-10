"""ResponseBuilder — assemble MCP tool responses from retrieval results.

Builds the MCP tool result shape:
- ``content``: an array whose first item is human-readable Markdown (answer +
  numbered citations), guaranteeing minimum client compatibility.
- ``structuredContent``: machine-readable payload with the citation list.

Multimodal (image) content is appended by MultimodalAssembler (E6); this builder
focuses on the text + citation portion.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.response.citation_generator import CitationGenerator

if TYPE_CHECKING:
    from src.core.response.citation_generator import Citation
    from src.core.types import RetrievalResult

_NO_RESULTS_MESSAGE = (
    "未找到相关文档。请先运行 ingest.py 摄取数据，或尝试调整查询关键词。"
)


class ResponseBuilder:
    """Build MCP-format responses with inline citations."""

    def __init__(self, citation_generator: CitationGenerator | None = None):
        self._citations = citation_generator or CitationGenerator()

    def build(
        self,
        retrieval_results: list["RetrievalResult"],
        query: str,
    ) -> dict[str, Any]:
        """Build an MCP tool result from retrieval results.

        Args:
            retrieval_results: Final ranked results.
            query: The original user query (echoed into the answer header).

        Returns:
            An MCP tool result dict with ``content`` and ``structuredContent``.
        """
        if not retrieval_results:
            return self._empty_response()

        citations = self._citations.generate(retrieval_results)
        markdown = self._render_markdown(query, citations)

        return {
            "content": [{"type": "text", "text": markdown}],
            "structuredContent": {
                "query": query,
                "citations": [c.to_dict() for c in citations],
            },
            "isError": False,
        }

    def _empty_response(self) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": _NO_RESULTS_MESSAGE}],
            "structuredContent": {"citations": []},
            "isError": False,
        }

    @staticmethod
    def _render_markdown(query: str, citations: list["Citation"]) -> str:
        """Render a human-readable Markdown answer with [n] citations."""
        lines = [f"### 检索结果：{query}", ""]
        for c in citations:
            page = f" p.{c.page}" if c.page != "" else ""
            lines.append(f"**[{c.id}]** `{c.source}{page}` (score={c.score:.4f})")
            if c.text:
                lines.append(f"> {c.text}")
            lines.append("")
        lines.append("---")
        sources = ", ".join(f"[{c.id}] {c.source}" for c in citations)
        lines.append(f"来源：{sources}")
        return "\n".join(lines)
