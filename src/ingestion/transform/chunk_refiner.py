"""ChunkRefiner — rule-based denoising + optional LLM enhancement.

Pipeline:
1. Rule-based cleaning (always runs): remove noise, normalize whitespace.
2. LLM refinement (optional): rewrite chunk for clarity and coherence.
3. Fallback: if LLM fails, use rule-based result and mark metadata.

Configuration via settings.yaml (future: ingestion.chunk_refiner section).
Currently controlled by whether an LLM instance is provided.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = "config/prompts/chunk_refinement.txt"

# ---------------------------------------------------------------------------
# Rule-based patterns
# ---------------------------------------------------------------------------

# Common page header/footer patterns
_HEADER_FOOTER_PATTERNS = [
    # Page numbers: "Page 1", "- 1 -", "1 / 10", "第1页"
    re.compile(r"^[\s]*[-—]?\s*\d+\s*[-—]?\s*$", re.MULTILINE),
    re.compile(r"^[\s]*Page\s+\d+(\s+of\s+\d+)?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^[\s]*第\s*\d+\s*页.*$", re.MULTILINE),
    re.compile(r"^[\s]*\d+\s*/\s*\d+\s*$", re.MULTILINE),
    # Separator lines (dashes, equals, underscores)
    re.compile(r"^[\s]*[-=_]{3,}\s*$", re.MULTILINE),
    # Common watermark/confidentiality headers
    re.compile(
        r"^[\s]*(Confidential|CONFIDENTIAL|Company Confidential|DRAFT|"
        r"Internal Use Only|DO NOT DISTRIBUTE|PROPRIETARY)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# HTML comments
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Excessive whitespace patterns
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_LEADING_SPACES_RE = re.compile(r"^[ \t]+", re.MULTILINE)
_MULTI_SPACES_RE = re.compile(r"[ \t]{2,}")

# Format markers that are noise (not content)
_FORMAT_MARKERS = [
    re.compile(r"</?(?:div|span|br|hr|p|table|tr|td|th)\s*/?>", re.IGNORECASE),
    re.compile(r"&(?:nbsp|amp|lt|gt|quot);"),
]


class ChunkRefiner(BaseTransform):
    """Refine chunks via rule-based denoising and optional LLM enhancement."""

    def __init__(
        self,
        llm: "BaseLLM | None" = None,
        use_llm: bool = True,
        prompt_path: str | None = None,
    ):
        """Initialize ChunkRefiner.

        Args:
            llm: Optional LLM instance for enhancement. If None, only rules apply.
            use_llm: Whether to attempt LLM refinement (can be disabled via config).
            prompt_path: Path to prompt template file. Uses default if not provided.
        """
        self._llm = llm
        self._use_llm = use_llm and (llm is not None)
        self._prompt_template = self._load_prompt(prompt_path)

    def transform(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None" = None,
    ) -> list[Chunk]:
        """Apply refinement to each chunk.

        Each chunk is processed independently. Failures on individual chunks
        are logged but do not affect other chunks.
        """
        if trace:
            trace.start_stage("chunk_refiner")

        refined_chunks: list[Chunk] = []
        stats = {"rule_refined": 0, "llm_refined": 0, "fallback": 0, "errors": 0}

        for chunk in chunks:
            try:
                refined = self._refine_single(chunk, trace)
                refined_chunks.append(refined)
                # Count stats
                refined_by = refined.metadata.get("refined_by", "")
                if refined_by == "llm":
                    stats["llm_refined"] += 1
                elif refined_by == "rule":
                    stats["rule_refined"] += 1
            except Exception as exc:
                logger.warning(
                    "ChunkRefiner failed on chunk %s: %s — preserving original",
                    chunk.id, exc,
                )
                stats["errors"] += 1
                # Preserve original chunk on unexpected errors
                chunk.metadata["refined_by"] = "none"
                chunk.metadata["refine_error"] = str(exc)
                refined_chunks.append(chunk)

        if trace:
            trace.end_stage(details=stats)

        return refined_chunks

    def _refine_single(
        self, chunk: Chunk, trace: "TraceContext | None" = None
    ) -> Chunk:
        """Refine a single chunk: rule-based first, then optional LLM."""
        original_text = chunk.text

        # Step 1: Rule-based denoising (always)
        rule_cleaned = self._rule_based_refine(original_text)

        # Step 2: Optional LLM refinement
        if self._use_llm:
            llm_result = self._llm_refine(rule_cleaned, trace)
            if llm_result is not None:
                chunk.text = llm_result
                chunk.metadata["refined_by"] = "llm"
                return chunk

            # LLM failed — fallback to rule result
            logger.info("LLM refinement failed for chunk %s, using rule result", chunk.id)
            chunk.text = rule_cleaned
            chunk.metadata["refined_by"] = "rule"
            chunk.metadata["refine_fallback"] = "llm_failed"
            return chunk

        # No LLM — use rule result only
        chunk.text = rule_cleaned
        chunk.metadata["refined_by"] = "rule"
        return chunk

    def _rule_based_refine(self, text: str) -> str:
        """Apply rule-based denoising to text.

        Preserves code blocks and meaningful Markdown structure.
        """
        if not text or not text.strip():
            return text

        # Protect code blocks from cleaning
        code_blocks: list[str] = []
        code_block_re = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")

        def _protect_code(match: re.Match) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        protected = code_block_re.sub(_protect_code, text)

        # Remove HTML comments
        protected = _HTML_COMMENT_RE.sub("", protected)

        # Remove format markers (HTML tags, entities)
        for pattern in _FORMAT_MARKERS:
            protected = pattern.sub("", protected)

        # Remove page header/footer patterns
        for pattern in _HEADER_FOOTER_PATTERNS:
            protected = pattern.sub("", protected)

        # Normalize whitespace
        protected = _TRAILING_SPACES_RE.sub("", protected)
        protected = _MULTI_SPACES_RE.sub(" ", protected)
        protected = _MULTI_BLANK_LINES_RE.sub("\n\n", protected)

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            protected = protected.replace(f"__CODE_BLOCK_{i}__", block)

        return protected.strip()

    def _llm_refine(self, text: str, trace: "TraceContext | None" = None) -> str | None:
        """Attempt LLM-based refinement. Returns None on failure."""
        if not self._llm:
            return None

        # Skip very short chunks (not worth LLM call)
        if len(text.strip()) < 30:
            return None  # Signal to use rule-based result

        prompt = self._prompt_template.replace("{text}", text)

        try:
            from src.libs.llm.base_llm import ChatMessage

            messages = [ChatMessage(role="user", content=prompt)]
            response = self._llm.chat(messages)
            refined = response.content.strip()

            # Sanity check: LLM should not return empty or drastically shorter
            if not refined:
                logger.warning("LLM returned empty response, falling back")
                return None
            if len(refined) < len(text) * 0.2:
                logger.warning(
                    "LLM output too short (%d vs %d chars), falling back",
                    len(refined), len(text),
                )
                return None

            return refined

        except Exception as exc:
            logger.warning("LLM refinement error: %s", exc)
            return None

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        """Load prompt template from file, with fallback to default."""
        path = Path(prompt_path) if prompt_path else Path(_DEFAULT_PROMPT_PATH)

        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            # Ensure the prompt has a {text} placeholder
            if "{text}" in content:
                return content
            # Append placeholder if missing
            return content + "\n\n{text}"

        # Hardcoded fallback prompt
        return (
            "You are a document processing assistant. "
            "Clean and refine the following text chunk to be self-contained "
            "and semantically coherent. Remove noise (page headers/footers, "
            "formatting artifacts, OCR errors) while preserving all meaningful "
            "content. Keep code blocks intact. Output only the refined text, "
            "nothing else.\n\n{text}"
        )

    @property
    def name(self) -> str:
        return "chunk_refiner"
