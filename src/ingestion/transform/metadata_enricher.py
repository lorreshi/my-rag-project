"""MetadataEnricher — rule-based defaults + optional LLM semantic enrichment.

Generates `title`, `summary`, and `tags` for each chunk and injects them into
chunk.metadata. Strategy:
1. Rule-based baseline (always available): derive title/summary/tags from text
   using simple heuristics (first heading/line, leading sentences, word freq).
2. LLM enrichment (preferred when enabled): ask the LLM for high-quality
   structured metadata (JSON).
3. Graceful fallback: if the LLM call or parsing fails, keep the rule-based
   result and mark the degradation reason in metadata. Never blocks ingestion.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = "config/prompts/metadata_enrichment.txt"

# Markdown heading: "# Title" / "## Title" ...
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
# Sentence boundary splitter (keeps it simple, multilingual-ish)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
# Word tokenizer for tag extraction (ASCII words + CJK runs)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

# Minimal English stopword set for rule-based tag extraction
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "her", "was", "one", "our", "out", "his", "has", "had", "him", "how",
    "its", "may", "new", "now", "old", "see", "two", "way", "who", "did",
    "get", "use", "this", "that", "with", "from", "they", "have", "will",
    "your", "what", "when", "which", "their", "there", "would", "about",
    "into", "than", "then", "them", "these", "those", "such", "also",
    "been", "more", "most", "some", "only", "over", "very", "each", "other",
}


class MetadataEnricher(BaseTransform):
    """Enrich chunks with title / summary / tags metadata."""

    def __init__(
        self,
        llm: "BaseLLM | None" = None,
        use_llm: bool = True,
        prompt_path: str | None = None,
        max_tags: int = 6,
    ):
        """Initialize MetadataEnricher.

        Args:
            llm: Optional LLM instance for semantic enrichment.
            use_llm: Whether to attempt LLM enrichment (config switch).
            prompt_path: Path to prompt template; uses default if not provided.
            max_tags: Maximum number of tags to keep.
        """
        self._llm = llm
        self._use_llm = use_llm and (llm is not None)
        self._max_tags = max_tags
        self._prompt_template = self._load_prompt(prompt_path)

    def transform(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None" = None,
    ) -> list[Chunk]:
        """Enrich each chunk with metadata. Per-chunk failures are isolated."""
        if trace:
            trace.start_stage("metadata_enricher")

        enriched: list[Chunk] = []
        stats = {"llm_enriched": 0, "rule_enriched": 0, "errors": 0}

        for chunk in chunks:
            try:
                result = self._enrich_single(chunk, trace)
                enriched.append(result)
                by = result.metadata.get("enriched_by", "")
                if by == "llm":
                    stats["llm_enriched"] += 1
                elif by == "rule":
                    stats["rule_enriched"] += 1
            except Exception as exc:
                logger.warning(
                    "MetadataEnricher failed on chunk %s: %s — applying rule baseline",
                    chunk.id, exc,
                )
                stats["errors"] += 1
                # Last-resort: ensure metadata fields exist
                self._apply_metadata(chunk, self._rule_based_enrich(chunk.text))
                chunk.metadata["enriched_by"] = "rule"
                chunk.metadata["enrich_error"] = str(exc)
                enriched.append(chunk)

        if trace:
            trace.end_stage(details=stats)

        return enriched

    def _enrich_single(
        self, chunk: Chunk, trace: "TraceContext | None" = None
    ) -> Chunk:
        """Enrich a single chunk: rule baseline first, then optional LLM."""
        rule_meta = self._rule_based_enrich(chunk.text)

        if self._use_llm:
            llm_meta = self._llm_enrich(chunk.text, trace)
            if llm_meta is not None:
                self._apply_metadata(chunk, llm_meta)
                chunk.metadata["enriched_by"] = "llm"
                return chunk

            # LLM failed — fall back to rule baseline
            self._apply_metadata(chunk, rule_meta)
            chunk.metadata["enriched_by"] = "rule"
            chunk.metadata["enrich_fallback"] = "llm_failed"
            return chunk

        self._apply_metadata(chunk, rule_meta)
        chunk.metadata["enriched_by"] = "rule"
        return chunk

    # ------------------------------------------------------------------
    # Rule-based baseline
    # ------------------------------------------------------------------

    def _rule_based_enrich(self, text: str) -> dict[str, Any]:
        """Derive title/summary/tags using simple heuristics.

        Guarantees non-empty title, summary, and tags for any non-trivial text.
        """
        stripped = (text or "").strip()
        if not stripped:
            return {"title": "Untitled", "summary": "", "tags": []}

        title = self._rule_title(stripped)
        summary = self._rule_summary(stripped)
        tags = self._rule_tags(stripped)
        return {"title": title, "summary": summary, "tags": tags}

    @staticmethod
    def _rule_title(text: str) -> str:
        """Use the first Markdown heading, else the first non-empty line."""
        heading = _HEADING_RE.search(text)
        if heading:
            return heading.group(1).strip()[:120]

        for line in text.splitlines():
            line = line.strip()
            if line:
                # Trim to a reasonable title length at a word boundary
                if len(line) <= 80:
                    return line
                return line[:80].rsplit(" ", 1)[0] + "…"
        return "Untitled"

    @staticmethod
    def _rule_summary(text: str) -> str:
        """Take the first 1-2 sentences as a summary."""
        # Drop heading lines for summary purposes
        body = _HEADING_RE.sub("", text).strip()
        if not body:
            body = text.strip()
        # Collapse whitespace
        body = re.sub(r"\s+", " ", body)
        sentences = _SENTENCE_SPLIT_RE.split(body)
        summary = " ".join(sentences[:2]).strip()
        if len(summary) > 300:
            summary = summary[:300].rsplit(" ", 1)[0] + "…"
        return summary

    def _rule_tags(self, text: str) -> list[str]:
        """Extract frequent non-stopword terms as tags."""
        words = [w.lower() for w in _WORD_RE.findall(text)]
        words = [w for w in words if w not in _STOPWORDS]
        if not words:
            return []
        freq = Counter(words)
        # Most common, stable order: by count desc then first-seen
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], words.index(kv[0])))
        return [w for w, _ in ranked[: self._max_tags]]

    # ------------------------------------------------------------------
    # LLM enrichment
    # ------------------------------------------------------------------

    def _llm_enrich(
        self, text: str, trace: "TraceContext | None" = None
    ) -> dict[str, Any] | None:
        """Attempt LLM enrichment. Returns parsed dict or None on failure."""
        if not self._llm:
            return None
        if len(text.strip()) < 30:
            # Too short to be worth an LLM call; let rule baseline handle it.
            return None

        prompt = self._prompt_template.replace("{text}", text)

        try:
            from src.libs.llm.base_llm import ChatMessage

            messages = [ChatMessage(role="user", content=prompt)]
            response = self._llm.chat(messages)
            content = (response.content or "").strip()
            if not content:
                logger.warning("LLM returned empty metadata response")
                return None
            return self._parse_llm_metadata(content)
        except Exception as exc:
            logger.warning("LLM metadata enrichment error: %s", exc)
            return None

    def _parse_llm_metadata(self, content: str) -> dict[str, Any] | None:
        """Parse and validate the LLM JSON response."""
        raw = self._extract_json(content)
        if raw is None:
            logger.warning("Could not locate JSON in LLM metadata response")
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM metadata JSON: %s", exc)
            return None

        if not isinstance(data, dict):
            return None

        title = data.get("title")
        summary = data.get("summary")
        tags = data.get("tags")

        # Validate / normalize
        if not isinstance(title, str) or not title.strip():
            return None
        if not isinstance(summary, str):
            summary = ""
        if not isinstance(tags, list):
            tags = []
        norm_tags = [
            str(t).strip().lower()
            for t in tags
            if isinstance(t, (str, int, float)) and str(t).strip()
        ]
        # Dedup preserving order, cap length
        seen: set[str] = set()
        unique_tags: list[str] = []
        for t in norm_tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)

        return {
            "title": title.strip()[:120],
            "summary": summary.strip(),
            "tags": unique_tags[: self._max_tags],
        }

    @staticmethod
    def _extract_json(content: str) -> str | None:
        """Extract the first JSON object from text (handles ```json fences)."""
        # Strip markdown code fences if present
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fence:
            return fence.group(1)
        # Otherwise grab the first balanced-looking { ... } block
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return content[start : end + 1]
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_metadata(chunk: Chunk, meta: dict[str, Any]) -> None:
        """Write title/summary/tags into chunk.metadata."""
        chunk.metadata["title"] = meta.get("title", "")
        chunk.metadata["summary"] = meta.get("summary", "")
        chunk.metadata["tags"] = meta.get("tags", [])

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        """Load prompt template from file, with fallback to default."""
        path = Path(prompt_path) if prompt_path else Path(_DEFAULT_PROMPT_PATH)
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if "{text}" in content:
                return content
            return content + "\n\n{text}"

        return (
            "Analyze the following text and return ONLY a JSON object with keys "
            '"title" (string), "summary" (string), and "tags" (array of strings). '
            "Base everything on the text; do not invent facts.\n\n{text}"
        )

    @property
    def name(self) -> str:
        return "metadata_enricher"
