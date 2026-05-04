"""LLM-based Reranker implementation.

Uses an LLM to rerank candidates by relevance. Reads the prompt template
from config/prompts/rerank.txt and asks the LLM to return a JSON array
of ranked candidate IDs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.libs.reranker.base_reranker import BaseReranker, RerankCandidate
from src.libs.reranker.reranker_factory import register_backend
from src.libs.llm.base_llm import BaseLLM, ChatMessage

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

_DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"


class LLMReranker(BaseReranker):
    """Reranker that uses an LLM to sort candidates by relevance."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str | None = None,
        prompt_path: str = _DEFAULT_PROMPT_PATH,
    ):
        self._llm = llm
        if prompt_template is not None:
            self._prompt_template = prompt_template
        else:
            p = Path(prompt_path)
            if p.exists():
                self._prompt_template = p.read_text(encoding="utf-8").strip()
            else:
                self._prompt_template = (
                    "Rank the following passages by relevance to the query. "
                    "Return a JSON array of IDs sorted most to least relevant."
                )

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: "TraceContext | None" = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        # Build the user message with candidates
        candidate_lines = "\n".join(
            f"- ID: {c.id} | Text: {c.text[:200]}" for c in candidates
        )
        user_msg = (
            f"Query: {query}\n\n"
            f"Candidates:\n{candidate_lines}\n\n"
            f"Return the ranked JSON array of IDs."
        )

        messages = [
            ChatMessage(role="system", content=self._prompt_template),
            ChatMessage(role="user", content=user_msg),
        ]

        try:
            response = self._llm.chat(messages)
            ranked_ids = self._parse_response(response.content)
        except Exception:
            # On any failure, return candidates unchanged (fallback signal)
            return list(candidates)

        # Reorder candidates by the LLM's ranking
        id_to_candidate = {c.id: c for c in candidates}
        ranked: list[RerankCandidate] = []
        seen: set[str] = set()

        for rank, rid in enumerate(ranked_ids):
            if rid in id_to_candidate and rid not in seen:
                c = id_to_candidate[rid]
                ranked.append(RerankCandidate(
                    id=c.id,
                    text=c.text,
                    score=1.0 - (rank / max(len(ranked_ids), 1)),
                ))
                seen.add(rid)

        # Append any candidates the LLM missed (preserve original order)
        for c in candidates:
            if c.id not in seen:
                ranked.append(c)

        return ranked

    @staticmethod
    def _parse_response(content: str) -> list[str]:
        """Parse the LLM response as a JSON array of IDs.

        Raises ValueError if the response is not valid JSON or not a list of strings.
        """
        content = content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            content = content.strip()

        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
        return [str(item) for item in parsed]

    @property
    def backend_name(self) -> str:
        return "llm"


def _create_llm_reranker(settings: Any) -> LLMReranker:
    # Import here to avoid circular imports; LLM must be created first
    from src.libs.llm.llm_factory import LLMFactory
    llm = LLMFactory.create(settings)
    return LLMReranker(llm=llm)


register_backend("llm", _create_llm_reranker)
