"""LLMJudgeEvaluator — generation-quality metrics via LLM-as-judge.

A dependency-free reimplementation of the three classic Ragas generation
metrics, using this project's own ``BaseLLM`` client (which already talks to
the configured endpoint) instead of the heavy ragas + langchain stack.

Metrics
-------
faithfulness        : 答案的每条事实主张是否都能由检索到的上下文支撑（防幻觉）。
                      做法：让 LLM 把答案拆成原子主张并逐条判定是否被上下文支撑，
                      score = 被支撑主张数 / 总主张数。
answer_relevancy    : 答案是否切题地回答了问题（防跑题）。
                      做法：让 LLM 从答案反推 N 个可能的问题，把它们与原问题做
                      embedding 余弦相似度，取均值。答案越切题，反推问题越接近原问题。
context_precision   : 检索到的上下文是否“有用且排序靠前”。
                      做法：逐个上下文判定其对得出标准答案是否有用，再以
                      Average Precision（排名加权）聚合，奖励把有用块排在前面。

每个指标只用 1 次（必要时批量）LLM 调用，便于控制成本与时延。所有 LLM 输出
要求为 JSON，解析失败时安全降级（记 0 分而非崩溃）。
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_METRICS = ("faithfulness", "answer_relevancy", "context_precision")

# Phrases that mark a (correct) refusal on unanswerable questions. A refusal
# makes no factual claims, so it is treated as vacuously faithful.
_REFUSAL_MARKERS = ("未提供", "不可答", "无法回答", "没有相关", "未找到", "无法确定")


class LLMJudgeEvaluator(BaseEvaluator):
    """Compute faithfulness / answer_relevancy / context_precision via an LLM."""

    def __init__(
        self,
        llm: "BaseLLM",
        embedding: "BaseEmbedding | None" = None,
        metrics: tuple[str, ...] = _METRICS,
        n_questions: int = 3,
    ):
        self._llm = llm
        self._embedding = embedding
        self._metrics = metrics
        self._n_questions = n_questions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        generated_answer: str = "",
        ground_truth: str = "",
        contexts: list[str] | None = None,
    ) -> dict[str, float]:
        contexts = contexts or []
        out: dict[str, float] = {}

        if "faithfulness" in self._metrics:
            out["faithfulness"] = self._faithfulness(generated_answer, contexts)
        if "answer_relevancy" in self._metrics:
            out["answer_relevancy"] = self._answer_relevancy(query, generated_answer)
        if "context_precision" in self._metrics:
            reference = ground_truth or generated_answer
            out["context_precision"] = self._context_precision(
                query, reference, contexts
            )
        return out

    @property
    def evaluator_name(self) -> str:
        return "llm_judge"

    # ------------------------------------------------------------------
    # faithfulness
    # ------------------------------------------------------------------

    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        answer = (answer or "").strip()
        if not answer:
            return 0.0
        # A correct refusal makes no verifiable factual claims -> vacuously faithful.
        if any(m in answer for m in _REFUSAL_MARKERS):
            return 1.0
        if not contexts:
            return 0.0

        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "你是严格的事实核查员。下面给出【上下文】与一段【答案】。\n"
            "请把【答案】拆解为若干条原子事实主张（每条只含一个可独立验证的事实），"
            "并逐条判断该主张是否能【仅依据上下文】推得（supported=true/false）。\n"
            "只输出 JSON，格式：{\"claims\":[{\"claim\":\"...\",\"supported\":true}]}。\n\n"
            f"【上下文】\n{context_block}\n\n【答案】\n{answer}"
        )
        data = self._ask_json(prompt)
        claims = data.get("claims") if isinstance(data, dict) else None
        if not claims:
            return 0.0
        supported = sum(1 for c in claims if c.get("supported") is True)
        return round(supported / len(claims), 4)

    # ------------------------------------------------------------------
    # answer_relevancy
    # ------------------------------------------------------------------

    def _answer_relevancy(self, query: str, answer: str) -> float:
        answer = (answer or "").strip()
        if not answer:
            return 0.0
        # A refusal is, by design, not a topical answer to the question.
        if any(m in answer for m in _REFUSAL_MARKERS):
            return 0.0

        prompt = (
            f"给定下面这段【答案】，请反推出 {self._n_questions} 个该答案最可能"
            "在回答的问题（中文，彼此略有差异）。\n"
            "只输出 JSON，格式：{\"questions\":[\"...\"]}。\n\n"
            f"【答案】\n{answer}"
        )
        data = self._ask_json(prompt)
        questions = data.get("questions") if isinstance(data, dict) else None
        if not questions:
            return 0.0
        questions = [q for q in questions if isinstance(q, str) and q.strip()]
        if not questions:
            return 0.0

        # Authentic ragas approach: embed reverse-questions + original query,
        # average their cosine similarity. Falls back to an LLM 0-1 rating if
        # no embedding client is wired.
        if self._embedding is None:
            return self._relevancy_via_llm(query, answer)

        vectors = self._embedding.embed([query] + questions)
        q_vec = vectors[0]
        sims = [_cosine(q_vec, v) for v in vectors[1:]]
        sims = [s for s in sims if s is not None]
        if not sims:
            return 0.0
        return round(max(0.0, sum(sims) / len(sims)), 4)

    def _relevancy_via_llm(self, query: str, answer: str) -> float:
        prompt = (
            "请评估【答案】对【问题】的切题程度，0 表示完全跑题，1 表示完全切题。\n"
            "只输出 JSON：{\"score\": 0.0~1.0}。\n\n"
            f"【问题】{query}\n【答案】{answer}"
        )
        data = self._ask_json(prompt)
        try:
            return round(max(0.0, min(1.0, float(data.get("score", 0.0)))), 4)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # context_precision
    # ------------------------------------------------------------------

    def _context_precision(
        self, query: str, reference: str, contexts: list[str]
    ) -> float:
        if not contexts or not (reference or "").strip():
            return 0.0

        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "下面给出一个【问题】、一个【参考答案】，以及按检索排名排列的若干【上下文】。\n"
            "请逐个判断每段上下文对于得出该参考答案是否有用（useful=true/false）。\n"
            "按上下文给出的顺序返回，只输出 JSON："
            "{\"verdicts\":[true,false,...]}（长度需等于上下文数量）。\n\n"
            f"【问题】{query}\n【参考答案】{reference}\n\n【上下文】\n{context_block}"
        )
        data = self._ask_json(prompt)
        verdicts = data.get("verdicts") if isinstance(data, dict) else None
        if not isinstance(verdicts, list) or not verdicts:
            return 0.0

        rel = [1 if bool(v) else 0 for v in verdicts[: len(contexts)]]
        total_relevant = sum(rel)
        if total_relevant == 0:
            return 0.0

        # Average Precision: 奖励把有用上下文排在更靠前的位置。
        # AP = Σ_k (Precision@k * rel_k) / 有用总数
        cumulative_hits = 0
        ap = 0.0
        for k, r in enumerate(rel, start=1):
            if r:
                cumulative_hits += 1
                ap += cumulative_hits / k
        return round(ap / total_relevant, 4)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _ask_json(self, prompt: str) -> dict[str, Any]:
        """Send a single chat request and parse the JSON object from the reply."""
        from src.libs.llm.base_llm import ChatMessage

        try:
            resp = self._llm.chat([
                ChatMessage(role="system", content="你只输出合法 JSON，不要解释。"),
                ChatMessage(role="user", content=prompt),
            ])
        except Exception as exc:
            logger.warning("LLM judge call failed: %s", exc)
            return {}
        return _parse_json(resp.content)


def _parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply (tolerates code fences)."""
    if not text:
        return {}
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost {...} span.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def _create_llm_judge(settings: "Settings") -> LLMJudgeEvaluator:
    """Factory: wire the project's LLM (+ embedding) into an LLMJudgeEvaluator."""
    from src.libs.llm.llm_factory import LLMFactory

    llm = LLMFactory.create(settings)
    embedding = None
    try:
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        embedding = EmbeddingFactory.create(settings)
    except Exception:  # pragma: no cover - embedding optional for relevancy
        embedding = None
    return LLMJudgeEvaluator(llm=llm, embedding=embedding)
