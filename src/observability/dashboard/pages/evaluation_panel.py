"""Evaluation panel page (H4) — run golden-set evaluation, view metrics."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.core.settings import load_settings

_DEFAULT_TEST_SET = "tests/fixtures/golden_test_set.json"


def _run_evaluation(test_set_path: str, top_k: int, generate: bool):
    """Build an EvalRunner from settings and run it (best effort).

    When *generate* is True the LLM generation step runs, enabling the
    generation-quality metrics (faithfulness / answer_relevancy /
    context_precision); otherwise only retrieval metrics are meaningful.
    """
    from src.observability.evaluation.eval_runner import EvalRunner

    settings = load_settings()
    runner = EvalRunner.from_settings(settings, top_k=top_k, generate=generate)
    return runner.run(test_set_path)


def _render_metrics(metrics: dict) -> None:
    """Show aggregate metrics, max 4 per row to stay readable."""
    items = list(metrics.items())
    for start in range(0, len(items), 4):
        row = items[start : start + 4]
        cols = st.columns(len(row))
        for col, (name, value) in zip(cols, row):
            col.metric(name, f"{value:.4f}")


def render() -> None:
    st.title("📈 评估面板")
    st.caption("运行黄金测试集评估，查看检索与生成质量指标")

    test_set = st.text_input("Golden Test Set 路径", value=_DEFAULT_TEST_SET)
    top_k = st.slider("Top-K", min_value=1, max_value=50, value=10)
    generate = st.checkbox(
        "运行生成（启用 faithfulness / answer_relevancy / context_precision）",
        value=False,
    )
    if generate:
        st.warning(
            "已开启生成：每条用例会调用 LLM 生成答案并由 LLM 评分，"
            "约 1 分钟/条，整套用例可能需要数分钟，页面在此期间会保持等待。"
        )

    if not Path(test_set).exists():
        st.warning(f"测试集文件不存在: {test_set}")

    if st.button("运行评估"):
        with st.spinner("评估运行中…"):
            try:
                report = _run_evaluation(test_set, top_k, generate)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"评估失败: {exc}")
                return

        st.session_state["last_eval_report"] = report.to_dict()

    report = st.session_state.get("last_eval_report")
    if not report:
        st.info("点击「运行评估」开始。")
        return

    st.subheader("汇总指标")
    metrics = report["aggregate_metrics"]
    if metrics:
        _render_metrics(metrics)
    st.caption(f"共 {report['num_cases']} 个测试用例")

    st.subheader("各 Query 明细")
    for q in report["per_query"]:
        flag = "" if q.get("answerable", True) else " · 🚫不可答"
        with st.expander(f"{q['query']}{flag} · {q['metrics']}"):
            answer = q.get("generated_answer", "")
            if answer:
                st.write("**Generated Answer:**")
                st.markdown(answer)
            st.write("**Retrieved:**", q["retrieved_ids"])
            st.write("**Expected:**", q["expected_ids"])
