"""Evaluation panel page (H4) — run golden-set evaluation, view metrics."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.core.settings import load_settings

_DEFAULT_TEST_SET = "tests/fixtures/golden_test_set.json"


def _run_evaluation(test_set_path: str, top_k: int):
    """Build an EvalRunner from settings and run it (best effort)."""
    from src.observability.evaluation.eval_runner import EvalRunner

    settings = load_settings()
    runner = EvalRunner.from_settings(settings, top_k=top_k)
    return runner.run(test_set_path)


def render() -> None:
    st.title("📈 评估面板")
    st.caption("运行黄金测试集评估，查看 hit_rate / mrr 等指标")

    test_set = st.text_input("Golden Test Set 路径", value=_DEFAULT_TEST_SET)
    top_k = st.slider("Top-K", min_value=1, max_value=50, value=10)

    if not Path(test_set).exists():
        st.warning(f"测试集文件不存在: {test_set}")

    if st.button("运行评估"):
        with st.spinner("评估运行中…"):
            try:
                report = _run_evaluation(test_set, top_k)
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
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")
    st.caption(f"共 {report['num_cases']} 个测试用例")

    st.subheader("各 Query 明细")
    for q in report["per_query"]:
        with st.expander(f"{q['query']} · {q['metrics']}"):
            st.write("**Retrieved:**", q["retrieved_ids"])
            st.write("**Expected:**", q["expected_ids"])
