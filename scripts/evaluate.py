"""Evaluation runner CLI.

Runs the golden test set through the retrieval pipeline and prints aggregate
metrics (hit_rate, mrr, plus any evaluator-specific metrics).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys

from src.core.settings import load_settings
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.logger import get_logger

logger = get_logger()

_DEFAULT_TEST_SET = "tests/fixtures/golden_test_set.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation.")
    parser.add_argument("--test-set", default=_DEFAULT_TEST_SET, help="Path to golden test set JSON.")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth.")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings YAML path.")
    parser.add_argument(
        "--generate", action="store_true",
        help="Run the LLM generation step (required for generation-quality metrics).",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        settings = load_settings(args.config)
        runner = EvalRunner.from_settings(settings, top_k=args.top_k, generate=args.generate)
        report = runner.run(args.test_set)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - surfaced to user
        logger.error("Evaluation failed: %s", exc)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"\nEvaluation over {report.num_cases} cases")
    print("Aggregate metrics:")
    for name, value in report.aggregate_metrics.items():
        print(f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
