"""Tests for Evaluator interface, CustomEvaluator, and factory (B6)."""

import pytest

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import (
    EvaluatorFactory,
    register_backend,
    _REGISTRY,
)
from src.core.settings import Settings


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    _REGISTRY["custom"] = lambda _s: CustomEvaluator()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


@pytest.mark.unit
class TestBaseEvaluatorInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseEvaluator()


@pytest.mark.unit
class TestCustomEvaluator:

    def test_perfect_hit(self):
        ev = CustomEvaluator()
        m = ev.evaluate("q", ["a", "b"], ["a"])
        assert m["hit_rate"] == 1.0
        assert m["mrr"] == 1.0

    def test_hit_at_rank_2(self):
        ev = CustomEvaluator()
        m = ev.evaluate("q", ["x", "a", "b"], ["a"])
        assert m["hit_rate"] == 1.0
        assert m["mrr"] == pytest.approx(0.5)

    def test_no_hit(self):
        ev = CustomEvaluator()
        m = ev.evaluate("q", ["x", "y"], ["a"])
        assert m["hit_rate"] == 0.0
        assert m["mrr"] == 0.0

    def test_empty_retrieved(self):
        ev = CustomEvaluator()
        m = ev.evaluate("q", [], ["a"])
        assert m["hit_rate"] == 0.0
        assert m["mrr"] == 0.0


    def test_empty_golden(self):
        ev = CustomEvaluator()
        m = ev.evaluate("q", ["a", "b"], [])
        assert m["hit_rate"] == 0.0
        assert m["mrr"] == 0.0

    def test_multiple_golden(self):
        ev = CustomEvaluator()
        m = ev.evaluate("q", ["x", "b", "a"], ["a", "b"])
        assert m["hit_rate"] == 1.0
        # MRR = 1/rank of first golden hit; "b" is at rank 2
        assert m["mrr"] == pytest.approx(0.5)

    def test_evaluator_name(self):
        assert CustomEvaluator().evaluator_name == "custom"

    def test_output_keys_stable(self):
        m = CustomEvaluator().evaluate("q", ["a"], ["a"])
        assert set(m.keys()) == {"hit_rate", "mrr"}


@pytest.mark.unit
class TestEvaluatorFactory:

    def test_create_custom(self):
        ev = EvaluatorFactory.create(Settings(), "custom")
        assert isinstance(ev, CustomEvaluator)

    def test_create_case_insensitive(self):
        ev = EvaluatorFactory.create(Settings(), "CUSTOM")
        assert isinstance(ev, CustomEvaluator)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown evaluator backend 'nope'"):
            EvaluatorFactory.create(Settings(), "nope")
