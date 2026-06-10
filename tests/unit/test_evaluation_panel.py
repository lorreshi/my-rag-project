"""Unit tests for the evaluation panel page (H4) — import + helper."""
from __future__ import annotations

from src.observability.dashboard.pages import evaluation_panel


class TestPage:
    def test_render_callable(self):
        assert callable(evaluation_panel.render)

    def test_has_run_helper(self):
        assert callable(evaluation_panel._run_evaluation)

    def test_default_test_set_constant(self):
        assert evaluation_panel._DEFAULT_TEST_SET.endswith("golden_test_set.json")
