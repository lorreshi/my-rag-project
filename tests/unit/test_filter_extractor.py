"""Tests for T7: BaseFilterExtractor + RuleBasedFilterExtractor + QueryProcessor merge.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
"""
from __future__ import annotations

import pytest

from src.core.query_engine.filter_extractor import (
    BaseFilterExtractor,
    RuleBasedFilterExtractor,
)
from src.core.query_engine.query_processor import QueryProcessor


@pytest.mark.unit
class TestRuleBasedFilterExtractor:
    def setup_method(self):
        self.ex = RuleBasedFilterExtractor()

    def test_explicit_key_value(self):
        assert self.ex.extract("salary collection:hr") == {"collection": "hr"}

    def test_fullwidth_colon(self):
        assert self.ex.extract("查询 doc_type：pdf") == {"doc_type": "pdf"}

    def test_quoted_value(self):
        assert self.ex.extract('sheet_name:"Q1 Sales"') == {"sheet_name": "Q1 Sales"}

    def test_is_table_bool_coercion(self):
        assert self.ex.extract("is_table:true") == {"is_table": True}
        assert self.ex.extract("is_table:false") == {"is_table": False}

    def test_int_coercion(self):
        assert self.ex.extract("row_start:5 row_end:9") == {"row_start": 5, "row_end": 9}

    def test_unknown_keys_ignored(self):
        assert self.ex.extract("foo:bar baz:qux") == {}

    def test_no_match_returns_empty(self):
        assert self.ex.extract("just a normal question") == {}

    def test_empty_query(self):
        assert self.ex.extract("") == {}

    def test_malformed_int_skipped(self):
        # non-int value for int key is skipped, not raised
        assert self.ex.extract("row_start:abc collection:x") == {"collection": "x"}

    def test_is_base_filter_extractor(self):
        assert isinstance(self.ex, BaseFilterExtractor)


@pytest.mark.unit
class TestQueryProcessorMerge:
    def test_no_extractor_is_unchanged_behaviour(self):
        qp = QueryProcessor()  # no extractor
        result = qp.process("collection:x some query", filters={"doc_type": "pdf"})
        # extraction OFF -> only external filters (minus None)
        assert result.filters == {"doc_type": "pdf"}

    def test_extractor_fills_filters(self):
        qp = QueryProcessor(filter_extractor=RuleBasedFilterExtractor())
        result = qp.process("salary collection:hr")
        assert result.filters == {"collection": "hr"}

    def test_external_overrides_extracted(self):
        qp = QueryProcessor(filter_extractor=RuleBasedFilterExtractor())
        # query says collection:hr but caller explicitly passes collection=fin
        result = qp.process("collection:hr", filters={"collection": "fin"})
        assert result.filters == {"collection": "fin"}

    def test_extracted_fills_only_missing_keys(self):
        qp = QueryProcessor(filter_extractor=RuleBasedFilterExtractor())
        result = qp.process("collection:hr doc_type:pdf", filters={"collection": "fin"})
        assert result.filters == {"collection": "fin", "doc_type": "pdf"}

    def test_none_values_dropped(self):
        qp = QueryProcessor(filter_extractor=RuleBasedFilterExtractor())
        result = qp.process("hello", filters={"collection": None})
        assert result.filters == {}

    def test_extractor_exception_does_not_break(self):
        class Boom(BaseFilterExtractor):
            def extract(self, query):
                raise RuntimeError("boom")

        qp = QueryProcessor(filter_extractor=Boom())
        # process must still succeed; filters fall back to external only
        result = qp.process("hello", filters={"doc_type": "pdf"})
        assert result.filters == {"doc_type": "pdf"}
