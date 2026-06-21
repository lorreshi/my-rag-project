"""Tests for T1: RetrievalConfig field expansion + settings.yaml wiring.

Covers:
- New RetrievalConfig fields and their backward-compatible defaults.
- Parsing a fully-specified retrieval section from a raw dict.
- Graceful fallback when fields / the whole section are missing.
- Unknown keys are ignored by _parse_raw.

Validates: Requirements 3.2, 3.4, 12.4
"""
from __future__ import annotations

import pytest

from src.core.settings import RetrievalConfig, Settings, _parse_raw, load_settings


@pytest.mark.unit
class TestRetrievalConfigDefaults:
    def test_defaults_preserve_current_behaviour(self):
        cfg = RetrievalConfig()
        # pre-existing
        assert cfg.sparse_backend == "bm25"
        assert cfg.fusion_algorithm == "rrf"
        assert cfg.top_k_dense == 20
        assert cfg.top_k_sparse == 20
        assert cfg.top_k_final == 10
        assert cfg.tokenizer == "jieba"
        # candidate pool / fusion (#3, #4) — equivalent to old hardcoded behaviour
        assert cfg.candidate_multiplier == 2
        assert cfg.rrf_k == 60
        assert cfg.fusion_weights == {}
        # normalization (#1)
        assert cfg.enable_nfkc is True
        assert cfg.normalize_casefold is True
        assert cfg.normalize_to_simplified is False
        # advanced enhancements default OFF (#2, #5, #7, #8-#11)
        assert cfg.enable_filter_extraction is False
        assert cfg.sparse_filter_overfetch == 4
        assert cfg.enable_synonym_expansion is False
        assert cfg.synonym_source == ""
        assert cfg.query_transform == "none"
        assert cfg.multi_query_count == 3
        assert cfg.query_transform_concurrency == 4
        assert cfg.query_transform_cache is True
        assert cfg.hyde_augment is True
        assert cfg.hyde_skip_doc_types == ["xlsx"]
        assert cfg.enable_mmr is False
        assert cfg.mmr_lambda == 0.5
        assert cfg.min_score_threshold == 0.0

    def test_settings_has_retrieval_default(self):
        settings = Settings()
        assert isinstance(settings.retrieval, RetrievalConfig)
        assert settings.retrieval.candidate_multiplier == 2

    def test_independent_default_containers(self):
        a = RetrievalConfig()
        b = RetrievalConfig()
        a.fusion_weights["dense"] = 2.0
        a.hyde_skip_doc_types.append("pdf")
        assert b.fusion_weights == {}
        assert b.hyde_skip_doc_types == ["xlsx"]


@pytest.mark.unit
class TestRetrievalConfigParsing:
    def test_parse_full_retrieval_section(self):
        raw = {
            "llm": {"provider": "openai", "model": "x"},
            "embedding": {"provider": "openai", "model": "y"},
            "vector_store": {"backend": "chroma"},
            "retrieval": {
                "fusion_algorithm": "weighted_sum",
                "top_k_dense": 30,
                "top_k_sparse": 40,
                "candidate_multiplier": 3,
                "rrf_k": 50,
                "fusion_weights": {"dense": 1.5, "sparse": 0.5},
                "enable_nfkc": False,
                "normalize_casefold": False,
                "normalize_to_simplified": True,
                "enable_filter_extraction": True,
                "sparse_filter_overfetch": 6,
                "enable_synonym_expansion": True,
                "synonym_source": "data/synonyms.json",
                "query_transform": "multi_query",
                "multi_query_count": 5,
                "query_transform_concurrency": 8,
                "query_transform_cache": False,
                "hyde_augment": False,
                "hyde_skip_doc_types": ["xlsx", "csv"],
                "enable_mmr": True,
                "mmr_lambda": 0.7,
                "min_score_threshold": 0.25,
            },
        }
        r = _parse_raw(raw).retrieval
        assert r.fusion_algorithm == "weighted_sum"
        assert r.top_k_dense == 30
        assert r.top_k_sparse == 40
        assert r.candidate_multiplier == 3
        assert r.rrf_k == 50
        assert r.fusion_weights == {"dense": 1.5, "sparse": 0.5}
        assert r.enable_nfkc is False
        assert r.normalize_casefold is False
        assert r.normalize_to_simplified is True
        assert r.enable_filter_extraction is True
        assert r.sparse_filter_overfetch == 6
        assert r.enable_synonym_expansion is True
        assert r.synonym_source == "data/synonyms.json"
        assert r.query_transform == "multi_query"
        assert r.multi_query_count == 5
        assert r.query_transform_concurrency == 8
        assert r.query_transform_cache is False
        assert r.hyde_augment is False
        assert r.hyde_skip_doc_types == ["xlsx", "csv"]
        assert r.enable_mmr is True
        assert r.mmr_lambda == 0.7
        assert r.min_score_threshold == 0.25

    def test_parse_partial_keeps_other_defaults(self):
        raw = {"retrieval": {"candidate_multiplier": 4}}
        r = _parse_raw(raw).retrieval
        assert r.candidate_multiplier == 4
        # unspecified fields fall back to defaults
        assert r.rrf_k == 60
        assert r.query_transform == "none"
        assert r.enable_mmr is False

    def test_missing_retrieval_section_uses_defaults(self):
        raw = {"llm": {"provider": "openai", "model": "x"}}
        r = _parse_raw(raw).retrieval
        assert isinstance(r, RetrievalConfig)
        assert r.top_k_dense == 20
        assert r.min_score_threshold == 0.0

    def test_unknown_keys_are_ignored(self):
        raw = {"retrieval": {"candidate_multiplier": 2, "totally_unknown": 123}}
        r = _parse_raw(raw).retrieval
        assert r.candidate_multiplier == 2
        assert not hasattr(r, "totally_unknown")


@pytest.mark.unit
class TestShippedYaml:
    def test_example_yaml_parses_new_fields(self):
        settings = load_settings("config/settings.yaml.example")
        r = settings.retrieval
        assert r.candidate_multiplier == 2
        assert r.rrf_k == 60
        assert r.fusion_weights == {}
        assert r.query_transform == "none"
        assert r.hyde_skip_doc_types == ["xlsx"]
        assert r.min_score_threshold == 0.0
