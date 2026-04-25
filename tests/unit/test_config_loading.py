"""Tests for config loading & validation (A3)."""

import pytest
import yaml
from pathlib import Path

from src.core.settings import (
    Settings,
    load_settings,
    validate_settings,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
)


@pytest.mark.unit
class TestLoadSettings:
    """load_settings reads YAML and returns a Settings object."""

    def test_load_default_config(self):
        """The shipped config/settings.yaml should load without error."""
        settings = load_settings("config/settings.yaml")
        assert isinstance(settings, Settings)
        assert settings.llm.provider == "azure"
        assert settings.embedding.model == "text-embedding-3-small"
        assert settings.vector_store.backend == "chroma"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_settings("nonexistent.yaml")

    def test_load_from_tmp(self, tmp_path: Path):
        """Load a minimal valid config from a temp file."""
        cfg = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "vector_store": {"backend": "chroma"},
        }
        p = tmp_path / "settings.yaml"
        p.write_text(yaml.dump(cfg))
        settings = load_settings(str(p))
        assert settings.llm.provider == "openai"


@pytest.mark.unit
class TestValidateSettings:
    """validate_settings catches missing required fields."""

    def test_valid_settings_passes(self):
        s = Settings(
            llm=LLMConfig(provider="azure", model="gpt-4o"),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(backend="chroma"),
        )
        validate_settings(s)  # should not raise

    def test_missing_llm_provider(self):
        s = Settings(
            llm=LLMConfig(provider="", model="gpt-4o"),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(backend="chroma"),
        )
        with pytest.raises(ValueError, match="llm.provider"):
            validate_settings(s)

    def test_missing_embedding_provider(self):
        s = Settings(
            llm=LLMConfig(provider="azure", model="gpt-4o"),
            embedding=EmbeddingConfig(provider="", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(backend="chroma"),
        )
        with pytest.raises(ValueError, match="embedding.provider"):
            validate_settings(s)

    def test_missing_embedding_model(self):
        s = Settings(
            llm=LLMConfig(provider="azure", model="gpt-4o"),
            embedding=EmbeddingConfig(provider="openai", model=""),
            vector_store=VectorStoreConfig(backend="chroma"),
        )
        with pytest.raises(ValueError, match="embedding.model"):
            validate_settings(s)

    def test_missing_vector_store_backend(self):
        s = Settings(
            llm=LLMConfig(provider="azure", model="gpt-4o"),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(backend=""),
        )
        with pytest.raises(ValueError, match="vector_store.backend"):
            validate_settings(s)

    def test_multiple_missing_fields(self):
        s = Settings()  # all defaults are empty
        with pytest.raises(ValueError) as exc_info:
            validate_settings(s)
        msg = str(exc_info.value)
        assert "llm.provider" in msg
        assert "embedding.provider" in msg
        assert "vector_store.backend" in msg


@pytest.mark.unit
class TestSettingsDefaults:
    """Verify sensible defaults for optional sections."""

    def test_retrieval_defaults(self):
        s = Settings()
        assert s.retrieval.fusion_algorithm == "rrf"
        assert s.retrieval.top_k_final == 10

    def test_rerank_defaults(self):
        s = Settings()
        assert s.rerank.backend == "none"

    def test_observability_defaults(self):
        s = Settings()
        assert s.observability.enabled is True

    def test_dashboard_defaults(self):
        s = Settings()
        assert s.dashboard.port == 8501
