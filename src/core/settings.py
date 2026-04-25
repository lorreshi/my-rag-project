"""Configuration loading & validation.

Provides Settings dataclass and load_settings / validate_settings functions.
Settings is a pure data container — no network/IO initialization here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = ""
    model: str = ""
    azure_endpoint: str = ""
    api_key: str = ""


@dataclass
class EmbeddingConfig:
    provider: str = ""
    model: str = ""


@dataclass
class VisionLLMConfig:
    provider: str = ""
    model: str = ""


@dataclass
class VectorStoreConfig:
    backend: str = ""
    persist_path: str = "./data/db/chroma"


@dataclass
class RetrievalConfig:
    sparse_backend: str = "bm25"
    fusion_algorithm: str = "rrf"
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10


@dataclass
class RerankConfig:
    backend: str = "none"
    model: str = ""
    top_m: int = 30


@dataclass
class EvaluationConfig:
    backends: list[str] = field(default_factory=lambda: ["custom"])


@dataclass
class ObservabilityConfig:
    enabled: bool = True
    log_file: str = "./logs/traces.jsonl"


@dataclass
class DashboardConfig:
    enabled: bool = True
    port: int = 8501
    traces_dir: str = "./logs"
    auto_refresh: bool = True
    refresh_interval: int = 5


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """Top-level configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vision_llm: VisionLLMConfig = field(default_factory=VisionLLMConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# ---------------------------------------------------------------------------
# Required fields — (section, field) pairs that must be non-empty strings
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: list[tuple[str, str]] = [
    ("llm", "provider"),
    ("llm", "model"),
    ("embedding", "provider"),
    ("embedding", "model"),
    ("vector_store", "backend"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Read YAML, parse into Settings, validate required fields.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if required fields are missing or empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    settings = _parse_raw(raw)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """Check that all required fields are present and non-empty.

    Raises ValueError with a human-readable message including the field path.
    """
    missing: list[str] = []
    for section, key in _REQUIRED_FIELDS:
        sub = getattr(settings, section, None)
        if sub is None:
            missing.append(f"{section}")
            continue
        val = getattr(sub, key, None)
        if not val:  # None or empty string
            missing.append(f"{section}.{key}")

    if missing:
        raise ValueError(
            "Missing required configuration field(s): "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_raw(raw: dict[str, Any]) -> Settings:
    """Convert a raw YAML dict into a Settings dataclass tree."""

    def _build(cls: type, data: Any) -> Any:
        if not isinstance(data, dict):
            return cls()
        # Only pass keys that the dataclass actually declares
        valid = {k: v for k, v in data.items() if k in {f.name for f in cls.__dataclass_fields__.values()}}
        return cls(**valid)

    return Settings(
        llm=_build(LLMConfig, raw.get("llm")),
        embedding=_build(EmbeddingConfig, raw.get("embedding")),
        vision_llm=_build(VisionLLMConfig, raw.get("vision_llm")),
        vector_store=_build(VectorStoreConfig, raw.get("vector_store")),
        retrieval=_build(RetrievalConfig, raw.get("retrieval")),
        rerank=_build(RerankConfig, raw.get("rerank")),
        evaluation=_build(EvaluationConfig, raw.get("evaluation")),
        observability=_build(ObservabilityConfig, raw.get("observability")),
        dashboard=_build(DashboardConfig, raw.get("dashboard")),
    )
