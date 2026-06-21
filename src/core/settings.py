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
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = "2024-06-01"
    base_url: str = ""  # For OpenAI-compatible endpoints (DeepSeek, etc.)
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingConfig:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = "2024-06-01"
    base_url: str = ""  # For Ollama or custom endpoints


@dataclass
class VisionLLMConfig:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = "2024-06-01"
    max_image_size: int = 2048  # max edge length in pixels


@dataclass
class VectorStoreConfig:
    backend: str = ""
    persist_path: str = "./data/db/chroma"


@dataclass
class RetrievalConfig:
    sparse_backend: str = "bm25"
    fusion_algorithm: str = "rrf"  # rrf | weighted_sum
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10
    tokenizer: str = "jieba"  # BM25 tokenizer: jieba | regex

    # --- Phase D retrieval enhancements (all default to current behaviour) ---
    # Candidate pool width / fusion (#3, #4)
    candidate_multiplier: int = 2
    rrf_k: int = 60
    fusion_weights: dict[str, float] = field(default_factory=dict)
    # Deterministic text normalization (#1)
    enable_nfkc: bool = True
    normalize_casefold: bool = True
    normalize_to_simplified: bool = False  # needs OpenCC; degrades if missing
    # Filter extraction from query text (#2)
    enable_filter_extraction: bool = False
    # Sparse pre-filter (#5)
    sparse_filter_overfetch: int = 4
    # Synonym/alias OR-expansion into BM25 (#7)
    enable_synonym_expansion: bool = False
    synonym_source: str = ""
    # Query transform: multi-query / HyDE (#8, #9)
    query_transform: str = "none"  # none | multi_query | hyde
    multi_query_count: int = 3
    query_transform_concurrency: int = 4
    query_transform_cache: bool = True
    hyde_augment: bool = True
    hyde_skip_doc_types: list[str] = field(default_factory=lambda: ["xlsx"])
    # MMR diversity (#10)
    enable_mmr: bool = False
    mmr_lambda: float = 0.5
    # Relevance threshold / abstain (#11)
    min_score_threshold: float = 0.0


@dataclass
class SplitterConfig:
    """Text splitting configuration.

    ``chunk_size``/``chunk_overlap`` are measured in ``size_unit`` (``token`` by
    default, aligned with the embedding model's tiktoken encoding).

    ``by_doc_type`` maps a document ``doc_type`` to a splitter type (routing is
    consumed by the chunker in a later task). ``overrides`` maps a collection
    name to a partial config dict (e.g. ``{"chunk_size": 256}``) that overrides
    the defaults for that collection only.
    """

    type: str = "recursive"
    size_unit: str = "token"
    chunk_size: int = 512
    chunk_overlap: int = 64
    token_encoding: str = "cl100k_base"
    by_doc_type: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, dict] = field(default_factory=dict)


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
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
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
        splitter=_build(SplitterConfig, raw.get("splitter")),
        rerank=_build(RerankConfig, raw.get("rerank")),
        evaluation=_build(EvaluationConfig, raw.get("evaluation")),
        observability=_build(ObservabilityConfig, raw.get("observability")),
        dashboard=_build(DashboardConfig, raw.get("dashboard")),
    )
