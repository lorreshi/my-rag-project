"""Tests for T2: shared deterministic text normalization (#1).

Covers:
- normalize_text pipeline: NFKC (full/half-width), casefold, optional t2s.
- Idempotence (Property 2).
- Index/query symmetry through the shared tokenizer (Property 1).
- TokenizerFactory forwarding the normalization flags from settings.
- Graceful degradation when OpenCC is unavailable.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
"""
from __future__ import annotations

import importlib.util

import pytest

from src.core.query_engine.query_processor import QueryProcessor
from src.core.settings import RetrievalConfig, Settings
from src.core.types import Chunk
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.libs.tokenizer import JiebaTokenizer, TokenizerFactory, normalize_text

_HAS_OPENCC = importlib.util.find_spec("opencc") is not None


@pytest.mark.unit
class TestNormalizeText:
    def test_nfkc_full_width_to_half_width(self):
        # Full-width letters/digits fold to ASCII; casefold lowercases.
        assert normalize_text("ＡＢＣ１２３") == "abc123"

    def test_casefold(self):
        assert normalize_text("HeLLo WORLD") == "hello world"

    def test_casefold_can_be_disabled(self):
        assert normalize_text("ABC", casefold=False) == "ABC"

    def test_nfkc_can_be_disabled(self):
        # Without NFKC the full-width form is preserved (only casefold applies).
        out = normalize_text("ＡＢＣ", nfkc=False)
        assert out == "ＡＢＣ".casefold()

    def test_empty_input(self):
        assert normalize_text("") == ""

    @pytest.mark.parametrize(
        "text",
        [
            "ＡＢＣ１２３",
            "HeLLo 世界 World",
            "繁體中文 Traditional",
            "ｆｕｌｌwidth ＭＩＸ 混合 123",
        ],
    )
    def test_idempotent(self, text):
        once = normalize_text(text, to_simplified=True)
        twice = normalize_text(once, to_simplified=True)
        assert once == twice

    def test_to_simplified_graceful_without_opencc(self):
        """t2s must never crash; degrades to NFKC+casefold when OpenCC missing."""
        text = "繁體字"
        out = normalize_text(text, to_simplified=True)
        if not _HAS_OPENCC:
            # Degraded: equals the non-t2s pipeline output.
            assert out == normalize_text(text, to_simplified=False)
        else:
            assert out == "繁体字"


@pytest.mark.unit
class TestTokenizerNormalization:
    def test_full_width_and_half_width_tokenize_equal(self):
        tok = JiebaTokenizer()
        assert tok.tokenize("ＡＰＰ") == tok.tokenize("app")

    def test_mixed_case_tokenize_equal(self):
        tok = JiebaTokenizer()
        assert tok.tokenize("Hybrid Search") == tok.tokenize("hybrid search")

    def test_nfkc_disabled_breaks_full_half_equivalence(self):
        tok = JiebaTokenizer(nfkc=False)
        # Without NFKC the full-width run is not folded to ASCII letters.
        assert tok.tokenize("ＡＰＰ") != tok.tokenize("app")


def _settings(nfkc: bool = True, casefold: bool = True, to_simplified: bool = False) -> Settings:
    s = Settings()
    s.retrieval = RetrievalConfig(
        enable_nfkc=nfkc,
        normalize_casefold=casefold,
        normalize_to_simplified=to_simplified,
    )
    return s


@pytest.mark.unit
class TestFactoryForwardsFlags:
    def test_factory_default_applies_nfkc_and_casefold(self):
        tok = TokenizerFactory.create(_settings())
        assert tok.tokenize("ＡＰＰ") == ["app"]

    def test_factory_can_disable_nfkc(self):
        tok = TokenizerFactory.create(_settings(nfkc=False))
        assert tok.tokenize("ＡＰＰ") != ["app"]


# Index/query symmetry (Property 1), including full-width / case differences.
SYMMETRY_TEXTS = [
    "ＡＰＰ 应用 Server",
    "Hybrid SEARCH 混合检索 BM25",
    "ｔｅｘｔ-embedding-3-small 维度 1536",
]


@pytest.mark.unit
@pytest.mark.parametrize("text", SYMMETRY_TEXTS)
def test_index_query_symmetry_with_normalization(text):
    """SparseEncoder (index) and QueryProcessor (query) cover the same terms."""
    tokenizer = JiebaTokenizer()
    encoder = SparseEncoder(tokenizer=tokenizer)
    processor = QueryProcessor(tokenizer=tokenizer)

    sparse_vector = encoder.encode([Chunk(id="c0", text=text, metadata={})])[0]
    result = processor.process(text)

    assert set(sparse_vector.term_freqs) == set(result.keywords)


@pytest.mark.unit
def test_dense_normalized_query_matches_token_form():
    """normalized_query (dense) uses the same NFKC+casefold form as keywords."""
    processor = QueryProcessor()
    result = processor.process("ＨｙｂｒｉＤ 检索")
    # Dense side: full-width folded + casefolded.
    assert "hybrid" in result.normalized_query
    assert "ＨｙｂｒｉＤ" not in result.normalized_query
