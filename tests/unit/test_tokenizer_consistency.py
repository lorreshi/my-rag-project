"""Property 8: tokenizer consistency between ingestion and query sides.

Validates: Requirements 2.2

The ingestion-side ``SparseEncoder`` and the query-side ``QueryProcessor`` must
produce aligned BM25 vocabularies for the same text, because both now delegate
tokenization to a single shared ``BaseTokenizer``.

Contract differences (intentional, asserted here):
- ``SparseEncoder`` emits ``term_freqs`` (a count map -> unordered, keeps
  duplicates as counts).
- ``QueryProcessor`` emits ``keywords`` (deduped, first-seen order preserved).

Both are derived from the very same ``tokenizer.tokenize(text)`` call, so the
correct way to pin consistency is:
- ``QueryProcessor.keywords == dedup_preserve_order(tokenizer.tokenize(text))``
- ``set(SparseEncoder.term_freqs) == set(QueryProcessor.keywords)``

i.e. both sides cover exactly the same term set and both originate from the
same segmentation output. This proves "same text -> same segmentation -> BM25
vocabulary alignment".
"""
from __future__ import annotations

import pytest

from src.core.query_engine.query_processor import QueryProcessor
from src.core.types import Chunk
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.libs.tokenizer import JiebaTokenizer, TokenizerFactory


def _dedup_preserve_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# Chinese / English / mixed text covering the realistic input space.
CONSISTENCY_TEXTS = [
    "检索增强生成是一种结合向量检索与大语言模型的方法",
    "vector database similarity search with hybrid retrieval",
    "使用 BM25 和 dense embedding 做 hybrid 混合检索 rerank",
    "Azure OpenAI 的 text-embedding-3-small 模型维度是 1536",
    "中文分词 jieba 与 tiktoken 是两个不同的组件",
]


def _make_sparse_encoder_and_query_processor():
    """Drive both sides from the SAME tokenizer instance."""
    tokenizer = JiebaTokenizer()
    encoder = SparseEncoder(tokenizer=tokenizer)
    processor = QueryProcessor(tokenizer=tokenizer)
    return tokenizer, encoder, processor


@pytest.mark.parametrize("text", CONSISTENCY_TEXTS)
def test_query_keywords_are_dedup_of_shared_tokenization(text):
    """QueryProcessor.keywords == dedup_preserve_order(tokenizer.tokenize)."""
    tokenizer, _encoder, processor = _make_sparse_encoder_and_query_processor()

    result = processor.process(text)
    expected = _dedup_preserve_order(tokenizer.tokenize(text))

    assert result.keywords == expected


@pytest.mark.parametrize("text", CONSISTENCY_TEXTS)
def test_sparse_and_query_cover_identical_term_set(text):
    """Both sides cover exactly the same term set from the same segmentation."""
    _tokenizer, encoder, processor = _make_sparse_encoder_and_query_processor()

    chunk = Chunk(id="c0", text=text, metadata={})
    sparse_vector = encoder.encode([chunk])[0]

    result = processor.process(text)

    # SparseEncoder term_freqs keys == QueryProcessor keywords as a set.
    assert set(sparse_vector.term_freqs) == set(result.keywords)


@pytest.mark.parametrize("text", CONSISTENCY_TEXTS)
def test_both_sides_originate_from_same_tokenization(text):
    """Term set on both sides equals the unique tokens of the shared output."""
    tokenizer, encoder, processor = _make_sparse_encoder_and_query_processor()

    tokens = tokenizer.tokenize(text)
    unique_terms = set(tokens)

    chunk = Chunk(id="c0", text=text, metadata={})
    sparse_vector = encoder.encode([chunk])[0]
    result = processor.process(text)

    assert set(sparse_vector.term_freqs) == unique_terms
    assert set(result.keywords) == unique_terms


@pytest.mark.parametrize("text", CONSISTENCY_TEXTS)
def test_consistency_via_factory_created_instances(text):
    """Two tokenizers built from the SAME config stay aligned (not just same instance)."""

    class _Retrieval:
        tokenizer = "jieba"

    class _Settings:
        retrieval = _Retrieval()

    settings = _Settings()
    encoder = SparseEncoder(tokenizer=TokenizerFactory.create(settings))
    processor = QueryProcessor(tokenizer=TokenizerFactory.create(settings))

    chunk = Chunk(id="c0", text=text, metadata={})
    sparse_vector = encoder.encode([chunk])[0]
    result = processor.process(text)

    assert set(sparse_vector.term_freqs) == set(result.keywords)
