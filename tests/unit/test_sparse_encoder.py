"""Unit tests for SparseEncoder — BM25 term statistics output contract.

Tokenization is delegated to an injected ``BaseTokenizer`` (Task 2). Tests
either inject a deterministic ``FakeTokenizer`` to assert the encoder's
delegation/counting contract precisely, or inject a ``JiebaTokenizer`` to
assert real Chinese word-level term_freqs.
"""
from __future__ import annotations

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.sparse_encoder import SparseEncoder, SparseVector
from src.libs.tokenizer import BaseTokenizer, JiebaTokenizer


def _chunk(text: str, chunk_id: str = "c0") -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={}, source_ref="doc")


class FakeTokenizer(BaseTokenizer):
    """Deterministic whitespace tokenizer for asserting delegation.

    Splits on whitespace, lowercases, and drops an optional stopword set.
    Empty/whitespace-only text yields an empty list (matching the contract).
    """

    def __init__(self, stopwords: set[str] | None = None):
        self._stopwords = stopwords or set()

    def tokenize(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [t for t in text.lower().split() if t not in self._stopwords]


# A jieba tokenizer with no stopword filtering, for raw word-level assertions.
def _no_stopword_jieba() -> JiebaTokenizer:
    return JiebaTokenizer(stopwords=set())


class TestEncoding:
    def test_output_count_matches_input(self):
        enc = SparseEncoder()
        chunks = [_chunk("hello world", "c0"), _chunk("foo bar baz", "c1")]
        result = enc.encode(chunks)
        assert len(result) == 2
        assert all(isinstance(r, SparseVector) for r in result)

    def test_term_freqs_counted(self):
        enc = SparseEncoder(tokenizer=FakeTokenizer())
        result = enc.encode([_chunk("alpha beta alpha gamma alpha")])
        sv = result[0]
        assert sv.term_freqs["alpha"] == 3
        assert sv.term_freqs["beta"] == 1
        assert sv.term_freqs["gamma"] == 1

    def test_doc_length(self):
        enc = SparseEncoder(tokenizer=FakeTokenizer())
        result = enc.encode([_chunk("one two three four")])
        assert result[0].doc_length == 4

    def test_chunk_id_preserved(self):
        enc = SparseEncoder()
        result = enc.encode([_chunk("text here", "my_chunk")])
        assert result[0].chunk_id == "my_chunk"

    def test_lowercase_via_tokenizer(self):
        enc = SparseEncoder(tokenizer=FakeTokenizer())
        result = enc.encode([_chunk("Apple APPLE apple")])
        assert result[0].term_freqs["apple"] == 3

    def test_stopwords_removed_via_tokenizer(self):
        enc = SparseEncoder(tokenizer=FakeTokenizer(stopwords={"the", "and"}))
        result = enc.encode([_chunk("the cat and the dog")])
        assert "the" not in result[0].term_freqs
        assert "and" not in result[0].term_freqs
        assert result[0].term_freqs["cat"] == 1
        assert result[0].term_freqs["dog"] == 1

    def test_delegates_to_injected_tokenizer(self):
        """term_freqs are exactly Counter(tokenizer.tokenize(text))."""
        tok = _no_stopword_jieba()
        text = "检索增强 hello 检索"
        enc = SparseEncoder(tokenizer=tok)
        sv = enc.encode([_chunk(text)])[0]
        expected = tok.tokenize(text)
        assert sv.doc_length == len(expected)
        for term in set(expected):
            assert sv.term_freqs[term] == expected.count(term)


class TestChineseWordLevel:
    def test_cjk_tokenized_word_level(self):
        """With jieba, Chinese is segmented into words, not single chars."""
        tok = _no_stopword_jieba()
        text = "检索增强生成"
        enc = SparseEncoder(tokenizer=tok)
        sv = enc.encode([_chunk(text)])[0]
        expected = tok.tokenize(text)
        # jieba yields word-level tokens (fewer than the 6 characters).
        assert sv.doc_length == len(expected)
        assert sv.term_freqs == {t: expected.count(t) for t in set(expected)}

    def test_char_level_via_injected_tokenizer(self):
        """A char-level tokenizer can be injected to assert per-char tokens."""

        class CharTokenizer(BaseTokenizer):
            def tokenize(self, text: str) -> list[str]:
                return [c for c in text if c.strip()]

        enc = SparseEncoder(tokenizer=CharTokenizer())
        result = enc.encode([_chunk("检索增强")])
        assert result[0].doc_length == 4
        assert result[0].term_freqs["检"] == 1


class TestEmptyText:
    def test_empty_string(self):
        enc = SparseEncoder()
        result = enc.encode([_chunk("", "empty")])
        assert result[0].chunk_id == "empty"
        assert result[0].term_freqs == {}
        assert result[0].doc_length == 0

    def test_whitespace_only(self):
        enc = SparseEncoder()
        result = enc.encode([_chunk("   \n\t  ")])
        assert result[0].term_freqs == {}
        assert result[0].doc_length == 0

    def test_stopwords_only_yields_empty(self):
        enc = SparseEncoder()  # default jieba tokenizer with stopwords
        result = enc.encode([_chunk("the and of to")])
        assert result[0].term_freqs == {}
        assert result[0].doc_length == 0

    def test_empty_chunk_list(self):
        enc = SparseEncoder()
        assert enc.encode([]) == []


class TestContractForIndexer:
    def test_to_dict_structure(self):
        enc = SparseEncoder(tokenizer=FakeTokenizer())
        sv = enc.encode([_chunk("alpha beta")])[0]
        d = sv.to_dict()
        assert set(d.keys()) == {"chunk_id", "term_freqs", "doc_length"}
        assert d["doc_length"] == 2

    def test_usable_for_idf_aggregation(self):
        """Document frequency can be derived across SparseVectors."""
        enc = SparseEncoder(tokenizer=FakeTokenizer())
        results = enc.encode([
            _chunk("apple banana", "c0"),
            _chunk("apple cherry", "c1"),
            _chunk("banana cherry", "c2"),
        ])
        # df(apple) = number of chunks containing 'apple' = 2
        df_apple = sum(1 for r in results if "apple" in r.term_freqs)
        assert df_apple == 2


class TestTrace:
    def test_trace_records_stage(self):
        enc = SparseEncoder()
        trace = TraceContext(trace_type="ingestion")
        enc.encode([_chunk("hello world"), _chunk("")], trace=trace)
        stage = trace.stages[0]
        assert stage.name == "sparse_encoder"
        assert stage.details["count"] == 2
        assert stage.details["empty"] == 1
