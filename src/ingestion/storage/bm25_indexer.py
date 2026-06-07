"""BM25Indexer — inverted index construction, persistence, and querying.

Consumes per-chunk term statistics from SparseEncoder (SparseVector), computes
IDF over the corpus, builds an inverted index, and persists it to
``data/db/bm25/``. Provides query() for BM25 scoring (used later by D3
SparseRetriever).

Index structure (in memory and serialized):

    {
      "N": <num_docs>,
      "avgdl": <average doc length>,
      "doc_lengths": {chunk_id: length},
      "index": {
        term: {
          "idf": <float>,
          "postings": [[chunk_id, tf], ...]
        }
      }
    }

IDF uses the BM25 (Robertson) formula:
    IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)

The ``+ 1`` inside log keeps IDF non-negative (the Lucene/standard variant),
avoiding negative weights for very common terms.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from src.ingestion.embedding.sparse_encoder import SparseVector

logger = logging.getLogger(__name__)

# BM25 free parameters (standard defaults)
_K1 = 1.5
_B = 0.75


class BM25Indexer:
    """Build, persist, load, and query a BM25 inverted index."""

    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        k1: float = _K1,
        b: float = _B,
    ):
        self._index_dir = Path(index_dir)
        self._k1 = k1
        self._b = b
        # Index state
        self._index: dict[str, dict] = {}
        self._doc_lengths: dict[str, int] = {}
        self._n_docs: int = 0
        self._avgdl: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, sparse_vectors: Iterable["SparseVector"]) -> None:
        """Build the inverted index from scratch over the given corpus.

        Empty chunks (no terms) still count toward N and doc_lengths so that
        corpus statistics remain accurate.
        """
        self._index = {}
        self._doc_lengths = {}

        # term -> {chunk_id -> tf}
        postings: dict[str, dict[str, int]] = {}
        # term -> document frequency
        df: dict[str, int] = {}

        n = 0
        for sv in sparse_vectors:
            n += 1
            self._doc_lengths[sv.chunk_id] = sv.doc_length
            for term, tf in sv.term_freqs.items():
                postings.setdefault(term, {})[sv.chunk_id] = tf
                df[term] = df.get(term, 0) + 1

        self._n_docs = n
        total_len = sum(self._doc_lengths.values())
        self._avgdl = (total_len / n) if n > 0 else 0.0

        for term, plist in postings.items():
            self._index[term] = {
                "idf": self._compute_idf(df[term], n),
                "postings": [[cid, tf] for cid, tf in plist.items()],
            }

        logger.info("Built BM25 index: %d docs, %d terms", n, len(self._index))

    def add_documents(self, sparse_vectors: Iterable["SparseVector"]) -> None:
        """Incrementally add documents and recompute corpus statistics.

        New chunk_ids are added; existing chunk_ids are replaced (idempotent
        update). IDF values are recomputed across the full corpus afterwards.
        """
        # Reconstruct per-doc term freqs from current index to merge cleanly.
        doc_terms: dict[str, dict[str, int]] = {}
        for term, entry in self._index.items():
            for cid, tf in entry["postings"]:
                doc_terms.setdefault(cid, {})[term] = tf

        # Apply updates
        for sv in sparse_vectors:
            doc_terms[sv.chunk_id] = dict(sv.term_freqs)
            self._doc_lengths[sv.chunk_id] = sv.doc_length

        # Rebuild from merged doc_terms
        self._rebuild_from_doc_terms(doc_terms)

    def remove_document(self, chunk_id: str) -> None:
        """Remove a document from the index and recompute statistics."""
        doc_terms: dict[str, dict[str, int]] = {}
        for term, entry in self._index.items():
            for cid, tf in entry["postings"]:
                if cid == chunk_id:
                    continue
                doc_terms.setdefault(cid, {})[term] = tf
        self._doc_lengths.pop(chunk_id, None)
        self._rebuild_from_doc_terms(doc_terms)

    def _rebuild_from_doc_terms(
        self, doc_terms: dict[str, dict[str, int]]
    ) -> None:
        """Rebuild index/idf/avgdl from a {chunk_id: {term: tf}} mapping."""
        postings: dict[str, dict[str, int]] = {}
        df: dict[str, int] = {}
        for cid, terms in doc_terms.items():
            for term, tf in terms.items():
                postings.setdefault(term, {})[cid] = tf
                df[term] = df.get(term, 0) + 1

        n = len(doc_terms)
        self._n_docs = n
        # Ensure doc_lengths only contains current docs
        self._doc_lengths = {
            cid: self._doc_lengths.get(cid, sum(terms.values()))
            for cid, terms in doc_terms.items()
        }
        total_len = sum(self._doc_lengths.values())
        self._avgdl = (total_len / n) if n > 0 else 0.0

        self._index = {}
        for term, plist in postings.items():
            self._index[term] = {
                "idf": self._compute_idf(df[term], n),
                "postings": [[cid, tf] for cid, tf in plist.items()],
            }

    def _compute_idf(self, df: int, n: int) -> float:
        """BM25 IDF with +1 smoothing (non-negative)."""
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the index to ``{index_dir}/bm25_index.json``."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        path = self._index_dir / "bm25_index.json"
        payload = {
            "N": self._n_docs,
            "avgdl": self._avgdl,
            "k1": self._k1,
            "b": self._b,
            "doc_lengths": self._doc_lengths,
            "index": self._index,
        }
        # Atomic write
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        logger.info("Saved BM25 index to %s", path)

    def load(self) -> None:
        """Load the index from disk. Raises FileNotFoundError if absent."""
        path = self._index_dir / "bm25_index.json"
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._n_docs = payload["N"]
        self._avgdl = payload["avgdl"]
        self._k1 = payload.get("k1", _K1)
        self._b = payload.get("b", _B)
        self._doc_lengths = payload["doc_lengths"]
        self._index = payload["index"]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_terms: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """Score the corpus against query terms using BM25.

        Args:
            query_terms: Already-tokenized query terms (lowercased to match index).
            top_k: Maximum number of results.

        Returns:
            List of (chunk_id, score) sorted by descending score, then chunk_id.
        """
        if not query_terms or self._n_docs == 0:
            return []

        scores: dict[str, float] = {}
        for term in query_terms:
            entry = self._index.get(term)
            if entry is None:
                continue
            idf = entry["idf"]
            for chunk_id, tf in entry["postings"]:
                dl = self._doc_lengths.get(chunk_id, 0)
                denom = tf + self._k1 * (
                    1 - self._b + self._b * (dl / self._avgdl if self._avgdl else 0)
                )
                score = idf * (tf * (self._k1 + 1)) / denom if denom else 0.0
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        return self._n_docs

    @property
    def num_terms(self) -> int:
        return len(self._index)

    def get_idf(self, term: str) -> float:
        """Return the IDF of a term (0.0 if not present)."""
        entry = self._index.get(term.lower())
        return entry["idf"] if entry else 0.0
