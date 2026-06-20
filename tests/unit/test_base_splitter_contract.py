"""Contract tests for SplitPiece + BaseSplitter.split()/split_text() (T4).

Validates the backward-compatible two-way delegation contract:

* A splitter that implements only ``split()`` gets a correct derived
  ``split_text()`` (text only).
* A legacy splitter that implements only ``split_text()`` gets a derived
  ``split()`` whose pieces carry empty metadata.
* ``SplitPiece`` metadata defaults to an independent empty dict.
* A subclass overriding neither method fails at definition time.

Validates: Requirements 3.1, 3.2
"""
from __future__ import annotations

import pytest

from src.libs.splitter.base_splitter import BaseSplitter, SplitPiece


class SplitOnlySplitter(BaseSplitter):
    """Minimal splitter that implements only split() with metadata."""

    def split(self, text, trace=None):
        pieces = []
        for i, part in enumerate(p for p in text.split("\n") if p.strip()):
            # Even-indexed pieces carry metadata, odd-indexed do not.
            meta = {"idx": i} if i % 2 == 0 else {}
            pieces.append(SplitPiece(part, meta))
        return pieces

    @property
    def splitter_type(self) -> str:
        return "split_only"


class SplitTextOnlySplitter(BaseSplitter):
    """Legacy splitter that implements only split_text()."""

    def split_text(self, text, trace=None):
        return [chunk for chunk in text.split("\n") if chunk.strip()]

    @property
    def splitter_type(self) -> str:
        return "split_text_only"


@pytest.mark.unit
class TestSplitPiece:

    def test_default_metadata_is_empty_dict(self):
        piece = SplitPiece("hello")
        assert piece.text == "hello"
        assert piece.metadata == {}

    def test_instances_do_not_share_metadata(self):
        a = SplitPiece("a")
        b = SplitPiece("b")
        a.metadata["k"] = "v"
        # Mutating one instance's metadata must not leak into the other.
        assert b.metadata == {}
        assert a.metadata == {"k": "v"}


@pytest.mark.unit
class TestSplitImpliesSplitText:
    """A split()-only splitter derives a correct text-only split_text()."""

    def test_split_returns_pieces(self):
        s = SplitOnlySplitter()
        pieces = s.split("alpha\nbeta\ngamma")
        assert [p.text for p in pieces] == ["alpha", "beta", "gamma"]
        assert pieces[0].metadata == {"idx": 0}
        assert pieces[1].metadata == {}
        assert pieces[2].metadata == {"idx": 2}

    def test_split_text_derived_takes_only_text(self):
        s = SplitOnlySplitter()
        chunks = s.split_text("alpha\nbeta\ngamma")
        assert chunks == ["alpha", "beta", "gamma"]
        assert all(isinstance(c, str) for c in chunks)


@pytest.mark.unit
class TestSplitTextImpliesSplit:
    """A legacy split_text()-only splitter derives split() with empty meta."""

    def test_split_text_still_works(self):
        s = SplitTextOnlySplitter()
        assert s.split_text("hello\nworld") == ["hello", "world"]

    def test_split_derived_pieces_have_empty_metadata(self):
        s = SplitTextOnlySplitter()
        pieces = s.split("hello\nworld")
        assert [p.text for p in pieces] == ["hello", "world"]
        assert all(isinstance(p, SplitPiece) for p in pieces)
        assert all(p.metadata == {} for p in pieces)
        # Derived pieces must not share a single metadata dict.
        assert pieces[0].metadata is not pieces[1].metadata


@pytest.mark.unit
class TestSubclassGuard:

    def test_subclass_overriding_neither_raises_typeerror(self):
        with pytest.raises(TypeError, match="must override at least one"):

            class BadSplitter(BaseSplitter):
                @property
                def splitter_type(self) -> str:
                    return "bad"
