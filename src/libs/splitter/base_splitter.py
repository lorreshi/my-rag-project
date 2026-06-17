"""Splitter abstract base class.

All splitter implementations must implement this interface so the pipeline
can call ``splitter.split_text(text)`` *or* the richer ``splitter.split(text)``
without knowing which strategy is in use.

Backward-compatibility contract
-------------------------------
The splitting interface comes in two flavours that delegate to each other:

* ``split(text) -> list[SplitPiece]`` carries structured per-chunk metadata.
* ``split_text(text) -> list[str]`` returns plain text chunks.

Both have default implementations, so a subclass only needs to override **one**
of them:

* A subclass that implements ``split()`` automatically gets a correct
  ``split_text()`` (it just extracts ``piece.text``).
* A legacy subclass that implements only ``split_text()`` automatically gets a
  ``split()`` whose pieces carry an empty ``metadata`` dict.

A ``__init_subclass__`` guard forbids subclasses that override *neither*
method, which would otherwise cause infinite mutual recursion at call time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


@dataclass
class SplitPiece:
    """A single split result carrying text plus optional structured metadata.

    Attributes:
        text: The chunk text.
        metadata: Structured per-chunk metadata (e.g. sheet name, row range).
            Defaults to an empty dict; each instance gets its own dict so
            instances never share mutable state.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSplitter(ABC):
    """Abstract base class for text splitting strategies.

    Subclasses MUST override at least one of ``split`` or ``split_text`` and
    MUST provide the ``splitter_type`` property.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Guard against subclasses that override neither split method.

        If a subclass overrides neither ``split`` nor ``split_text``, the two
        default implementations would delegate to each other forever. Catch
        that mistake early — at class definition time — with a clear error.
        """
        super().__init_subclass__(**kwargs)

        # Skip intermediate abstract subclasses (they may legitimately defer
        # the implementation to their own concrete subclasses).
        if getattr(cls, "__abstractmethods__", None):
            return

        overrides_split = cls.split is not BaseSplitter.split
        overrides_split_text = cls.split_text is not BaseSplitter.split_text
        if not (overrides_split or overrides_split_text):
            raise TypeError(
                f"{cls.__name__} must override at least one of "
                f"'split' or 'split_text'."
            )

    def split(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[SplitPiece]:
        """Split *text* into a list of :class:`SplitPiece`.

        Default implementation derives pieces from :meth:`split_text`, giving
        each piece an empty ``metadata`` dict. Subclasses that produce
        structured metadata should override this method.

        Args:
            text: The full document text to split.
            trace: Optional trace context for observability.

        Returns:
            List of split pieces.
        """
        return [SplitPiece(t) for t in self.split_text(text, trace)]

    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        """Split *text* into a list of plain text chunks.

        Default implementation derives chunks from :meth:`split`, keeping only
        the ``text`` of each piece (metadata is dropped). Subclasses that only
        produce plain text may override this method instead of ``split``.

        Args:
            text: The full document text to split.
            trace: Optional trace context for observability.

        Returns:
            List of text chunks.
        """
        return [piece.text for piece in self.split(text, trace)]

    @property
    @abstractmethod
    def splitter_type(self) -> str:
        """Return a human-readable splitter identifier (e.g. 'recursive')."""
        ...
