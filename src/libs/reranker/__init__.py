"""Reranker abstractions.

Importing this package registers the built-in reranker backends
(cross_encoder / llm) with the RerankerFactory via import side-effects.
"""

# Register built-in backends (side-effect imports).
from src.libs.reranker import (  # noqa: F401
    cross_encoder_reranker,
    llm_reranker,
)
