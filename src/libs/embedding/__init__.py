"""Embedding abstractions.

Importing this package registers all built-in embedding providers
(openai / azure / ollama) with the EmbeddingFactory via import side-effects.
"""

# Register built-in providers (side-effect imports).
from src.libs.embedding import (  # noqa: F401
    azure_embedding,
    ollama_embedding,
    openai_embedding,
)
