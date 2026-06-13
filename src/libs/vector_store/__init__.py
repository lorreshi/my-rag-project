"""VectorStore abstractions.

Importing this package registers the built-in vector store backends
(chroma) with the VectorStoreFactory via import side-effects.
"""

# Register built-in backends (side-effect imports).
from src.libs.vector_store import chroma_store  # noqa: F401
