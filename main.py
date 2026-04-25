"""Smart Knowledge Hub — MCP Server entry point."""

import sys

from src.core.settings import load_settings
from src.observability.logger import get_logger

logger = get_logger()


def main() -> None:
    """Load config and start the MCP Server."""
    try:
        settings = load_settings()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load configuration: %s", exc)
        sys.exit(1)

    logger.info(
        "Configuration loaded — LLM=%s/%s, Embedding=%s/%s, VectorStore=%s",
        settings.llm.provider,
        settings.llm.model,
        settings.embedding.provider,
        settings.embedding.model,
        settings.vector_store.backend,
    )
    # MCP Server startup will be implemented in Phase E
    print("Smart Knowledge Hub — MCP Server (not yet implemented)", file=sys.stderr)


if __name__ == "__main__":
    main()
