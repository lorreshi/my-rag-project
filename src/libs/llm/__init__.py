"""LLM abstractions.

Importing this package registers all built-in LLM providers
(openai / azure / ollama / deepseek) with the LLMFactory via import
side-effects.
"""

# Register built-in providers (side-effect imports).
from src.libs.llm import (  # noqa: F401
    azure_llm,
    azure_vision_llm,
    deepseek_llm,
    ollama_llm,
    openai_llm,
)
