"""LLM provider abstraction layer.

Centralizes model-id → provider resolution, client construction, a uniform
complete/stream wrapper, and provider-aware token counting so the rest of the
codebase can treat OpenAI / Anthropic / Ollama Cloud interchangeably.
"""

from .provider import (
    Provider,
    resolve_provider,
    get_base_url,
    pick_helper_model,
    pick_smart_model,
    CATALOG,
)
from .llm_factory import get_client
from .llm_client import LLMClient, get_llm_client
from .token_count import count_tokens

__all__ = [
    "Provider",
    "resolve_provider",
    "get_base_url",
    "pick_helper_model",
    "pick_smart_model",
    "CATALOG",
    "get_client",
    "LLMClient",
    "get_llm_client",
    "count_tokens",
]
