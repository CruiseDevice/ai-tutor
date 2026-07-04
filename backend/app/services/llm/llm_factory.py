"""LLM client construction.

Returns the appropriate async SDK client for a provider. OpenAI and Ollama both
use the OpenAI SDK (Ollama via base_url); Anthropic uses its own SDK.
"""

import logging

from openai import AsyncOpenAI

from .provider import Provider, get_base_url

logger = logging.getLogger(__name__)


def get_client(
    provider: Provider,
    api_key: str,
    base_url: str | None = None,
) -> object:
    """Construct an async LLM client for the given provider.

    Args:
        provider: Target provider.
        api_key: Decrypted user API key for the provider.
        base_url: Optional override; falls back to the provider's configured url.

    Returns:
        AsyncOpenAI for openai/ollama, AsyncAnthropic for anthropic. The return
        type is `object` because the two SDKs don't share a base class — callers
        should use LLMClient for a uniform interface rather than the raw client.
    """
    if provider == Provider.ANTHROPIC:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "anthropic package is not installed. "
                "Run: pip install anthropic"
            ) from e
        return AsyncAnthropic(api_key=api_key)

    # OpenAI or Ollama — both speak the OpenAI API.
    url = base_url or get_base_url(provider)
    return AsyncOpenAI(api_key=api_key, base_url=url) if url else AsyncOpenAI(api_key=api_key)
