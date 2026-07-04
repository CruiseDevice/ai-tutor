"""Uniform LLM client wrapper.

Hides the differences between the OpenAI and Anthropic async SDKs behind two
methods — complete() and stream() — so call sites stay provider-agnostic.

Differences normalized:
  - Message format: OpenAI takes system as the first message; Anthropic takes
    system as a top-level `system=` parameter.
  - Max tokens: OpenAI uses `max_completion_tokens`; Anthropic uses `max_tokens`.
  - Streaming events: OpenAI yields choices[0].delta.content; Anthropic yields
    content_block_delta events with delta.text.
"""

import logging
from typing import AsyncIterator, List, Dict

from .llm_factory import get_client
from .provider import Provider

logger = logging.getLogger(__name__)


class LLMClient:
    """Provider-agnostic wrapper around an underlying async SDK client."""

    def __init__(self, provider: Provider, api_key: str, base_url: str | None = None):
        self.provider = provider
        self.client = get_client(provider, api_key, base_url)

    async def complete(
        self,
        system_prompt: str,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Non-streaming completion. Returns the full text response."""
        if self.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(
                system_prompt, messages, model, temperature, max_tokens
            )
        return await self._complete_openai(
            system_prompt, messages, model, temperature, max_tokens
        )

    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Streaming completion. Yields plain text deltas."""
        if self.provider == Provider.ANTHROPIC:
            async for delta in self._stream_anthropic(
                system_prompt, messages, model, temperature, max_tokens
            ):
                yield delta
            return
        async for delta in self._stream_openai(
            system_prompt, messages, model, temperature, max_tokens
        ):
            yield delta

    # ------------------------------------------------------------------
    # OpenAI-compatible (also serves Ollama Cloud)
    # ------------------------------------------------------------------
    async def _complete_openai(
        self, system_prompt, messages, model, temperature, max_tokens
    ) -> str:
        full = [{"role": "system", "content": system_prompt}] + messages
        completion = await self.client.chat.completions.create(
            model=model,
            messages=full,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    async def _stream_openai(
        self, system_prompt, messages, model, temperature, max_tokens
    ) -> AsyncIterator[str]:
        full = [{"role": "system", "content": system_prompt}] + messages
        stream = await self.client.chat.completions.create(
            model=model,
            messages=full,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    async def _complete_anthropic(
        self, system_prompt, messages, model, temperature, max_tokens
    ) -> str:
        response = await self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Anthropic returns a list of content blocks; concatenate text blocks.
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    async def _stream_anthropic(
        self, system_prompt, messages, model, temperature, max_tokens
    ) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=model,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text


def get_llm_client(
    provider: Provider, api_key: str, base_url: str | None = None
) -> LLMClient:
    """Convenience constructor."""
    return LLMClient(provider, api_key, base_url)
