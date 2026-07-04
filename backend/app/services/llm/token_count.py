"""Provider-aware token counting.

OpenAI/Ollama use tiktoken (exact). Anthropic has no offline tokenizer in the
standard SDK, so we use a chars/4 heuristic — coarse, but good enough for the
budget/truncation decisions the codebase makes (it never needs to be exact).
"""

import logging

from .provider import Provider

logger = logging.getLogger(__name__)

# Lazily initialized tiktoken encoder
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # pragma: no cover - tiktoken always available in this repo
            logger.warning(f"tiktoken unavailable, falling back to heuristic: {e}")
            return None
    return _encoder


def count_tokens(text: str, provider: Provider) -> int:
    """Estimate the token count of `text` for the given provider.

    For OpenAI/Ollama: exact via tiktoken cl100k_base.
    For Anthropic: ~len(text)/4 heuristic (no offline tokenizer available).
    """
    if not text:
        return 0

    if provider == Provider.ANTHROPIC:
        return max(1, len(text) // 4)

    enc = _get_encoder()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))
