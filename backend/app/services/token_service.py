"""
Token counting and management service for LLM context windows.

Provides utilities for:
- Counting tokens in text using tiktoken
- Mapping model names to appropriate encodings
- Estimating token usage for message lists
- Truncating text to fit token limits while preserving sentence boundaries
"""

import re
import logging
from typing import List, Dict, Optional
import tiktoken

logger = logging.getLogger(__name__)


class TokenService:
    """Service for token counting and text truncation."""

    # Model to encoding mapping for tiktoken
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }

    @staticmethod
    def get_model_encoding(model: str) -> str:
        """
        Get the tiktoken encoding name for a given model.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o-mini")

        Returns:
            Encoding name for tiktoken (e.g., "cl100k_base")
        """
        # Handle model names with versions or suffixes
        for model_prefix, encoding in TokenService.MODEL_ENCODINGS.items():
            if model.startswith(model_prefix):
                return encoding

        # Default to cl100k_base for unknown models (used by most modern OpenAI models)
        logger.warning(f"Unknown model '{model}', defaulting to cl100k_base encoding")
        return "cl100k_base"

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for
            model: Model name to determine encoding

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0

        try:
            encoding_name = TokenService.get_model_encoding(model)
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            # Fallback to character-based estimation if tiktoken fails
            # Rough estimate: 1 token ≈ 4 characters for English text
            logger.warning(f"Token counting failed, using character-based estimation: {e}")
            return len(text) // 4

    @staticmethod
    def estimate_context_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
        """
        Estimate total tokens for a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name to determine encoding

        Returns:
            Estimated total tokens for all messages
        """
        if not messages:
            return 0

        try:
            encoding_name = TokenService.get_model_encoding(model)
            encoding = tiktoken.get_encoding(encoding_name)

            # According to OpenAI's token counting guide:
            # Each message has overhead: <im_start>{role}\n{content}<im_end>\n
            # Approximate: 4 tokens per message for formatting
            total_tokens = 0
            for message in messages:
                content = message.get("content", "")
                total_tokens += len(encoding.encode(content))
                total_tokens += 4  # Message formatting overhead

            total_tokens += 2  # Conversation start/end tokens
            return total_tokens

        except Exception as e:
            # Fallback estimation
            logger.warning(f"Message token estimation failed, using fallback: {e}")
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            return (total_chars // 4) + (len(messages) * 4)

    @staticmethod
    def truncate_text_to_tokens(
        text: str,
        max_tokens: int,
        model: str = "gpt-4",
        prefer_sentence_boundaries: bool = True
    ) -> str:
        """
        Truncate text to fit within a token limit.

        Attempts to preserve sentence boundaries when possible.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens allowed
            model: Model name to determine encoding
            prefer_sentence_boundaries: If True, try to cut at sentence boundaries

        Returns:
            Truncated text that fits within max_tokens
        """
        if not text:
            return text

        # Quick check: if text is already under limit, return as-is
        token_count = TokenService.count_tokens(text, model)
        if token_count <= max_tokens:
            return text

        try:
            encoding_name = TokenService.get_model_encoding(model)
            encoding = tiktoken.get_encoding(encoding_name)

            # Encode the text
            tokens = encoding.encode(text)

            # If already within limit, return original
            if len(tokens) <= max_tokens:
                return text

            # Truncate to max_tokens
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)

            # If prefer_sentence_boundaries, try to cut at last complete sentence
            if prefer_sentence_boundaries:
                # Find the last sentence boundary (., !, ?, or newline)
                sentence_end_pattern = r'[.!?]\s+'
                matches = list(re.finditer(sentence_end_pattern, truncated_text))

                if matches:
                    # Cut at the last sentence boundary
                    last_match = matches[-1]
                    truncated_text = truncated_text[:last_match.end()].rstrip()
                    logger.debug(f"Truncated at sentence boundary: {token_count} -> ~{max_tokens} tokens")
                else:
                    # No sentence boundary found, try word boundary
                    last_space = truncated_text.rfind(' ')
                    if last_space > len(truncated_text) * 0.5:  # Only if we're keeping >50% of text
                        truncated_text = truncated_text[:last_space].rstrip()
                        logger.debug(f"Truncated at word boundary: {token_count} -> ~{max_tokens} tokens")
                    else:
                        logger.debug(f"Truncated at token boundary: {token_count} -> {max_tokens} tokens")
            else:
                logger.debug(f"Truncated at token boundary: {token_count} -> {max_tokens} tokens")

            return truncated_text

        except Exception as e:
            # Fallback: character-based truncation
            logger.warning(f"Token-based truncation failed, using character estimation: {e}")
            estimated_chars = max_tokens * 4  # Rough estimate: 4 chars per token

            if len(text) <= estimated_chars:
                return text

            truncated = text[:estimated_chars]

            if prefer_sentence_boundaries:
                # Try sentence boundary
                sentence_end_pattern = r'[.!?]\s+'
                matches = list(re.finditer(sentence_end_pattern, truncated))
                if matches:
                    last_match = matches[-1]
                    return truncated[:last_match.end()].rstrip()

                # Try word boundary
                last_space = truncated.rfind(' ')
                if last_space > len(truncated) * 0.5:
                    return truncated[:last_space].rstrip()

            return truncated

    @staticmethod
    def calculate_available_tokens(
        max_context_tokens: int,
        system_prompt_tokens: int,
        user_message_tokens: int,
        history_tokens: int,
        response_reserve_tokens: int,
        buffer_tokens: int
    ) -> int:
        """
        Calculate available tokens for chunk context.

        Args:
            max_context_tokens: Maximum context window size
            system_prompt_tokens: Tokens used by system prompt
            user_message_tokens: Tokens in current user message
            history_tokens: Tokens in conversation history
            response_reserve_tokens: Tokens reserved for model response
            buffer_tokens: Safety buffer for token counting inaccuracy

        Returns:
            Number of tokens available for chunk context (>= 0)
        """
        available = (
            max_context_tokens
            - system_prompt_tokens
            - user_message_tokens
            - history_tokens
            - response_reserve_tokens
            - buffer_tokens
        )

        return max(0, available)
