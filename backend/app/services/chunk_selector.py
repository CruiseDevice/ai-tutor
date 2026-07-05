"""Token-budgeted chunk selection.

Extracted from ChatService. Owns:
- `format_chunk_for_context`: pure formatter that renders a chunk (text or
  image) the way it will appear in the LLM context, with a page header.
- `ChunkSelector.select_by_token_limit`: greedily packs chunks into the
  available token budget, truncating the last-fitting chunk at a sentence
  boundary when needed.

The formatter is shared by the entry points (which build context text for the
LLM) and by the selector (which counts tokens in the formatted form).
"""
import logging
from typing import Dict, List

from ..config import settings
from .token_service import TokenService

logger = logging.getLogger(__name__)


def format_chunk_for_context(chunk: Dict) -> str:
    """Format a chunk for inclusion in the LLM context."""
    page_num = chunk.get("pageNumber", chunk.get("page_number", "?"))
    content = chunk.get("content", "")
    chunk_type = chunk.get("chunk_type", "text")
    position_data = chunk.get("positionData") or chunk.get("position_data") or {}

    if chunk_type == "image" and position_data:
        bbox = position_data.get("bbox") or [0, 0, 0, 0]
        image_id = chunk.get("id", "")

        try:
            x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        except (TypeError, ValueError):
            x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0

        header = (
            f"[Page {page_num} - Image at bbox ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})]"
        )
        metadata_line = f"Image ID: {image_id}"

        return f"{header}:\n{metadata_line}\n{content}"

    return f"[Page {page_num}]: {content}"


class ChunkSelector:
    """Select chunks that fit within a model's token budget."""

    def select_by_token_limit(
        self,
        chunks: List[Dict],
        max_tokens: int,
        model: str,
        system_prompt_tokens: int,
        user_message_tokens: int,
        history_tokens: int,
        response_reserve_tokens: int
    ) -> tuple[List[Dict], Dict[str, int]]:
        """
        Dynamically select chunks based on token limits instead of fixed count.

        Args:
            chunks: List of chunks sorted by relevance (from find_similar_chunks)
            max_tokens: Maximum context window size
            model: Model name for token counting
            system_prompt_tokens: Tokens used by system prompt template (excluding chunks)
            user_message_tokens: Tokens in current user message
            history_tokens: Tokens in conversation history
            response_reserve_tokens: Tokens reserved for model response

        Returns:
            Tuple of (selected_chunks, token_usage_stats)
        """
        from ..config import settings

        # Get configuration
        buffer_tokens = getattr(settings, 'TOKEN_RESERVE_BUFFER', 20000)
        truncation_enabled = getattr(settings, 'CHUNK_TRUNCATION_ENABLED', True)

        # Calculate available tokens for chunks
        available_tokens = TokenService.calculate_available_tokens(
            max_context_tokens=max_tokens,
            system_prompt_tokens=system_prompt_tokens,
            user_message_tokens=user_message_tokens,
            history_tokens=history_tokens,
            response_reserve_tokens=response_reserve_tokens,
            buffer_tokens=buffer_tokens
        )

        logger.info(
            f"Token budget: max={max_tokens}, system={system_prompt_tokens}, "
            f"user={user_message_tokens}, history={history_tokens}, "
            f"response_reserve={response_reserve_tokens}, buffer={buffer_tokens}, "
            f"available_for_chunks={available_tokens}"
        )

        if available_tokens <= 0:
            logger.warning(
                f"No tokens available for chunks! "
                f"Consider reducing history or using a larger context model."
            )
            return [], {
                "selected_chunks": 0,
                "total_chunk_tokens": 0,
                "available_tokens": available_tokens,
                "truncated_chunks": 0,
                "skipped_chunks": len(chunks)
            }

        selected_chunks = []
        tokens_used = 0
        truncated_count = 0
        skipped_count = 0

        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            page_number = chunk.get("pageNumber", "Unknown")

            # Format chunk as it would appear in context (with page number)
            formatted_chunk = format_chunk_for_context(chunk)

            # Count tokens in formatted chunk
            chunk_tokens = TokenService.count_tokens(formatted_chunk, model)

            # Check if chunk fits
            if tokens_used + chunk_tokens <= available_tokens:
                # Chunk fits completely
                selected_chunks.append(chunk)
                tokens_used += chunk_tokens
                logger.debug(
                    f"Added chunk {i+1}/{len(chunks)} "
                    f"(page {page_number}, {chunk_tokens} tokens, "
                    f"total: {tokens_used}/{available_tokens})"
                )
            elif truncation_enabled and tokens_used < available_tokens:
                # Chunk doesn't fit, but we have room and truncation is enabled
                remaining_tokens = available_tokens - tokens_used

                # Need at least some minimum tokens to make truncation worthwhile
                min_useful_tokens = 50
                if remaining_tokens >= min_useful_tokens:
                    # Truncate the chunk content to fit
                    truncated_content = TokenService.truncate_text_to_tokens(
                        chunk_content,
                        max_tokens=remaining_tokens - 20,  # Reserve tokens for page number formatting
                        model=model,
                        prefer_sentence_boundaries=True
                    )

                    # Create truncated chunk
                    truncated_chunk = chunk.copy()
                    truncated_chunk["content"] = truncated_content
                    truncated_chunk["truncated"] = True

                    selected_chunks.append(truncated_chunk)
                    truncated_count += 1

                    # Count actual tokens in truncated formatted chunk
                    truncated_formatted = format_chunk_for_context(truncated_chunk)
                    actual_tokens = TokenService.count_tokens(truncated_formatted, model)
                    tokens_used += actual_tokens

                    logger.info(
                        f"Truncated chunk {i+1}/{len(chunks)} "
                        f"(page {page_number}, {chunk_tokens} -> {actual_tokens} tokens, "
                        f"total: {tokens_used}/{available_tokens})"
                    )
                else:
                    # Not enough room even for truncation
                    skipped_count += 1
                    logger.debug(
                        f"Skipped chunk {i+1}/{len(chunks)} - insufficient remaining tokens "
                        f"({remaining_tokens} < {min_useful_tokens})"
                    )
                break  # No more room for additional chunks
            else:
                # Chunk doesn't fit and truncation is disabled
                skipped_count += 1
                logger.debug(
                    f"Skipped chunk {i+1}/{len(chunks)} "
                    f"(page {page_number}, {chunk_tokens} tokens) - would exceed limit"
                )

        # Token usage statistics
        stats = {
            "selected_chunks": len(selected_chunks),
            "total_chunk_tokens": tokens_used,
            "available_tokens": available_tokens,
            "truncated_chunks": truncated_count,
            "skipped_chunks": skipped_count
        }

        logger.info(
            f"Chunk selection complete: {len(selected_chunks)}/{len(chunks)} chunks selected, "
            f"{tokens_used}/{available_tokens} tokens used, "
            f"{truncated_count} truncated, {skipped_count} skipped"
        )

        return selected_chunks, stats


# Process-wide singleton. ChunkSelector holds no instance state.
_selector: "ChunkSelector | None" = None


def get_chunk_selector() -> ChunkSelector:
    """Return the process-wide ChunkSelector singleton."""
    global _selector
    if _selector is None:
        _selector = ChunkSelector()
    return _selector
