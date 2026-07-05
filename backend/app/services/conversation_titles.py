"""Conversation title generation.

Extracted from ChatService. Generates a concise (3-6 word) title from the
first user message using a helper-LLM call, with a deterministic word-
truncation fallback when the LLM is unavailable or errors.

Stateless; a process-wide singleton is exposed via get_conversation_title_service(),
mirroring the other extracted services.
"""
import logging

from openai import APIError

from .llm import Provider, get_llm_client, pick_helper_model
from .retry_utils import async_retry_openai_call

logger = logging.getLogger(__name__)


class ConversationTitleService:
    """Generate concise conversation titles from the opening user message."""

    async def generate(
        self,
        user_message: str,
        user_api_key: str,
        provider: Provider | None = None,
    ) -> str:
        """
        Generate a smart, concise title for the conversation based on the first user message.
        Uses LLM to create a title that's 3-6 words.
        """
        try:
            prov = provider or Provider.OPENAI
            client = get_llm_client(prov, user_api_key)
            model = pick_helper_model(prov)

            prompt = f"""Generate a concise, descriptive title for this conversation based on the user's question.
The title should be 3-6 words and capture the main topic or question.

User question: "{user_message}"

Return ONLY the title, nothing else. Make it specific and informative.
Examples:
- "What is a virus?" → "Understanding Viruses"
- "Explain photosynthesis" → "Photosynthesis Explanation"
- "Summarize chapter 3" → "Chapter 3 Summary"
- "How does DNA replication work?" → "DNA Replication Process"

Title:"""

            async def _create_completion():
                return await client.complete(
                    system_prompt="You are a helpful assistant that generates concise conversation titles.",
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.7,
                    max_tokens=20,
                )

            title = await async_retry_openai_call(
                _create_completion,
                max_attempts=3,  # Fewer retries for title generation (non-critical)
                initial_wait=1.0,
                max_wait=30.0
            )

            title = title.strip()
            # Remove quotes if present
            title = title.strip('"\'')
            # Limit to 50 characters
            title = title[:50] if len(title) > 50 else title

            logger.info(f"Generated conversation title: {title}")
            return title

        except APIError as e:
            logger.warning(f"Failed to generate title with LLM after retries: {e}")
            return self._fallback_title(user_message)
        except Exception as e:
            logger.warning(f"Failed to generate title with LLM: {e}")
            return self._fallback_title(user_message)

    @staticmethod
    def _fallback_title(user_message: str) -> str:
        """Derive a title from the first few words when LLM generation fails."""
        words = user_message.split()[:6]
        title = " ".join(words)
        if len(user_message) > len(title):
            title += "..."
        return title[:50]


# Process-wide singleton. The service holds no instance state.
_service: ConversationTitleService | None = None


def get_conversation_title_service() -> ConversationTitleService:
    """Return the process-wide ConversationTitleService singleton."""
    global _service
    if _service is None:
        _service = ConversationTitleService()
    return _service
