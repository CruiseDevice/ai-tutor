"""Unit tests for ConversationTitleService.

The LLM-backed generate() happy path needs a live model; here we cover the
deterministic fallback (LLM unavailable / erroring) and the basic shape of
a successful generation via a monkeypatched client.
"""
import pytest

from app.services.conversation_titles import (
    ConversationTitleService,
    get_conversation_title_service,
)


@pytest.fixture
def service() -> ConversationTitleService:
    return ConversationTitleService()


def test_factory_returns_singleton():
    a = get_conversation_title_service()
    b = get_conversation_title_service()
    assert a is b


def test_fallback_title_truncates_to_six_words(service):
    """The fallback derives a title from the first 6 words + ellipsis."""
    msg = "How does the process of photosynthesis convert sunlight into chemical energy?"
    title = service._fallback_title(msg)
    assert title.count(" ") <= 5  # at most 6 words
    assert title.endswith("...")
    assert len(title) <= 53  # 50-char cap + "..." (cap applies after ellipsis)


def test_fallback_title_short_message_no_ellipsis(service):
    """A message shorter than 6 words is returned whole without ellipsis."""
    msg = "What is a virus?"
    title = service._fallback_title(msg)
    assert title == "What is a virus?"
    assert "..." not in title


def test_fallback_title_respects_50_char_cap(service):
    """Long first words get hard-truncated at 50 chars."""
    msg = "antidisestablishmentarianism " * 3  # very long single tokens
    title = service._fallback_title(msg)
    assert len(title) <= 53  # 50 + "..."


@pytest.mark.asyncio
async def test_generate_uses_llm_when_available(service, monkeypatch):
    """A working LLM client returns its (cleaned) title output."""
    captured = {}

    class FakeClient:
        async def complete(self, *, system_prompt, messages, model, temperature, max_tokens):
            captured["model"] = model
            return '  "Photosynthesis Explained"  '

    monkeypatch.setattr(
        "app.services.conversation_titles.get_llm_client",
        lambda prov, key: FakeClient(),
    )
    monkeypatch.setattr(
        "app.services.conversation_titles.pick_helper_model",
        lambda prov: "helper-model",
    )

    title = await service.generate("Explain photosynthesis", user_api_key="k")
    assert title == "Photosynthesis Explained"  # whitespace + surrounding quotes stripped
    assert captured["model"] == "helper-model"


@pytest.mark.asyncio
async def test_generate_falls_back_when_llm_errors(service, monkeypatch):
    """If the LLM call raises, generate() returns the word-truncation fallback."""
    def boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(
        "app.services.conversation_titles.async_retry_openai_call",
        boom,
    )
    monkeypatch.setattr(
        "app.services.conversation_titles.get_llm_client",
        lambda prov, key: type("C", (), {"complete": boom})(),
    )
    monkeypatch.setattr(
        "app.services.conversation_titles.pick_helper_model",
        lambda prov: "m",
    )

    title = await service.generate("What is mitosis and how does it work?", user_api_key="k")
    # Fallback: first 6 words, ellipsis if truncated
    assert title.startswith("What is mitosis and how does")
    assert len(title) <= 53
