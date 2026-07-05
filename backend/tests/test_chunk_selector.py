"""Unit tests for the chunk selector and formatter.

These cover the pure CPU logic extracted from ChatService: the chunk
formatter (text vs image) and the token-budget selection (greedy packing
with truncation). No DB or LLM is required.
"""
import pytest

from app.services.chunk_selector import (
    ChunkSelector,
    format_chunk_for_context,
    get_chunk_selector,
)


@pytest.fixture
def selector() -> ChunkSelector:
    return ChunkSelector()


def test_factory_returns_singleton():
    a = get_chunk_selector()
    b = get_chunk_selector()
    assert a is b


def test_format_text_chunk_includes_page_header():
    out = format_chunk_for_context({"pageNumber": 7, "content": "hello world", "chunk_type": "text"})
    assert out == "[Page 7]: hello world"


def test_format_text_chunk_falls_back_to_page_number_key():
    """Chunks using the snake_case `page_number` key are still formatted."""
    out = format_chunk_for_context({"page_number": 3, "content": "x", "chunk_type": "text"})
    assert out.startswith("[Page 3]:")


def test_format_image_chunk_uses_bbox_header():
    out = format_chunk_for_context({
        "pageNumber": 5,
        "content": "diagram caption",
        "chunk_type": "image",
        "id": "img-1",
        "positionData": {"bbox": [10.0, 20.0, 30.0, 40.0]},
    })
    assert "Image at bbox (10.0, 20.0, 30.0, 40.0)" in out
    assert "Image ID: img-1" in out
    assert "diagram caption" in out


def test_format_image_chunk_handles_bad_bbox():
    """Non-numeric bbox coordinates fall back to 0.0, not raise."""
    out = format_chunk_for_context({
        "pageNumber": 1,
        "content": "c",
        "chunk_type": "image",
        "id": "i",
        "positionData": {"bbox": ["a", "b", "c"]},
    })
    assert "Image at bbox (0.0, 0.0, 0.0, 0.0)" in out


def test_select_by_token_limit_packs_chunks_in_order(selector):
    """Chunks are added in given order until the budget is exhausted."""
    chunks = [
        {"pageNumber": 1, "content": "alpha beta gamma delta", "chunk_type": "text"},
        {"pageNumber": 2, "content": "epsilon zeta eta theta", "chunk_type": "text"},
        {"pageNumber": 3, "content": "iota kappa lambda mu", "chunk_type": "text"},
    ]
    selected, stats = selector.select_by_token_limit(
        chunks=chunks,
        max_tokens=100000,
        model="gpt-4",
        system_prompt_tokens=10,
        user_message_tokens=10,
        history_tokens=0,
        response_reserve_tokens=100,
    )
    # With a generous budget, all chunks fit.
    assert len(selected) == 3
    assert stats["selected_chunks"] == 3
    assert stats["skipped_chunks"] == 0


def test_select_by_token_limit_skips_when_budget_exhausted(selector, monkeypatch):
    """When there are zero available tokens, nothing is selected."""
    # Force available_tokens to 0 by reserving the entire budget.
    selected, stats = selector.select_by_token_limit(
        chunks=[{"pageNumber": 1, "content": "x", "chunk_type": "text"}],
        max_tokens=100,
        model="gpt-4",
        system_prompt_tokens=50,
        user_message_tokens=50,
        history_tokens=50,
        response_reserve_tokens=50,
    )
    assert selected == []
    assert stats["selected_chunks"] == 0
    assert stats["skipped_chunks"] == 1
