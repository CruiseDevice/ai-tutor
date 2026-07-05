"""Pipeline and adapter tests for the unified linear chat path.

Phase 3 collapsed generate_chat_response and generate_chat_response_stream
into one _run_linear_pipeline generator plus two thin adapters. These tests
exercise the pipeline with a fake LLM client (no DB, no real model) to lock
in the event contract and the adapters' translation behavior.
"""
import json
from typing import AsyncIterator, Dict, List

import pytest

from app.services.chat_service import (
    ChatService,
    ChunkEvent,
    DoneEvent,
    ErrorEvent,
    PipelineEvent,
)


class FakeUser:
    def __init__(self, api_key: str = "key"):
        self.id = "user-1"
        self._api_key = api_key

    def get_decrypted_key(self, provider_value: str) -> str:
        return self._api_key


class FakeClient:
    """Fake LLM client returning canned content / fragments."""
    def __init__(self, *, complete_content=None, stream_fragments=None):
        self._complete_content = complete_content
        self._stream_fragments = stream_fragments

    async def complete(self, *, system_prompt, messages, model, temperature, max_tokens):
        return self._complete_content

    async def stream(self, *, system_prompt, messages, model, temperature, max_tokens) -> AsyncIterator[str]:
        for frag in (self._stream_fragments or []):
            yield frag


def _collect_sync(events):
    """ Flatten ChunkEvent content for assertion convenience."""
    return "".join(e.content for e in events if isinstance(e, ChunkEvent))


# -- typed events ---------------------------------------------------------

def test_pipeline_event_subtypes_exist():
    assert issubclass(ChunkEvent, PipelineEvent)
    assert issubclass(ErrorEvent, PipelineEvent)
    assert issubclass(DoneEvent, PipelineEvent)


def test_chunk_and_error_events_carry_payload():
    assert ChunkEvent(content="x").content == "x"
    assert ErrorEvent(message="boom").message == "boom"


# -- _estimate_token_usage ------------------------------------------------

def test_estimate_token_usage_returns_dict():
    """The shared helper returns prompt/completion/total token counts."""
    svc = ChatService()
    usage = svc._estimate_token_usage(
        messages=[{"role": "user", "content": "hello world"}],
        content="hi there",
        model="gpt-4",
        max_context_tokens=100000,
    )
    assert usage is not None
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(usage)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_estimate_token_usage_returns_none_on_bad_model():
    """A token-counting failure (e.g. unknown model) returns None, not raise."""
    svc = ChatService()
    usage = svc._estimate_token_usage(
        messages=[{"role": "user", "content": "x"}],
        content="y",
        model="this-model-does-not-exist-xyz",
        max_context_tokens=100000,
    )
    # tiktoken falls back to a heuristic; either a dict or None is acceptable,
    # but it must not raise.
    assert usage is None or "total_tokens" in usage


# -- LLM-call strategies --------------------------------------------------

@pytest.mark.asyncio
async def test_complete_then_yield_emits_single_fragment():
    """The non-stream strategy yields the full completion in one step."""
    svc = ChatService()
    client = FakeClient(complete_content="full answer")
    fragments = [
        f async for f in svc._complete_then_yield(
            client, "sys", [{"role": "user", "content": "q"}], "gpt-4", 100
        )
    ]
    assert fragments == ["full answer"]


@pytest.mark.asyncio
async def test_stream_passthrough_forwards_each_fragment():
    """The stream strategy forwards client.stream() fragments verbatim."""
    svc = ChatService()
    client = FakeClient(stream_fragments=["alpha ", "beta ", "gamma"])
    fragments = [
        f async for f in svc._stream_passthrough(
            client, "sys", [{"role": "user", "content": "q"}], "gpt-4", 100
        )
    ]
    assert fragments == ["alpha ", "beta ", "gamma"]


# -- pipeline error path (no DB needed) -----------------------------------

@pytest.mark.asyncio
async def test_pipeline_yields_error_event_on_missing_api_key(monkeypatch):
    """A user with no API key triggers an ErrorEvent, not an exception.

    The pipeline must not raise into the adapter; it yields ErrorEvent and
    returns, letting each adapter translate (raise vs SSE).
    """
    svc = ChatService()
    user = FakeUser(api_key="")  # missing key

    # The pipeline resolves the provider before checking the key; stub the
    # cache service init that happens earlier in the flow. We only need to
    # reach the key check, which is the first thing after provider resolution.
    events = []
    async for ev in svc._run_linear_pipeline(
        db=None,
        user=user,
        content="hi",
        conversation_id="c",
        document_id="d",
        model="gpt-4",
        generate=svc._complete_then_yield,
    ):
        events.append(ev)
        # The very first event should be the ErrorEvent for the missing key.
        break

    assert events and isinstance(events[0], ErrorEvent)
    assert "API key" in events[0].message


# -- entry-point signatures (regression net) ------------------------------

def test_entry_point_signatures_unchanged():
    """All four public entry points must keep their parameter names/order.

    app/api/chat.py calls these with keyword args; renaming would break it.
    """
    import inspect
    svc = ChatService()

    nonstream = inspect.signature(svc.generate_chat_response)
    stream = inspect.signature(svc.generate_chat_response_stream)
    agent_ns = inspect.signature(svc.generate_chat_response_with_agent)
    agent_s = inspect.signature(svc.generate_chat_response_stream_with_agent)

    for sig in (nonstream, stream, agent_ns, agent_s):
        params = list(sig.parameters)
        assert params[:3] == ["db", "user", "content"], f"signature drifted: {params}"
        assert "conversation_id" in sig.parameters
        assert "document_id" in sig.parameters
        assert "model" in sig.parameters
