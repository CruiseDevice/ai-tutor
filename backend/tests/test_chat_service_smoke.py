"""Wiring smoke tests for ChatService and its extracted collaborators.

These tests do NOT exercise the LLM or database. They assert that:
- `ChatService` imports and instantiates without external dependencies,
- its public entry points exist with the expected signatures, and
- each extracted collaborator module (`annotation_service`, `prompt_builder`,
  `quality_service`) is importable and exposes its factory.

They exist as a regression net for the chat_service decomposition: if a later
extraction breaks import wiring or drops a delegating method, this file fails
fast.
"""
import inspect
from typing import AsyncGenerator

import pytest

from app.services.chat_service import ChatService


def _coro(fn) -> bool:
    """True if `fn` is an async function (coroutine or async generator)."""
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def test_chat_service_instantiates_without_external_deps():
    """ChatService should construct with no DB session or API keys."""
    svc = ChatService()
    assert svc is not None
    # Embedding service is pulled in at construction; verify it's set.
    assert svc.embedding_service is not None


def test_chat_service_exposes_public_entry_points():
    svc = ChatService()
    # Retrieval
    assert _coro(svc.find_similar_chunks)
    # The four orchestration entry points
    assert _coro(svc.generate_chat_response)
    assert _coro(svc.generate_chat_response_with_agent)
    assert _coro(svc.generate_chat_response_stream)
    assert _coro(svc.generate_chat_response_stream_with_agent)


def test_chat_response_streams_are_async_generators():
    """Streaming entry points must be async generators (return AsyncGenerator)."""
    svc = ChatService()
    assert inspect.isasyncgenfunction(svc.generate_chat_response_stream)
    assert inspect.isasyncgenfunction(svc.generate_chat_response_stream_with_agent)


def test_chat_service_still_delegates_extracted_methods():
    """ChatService must keep thin delegators for backward compatibility.

    agent_service.py (and any other consumer) may still call these; the
    extracted modules do the real work.
    """
    svc = ChatService()
    # Annotation parsing
    assert callable(getattr(svc, "_parse_annotations", None))
    # Prompt building / classification
    assert _coro(getattr(svc, "_classify_query_type", None))
    assert callable(getattr(svc, "_build_system_prompt", None))
    # Quality / citations
    assert callable(getattr(svc, "_verify_citations", None))
    assert _coro(getattr(svc, "_score_answer_quality", None))
    # Phase 2 pipeline helpers
    assert callable(getattr(svc, "_llm_error_message", None))
    assert _coro(getattr(svc, "_set_title_if_first_message", None))
    assert _coro(getattr(svc, "_prepare_generation_context", None))
    # Phase 3 unified pipeline
    assert callable(getattr(svc, "_estimate_token_usage", None))
    assert _coro(getattr(svc, "_complete_then_yield", None))
    assert _coro(getattr(svc, "_stream_passthrough", None))
    assert inspect.isasyncgenfunction(svc._run_linear_pipeline)


def test_llm_error_message_maps_status_codes():
    """The shared status-code map covers 429/401/403/5xx + a generic fallback.

    Both linear entry points previously kept their own copy of this map;
    it's now centralized in _llm_error_message (regression net).
    """
    def err(code, message="boom"):
        # _llm_error_message reads status_code via getattr, so a lightweight
        # stand-in is sufficient (the helper never imports the real APIError).
        e = Exception(message)
        e.status_code = code
        return e

    svc = ChatService()
    assert "Rate limit" in svc._llm_error_message(err(429))
    assert "API key" in svc._llm_error_message(err(401))
    assert "forbidden" in svc._llm_error_message(err(403))
    assert "temporarily unavailable" in svc._llm_error_message(err(503))
    # Missing/unknown status code -> generic message
    assert "LLM API error" in svc._llm_error_message(err(None))


_PHASE1_MODULES = [
    ("app.services.annotation_service", "get_annotation_service"),
    ("app.services.prompt_builder", "get_prompt_builder"),
    ("app.services.quality_service", "get_quality_service"),
]


@pytest.mark.parametrize("module_name, factory_name", _PHASE1_MODULES)
def test_extracted_collaborator_modules_importable(module_name, factory_name):
    """Each Phase-1 collaborator module must import and expose its factory.

    Skipped until the module lands — once extracted, this becomes a real
    assertion so we catch import/factory regressions in later steps.
    """
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.skip(f"{module_name} not extracted yet (Phase 1 step pending)")
    assert callable(getattr(mod, factory_name, None)), (
        f"{module_name} is missing {factory_name}()"
    )
