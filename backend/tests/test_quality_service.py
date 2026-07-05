"""Unit tests for QualityService.

Covers the pure citation-verification logic extracted from ChatService.
The LLM-backed score_answer_quality() is exercised only via its disabled
short-circuit; full LLM scoring belongs in integration tests.
"""
import pytest

from app.services.quality_service import QualityService, get_quality_service


@pytest.fixture
def service() -> QualityService:
    return QualityService()


def test_factory_returns_singleton():
    a = get_quality_service()
    b = get_quality_service()
    assert a is b


def _ann(page_number):
    return {"pageNumber": page_number, "annotations": []}


def test_verify_citations_passes_when_pages_match(service):
    """All cited and annotated pages exist in the chunks -> no warnings."""
    response = "Mitosis divides once [Page 12]. Meiosis divides twice [Page 79]."
    chunks = [{"pageNumber": 12}, {"pageNumber": 79}]
    annotations = [_ann(12), _ann(79)]

    warnings = service.verify_citations(response, annotations, chunks)
    assert warnings == []


def test_verify_citations_flags_page_not_in_chunks(service):
    """A [Page X] citation with no matching chunk is flagged."""
    response = "Something on [Page 99]."
    chunks = [{"pageNumber": 12}]
    warnings = service.verify_citations(response, [], chunks)
    assert any("Page 99" in w for w in warnings)


def test_verify_citations_flags_annotation_page_not_in_chunks(service):
    """An annotation whose page isn't in the chunks is flagged by index."""
    chunks = [{"pageNumber": 12}]
    annotations = [_ann(12), _ann(200)]
    warnings = service.verify_citations("", annotations, chunks)
    assert any("Annotation #2" in w and "200" in w for w in warnings)


def test_verify_citations_handles_invalid_annotation_page(service):
    """Non-integer annotation page numbers produce a warning, not a crash."""
    chunks = []
    annotations = [{"pageNumber": "not-a-number", "annotations": []}]
    warnings = service.verify_citations("", annotations, chunks)
    assert any("invalid page number" in w for w in warnings)


def test_verify_citations_flags_cited_but_unannotated_pages(service):
    """Pages cited in text but missing from annotations are reported."""
    response = "See [Page 12] and [Page 13]."
    chunks = [{"pageNumber": 12}, {"pageNumber": 13}]
    annotations = [_ann(12)]  # Page 13 cited but not annotated
    warnings = service.verify_citations(response, annotations, chunks)
    assert any("cited but not annotated" in w and "13" in w for w in warnings)


def test_verify_citations_disabled_returns_empty(service, monkeypatch):
    """When ENABLE_CITATION_VERIFICATION is False, returns [] with no work."""
    from app.services import quality_service as qs_mod
    monkeypatch.setattr(qs_mod.settings, "ENABLE_CITATION_VERIFICATION", False)
    # Grossly inconsistent inputs that would otherwise produce many warnings
    warnings = service.verify_citations(
        "[Page 999] body", [_ann(999)], []
    )
    assert warnings == []


@pytest.mark.asyncio
async def test_score_answer_quality_disabled_returns_nulls(service, monkeypatch):
    """When ENABLE_ANSWER_QUALITY_SCORING is False, returns null scores, no LLM call."""
    from app.services import quality_service as qs_mod
    monkeypatch.setattr(qs_mod.settings, "ENABLE_ANSWER_QUALITY_SCORING", False)

    scores = await service.score_answer_quality(
        query="q", answer="a", context_chunks=[], user_api_key="k"
    )
    assert scores["accuracy"] is None
    assert scores["overall"] is None
    assert scores["feedback"] == "Quality scoring disabled"
