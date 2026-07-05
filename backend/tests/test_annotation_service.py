"""Unit tests for AnnotationService.

These exercise the pure parsing logic that was previously buried as private
methods on ChatService. No DB or LLM is required.
"""
import pytest

from app.services.annotation_service import AnnotationService, get_annotation_service


@pytest.fixture
def service() -> AnnotationService:
    return AnnotationService()


def test_factory_returns_singleton():
    a = get_annotation_service()
    b = get_annotation_service()
    assert a is b


def test_parse_returns_original_when_no_annotations_block(service):
    """A response with no fenced annotation block passes through untouched."""
    response = "Photosynthesis makes oxygen [Page 12]."
    cleaned, annotations = service.parse(response, relevant_chunks=[])
    assert cleaned == response
    assert annotations == []


def test_parse_extracts_text_annotation_and_strips_block(service):
    """A valid ```annotations block is parsed and removed from the response."""
    response = (
        'Cellular respiration produces ATP [Page 45].\n'
        '```annotations\n'
        '[\n'
        '  {\n'
        '    "type": "highlight",\n'
        '    "textToHighlight": "produces ATP",\n'
        '    "pageNumber": 45,\n'
        '    "explanation": "Core mechanism"\n'
        '  }\n'
        ']\n'
        '```\n'
    )
    chunks = [{"pageNumber": 45, "content": "Respiration produces ATP in the mitochondria."}]

    cleaned, annotations = service.parse(response, relevant_chunks=chunks)

    assert "```annotations" not in cleaned
    assert "produces ATP" in cleaned  # body text retained
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["pageNumber"] == 45
    assert ann["annotations"][0]["type"] == "highlight"
    assert ann["annotations"][0]["textContent"] == "produces ATP"
    assert ann["explanation"] == "Core mechanism"
    assert ann["annotations"][0]["label"] == "Core mechanism"  # backfilled from explanation


def test_parse_accepts_json_fenced_block_too(service):
    """Models sometimes emit ```json instead of ```annotations — both must work."""
    response = (
        '```json\n'
        '{"annotations": [{"type": "highlight", "textToHighlight": "x", "pageNumber": 1, "explanation": "e"}]}\n'
        '```'
    )
    chunks = [{"pageNumber": 1, "content": "x marks the spot"}]
    cleaned, annotations = service.parse(response, relevant_chunks=chunks)
    assert len(annotations) == 1


def test_parse_skips_text_annotation_when_chunk_missing(service):
    """If no source chunk contains the text, the annotation is dropped silently."""
    response = (
        '```annotations\n'
        '[{"type": "highlight", "textToHighlight": "missing phrase", "pageNumber": 99}]'
        '\n```'
    )
    cleaned, annotations = service.parse(response, relevant_chunks=[])
    assert annotations == []
    # Block is still stripped even when parsing yields nothing usable
    assert "```annotations" not in cleaned


def test_parse_handles_malformed_json(service):
    """Malformed JSON in the block logs and returns no annotations, no crash."""
    response = 'Answer [Page 1].\n```annotations\n{not valid json}\n```'
    cleaned, annotations = service.parse(response, relevant_chunks=[])
    assert annotations == []
    assert "```annotations" not in cleaned


def test_pdf_coords_to_percentage_bounds_converts(service):
    """PDF point coords are converted to percentage-of-page bounds."""
    # 612 x 792 is the assumed letter-size page.
    bounds = service.pdf_coords_to_percentage_bounds(
        bbox=[306.0, 396.0, 612.0, 792.0],  # bottom-right quadrant
        page_number=1,
        document_id="doc",
    )
    assert bounds["x"] == 50.0
    assert bounds["y"] == 50.0
    assert bounds["width"] == 50.0
    assert bounds["height"] == 50.0


def test_pdf_coords_falls_back_on_bad_bbox(service):
    """Non-numeric bbox returns a safe default bound instead of raising."""
    bounds = service.pdf_coords_to_percentage_bounds(
        bbox=["a", "b", "c"], page_number=1, document_id="doc"
    )
    assert bounds == {"x": 10, "y": 10, "width": 30, "height": 30}


def test_get_annotation_color_palette(service):
    assert service.get_annotation_color("highlight").startswith("rgba(255, 235, 59")
    assert service.get_annotation_color("circle").startswith("rgba(33, 150, 243")
    assert service.get_annotation_color("unknown") == service.get_annotation_color("highlight")
