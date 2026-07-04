"""Tests that internal error text is no longer leaked to API clients (Phase 1.3)."""
import pytest
from unittest.mock import MagicMock

from app.core.exceptions import (
    StudyFetchError,
    NotFoundError,
    ValidationError,
    ExternalServiceError,
)
from app.main import generic_exception_handler, study_fetch_error_handler


def _mock_request():
    return MagicMock(spec=["headers", "method", "url"])


@pytest.mark.asyncio
async def test_domain_handler_maps_not_found():
    req = _mock_request()
    exc = NotFoundError("sensitive db info")
    response = await study_fetch_error_handler(req, exc)
    assert response.status_code == 404
    body = response.body.decode()
    assert "Resource not found" in body
    assert "request_id" in body
    assert "sensitive db info" not in body


@pytest.mark.asyncio
async def test_domain_handler_maps_validation():
    req = _mock_request()
    exc = ValidationError("sensitive validation info")
    response = await study_fetch_error_handler(req, exc)
    assert response.status_code == 400
    body = response.body.decode()
    assert "Invalid request" in body
    assert "sensitive validation info" not in body


@pytest.mark.asyncio
async def test_domain_handler_maps_external_service():
    req = _mock_request()
    exc = ExternalServiceError("sensitive s3 traceback")
    response = await study_fetch_error_handler(req, exc)
    assert response.status_code == 502
    body = response.body.decode()
    assert "External service error" in body
    assert "sensitive s3 traceback" not in body


@pytest.mark.asyncio
async def test_generic_handler_hides_runtime_details():
    req = _mock_request()
    exc = RuntimeError("sensitive runtime traceback with SELECT password")
    response = await generic_exception_handler(req, exc)
    assert response.status_code == 500
    body = response.body.decode()
    assert "Internal server error" in body
    assert "request_id" in body
    assert "SELECT password" not in body
    assert "sensitive runtime traceback" not in body


def test_document_service_no_longer_imports_httpexception():
    """Verify HTTPException is gone from document_service.py imports."""
    with open("app/services/document_service.py") as f:
        source = f.read()
    assert "HTTPException" not in source
    assert "from ..core.exceptions import" in source
