"""Tests for Phase 1.4 file-upload hardening."""
import pytest
import os
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

from app.services.document_service import DocumentService, _sanitize_filename
from app.core.exceptions import ValidationError, ExternalServiceError


class FakeUploadFile:
    """Minimal UploadFile stand-in for tests."""

    def __init__(self, filename: str, content: bytes, content_type: str = "application/pdf"):
        self.filename = filename
        self._content = content
        self._position = 0
        self.content_type = content_type
        self.file = BytesIO(content)

    async def read(self, size: int = -1) -> bytes:
        return self.file.read(size)

    async def seek(self, offset: int) -> int:
        return self.file.seek(offset)

    async def close(self):
        self.file.close()


def test_sanitize_filename_strips_control_chars():
    assert _sanitize_filename("my\r\nfile.pdf") == "myfile.pdf"
    assert _sanitize_filename('ev"il\tpdf.pdf') == "evilpdf.pdf"
    assert _sanitize_filename("\x00\x1f\x7fhidden.pdf") == "hidden.pdf"


def test_sanitize_filename_empty_defaults_to_document_pdf():
    assert _sanitize_filename("") == "document.pdf"
    assert _sanitize_filename("  \t\n  ") == "document.pdf"


@pytest.mark.asyncio
async def test_upload_rejects_oversize_file(monkeypatch):
    svc = DocumentService()
    monkeypatch.setattr(svc, "s3_client", MagicMock())

    # 35 MB of valid PDF header bytes
    big_content = b"%PDF-1.4\n" + b"x" * (35 * 1024 * 1024)
    file = FakeUploadFile("big.pdf", big_content)

    from app import config
    monkeypatch.setattr(config.settings, "MAX_FILE_SIZE", 30 * 1024 * 1024)

    with pytest.raises(ValidationError):
        await svc.upload_to_s3(file, "user-1")


@pytest.mark.asyncio
async def test_upload_rejects_wrong_magic_bytes(monkeypatch):
    svc = DocumentService()
    monkeypatch.setattr(svc, "s3_client", MagicMock())

    file = FakeUploadFile("evil.pdf", b"NOT A PDF, just some text\n")

    with pytest.raises(ValidationError) as exc_info:
        await svc.upload_to_s3(file, "user-1")

    assert "valid PDF" in exc_info.value.detail


@pytest.mark.asyncio
async def test_upload_accepts_valid_pdf(monkeypatch):
    svc = DocumentService()
    fake_s3 = MagicMock()
    monkeypatch.setattr(svc, "s3_client", fake_s3)

    file = FakeUploadFile("valid.pdf", b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n")

    url, blob_path = await svc.upload_to_s3(file, "user-1")

    assert url.endswith(blob_path)
    assert blob_path.startswith("user-1/")
    assert blob_path.endswith(".pdf")
    fake_s3.put_object.assert_called_once()


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_extension(monkeypatch):
    svc = DocumentService()
    file = FakeUploadFile("malware.exe", b"%PDF-1.4\n")

    with pytest.raises(ValidationError) as exc_info:
        await svc.upload_to_s3(file, "user-1")

    assert "PDF" in exc_info.value.detail


def test_header_injection_filename_is_sanitized():
    malicious = "report\r\nContent-Type: text/html\r\n\r\n<script>alert(1)</script>.pdf"
    safe = _sanitize_filename(malicious)
    assert "\r" not in safe
    assert "\n" not in safe
    assert "<script>" not in safe
