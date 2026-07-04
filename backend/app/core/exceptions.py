"""Domain exceptions used by services and translated to HTTP responses by API handlers.

The goal is to stop leaking internal error text (SQL errors, SDK exceptions,
S3 tracebacks, etc.) to API clients. Services raise these typed exceptions;
the FastAPI exception handlers in main.py turn them into safe, generic
responses with a request_id for server-side correlation.
"""
from typing import Optional


class StudyFetchError(Exception):
    """Base class for all application-specific errors."""

    status_code: int = 500
    default_detail: str = "Internal server error"

    def __init__(self, detail: Optional[str] = None):
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


class NotFoundError(StudyFetchError):
    """Resource not found."""

    status_code = 404
    default_detail = "Resource not found"


class ValidationError(StudyFetchError):
    """Input validation or business-rule violation."""

    status_code = 400
    default_detail = "Invalid request"


class AccessDeniedError(StudyFetchError):
    """User does not have permission to access the resource."""

    status_code = 403
    default_detail = "Access denied"


class ExternalServiceError(StudyFetchError):
    """A downstream service (S3, LLM, embedding, etc.) failed."""

    status_code = 502
    default_detail = "External service error"


class LLMProviderError(ExternalServiceError):
    """LLM API call failed after retries."""

    status_code = 502
    default_detail = "AI provider error"


class RateLimitError(StudyFetchError):
    """Too many requests."""

    status_code = 429
    default_detail = "Rate limit exceeded. Please try again later."


class ConflictError(StudyFetchError):
    """Resource conflict (e.g. duplicate conversation)."""

    status_code = 409
    default_detail = "Conflict"
