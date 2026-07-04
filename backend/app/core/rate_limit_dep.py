"""
Rate limiting dependency for FastAPI endpoints.

Uses the shared slowapi Limiter attached to the app state. Endpoints should
prefer the @limiter.limit() decorator; this dependency remains for endpoints
that need a dynamic or programmatic check.
"""
from fastapi import Request, HTTPException, status
from .rate_limiting import get_client_ip
from ..config import settings


async def check_rate_limit(request: Request, limit_per_minute: int = None):
    """
    Check rate limits for an endpoint using the shared app limiter.

    Falls back to a local in-memory sliding window if the app-level limiter is
    not available. This path exists for tests and edge cases; production code
    should use the @limiter.limit() decorator.

    Raises:
        HTTPException: If rate limit is exceeded
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    limiter = getattr(request.app.state, "limiter", None)
    if limiter is not None:
        # Delegate to slowapi. It will raise RateLimitExceeded on violation,
        # which is mapped to a 429 by the global exception handler.
        await limiter._check_limits(request)
        return

    # Fallback: simple in-memory sliding window (per-process only).
    from collections import defaultdict, deque
    from time import time

    rate_limit = limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
    client_ip = get_client_ip(request)

    if not hasattr(check_rate_limit, "_store"):
        check_rate_limit._store = defaultdict(lambda: deque())

    now = time()
    timestamps = check_rate_limit._store[client_ip]
    while timestamps and timestamps[0] < now - 60:
        timestamps.popleft()

    if len(timestamps) >= rate_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )

    timestamps.append(now)
