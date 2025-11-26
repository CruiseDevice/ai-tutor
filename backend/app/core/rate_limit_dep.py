"""
Rate limiting dependency for FastAPI endpoints.
Uses simple in-memory storage (can be upgraded to Redis for production).
"""
from fastapi import Request, HTTPException, status
from collections import defaultdict, deque
from time import time
from ..config import settings
from .rate_limiting import get_client_ip

# In-memory storage for rate limits: {ip: deque of timestamps}
_rate_limit_store: dict[str, deque] = defaultdict(lambda: deque())


async def check_rate_limit(request: Request, limit_per_minute: int = None):
    """
    Check rate limits for an endpoint.

    Args:
        request: FastAPI request object
        limit_per_minute: Custom rate limit per minute (uses default from settings if None)

    Raises:
        HTTPException: If rate limit is exceeded
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    # Use custom limit or default from settings
    rate_limit = limit_per_minute or settings.RATE_LIMIT_PER_MINUTE
    client_ip = get_client_ip(request)

    # Clean old entries (older than 1 minute)
    now = time()
    timestamps = _rate_limit_store[client_ip]

    # Remove timestamps older than 1 minute
    while timestamps and timestamps[0] < now - 60:
        timestamps.popleft()

    # Check if limit exceeded
    if len(timestamps) >= rate_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {rate_limit} requests per minute. Please try again later."
        )

    # Add current timestamp
    timestamps.append(now)

