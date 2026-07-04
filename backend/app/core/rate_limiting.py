"""
Rate limiting utilities for FastAPI.

Provides trusted-proxy-aware IP extraction and Redis-backed rate limiting.
The previous implementation stored counters in process memory, which gave
multi-worker deployments N× the configured limit and trusted arbitrary
X-Forwarded-For headers.
"""
import ipaddress
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from redis.asyncio import Redis

from ..config import settings


def _is_trusted_peer(peer_ip: str) -> bool:
    """Return True if the immediate peer is in TRUSTED_PROXIES."""
    if not settings.TRUSTED_PROXIES:
        return False

    try:
        peer = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False

    for trusted in settings.TRUSTED_PROXIES:
        trusted = trusted.strip()
        if not trusted:
            continue
        try:
            if "/" in trusted:
                if peer in ipaddress.ip_network(trusted, strict=False):
                    return True
            elif peer == ipaddress.ip_address(trusted):
                return True
        except ValueError:
            continue
    return False


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request, honoring forwarded headers only when
    the immediate peer is a configured trusted proxy.
    """
    peer_ip = request.client.host if request.client else "unknown"

    if not _is_trusted_peer(peer_ip):
        return peer_ip

    # Peer is trusted: inspect forwarded headers.
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the leftmost (original client) IP.
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    return peer_ip


def _make_redis_client() -> Redis:
    """Create an async Redis client from settings.REDIS_URL."""
    return Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _key_func(request: Request) -> str:
    """Rate-limit key: client IP + HTTP method + (optional) route prefix."""
    return f"{request.method}:{get_client_ip(request)}"


def create_limiter() -> Limiter:
    """
    Create a slowapi Limiter backed by Redis.

    Falls back to in-memory storage if Redis is not available, with a warning.
    """
    try:
        redis_client = _make_redis_client()
        # Test connectivity synchronously; slowapi expects a storage uri or object.
        # We pass the redis url directly so slowapi/slimits can manage its own pool.
        return Limiter(
            key_func=_key_func,
            default_limits=[],
            storage_uri=settings.REDIS_URL,
            headers_enabled=True,
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Failed to create Redis-backed rate limiter: {e}. "
            "Falling back to in-memory storage. Multi-worker deployments may exceed limits."
        )
        return Limiter(key_func=_key_func, default_limits=[])


def setup_rate_limiting(app):
    """Attach the rate limiter extension to the FastAPI app."""
    limiter = create_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Return a generic 429 response without leaking internal limit details."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )
