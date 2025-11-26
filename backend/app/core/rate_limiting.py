"""
Rate limiting utilities for FastAPI.
Provides IP address extraction for rate limiting.
"""
from fastapi import Request


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    Handles proxies and load balancers.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address as string
    """
    # Check for forwarded IP (when behind proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return forwarded.split(",")[0].strip()

    # Check for real IP header (used by some proxies)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback to direct connection
    if request.client:
        return request.client.host

    # Last resort
    return "unknown"


def setup_rate_limiting(app):
    """
    Setup rate limiting for the FastAPI app.
    Currently a no-op as we use manual rate limiting in dependencies.
    Can be extended to use Redis or other storage backends.

    Args:
        app: FastAPI application instance
    """
    # Rate limiting is handled in rate_limit_dep.py
    # This function is kept for future enhancements (e.g., Redis backend)
    pass

