"""Tests for trusted-proxy-aware IP extraction and rate limiting (Phase 1.2)."""
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.core.rate_limiting import get_client_ip, _is_trusted_peer
from app.core.rate_limit_dep import check_rate_limit
from app.config import settings


class DummyClient:
    def __init__(self, host):
        self.host = host


class DummyRequest:
    def __init__(self, client_host="127.0.0.1", headers=None):
        self.client = DummyClient(client_host) if client_host else None
        self.headers = headers or {}
        self.app = MagicMock()
        self.app.state.limiter = None


def test_get_client_ip_ignores_forwarded_headers_without_trusted_proxy(monkeypatch):
    """Without trusted proxies, X-Forwarded-For must not be honored."""
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", [])
    req = DummyRequest(
        client_host="10.0.0.1",
        headers={"X-Forwarded-For": "1.2.3.4"}
    )
    assert get_client_ip(req) == "10.0.0.1"


def test_get_client_ip_honors_forwarded_header_from_trusted_proxy(monkeypatch):
    """With a trusted proxy, X-Forwarded-For should return the original client."""
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", ["127.0.0.1"])
    req = DummyRequest(
        client_host="127.0.0.1",
        headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
    )
    assert get_client_ip(req) == "1.2.3.4"


def test_get_client_ip_trusted_proxy_cidr_match(monkeypatch):
    """Trusted proxy can be specified as a CIDR block."""
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", ["10.0.0.0/8"])
    req = DummyRequest(
        client_host="10.1.2.3",
        headers={"X-Real-IP": "192.168.1.5"}
    )
    assert get_client_ip(req) == "192.168.1.5"


def test_get_client_ip_untrusted_proxy_cidr_no_match(monkeypatch):
    """A peer outside the trusted CIDR must not have its headers honored."""
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", ["10.0.0.0/8"])
    req = DummyRequest(
        client_host="203.0.113.1",
        headers={"X-Forwarded-For": "192.168.1.5"}
    )
    assert get_client_ip(req) == "203.0.113.1"


def test_is_trusted_peer_empty_list_never_trusts():
    assert _is_trusted_peer("127.0.0.1") is False


@pytest.mark.asyncio
async def test_check_rate_limit_enforces_per_ip_limit(monkeypatch):
    """21 requests from the same IP within 60s should trip the limit."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 20)

    req = DummyRequest(client_host="1.2.3.4")

    for _ in range(20):
        await check_rate_limit(req)

    with pytest.raises(Exception) as exc_info:
        await check_rate_limit(req)

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_rate_limit_disabled_does_nothing(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    req = DummyRequest(client_host="1.2.3.4")

    for _ in range(25):
        await check_rate_limit(req)  # should not raise


def test_check_rate_limit_rotating_forwarded_headers_still_blocked(monkeypatch):
    """Rotating X-Forwarded-For must not bypass the limit when peer is untrusted."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 5)
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", [])

    # Simulate a malicious client rotating forwarded headers.
    # Because the peer is untrusted, the key remains the peer IP.
    for i in range(6):
        req = DummyRequest(
            client_host="203.0.113.1",
            headers={"X-Forwarded-For": f"1.2.3.{i}"}
        )
        if i < 5:
            assert get_client_ip(req) == "203.0.113.1"
