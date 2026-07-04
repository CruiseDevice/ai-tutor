"""Tests for CacheService namespacing and safe clear behavior."""
import asyncio
import pytest

from app.services.cache_service import CacheService


class FakeAsyncRedis:
    """Minimal async Redis stub that records operations and supports SCAN/DELETE."""

    def __init__(self, initial_keys=None):
        self.store = set(initial_keys or [])
        self.deleted = []
        self.scan_calls = []

    async def setex(self, key, seconds, value):
        self.store.add(key)
        return True

    async def get(self, key):
        if key in self.store:
            return '{"cached": true}'
        return None

    async def scan_iter(self, match=None):
        self.scan_calls.append(match)
        for key in sorted(self.store):
            if match is None or self._match(key, match):
                yield key

    async def delete(self, *keys):
        if not keys:
            return 0
        count = 0
        for key in keys:
            if key in self.store:
                self.store.discard(key)
                self.deleted.append(key)
                count += 1
        return count

    @staticmethod
    def _match(key, pattern):
        # Very small wildcard matcher sufficient for these tests.
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return key == pattern


@pytest.fixture
def service(monkeypatch):
    svc = CacheService()
    svc.enabled = True
    svc.key_prefix = "sft:cache:"
    return svc


@pytest.mark.asyncio
async def test_clear_all_deletes_only_prefixed_keys(service):
    """Phase 0.4: clear_all must leave arq:* keys untouched."""
    redis = FakeAsyncRedis(initial_keys={
        "sft:cache:embedding:abc123",
        "sft:cache:chunks:doc-1:def456:no-rerank",
        "sft:cache:response:doc-1:xyz789",
        "arq:job:some-job-id",
        "arq:result:some-job-id",
        "arq:queue:default",
    })
    service.redis_client = redis

    await service.clear_all()

    remaining = set(redis.store)
    assert remaining == {
        "arq:job:some-job-id",
        "arq:result:some-job-id",
        "arq:queue:default",
    }
    assert set(redis.deleted) == {
        "sft:cache:embedding:abc123",
        "sft:cache:chunks:doc-1:def456:no-rerank",
        "sft:cache:response:doc-1:xyz789",
    }
    assert redis.scan_calls == ["sft:cache:*"]


@pytest.mark.asyncio
async def test_clear_all_is_no_op_when_no_cache_keys(service):
    """clear_all must not error or delete non-cache keys when the cache is empty."""
    redis = FakeAsyncRedis(initial_keys={
        "arq:job:some-job-id",
    })
    service.redis_client = redis

    await service.clear_all()

    assert redis.store == {"arq:job:some-job-id"}
    assert redis.deleted == []


@pytest.mark.asyncio
async def test_delete_keys_batches_large_key_sets(service):
    """_delete_keys should split huge key lists into Redis-safe batches."""
    keys = [f"sft:cache:key:{i}" for i in range(1200)]
    redis = FakeAsyncRedis(initial_keys=set(keys))
    service.redis_client = redis

    deleted = await service._delete_keys(keys)

    assert deleted == 1200
    assert len(redis.deleted) == 1200


@pytest.mark.asyncio
async def test_cache_keys_are_prefixed(service):
    """Individual cache helpers must emit namespaced keys."""
    service.redis_client = FakeAsyncRedis()

    assert service._prefix_key("embedding:abc") == "sft:cache:embedding:abc"
    assert service._scan_pattern("chunks:doc-1:*") == "sft:cache:chunks:doc-1:*"

    # set_embedding stores a namespaced key.
    await service.set_embedding("hello", [0.1, 0.2, 0.3])
    assert any(k.startswith("sft:cache:embedding:") for k in service.redis_client.store)

    # set_response stores a namespaced key; find_similar_response scans the response namespace.
    await service.set_response("doc-1", [0.1, 0.2], "answer", [], [])
    assert any(k.startswith("sft:cache:response:doc-1:") for k in service.redis_client.store)
    await service.find_similar_response([0.1, 0.2], "doc-1")
    assert "sft:cache:response:doc-1:*" in service.redis_client.scan_calls


@pytest.mark.asyncio
async def test_invalidate_document_chunks_uses_prefix(service):
    """Document chunk invalidation must scan only the cache namespace."""
    redis = FakeAsyncRedis(initial_keys={
        "sft:cache:chunks:doc-1:abc:no-rerank",
        "sft:cache:chunks:doc-2:def:rerank",
        "arq:job:other",
    })
    service.redis_client = redis

    await service.invalidate_document_chunks("doc-1")

    assert redis.store == {
        "sft:cache:chunks:doc-2:def:rerank",
        "arq:job:other",
    }
    assert redis.scan_calls == ["sft:cache:chunks:doc-1:*"]
