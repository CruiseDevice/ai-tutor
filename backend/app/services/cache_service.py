try:
    import redis.asyncio as aioredis
except ImportError:
    # Fallback for older redis versions or if redis is not installed
    aioredis = None

import json
import hashlib
import logging
import numpy as np
from typing import Optional, List, Dict, Any
from ..config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Service for managing Redis-based caching of embeddings, responses, and chunks."""

    def __init__(self):
        self.redis_client: Optional[Any] = None
        self._connection_pool: Optional[Any] = None
        self.enabled = settings.CACHE_ENABLED and aioredis is not None
        # Cache metrics
        self.stats = {
            'embedding_hits': 0,
            'embedding_misses': 0,
            'chunk_hits': 0,
            'chunk_misses': 0,
            'response_hits': 0,
            'response_misses': 0
        }

    async def connect(self):
        """Initialize Redis connection."""
        if not self.enabled:
            logger.info("Caching is disabled")
            return

        if aioredis is None:
            logger.warning("Redis async library not available. Caching disabled.")
            self.enabled = False
            return

        try:
            self._connection_pool = aioredis.ConnectionPool.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                max_connections=20  # Increased from 10 for better concurrency
            )
            self.redis_client = aioredis.Redis(connection_pool=self._connection_pool)
            # Test connection
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_URL}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
            self.enabled = False
            self.redis_client = None

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
        logger.info("Disconnected from Redis")

    def _hash_text(self, text: str) -> str:
        """Generate a hash for text (used in cache keys)."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _hash_embedding(self, embedding: List[float]) -> str:
        """Generate a hash for embedding vector."""
        # Convert embedding to string and hash it
        embedding_str = ','.join(map(str, embedding))
        return hashlib.sha256(embedding_str.encode('utf-8')).hexdigest()[:16]

    def _ensure_connected(self) -> bool:
        """Check if Redis is connected and enabled."""
        return self.enabled and self.redis_client is not None

    async def get_embedding(self, query_text: str) -> Optional[List[float]]:
        """Get cached embedding for a query text."""
        if not self._ensure_connected():
            return None

        try:
            cache_key = f"embedding:{self._hash_text(query_text)}"
            cached = await self.redis_client.get(cache_key)
            if cached:
                self.stats['embedding_hits'] += 1
                logger.debug(f"Cache hit for embedding: {query_text[:50]}...")
                return json.loads(cached)
            self.stats['embedding_misses'] += 1
            logger.debug(f"Cache miss for embedding: {query_text[:50]}...")
            return None
        except Exception as e:
            logger.warning(f"Error getting embedding from cache: {e}")
            return None

    async def set_embedding(self, query_text: str, embedding: List[float]):
        """Cache an embedding for a query text."""
        if not self._ensure_connected():
            return

        try:
            cache_key = f"embedding:{self._hash_text(query_text)}"
            await self.redis_client.setex(
                cache_key,
                settings.CACHE_EMBEDDING_TTL,
                json.dumps(embedding)
            )
            logger.debug(f"Cached embedding for: {query_text[:50]}...")
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")

    async def get_chunks(self, document_id: str, query_embedding: List[float]) -> Optional[List[Dict]]:
        """Get cached chunks for a document and query embedding."""
        if not self._ensure_connected():
            return None

        try:
            embedding_hash = self._hash_embedding(query_embedding)
            cache_key = f"chunks:{document_id}:{embedding_hash}"
            cached = await self.redis_client.get(cache_key)
            if cached:
                self.stats['chunk_hits'] += 1
                logger.debug(f"Cache hit for chunks: document_id={document_id}")
                return json.loads(cached)
            self.stats['chunk_misses'] += 1
            logger.debug(f"Cache miss for chunks: document_id={document_id}")
            return None
        except Exception as e:
            logger.warning(f"Error getting chunks from cache: {e}")
            return None

    async def set_chunks(self, document_id: str, query_embedding: List[float], chunks: List[Dict]):
        """Cache chunks for a document and query embedding."""
        if not self._ensure_connected():
            return

        try:
            embedding_hash = self._hash_embedding(query_embedding)
            cache_key = f"chunks:{document_id}:{embedding_hash}"
            await self.redis_client.setex(
                cache_key,
                settings.CACHE_CHUNK_TTL,
                json.dumps(chunks)
            )
            logger.debug(f"Cached chunks for document_id={document_id}")
        except Exception as e:
            logger.warning(f"Error caching chunks: {e}")

    async def invalidate_document_chunks(self, document_id: str):
        """Invalidate all cached chunks for a document."""
        if not self._ensure_connected():
            return

        try:
            pattern = f"chunks:{document_id}:*"
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} chunk cache entries for document_id={document_id}")
        except Exception as e:
            logger.warning(f"Error invalidating document chunks: {e}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    async def find_similar_response(
        self,
        query_embedding: List[float],
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find a cached response with similar query embedding."""
        if not self._ensure_connected():
            return None

        try:
            # Search for cached responses for this document
            pattern = f"response:{document_id}:*"
            best_match = None
            best_similarity = 0.0

            async for key in self.redis_client.scan_iter(match=pattern):
                try:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        data = json.loads(cached_data)
                        cached_embedding = data.get('query_embedding')
                        if cached_embedding:
                            similarity = self._cosine_similarity(query_embedding, cached_embedding)
                            if similarity >= settings.CACHE_SIMILARITY_THRESHOLD and similarity > best_similarity:
                                best_similarity = similarity
                                best_match = {
                                    'content': data.get('content'),
                                    'annotations': data.get('annotations', []),
                                    'chunks': data.get('chunks', []),
                                    'similarity': similarity
                                }
                except Exception as e:
                    logger.debug(f"Error processing cached response {key}: {e}")
                    continue

            if best_match:
                self.stats['response_hits'] += 1
                logger.info(f"Found similar cached response with similarity {best_similarity:.3f}")
            else:
                self.stats['response_misses'] += 1
                logger.debug("No similar cached response found")

            return best_match
        except Exception as e:
            logger.warning(f"Error finding similar response: {e}")
            return None

    async def set_response(
        self,
        document_id: str,
        query_embedding: List[float],
        content: str,
        annotations: List[Dict],
        chunks: List[Dict]
    ):
        """Cache a response with its query embedding."""
        if not self._ensure_connected():
            return

        try:
            embedding_hash = self._hash_embedding(query_embedding)
            cache_key = f"response:{document_id}:{embedding_hash}"
            cache_data = {
                'query_embedding': query_embedding,
                'content': content,
                'annotations': annotations,
                'chunks': chunks
            }
            await self.redis_client.setex(
                cache_key,
                settings.CACHE_RESPONSE_TTL,
                json.dumps(cache_data)
            )
            logger.debug(f"Cached response for document_id={document_id}")
        except Exception as e:
            logger.warning(f"Error caching response: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including hit rates."""
        total_embedding = self.stats['embedding_hits'] + self.stats['embedding_misses']
        total_chunk = self.stats['chunk_hits'] + self.stats['chunk_misses']
        total_response = self.stats['response_hits'] + self.stats['response_misses']

        return {
            'enabled': self.enabled,
            'embedding': {
                'hits': self.stats['embedding_hits'],
                'misses': self.stats['embedding_misses'],
                'total': total_embedding,
                'hit_rate': (self.stats['embedding_hits'] / total_embedding * 100) if total_embedding > 0 else 0
            },
            'chunk': {
                'hits': self.stats['chunk_hits'],
                'misses': self.stats['chunk_misses'],
                'total': total_chunk,
                'hit_rate': (self.stats['chunk_hits'] / total_chunk * 100) if total_chunk > 0 else 0
            },
            'response': {
                'hits': self.stats['response_hits'],
                'misses': self.stats['response_misses'],
                'total': total_response,
                'hit_rate': (self.stats['response_hits'] / total_response * 100) if total_response > 0 else 0
            }
        }

    async def clear_all(self):
        """Clear all cache entries (use with caution)."""
        if not self._ensure_connected():
            return

        try:
            await self.redis_client.flushdb()
            logger.warning("Cleared all cache entries")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Global cache service instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.connect()
    return _cache_service


async def close_cache_service():
    """Close the cache service connection."""
    global _cache_service
    if _cache_service:
        await _cache_service.disconnect()
        _cache_service = None

