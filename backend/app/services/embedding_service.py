from typing import List
import httpx
import logging
import os

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, embedding_service_url: str = None):
        """Initialize the embedding service client."""
        # Use environment variable or default to docker service name
        self.embedding_service_url = embedding_service_url or os.getenv(
            "EMBEDDING_SERVICE_URL", "http://embedding-service:8002"
        )
        self.dimensions = 768  # all-mpnet-base-v2 produces 768-dimensional embeddings
        logger.info(f"Embedding service client initialized: {self.embedding_service_url}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.generate_batch_embeddings([text])[0]

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.embedding_service_url}/embed",
                    json={"texts": texts}
                )
                response.raise_for_status()
                data = response.json()
                return data["embeddings"]
        except Exception as e:
            logger.error(f"Error calling embedding service: {e}")
            raise

    def get_dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.dimensions

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        return (await self.generate_batch_embeddings_async([text]))[0]

    async def generate_batch_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_service_url}/embed",
                    json={"texts": texts}
                )
                response.raise_for_status()
                data = response.json()
                return data["embeddings"]
        except Exception as e:
            logger.error(f"Error calling embedding service: {e}")
            raise


# Global instance to be initialized at startup
embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service

