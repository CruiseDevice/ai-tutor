from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import asyncio


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the embedding service with the specified model."""
        self.model = SentenceTransformer(model_name)
        self.dimensions = 768  # all-mpnet-base-v2 produces 768-dimensional embeddings

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def get_dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.dimensions

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        # Run the blocking embedding generation in a thread pool
        embedding = await asyncio.to_thread(self.model.encode, text)
        return embedding.tolist()

    async def generate_batch_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        # Run the blocking batch embedding generation in a thread pool
        embeddings = await asyncio.to_thread(self.model.encode, texts)
        return embeddings.tolist()


# Global instance to be initialized at startup
embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service

