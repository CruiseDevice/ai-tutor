from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


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


# Global instance to be initialized at startup
embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service

