from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import asyncio
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the embedding service with the specified model."""
        # Auto-detect and use best available device (GPU/MPS/CPU)
        device = self._detect_device()

        logger.info(f"Initializing embedding model '{model_name}' on device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimensions = 768  # all-mpnet-base-v2 produces 768-dimensional embeddings
        self.device = device

        logger.info(f"Embedding model loaded successfully on {device}")

    def _detect_device(self) -> str:
        """
        Auto-detect the best available device for embeddings.

        Priority:
        1. CUDA (NVIDIA GPU) - fastest
        2. MPS (Apple Silicon) - fast on M1/M2/M3 Macs
        3. CPU - slowest but always available

        Returns:
            Device name: 'cuda', 'mps', or 'cpu'
        """
        try:
            import torch

            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU detected: {gpu_name}")
                return "cuda"

            # Check for Apple Silicon MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple Silicon MPS detected")
                return "mps"

            # Fallback to CPU
            logger.info("No GPU detected, using CPU")
            return "cpu"

        except ImportError:
            logger.warning("PyTorch not found, defaulting to CPU")
            return "cpu"
        except Exception as e:
            logger.warning(f"Error detecting device: {e}, defaulting to CPU")
            return "cpu"

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

