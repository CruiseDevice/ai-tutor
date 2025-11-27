from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class RerankService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the re-ranking service with the specified cross-encoder model."""
        # Auto-detect and use best available device (GPU/MPS/CPU)
        device = self._detect_device()

        logger.info(f"Initializing cross-encoder model '{model_name}' on device: {device}")
        self.model = CrossEncoder(model_name, device=device)
        self.device = device

        logger.info(f"Cross-encoder model loaded successfully on {device}")

    def _detect_device(self) -> str:
        """
        Auto-detect the best available device for re-ranking.

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

    async def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using cross-encoder model based on query relevance.

        Args:
            query: The search query
            chunks: List of chunk dictionaries (must have 'content' field)
            top_k: Number of top chunks to return after re-ranking

        Returns:
            List of re-ranked chunks with updated similarity scores
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for re-ranking")
                return []

            # Prepare query-chunk pairs for the cross-encoder
            pairs = [[query, chunk.get('content', '')] for chunk in chunks]

            # Run prediction in a thread pool to avoid blocking
            scores = await asyncio.to_thread(self.model.predict, pairs)

            # Add re-ranking scores to chunks and sort by score
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)
                # Optionally preserve original similarity score
                if 'similarity' in chunk:
                    chunk['original_similarity'] = chunk['similarity']
                # Update similarity to rerank score for consistency
                chunk['similarity'] = float(score)

            # Sort by re-ranking score (descending) and return top_k
            reranked_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

            logger.info(
                f"Re-ranked {len(chunks)} chunks, returning top {min(top_k, len(reranked_chunks))}"
            )

            return reranked_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error during re-ranking: {e}", exc_info=True)
            # Return original chunks on error (fallback behavior)
            logger.warning("Falling back to original chunk ordering due to re-ranking error")
            return chunks[:top_k]


# Global instance to be initialized at startup
rerank_service: RerankService = None


def get_rerank_service() -> RerankService:
    """Get the global re-ranking service instance."""
    global rerank_service
    if rerank_service is None:
        rerank_service = RerankService()
    return rerank_service
