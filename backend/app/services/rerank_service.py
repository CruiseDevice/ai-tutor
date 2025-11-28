from typing import List, Dict, Any
import httpx
import logging
import os

logger = logging.getLogger(__name__)


class RerankService:
    def __init__(self, embedding_service_url: str = None):
        """Initialize the re-ranking service client."""
        # Use environment variable or default to docker service name
        self.embedding_service_url = embedding_service_url or os.getenv(
            "EMBEDDING_SERVICE_URL", "http://embedding-service:8002"
        )
        logger.info(f"Rerank service client initialized: {self.embedding_service_url}")

    async def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using the embedding service based on query relevance.

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

            # Extract document texts
            documents = [chunk.get('content', '') for chunk in chunks]

            # Call embedding service for reranking
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_service_url}/rerank",
                    json={
                        "query": query,
                        "documents": documents,
                        "top_k": top_k
                    }
                )
                response.raise_for_status()
                data = response.json()

            # Map scores back to chunks
            scores = data["scores"]
            indices = data["indices"]

            # Reorder chunks based on reranking results
            reranked_chunks = []
            for idx, score in zip(indices, scores):
                chunk = chunks[idx].copy()
                chunk['rerank_score'] = float(score)
                # Preserve original similarity score
                if 'similarity' in chunk:
                    chunk['original_similarity'] = chunk['similarity']
                # Update similarity to rerank score
                chunk['similarity'] = float(score)
                reranked_chunks.append(chunk)

            logger.info(
                f"Re-ranked {len(chunks)} chunks, returning top {len(reranked_chunks)}"
            )

            return reranked_chunks

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
