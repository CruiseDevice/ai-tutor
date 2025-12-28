from typing import List, Dict, Any, Optional
import logging
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)


class ContextualCompressionService:
    """
    Compress retrieved chunks by extracting only query-relevant sentences.

    This service uses a cross-encoder model to score each sentence in a chunk
    against the query, keeping only the most relevant sentences. This provides:

    Benefits:
    - Removes noise from retrieved chunks
    - Allows MORE chunks to fit in context window
    - Focuses LLM attention on relevant content
    - Reduces token usage while maintaining information quality

    Use cases:
    - When retrieving many chunks (>10)
    - For long-form documents with verbose chunks
    - To maximize information density in LLM context
    - When approaching token limits

    Example:
        Original chunk (5 sentences, 300 tokens):
        "Introduction paragraph. Relevant fact 1. Relevant fact 2.
         Unrelated anecdote. Conclusion."

        Compressed chunk (2 sentences, 120 tokens):
        "Relevant fact 1. Relevant fact 2."
    """

    def __init__(self):
        """Initialize the contextual compression service."""
        logger.info("ContextualCompressionService initialized")
        self._compressor = None

    def _get_compressor(self):
        """
        Lazy load the cross-encoder model.

        Uses cross-encoder/ms-marco-MiniLM-L-6-v2 which is:
        - Lightweight (~90MB)
        - Fast inference
        - Trained on MS MARCO passage ranking
        - Good for sentence-query relevance scoring

        Returns:
            CrossEncoder model instance
        """
        if self._compressor is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder model for contextual compression...")
                self._compressor = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder model loaded successfully")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Error loading cross-encoder model: {e}", exc_info=True)
                raise

        return self._compressor

    async def compress_chunks(
        self,
        query: str,
        chunks: List[Dict],
        min_sentences: Optional[int] = None,
        keep_percentage: Optional[float] = None
    ) -> List[Dict]:
        """
        Extract relevant sentences from chunks using cross-encoder model.

        For each chunk:
        1. Tokenize into sentences
        2. Score each sentence against the query
        3. Keep only top-scoring sentences
        4. Reconstruct compressed chunk

        Args:
            query: User query text
            chunks: List of chunk dictionaries with 'content' field
            min_sentences: Minimum sentences before compression (default from config)
            keep_percentage: Percentage of sentences to keep (default from config)

        Returns:
            List of compressed chunks with:
                - All original fields preserved
                - 'content' replaced with compressed version
                - 'compression_ratio' added (0.0-1.0)
                - 'original_length' added (character count before compression)
                - 'compressed_length' added (character count after compression)
        """
        try:
            # Use config defaults if not specified
            if min_sentences is None:
                min_sentences = settings.COMPRESSION_MIN_SENTENCES
            if keep_percentage is None:
                keep_percentage = settings.COMPRESSION_KEEP_PERCENTAGE

            # Import sentence tokenizer
            try:
                from nltk.tokenize import sent_tokenize
                logger.debug("Using nltk sentence tokenizer")
            except ImportError:
                logger.warning("NLTK not available, using simple sentence tokenizer")
                sent_tokenize = self._simple_sentence_tokenize

            # Get cross-encoder model
            compressor = self._get_compressor()

            compressed_chunks = []
            total_original_chars = 0
            total_compressed_chars = 0

            for chunk in chunks:
                content = chunk.get('content', '')

                if not content:
                    # Empty chunk, skip
                    compressed_chunks.append(chunk)
                    continue

                # Tokenize into sentences
                sentences = sent_tokenize(content)

                # Track original length
                original_length = len(content)
                total_original_chars += original_length

                if len(sentences) < min_sentences:
                    # Too short to compress meaningfully, keep as-is
                    chunk_copy = chunk.copy()
                    chunk_copy['compression_ratio'] = 1.0
                    chunk_copy['original_length'] = original_length
                    chunk_copy['compressed_length'] = original_length
                    compressed_chunks.append(chunk_copy)
                    total_compressed_chars += original_length
                    continue

                # Score each sentence against the query
                # cross-encoder.predict() expects list of [query, sentence] pairs
                sentence_pairs = [[query, sent] for sent in sentences]
                sentence_scores = compressor.predict(sentence_pairs)

                # Determine threshold to keep top N% of sentences
                # Use percentile: keep sentences scoring above the (100-keep_percentage) percentile
                percentile_threshold = (1.0 - keep_percentage) * 100
                score_threshold = np.percentile(sentence_scores, percentile_threshold)

                # Keep sentences above threshold
                relevant_sentences = [
                    sent for sent, score in zip(sentences, sentence_scores)
                    if score >= score_threshold
                ]

                # Ensure we keep at least one sentence
                if not relevant_sentences and sentences:
                    # If threshold was too high, just keep the best sentence
                    best_idx = np.argmax(sentence_scores)
                    relevant_sentences = [sentences[best_idx]]

                # Create compressed chunk
                compressed_content = ' '.join(relevant_sentences)
                compressed_length = len(compressed_content)
                total_compressed_chars += compressed_length

                compressed_chunk = chunk.copy()
                compressed_chunk['content'] = compressed_content
                compressed_chunk['compression_ratio'] = len(relevant_sentences) / len(sentences)
                compressed_chunk['original_length'] = original_length
                compressed_chunk['compressed_length'] = compressed_length
                compressed_chunk['sentences_kept'] = len(relevant_sentences)
                compressed_chunk['sentences_total'] = len(sentences)

                compressed_chunks.append(compressed_chunk)

            # Calculate overall compression statistics
            if compressed_chunks:
                avg_compression_ratio = sum(
                    c.get('compression_ratio', 1.0) for c in compressed_chunks
                ) / len(compressed_chunks)

                char_reduction = (
                    (total_original_chars - total_compressed_chars) / total_original_chars * 100
                    if total_original_chars > 0 else 0
                )

                logger.info(
                    f"Compressed {len(compressed_chunks)} chunks: "
                    f"avg sentence retention {avg_compression_ratio:.1%}, "
                    f"character reduction {char_reduction:.1f}% "
                    f"({total_original_chars} → {total_compressed_chars} chars)"
                )

            return compressed_chunks

        except Exception as e:
            logger.error(f"Error compressing chunks: {e}", exc_info=True)
            # On error, return original chunks uncompressed
            logger.warning("Returning original uncompressed chunks due to error")
            return chunks

    def _simple_sentence_tokenize(self, text: str) -> List[str]:
        """
        Simple sentence tokenizer as fallback when nltk is not available.

        Splits on common sentence boundaries: . ! ? followed by space and capital letter.

        Args:
            text: Text to tokenize

        Returns:
            List of sentences
        """
        import re

        # Split on period, exclamation, or question mark followed by space
        # Use positive lookahead to keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def calculate_compression_stats(self, compressed_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Calculate compression statistics for reporting.

        Args:
            compressed_chunks: List of chunks returned from compress_chunks()

        Returns:
            Dictionary with compression statistics:
                - total_chunks: Total number of chunks
                - avg_compression_ratio: Average sentence retention ratio
                - total_original_chars: Total characters before compression
                - total_compressed_chars: Total characters after compression
                - char_reduction_pct: Percentage reduction in characters
                - total_sentences_kept: Total sentences kept across all chunks
                - total_sentences_original: Total sentences before compression
        """
        if not compressed_chunks:
            return {
                'total_chunks': 0,
                'avg_compression_ratio': 0.0,
                'total_original_chars': 0,
                'total_compressed_chars': 0,
                'char_reduction_pct': 0.0,
                'total_sentences_kept': 0,
                'total_sentences_original': 0
            }

        total_original_chars = sum(c.get('original_length', 0) for c in compressed_chunks)
        total_compressed_chars = sum(c.get('compressed_length', 0) for c in compressed_chunks)
        total_sentences_kept = sum(c.get('sentences_kept', 0) for c in compressed_chunks)
        total_sentences_original = sum(c.get('sentences_total', 0) for c in compressed_chunks)

        avg_compression_ratio = sum(
            c.get('compression_ratio', 1.0) for c in compressed_chunks
        ) / len(compressed_chunks)

        char_reduction_pct = (
            (total_original_chars - total_compressed_chars) / total_original_chars * 100
            if total_original_chars > 0 else 0
        )

        return {
            'total_chunks': len(compressed_chunks),
            'avg_compression_ratio': avg_compression_ratio,
            'total_original_chars': total_original_chars,
            'total_compressed_chars': total_compressed_chars,
            'char_reduction_pct': char_reduction_pct,
            'total_sentences_kept': total_sentences_kept,
            'total_sentences_original': total_sentences_original
        }

    def should_compress(self, chunks: List[Dict], query: str) -> bool:
        """
        Determine if compression should be applied based on chunk characteristics.

        Compression is beneficial when:
        - Many chunks retrieved (>10)
        - Chunks are long (avg >500 chars)
        - Approaching token limits

        Args:
            chunks: List of chunk dictionaries
            query: User query (unused but kept for future query-based logic)

        Returns:
            True if compression is recommended, False otherwise
        """
        if not chunks:
            return False

        # Don't compress if disabled
        if not settings.ENABLE_CONTEXTUAL_COMPRESSION:
            return False

        # Compress if we have many chunks
        if len(chunks) > 10:
            logger.debug(f"Compression recommended: {len(chunks)} chunks > 10")
            return True

        # Compress if chunks are long on average
        avg_length = sum(len(c.get('content', '')) for c in chunks) / len(chunks)
        if avg_length > 500:
            logger.debug(f"Compression recommended: avg chunk length {avg_length:.0f} > 500 chars")
            return True

        logger.debug(f"Compression not needed: {len(chunks)} chunks, avg {avg_length:.0f} chars")
        return False
