"""Parallel embedding + DB-write machinery for document ingestion.

Extracted from DocumentService so the ingestion fan-out is independently
testable and reusable. This module owns:

- `generate_embeddings_parallel`: batched, concurrency-limited embedding generation.
- `save_chunks_parallel`: parallel batched DB writes, each in its own session.
- `extract_and_save_sentences`: critical-sentence extraction at the sentence chunk level.
- `save_parent_child_relationships`: persistence for hierarchical chunking.

`IngestionPipeline` holds the embedding service. It performs I/O against the
embedding microservice, the database, and (transitively, via
SentenceRetrievalService) the embedding service again. A process-wide
singleton is exposed via `get_ingestion_pipeline()`, mirroring the
convention used by `annotation_service` and `retrieval_service`.
"""
import asyncio
import logging
from itertools import islice
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..database import engine
from ..models.document import ParentChildRelationship
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Owns the parallel embedding + DB write fan-out used by every ingestion path."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    async def generate_embeddings_parallel(
        self,
        texts: List[str],
        batch_size: int = 50,
        max_concurrent: int = 4,
    ) -> List[List[float]]:
        """
        Generate embeddings in parallel batches for faster processing.

        Args:
            texts: List of text chunks to embed
            batch_size: Number of texts per batch
            max_concurrent: Maximum concurrent batch operations

        Returns:
            List of embeddings (same order as input texts)
        """
        # Split texts into batches
        def chunks(iterable, size):
            iterator = iter(iterable)
            while True:
                batch = list(islice(iterator, size))
                if not batch:
                    break
                yield batch

        text_batches = list(chunks(texts, batch_size))
        all_embeddings: List[List[float]] = []

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch):
            async with semaphore:
                return await self.embedding_service.generate_batch_embeddings_async(batch)

        # Process all batches concurrently
        logger.info(
            f"Processing {len(text_batches)} batches with max {max_concurrent} concurrent operations"
        )
        tasks = [process_batch(batch) for batch in text_batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)

        logger.info(f"Generated {len(all_embeddings)} embeddings in parallel")
        return all_embeddings

    async def save_chunks_parallel(
        self,
        db: Session,
        document_id: str,
        chunks_data: List[Dict],
        batch_size: int = 50,
    ) -> Dict[str, int]:
        """
        Save chunks to database in parallel batches.

        Args:
            db: Database session
            document_id: ID of the document
            chunks_data: List of dicts with 'content', 'page_number', 'embedding'
            batch_size: Number of chunks per batch

        Returns:
            Dict with success/failure counts
        """
        from sqlalchemy.orm import sessionmaker

        # Import here to avoid a circular import at module load (DocumentChunk
        # is fine, but keeping the import local matches the original file).
        from ..models.document import DocumentChunk

        # Create session factory for parallel operations
        SessionFactory = sessionmaker(bind=engine)

        async def save_batch(batch: List[Dict]):
            """Save a single batch in a separate DB session."""
            session = SessionFactory()
            try:
                for chunk_data in batch:
                    # Extract metadata if available (for semantic chunking)
                    metadata = chunk_data.get('metadata', None)

                    # Get chunk_type (default to 'text' if not specified)
                    chunk_type = chunk_data.get('chunk_type', 'text')

                    chunk = DocumentChunk(
                        content=chunk_data['content'],
                        page_number=chunk_data['page_number'],
                        embedding=chunk_data['embedding'],
                        document_id=document_id,
                        chunk_type=chunk_type,  # Set chunk type (text or image)
                        position_data=metadata,  # Store metadata in position_data JSONB field
                    )
                    session.add(chunk)
                session.commit()
                return len(batch), 0  # success_count, error_count
            except Exception as e:
                logger.error(f"Error saving batch: {e}", exc_info=True)
                session.rollback()
                return 0, len(batch)
            finally:
                session.close()

        # Split into batches
        batches = [chunks_data[i:i + batch_size] for i in range(0, len(chunks_data), batch_size)]

        logger.info(f"Saving {len(chunks_data)} chunks in {len(batches)} parallel batches")

        # Save batches concurrently
        results = await asyncio.gather(*[save_batch(batch) for batch in batches], return_exceptions=True)

        # Aggregate results
        total_success = sum(r[0] for r in results if not isinstance(r, Exception))
        total_errors = sum(r[1] for r in results if not isinstance(r, Exception))

        logger.info(f"Saved {total_success} chunks successfully, {total_errors} failed")

        return {
            "success": total_success,
            "errors": total_errors,
        }

    async def extract_and_save_sentences(
        self,
        db: Session,
        document_id: str,
        pages: List,
        course_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Extract critical sentences from document pages and save them as sentence-level chunks.

        Args:
            db: Database session
            document_id: ID of the document
            pages: List of LangChain Document objects from PyPDFLoader
            course_id: Optional course ID

        Returns:
            Dict with statistics: sentences_extracted, sentences_saved, sentences_failed
        """
        if not settings.ENABLE_SENTENCE_RETRIEVAL:
            logger.info("Sentence retrieval disabled, skipping sentence extraction")
            return {"sentences_extracted": 0, "sentences_saved": 0, "sentences_failed": 0}

        try:
            from .sentence_retrieval_service import SentenceRetrievalService

            logger.info(f"Extracting critical sentences from {len(pages)} pages")

            # Initialize sentence retrieval service
            sentence_service = SentenceRetrievalService()

            # Extract critical sentences
            sentences = sentence_service.extract_critical_sentences(
                pages=pages,
                min_sentence_length=settings.SENTENCE_MIN_LENGTH,
                max_sentence_length=settings.SENTENCE_MAX_LENGTH,
                include_short_sentences=settings.SENTENCE_INCLUDE_SHORT,
            )

            if not sentences:
                logger.info("No critical sentences extracted")
                return {"sentences_extracted": 0, "sentences_saved": 0, "sentences_failed": 0}

            # Deduplicate sentences
            sentences = sentence_service.deduplicate_sentences(sentences)

            logger.info(f"Extracted {len(sentences)} critical sentences (after deduplication)")

            # Generate embeddings for sentences
            sentence_texts = [s['content'] for s in sentences]
            sentence_embeddings = await self.generate_embeddings_parallel(
                sentence_texts, batch_size=50, max_concurrent=4
            )

            # Prepare sentence chunks for storage
            sentence_chunks = []
            for i, sentence in enumerate(sentences):
                chunk_data = {
                    'content': sentence['content'],
                    'page_number': sentence['page_number'],
                    'embedding': sentence_embeddings[i],
                    'chunk_type': 'sentence',
                    'chunk_level': 'sentence',
                    'metadata': sentence.get('metadata', {}),
                }
                sentence_chunks.append(chunk_data)

            # Save sentence chunks to database
            result = await self.save_chunks_parallel(
                db, document_id, sentence_chunks, batch_size=50
            )

            logger.info(
                f"Sentence extraction complete: extracted={len(sentences)}, "
                f"saved={result['success']}, failed={result['errors']}"
            )

            return {
                "sentences_extracted": len(sentences),
                "sentences_saved": result['success'],
                "sentences_failed": result['errors'],
            }

        except Exception as e:
            logger.error(f"Error extracting and saving sentences: {e}", exc_info=True)
            return {"sentences_extracted": 0, "sentences_saved": 0, "sentences_failed": 0}

    async def save_parent_child_relationships(
        self,
        db: Session,
        relationships: List[Dict],
    ) -> int:
        """
        Save parent-child relationships to database.

        Args:
            db: Database session
            relationships: List of relationship dicts with parent_chunk_id, child_chunk_id, child_index

        Returns:
            Number of relationships saved
        """
        try:
            saved_count = 0

            for rel_data in relationships:
                relationship = ParentChildRelationship(
                    id=rel_data['id'],
                    parent_chunk_id=rel_data['parent_chunk_id'],
                    child_chunk_id=rel_data['child_chunk_id'],
                    child_index=rel_data['child_index'],
                )
                db.add(relationship)
                saved_count += 1

            db.commit()
            logger.info(f"Successfully saved {saved_count} parent-child relationships")
            return saved_count

        except Exception as e:
            logger.error(f"Failed to save parent-child relationships: {e}", exc_info=True)
            db.rollback()
            raise


# --- Process-wide singleton -------------------------------------------------
_pipeline: IngestionPipeline | None = None


def get_ingestion_pipeline() -> IngestionPipeline:
    """Return the process-wide IngestionPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from .embedding_service import get_embedding_service

        _pipeline = IngestionPipeline(get_embedding_service())
    return _pipeline
