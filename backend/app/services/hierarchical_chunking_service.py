"""
Hierarchical Chunking Service for Phase 3: Parent-Child Chunk Relationships

This service implements a two-level chunking strategy:
1. Parent chunks: Larger chunks (1500 tokens) that provide context to the LLM
2. Child chunks: Smaller chunks (300 tokens) that improve retrieval precision

Strategy:
- Search/retrieval operates on child chunks for better precision
- LLM receives parent chunks for better context
- Best of both worlds: precision + context
"""
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import uuid

from ..config import settings

logger = logging.getLogger(__name__)


class HierarchicalChunkingService:
    """
    Service for creating hierarchical parent-child chunk relationships.

    This enables:
    - Precision retrieval: Search against small child chunks
    - Context preservation: Return large parent chunks to LLM
    - Flexible chunking: Different sizes for indexing vs delivery
    """

    def __init__(self):
        """Initialize the hierarchical chunking service."""
        self.parent_chunk_size = settings.HIERARCHICAL_PARENT_CHUNK_SIZE
        self.parent_overlap = settings.HIERARCHICAL_PARENT_OVERLAP
        self.child_chunk_size = settings.HIERARCHICAL_CHILD_CHUNK_SIZE
        self.child_overlap = settings.HIERARCHICAL_CHILD_OVERLAP

        logger.info(
            f"HierarchicalChunkingService initialized: "
            f"parent_size={self.parent_chunk_size}, child_size={self.child_chunk_size}"
        )

    def chunk_text(
        self,
        text: str,
        page_number: int,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Create parent chunks from text.

        Args:
            text: The text content to chunk
            page_number: The page number this text came from
            metadata: Optional metadata to attach to chunks

        Returns:
            List of parent chunk dictionaries with:
                - id: Unique chunk ID
                - content: Chunk text content
                - page_number: Page number
                - metadata: Chunk metadata
                - chunk_level: 'parent'
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for page {page_number}, skipping chunking")
            return []

        # Create parent chunks using RecursiveCharacterTextSplitter
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        parent_texts = parent_splitter.split_text(text)

        parent_chunks = []
        for idx, parent_text in enumerate(parent_texts):
            chunk_id = str(uuid.uuid4())
            parent_chunk = {
                'id': chunk_id,
                'content': parent_text,
                'page_number': page_number,
                'metadata': metadata or {},
                'chunk_level': 'parent',
                'parent_index': idx  # Track order of parent chunks
            }
            parent_chunks.append(parent_chunk)

        logger.debug(
            f"Created {len(parent_chunks)} parent chunks from page {page_number} "
            f"(avg size: {sum(len(c['content']) for c in parent_chunks) // len(parent_chunks) if parent_chunks else 0} chars)"
        )

        return parent_chunks

    def create_child_chunks(
        self,
        parent_chunk: Dict
    ) -> List[Dict]:
        """
        Create child chunks from a parent chunk.

        Args:
            parent_chunk: Parent chunk dictionary with 'content', 'page_number', etc.

        Returns:
            List of child chunk dictionaries with:
                - id: Unique chunk ID
                - content: Chunk text content
                - page_number: Page number (inherited from parent)
                - metadata: Chunk metadata (inherited from parent)
                - chunk_level: 'child'
                - parent_chunk_id: ID of the parent chunk
                - child_index: Order within parent (0, 1, 2, ...)
        """
        parent_content = parent_chunk['content']
        parent_id = parent_chunk['id']

        if not parent_content or not parent_content.strip():
            logger.warning(f"Empty parent chunk {parent_id}, skipping child creation")
            return []

        # Create child chunks using RecursiveCharacterTextSplitter
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        child_texts = child_splitter.split_text(parent_content)

        child_chunks = []
        for child_idx, child_text in enumerate(child_texts):
            chunk_id = str(uuid.uuid4())
            child_chunk = {
                'id': chunk_id,
                'content': child_text,
                'page_number': parent_chunk['page_number'],
                'metadata': parent_chunk.get('metadata', {}),
                'chunk_level': 'child',
                'parent_chunk_id': parent_id,
                'child_index': child_idx  # Order within parent
            }
            child_chunks.append(child_chunk)

        logger.debug(
            f"Created {len(child_chunks)} child chunks from parent {parent_id[:8]}... "
            f"(avg size: {sum(len(c['content']) for c in child_chunks) // len(child_chunks) if child_chunks else 0} chars)"
        )

        return child_chunks

    def process_document_hierarchically(
        self,
        pages: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process an entire document into hierarchical chunks.

        Args:
            pages: List of page dictionaries with 'content', 'page_number', 'metadata'

        Returns:
            Tuple of (parent_chunks, child_chunks, relationships):
                - parent_chunks: List of parent chunk dicts
                - child_chunks: List of child chunk dicts
                - relationships: List of parent-child relationship dicts with:
                    - parent_chunk_id
                    - child_chunk_id
                    - child_index
        """
        all_parent_chunks = []
        all_child_chunks = []
        all_relationships = []

        for page_data in pages:
            page_content = page_data.get('content', '')
            page_number = page_data.get('page_number', 1)
            page_metadata = page_data.get('metadata', {})

            # Step 1: Create parent chunks for this page
            parent_chunks = self.chunk_text(page_content, page_number, page_metadata)

            # Step 2: For each parent, create child chunks
            for parent_chunk in parent_chunks:
                child_chunks = self.create_child_chunks(parent_chunk)

                # Step 3: Create relationships
                for child_chunk in child_chunks:
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'parent_chunk_id': parent_chunk['id'],
                        'child_chunk_id': child_chunk['id'],
                        'child_index': child_chunk['child_index']
                    }
                    all_relationships.append(relationship)

                all_child_chunks.extend(child_chunks)

            all_parent_chunks.extend(parent_chunks)

        logger.info(
            f"Hierarchical processing complete: "
            f"{len(all_parent_chunks)} parents, {len(all_child_chunks)} children, "
            f"{len(all_relationships)} relationships"
        )

        return all_parent_chunks, all_child_chunks, all_relationships


def prepare_chunks_for_storage(
    parent_chunks: List[Dict],
    child_chunks: List[Dict],
    embeddings_parent: List[List[float]],
    embeddings_child: List[List[float]]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare hierarchical chunks for database storage by adding embeddings.

    Args:
        parent_chunks: List of parent chunk dictionaries
        child_chunks: List of child chunk dictionaries
        embeddings_parent: List of embedding vectors for parent chunks
        embeddings_child: List of embedding vectors for child chunks

    Returns:
        Tuple of (prepared_parent_chunks, prepared_child_chunks) ready for DB insertion
    """
    if len(parent_chunks) != len(embeddings_parent):
        raise ValueError(
            f"Mismatch: {len(parent_chunks)} parent chunks but {len(embeddings_parent)} embeddings"
        )

    if len(child_chunks) != len(embeddings_child):
        raise ValueError(
            f"Mismatch: {len(child_chunks)} child chunks but {len(embeddings_child)} embeddings"
        )

    # Add embeddings to parent chunks
    prepared_parents = []
    for chunk, embedding in zip(parent_chunks, embeddings_parent):
        chunk_copy = chunk.copy()
        chunk_copy['embedding'] = embedding
        prepared_parents.append(chunk_copy)

    # Add embeddings to child chunks
    prepared_children = []
    for chunk, embedding in zip(child_chunks, embeddings_child):
        chunk_copy = chunk.copy()
        chunk_copy['embedding'] = embedding
        prepared_children.append(chunk_copy)

    logger.info(
        f"Prepared {len(prepared_parents)} parent and {len(prepared_children)} child chunks for storage"
    )

    return prepared_parents, prepared_children
