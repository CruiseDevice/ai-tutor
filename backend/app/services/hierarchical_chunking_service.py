"""
Hierarchical Chunking Service for Parent-Child Chunk Relationships

This service implements a two-level chunking strategy:
1. Parent chunks: Larger chunks (1500 tokens) that provide context to the LLM
2. Child chunks: Smaller chunks (300 tokens) that improve retrieval precision

Strategy:
- Search/retrieval operates on child chunks for better precision
- LLM receives parent chunks for better context
- Best of both worlds: precision + context

Adaptive Chunk Sizing
- Analyzes content density before chunking
- Adjusts chunk sizes dynamically based on content type
- Dense content (tables, code) → smaller chunks
- Sparse content (narrative) → larger chunks
"""
from typing import List, Dict, Tuple, Optional
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

    Adaptive chunk sizing based on content density
    """

    def __init__(self, enable_adaptive_chunking: bool = None):
        """
        Initialize the hierarchical chunking service.

        Args:
            enable_adaptive_chunking: Whether to enable adaptive chunk sizing.
                                     If None, uses settings.ENABLE_ADAPTIVE_CHUNKING
        """
        self.parent_chunk_size = settings.HIERARCHICAL_PARENT_CHUNK_SIZE
        self.parent_overlap = settings.HIERARCHICAL_PARENT_OVERLAP
        self.child_chunk_size = settings.HIERARCHICAL_CHILD_CHUNK_SIZE
        self.child_overlap = settings.HIERARCHICAL_CHILD_OVERLAP

        # Adaptive chunking
        self.enable_adaptive_chunking = (
            enable_adaptive_chunking if enable_adaptive_chunking is not None
            else settings.ENABLE_ADAPTIVE_CHUNKING
        )

        # Initialize density calculator if adaptive chunking enabled
        self.density_calculator = None
        if self.enable_adaptive_chunking:
            from .density_calculator_service import get_density_calculator_service
            self.density_calculator = get_density_calculator_service()
            logger.info("Adaptive chunk sizing enabled - using density calculator")

        logger.info(
            f"HierarchicalChunkingService initialized: "
            f"parent_size={self.parent_chunk_size}, child_size={self.child_chunk_size}, "
            f"adaptive_chunking={self.enable_adaptive_chunking}"
        )

    def chunk_text(
        self,
        text: str,
        page_number: int,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Create parent chunks from text.

        Uses adaptive chunk sizing based on content density if enabled.

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
                - density_metrics: Density analysis (if adaptive chunking enabled)
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for page {page_number}, skipping chunking")
            return []

        # Analyze content density for adaptive chunk sizing
        density_metrics = None
        parent_size = self.parent_chunk_size
        parent_overlap = self.parent_overlap

        if self.enable_adaptive_chunking and self.density_calculator:
            try:
                density_metrics = self.density_calculator.calculate_density(text)

                # Use recommended parent chunk size (scale up from base recommendation)
                # Base recommendation is for general chunking, parent chunks should be ~3x larger
                base_size = density_metrics.recommended_chunk_size
                parent_size = int(base_size * 1.5)  # 1.5x scaling for parent chunks

                # Ensure within bounds
                parent_size = max(
                    settings.CHUNK_SIZE_MIN,
                    min(settings.HIERARCHICAL_PARENT_CHUNK_SIZE * 2, parent_size)
                )

                # Scale overlap proportionally
                parent_overlap = int(density_metrics.recommended_overlap * 1.5)

                logger.debug(
                    f"[Adaptive] Page {page_number}: density={density_metrics.overall_density:.2f}, "
                    f"type={density_metrics.content_type}, "
                    f"parent_size={parent_size} (base={base_size})"
                )

            except Exception as e:
                logger.warning(f"Density calculation failed for page {page_number}, using default sizes: {e}")
                # Fall back to default sizes
                parent_size = self.parent_chunk_size
                parent_overlap = self.parent_overlap

        # Create parent chunks using RecursiveCharacterTextSplitter
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        parent_texts = parent_splitter.split_text(text)

        parent_chunks = []
        for idx, parent_text in enumerate(parent_texts):
            chunk_id = str(uuid.uuid4())

            # Build chunk metadata
            chunk_metadata = metadata or {}
            if density_metrics:
                # Store density information in metadata
                chunk_metadata['density'] = {
                    'overall_density': density_metrics.overall_density,
                    'content_type': density_metrics.content_type,
                    'token_density': density_metrics.token_density,
                    'special_char_density': density_metrics.special_char_density,
                    'numeric_density': density_metrics.numeric_density,
                    'adaptive_chunk_size': parent_size,
                    'adaptive_overlap': parent_overlap
                }

            parent_chunk = {
                'id': chunk_id,
                'content': parent_text,
                'page_number': page_number,
                'metadata': chunk_metadata,
                'chunk_level': 'parent',
                'parent_index': idx  # Track order of parent chunks
            }
            parent_chunks.append(parent_chunk)

        avg_size = sum(len(c['content']) for c in parent_chunks) // len(parent_chunks) if parent_chunks else 0
        logger.debug(
            f"Created {len(parent_chunks)} parent chunks from page {page_number} "
            f"(avg size: {avg_size} chars, target: {parent_size})"
        )

        return parent_chunks

    def create_child_chunks(
        self,
        parent_chunk: Dict
    ) -> List[Dict]:
        """
        Create child chunks from a parent chunk.

        Uses adaptive chunk sizing based on parent content density if enabled.

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

        # Use adaptive child chunk sizing if enabled
        child_size = self.child_chunk_size
        child_overlap = self.child_overlap

        # Check if parent already has density metrics in metadata
        parent_metadata = parent_chunk.get('metadata', {})
        if self.enable_adaptive_chunking and 'density' in parent_metadata:
            # Reuse parent density metrics for child sizing
            density_info = parent_metadata['density']
            overall_density = density_info.get('overall_density', 0.5)
            content_type = density_info.get('content_type', 'narrative')

            # Calculate child size based on parent's density
            # Child chunks should be ~20% of parent size
            adaptive_parent_size = density_info.get('adaptive_chunk_size', self.parent_chunk_size)
            child_size = int(adaptive_parent_size * 0.2)

            # Ensure within bounds
            child_size = max(
                settings.CHUNK_SIZE_MIN // 2,  # Minimum child size is half of general minimum
                min(settings.HIERARCHICAL_CHILD_CHUNK_SIZE * 2, child_size)
            )

            # Scale overlap proportionally
            adaptive_parent_overlap = density_info.get('adaptive_overlap', self.parent_overlap)
            child_overlap = int(adaptive_parent_overlap * 0.2)

            logger.debug(
                f"[Adaptive] Child sizing for parent {parent_id[:8]}: "
                f"density={overall_density:.2f}, type={content_type}, child_size={child_size}"
            )

        # Create child chunks using RecursiveCharacterTextSplitter
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
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
                'metadata': parent_metadata.copy(),  # Inherit parent's metadata (including density)
                'chunk_level': 'child',
                'parent_chunk_id': parent_id,
                'child_index': child_idx  # Order within parent
            }
            child_chunks.append(child_chunk)

        avg_size = sum(len(c['content']) for c in child_chunks) // len(child_chunks) if child_chunks else 0
        logger.debug(
            f"Created {len(child_chunks)} child chunks from parent {parent_id[:8]}... "
            f"(avg size: {avg_size} chars, target: {child_size})"
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
