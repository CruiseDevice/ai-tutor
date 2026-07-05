"""Text chunking for document ingestion.

Extracted from DocumentService so the chunking pipeline is independently
testable and reusable. This module owns:

- NLTK punkt/punkt_tab data provisioning.
- Document structure detection (headers, sections, tables, lists).
- Per-chunk content-type classification + metadata extraction.
- Adaptive RecursiveCharacterTextSplitter configuration (content-type aware).
- LangChain SemanticChunker configuration (embedding-based boundaries).
- The public `chunk_with_semantic_boundaries` entry point, which implements
  the semantic → adaptive fallback strategy and sentence-boundary correction.

`DocumentChunker` holds the embedding service (required by the semantic
chunker). It performs no I/O and writes nothing to the database — it is a
pure transform from PDF pages to chunk dicts. A process-wide singleton is
exposed via `get_chunker()`, mirroring the convention used by
`annotation_service` and `retrieval_service`.
"""
import logging
import re
from functools import lru_cache
from typing import Dict, List

from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _ensure_nltk_data():
    """
    Ensure NLTK punkt tokenizer is downloaded.
    Uses lru_cache to only run once per application lifecycle.
    """
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
        logger.debug("NLTK punkt tokenizer already available")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK punkt tokenizer downloaded successfully")

    # Also ensure punkt_tab is available (required for newer NLTK versions)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)
        logger.info("NLTK punkt_tab tokenizer downloaded successfully")


class DocumentChunker:
    """Transform PDF pages into chunk dicts. Stateless beyond the embedding service."""

    def __init__(self, embedding_service: EmbeddingService):
        # Ensure NLTK data is present (punkt tokenizer for sentence splitting)
        _ensure_nltk_data()
        self.embedding_service = embedding_service

    def detect_document_structure(self, pages: List) -> Dict:
        """
        Analyze document pages to detect structural elements like headers, sections, tables, and lists.

        Args:
            pages: List of LangChain Document objects from PyPDFLoader

        Returns:
            Dict containing structural information:
            - headers: List of detected headers with their positions
            - sections: List of section boundaries
            - tables: List of detected table regions
            - lists: List of detected list regions
        """
        structure = {
            'headers': [],
            'sections': [],
            'tables': [],
            'lists': [],
        }

        for page_idx, page in enumerate(pages):
            text = page.page_content
            lines = text.split('\n')

            for line_idx, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # Detect headers (short lines, all caps, or numbered sections)
                if len(line_stripped) < 100:  # Headers are typically shorter
                    # Check for all caps (at least 70% uppercase)
                    alpha_chars = [c for c in line_stripped if c.isalpha()]
                    if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.7:
                        structure['headers'].append({
                            'page': page_idx,
                            'line': line_idx,
                            'text': line_stripped,
                            'type': 'all_caps',
                        })
                        continue

                    # Check for numbered sections (e.g., "1. Introduction", "Chapter 1", "Section 2.1")
                    if re.match(
                        r'^(\d+\.|\d+\.\d+\.?|\w+\s+\d+|Chapter\s+\d+|Section\s+\d+)',
                        line_stripped,
                        re.IGNORECASE,
                    ):
                        structure['headers'].append({
                            'page': page_idx,
                            'line': line_idx,
                            'text': line_stripped,
                            'type': 'numbered_section',
                        })
                        continue

                # Detect tables (aligned columns, multiple tabs/spaces, pipe characters)
                if '|' in line or '\t\t' in line or re.search(r'\s{4,}', line):
                    # Look for table-like patterns
                    structure['tables'].append({
                        'page': page_idx,
                        'line': line_idx,
                        'text': line_stripped,
                    })
                    continue

                # Detect lists (bullet points, numbered items)
                if re.match(r'^[\u2022\-\*\u25e6\u25aa]\s+', line_stripped) or re.match(
                    r'^\d+[\.\)]\s+', line_stripped
                ):
                    structure['lists'].append({
                        'page': page_idx,
                        'line': line_idx,
                        'text': line_stripped,
                    })

        # Detect section boundaries based on headers
        for header in structure['headers']:
            structure['sections'].append({
                'page': header['page'],
                'line': header['line'],
                'title': header['text'],
            })

        logger.debug(
            f"Detected structure: {len(structure['headers'])} headers, "
            f"{len(structure['sections'])} sections, {len(structure['tables'])} table lines, "
            f"{len(structure['lists'])} list items"
        )

        return structure

    def identify_content_type(self, text: str) -> str:
        """
        Classify text content type.

        Args:
            text: Text content to classify

        Returns:
            Content type: 'header', 'table', 'list', 'code', or 'paragraph'
        """
        text_stripped = text.strip()
        if not text_stripped:
            return 'paragraph'

        # Check for header patterns
        if len(text_stripped) < 100:
            # All caps header
            alpha_chars = [c for c in text_stripped if c.isalpha()]
            if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.7:
                return 'header'

            # Numbered section header
            if re.match(
                r'^(\d+\.|\d+\.\d+\.?|\w+\s+\d+|Chapter\s+\d+|Section\s+\d+)',
                text_stripped,
                re.IGNORECASE,
            ):
                return 'header'

        # Check for table content
        if '|' in text or '\t\t' in text or re.search(r'\s{4,}', text):
            # Count aligned spacing or pipe characters
            if text.count('|') >= 2 or len(re.findall(r'\s{4,}', text)) >= 2:
                return 'table'

        # Check for list items
        if re.match(r'^[\u2022\-\*\u25e6\u25aa]\s+', text_stripped) or re.match(
            r'^\d+[\.\)]\s+', text_stripped
        ):
            return 'list'

        # Check for code (high density of special chars, indentation)
        lines = text.split('\n')
        if len(lines) > 1:
            indented_lines = sum(
                1 for line in lines if line.startswith('    ') or line.startswith('\t')
            )
            if indented_lines / len(lines) > 0.5:  # More than 50% indented
                # Check for code-like characters
                special_chars = sum(1 for c in text if c in '{}[]()<>=;:')
                if special_chars / len(text) > 0.05:  # More than 5% special chars
                    return 'code'

        # Default to paragraph
        return 'paragraph'

    def extract_metadata(self, chunk_text: str, page_number: int, document_structure: Dict) -> Dict:
        """
        Extract metadata from a chunk based on its content and document structure.

        Args:
            chunk_text: The text content of the chunk
            page_number: Page number (1-indexed)
            document_structure: Document structure from detect_document_structure()

        Returns:
            Dict containing metadata:
            - headers: List of headers preceding this chunk
            - section_title: Current section title
            - content_type: Type of content
            - table_info: Table metadata if applicable
            - sentence_count: Number of sentences
            - char_count: Character count
            - word_count: Word count
        """
        metadata = {
            'headers': [],
            'section_title': None,
            'content_type': self.identify_content_type(chunk_text),
            'table_info': None,
            'sentence_count': 0,
            'char_count': len(chunk_text),
            'word_count': len(chunk_text.split()),
        }

        # Find headers that precede this chunk on the same or previous pages
        page_idx = page_number - 1  # Convert to 0-indexed
        for header in document_structure.get('headers', []):
            if header['page'] <= page_idx:
                metadata['headers'].append(header['text'])
                # Use the most recent header as section title
                if header['page'] == page_idx or not metadata['section_title']:
                    metadata['section_title'] = header['text']

        # Count sentences using NLTK
        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(chunk_text)
            metadata['sentence_count'] = len(sentences)
        except Exception as e:
            logger.debug(f"NLTK sentence tokenization failed: {e}, using simple count")
            # Fallback: count sentence-ending punctuation
            metadata['sentence_count'] = len(re.findall(r'[.!?]+', chunk_text))

        # Extract table info if content is a table
        if metadata['content_type'] == 'table':
            lines = chunk_text.split('\n')
            metadata['table_info'] = {
                'rows': len(lines),
                'has_pipes': '|' in chunk_text,
                'estimated_columns': (
                    max(line.count('|') for line in lines) + 1 if '|' in chunk_text else None
                ),
            }

        return metadata

    def create_semantic_chunker(self):
        """
        Create a LangChain SemanticChunker using the existing embedding service.

        Returns:
            SemanticChunker configured with our embedding service and settings
        """
        from langchain.embeddings.base import Embeddings
        from langchain_experimental.text_splitter import SemanticChunker

        # Create a wrapper class to adapt our EmbeddingService to LangChain's Embeddings interface
        class EmbeddingServiceAdapter(Embeddings):
            def __init__(self, embedding_service):
                self.embedding_service = embedding_service

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed a list of documents."""
                return self.embedding_service.generate_batch_embeddings(texts)

            def embed_query(self, text: str) -> List[float]:
                """Embed a single query."""
                return self.embedding_service.generate_embedding(text)

        # Create adapter and semantic chunker
        embeddings_adapter = EmbeddingServiceAdapter(self.embedding_service)

        semantic_chunker = SemanticChunker(
            embeddings=embeddings_adapter,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=settings.SEMANTIC_CHUNKING_BREAKPOINT_THRESHOLD,
        )

        logger.debug("Created SemanticChunker with embedding service adapter")
        return semantic_chunker

    def create_adaptive_splitter(self, content_type: str = 'paragraph') -> RecursiveCharacterTextSplitter:
        """
        Create an adaptive RecursiveCharacterTextSplitter with structure-aware separators.

        Args:
            content_type: Type of content to optimize for ('header', 'paragraph', 'table', 'list', 'code')

        Returns:
            Configured RecursiveCharacterTextSplitter
        """
        # Get adaptive chunk size and overlap based on content type
        chunk_size = self.get_adaptive_chunk_size(content_type)
        chunk_overlap = self.calculate_adaptive_overlap(chunk_size, content_type)

        # Define separators based on content type
        if content_type == 'code':
            separators = ["\n\n\n", "\n\n", "\n", " ", ""]
        elif content_type == 'table':
            separators = ["\n\n", "\n", "|", " ", ""]
        elif content_type == 'list':
            separators = ["\n\n", "\n", ". ", ", ", " ", ""]
        else:  # paragraph, header, or default
            separators = [
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",  # Line breaks
                ". ",  # Sentence endings
                "! ",  # Exclamation sentences
                "? ",  # Question sentences
                "; ",  # Semicolons
                ", ",  # Commas
                " ",  # Spaces
                "",  # Characters (fallback)
            ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.debug(
            f"Created adaptive splitter for {content_type}: "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

        return splitter

    def get_adaptive_chunk_size(self, content_type: str) -> int:
        """
        Determine adaptive chunk size based on content type.

        Args:
            content_type: Type of content

        Returns:
            Chunk size in characters
        """
        if not settings.CHUNK_BY_CONTENT_TYPE:
            return settings.CHUNK_SIZE_DEFAULT

        size_map = {
            'header': 650,  # Headers: 500-800 chars (using mid-range)
            'paragraph': 1250,  # Paragraphs: 1000-1500 chars (using mid-range)
            'table': 1000,  # Tables: 800-1200 chars (using mid-range)
            'list': 1000,  # Lists: Similar to paragraphs
            'code': 1200,  # Code: Larger to preserve complete blocks
        }

        chunk_size = size_map.get(content_type, settings.CHUNK_SIZE_DEFAULT)

        # Clamp to configured min/max
        chunk_size = max(settings.CHUNK_SIZE_MIN, min(chunk_size, settings.CHUNK_SIZE_MAX))

        return chunk_size

    def calculate_adaptive_overlap(self, chunk_size: int, content_type: str) -> int:
        """
        Calculate adaptive overlap based on chunk size and content type.

        Args:
            chunk_size: Size of the chunk in characters
            content_type: Type of content

        Returns:
            Overlap size in characters
        """
        # Start with percentage-based overlap
        overlap = int(chunk_size * settings.CHUNK_OVERLAP_PERCENTAGE)

        # Adjust based on content type
        if content_type == 'header':
            # Smaller overlap for headers (they're already small)
            overlap = int(overlap * 0.7)
        elif content_type in ['table', 'code']:
            # Larger overlap for structured content to preserve context
            overlap = int(overlap * 1.2)

        # Clamp to configured min/max
        overlap = max(settings.CHUNK_OVERLAP_MIN, min(overlap, settings.CHUNK_OVERLAP_MAX))

        # Ensure overlap is less than chunk size
        overlap = min(overlap, chunk_size // 2)

        return overlap

    def ensure_sentence_boundaries(self, chunks: List[str]) -> List[str]:
        """
        Ensure chunks don't break mid-sentence using NLTK sentence tokenizer.

        Args:
            chunks: List of text chunks that may have mid-sentence breaks

        Returns:
            List of chunks with complete sentences
        """
        from nltk.tokenize import sent_tokenize

        processed_chunks = []
        carry_over = ""

        for chunk in chunks:
            # Combine with carry-over from previous chunk
            full_text = carry_over + chunk

            # Tokenize into sentences
            try:
                sentences = sent_tokenize(full_text)
            except Exception as e:
                logger.warning(f"Sentence tokenization failed: {e}, using chunk as-is")
                processed_chunks.append(chunk)
                carry_over = ""
                continue

            if not sentences:
                carry_over = full_text
                continue

            # Check if last sentence is complete
            # A sentence is complete if it ends with sentence-ending punctuation
            last_sentence = sentences[-1]
            is_complete = bool(re.search(r'[.!?]\s*$', last_sentence))

            if is_complete:
                # All sentences are complete, join them
                processed_chunks.append(' '.join(sentences))
                carry_over = ""
            else:
                # Last sentence is incomplete, carry it over
                if len(sentences) > 1:
                    processed_chunks.append(' '.join(sentences[:-1]))
                    carry_over = sentences[-1] + " "
                else:
                    # Only one incomplete sentence, carry it all over
                    carry_over = full_text + " "

        # Add any remaining carry-over as final chunk
        if carry_over.strip():
            processed_chunks.append(carry_over.strip())

        logger.debug(
            f"Sentence boundary processing: {len(chunks)} chunks -> {len(processed_chunks)} chunks"
        )

        return processed_chunks

    def chunk_with_semantic_boundaries(
        self,
        pages: List,
        document_structure: Dict,
    ) -> List[Dict]:
        """
        Main chunking method with semantic boundaries, fallback logic, and metadata preservation.

        This method:
        1. Attempts semantic chunking using embeddings (if enabled)
        2. Falls back to adaptive RecursiveCharacterTextSplitter if semantic chunking fails
        3. Applies sentence-aware splitting to prevent mid-sentence breaks
        4. Extracts and preserves metadata for each chunk

        Args:
            pages: List of LangChain Document objects from PyPDFLoader
            document_structure: Document structure from detect_document_structure()

        Returns:
            List of dicts with 'content', 'page_number', and 'metadata' keys
        """
        chunk_data = []

        try:
            # Strategy 1: Try semantic chunking (if enabled)
            if settings.USE_SEMANTIC_CHUNKING:
                try:
                    logger.info("Attempting semantic chunking with embedding-based boundary detection")
                    semantic_chunker = self.create_semantic_chunker()
                    chunks = semantic_chunker.split_documents(pages)

                    # Process semantic chunks
                    for chunk in chunks:
                        text_content = chunk.page_content
                        page_number = chunk.metadata.get("page", 0) + 1

                        # Extract metadata
                        metadata = self.extract_metadata(text_content, page_number, document_structure)

                        chunk_data.append({
                            'content': text_content,
                            'page_number': page_number,
                            'metadata': metadata,
                        })

                    logger.info(f"Semantic chunking successful: created {len(chunk_data)} chunks")
                    return chunk_data

                except Exception as e:
                    logger.warning(
                        f"Semantic chunking failed: {e}. Falling back to adaptive splitter",
                        exc_info=True,
                    )

            # Strategy 2: Adaptive RecursiveCharacterTextSplitter (fallback or default)
            logger.info("Using adaptive RecursiveCharacterTextSplitter")

            # Group pages by detected content type for optimized splitting
            if settings.CHUNK_BY_CONTENT_TYPE:
                # Split each page with content-type-aware settings
                for page in pages:
                    text_content = page.page_content
                    page_number = page.metadata.get("page", 0) + 1
                    content_type = self.identify_content_type(text_content)

                    # Create adaptive splitter for this content type
                    splitter = self.create_adaptive_splitter(content_type)

                    # Split the page
                    page_chunks = splitter.split_text(text_content)

                    # Apply sentence boundary correction
                    page_chunks = self.ensure_sentence_boundaries(page_chunks)

                    # Extract metadata for each chunk
                    for chunk_text in page_chunks:
                        metadata = self.extract_metadata(chunk_text, page_number, document_structure)
                        chunk_data.append({
                            'content': chunk_text,
                            'page_number': page_number,
                            'metadata': metadata,
                        })
            else:
                # Use default splitter for all content
                splitter = self.create_adaptive_splitter('paragraph')
                chunks = splitter.split_documents(pages)

                # Apply sentence boundary correction
                chunk_texts = [chunk.page_content for chunk in chunks]
                chunk_texts = self.ensure_sentence_boundaries(chunk_texts)

                # Extract metadata
                for idx, chunk in enumerate(chunks):
                    if idx < len(chunk_texts):
                        text_content = chunk_texts[idx]
                        page_number = chunk.metadata.get("page", 0) + 1
                        metadata = self.extract_metadata(text_content, page_number, document_structure)

                        chunk_data.append({
                            'content': text_content,
                            'page_number': page_number,
                            'metadata': metadata,
                        })

            logger.info(f"Adaptive chunking successful: created {len(chunk_data)} chunks")

        except Exception as e:
            logger.error(f"All chunking strategies failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to chunk document: {str(e)}")

        return chunk_data


# --- Process-wide singleton -------------------------------------------------
_chunker: DocumentChunker | None = None


def get_chunker() -> DocumentChunker:
    """Return the process-wide DocumentChunker singleton."""
    global _chunker
    if _chunker is None:
        from .embedding_service import get_embedding_service

        _chunker = DocumentChunker(get_embedding_service())
    return _chunker
