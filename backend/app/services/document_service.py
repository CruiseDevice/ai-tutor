from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException
from typing import List, Dict, Optional
import tempfile
import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone
import logging
from ..models.document import Document, DocumentChunk
from ..models.conversation import Conversation
from ..config import settings
from .embedding_service import get_embedding_service
from .cache_service import get_cache_service
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import nltk
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _ensure_nltk_data():
    """
    Ensure NLTK punkt tokenizer is downloaded.
    Uses lru_cache to only run once per application lifecycle.
    """
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


class DocumentService:
    def __init__(self):
        # Initialize NLTK data (punkt tokenizer for sentence splitting)
        _ensure_nltk_data()

        self.s3_client = boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.bucket_name = settings.S3_PDFBUCKET_NAME
        self.embedding_service = get_embedding_service()

    async def _generate_embeddings_parallel(
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
        import asyncio
        from itertools import islice

        # Split texts into batches
        def chunks(iterable, size):
            iterator = iter(iterable)
            while True:
                batch = list(islice(iterator, size))
                if not batch:
                    break
                yield batch

        text_batches = list(chunks(texts, batch_size))
        all_embeddings = []

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch):
            async with semaphore:
                return await self.embedding_service.generate_batch_embeddings_async(batch)

        # Process all batches concurrently
        logger.info(f"Processing {len(text_batches)} batches with max {max_concurrent} concurrent operations")
        tasks = [process_batch(batch) for batch in text_batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)

        logger.info(f"Generated {len(all_embeddings)} embeddings in parallel")
        return all_embeddings

    async def _save_chunks_parallel(
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
        import asyncio
        from sqlalchemy.orm import sessionmaker
        from ..database import engine

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
                        position_data=metadata  # Store metadata in position_data JSONB field
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
        batches = [
            chunks_data[i:i + batch_size]
            for i in range(0, len(chunks_data), batch_size)
        ]

        logger.info(f"Saving {len(chunks_data)} chunks in {len(batches)} parallel batches")

        # Save batches concurrently
        results = await asyncio.gather(
            *[save_batch(batch) for batch in batches],
            return_exceptions=True
        )

        # Aggregate results
        total_success = sum(r[0] for r in results if not isinstance(r, Exception))
        total_errors = sum(r[1] for r in results if not isinstance(r, Exception))

        logger.info(f"Saved {total_success} chunks successfully, {total_errors} failed")

        return {
            "success": total_success,
            "errors": total_errors,
        }

    def _detect_document_structure(self, pages: List) -> Dict:
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
            'lists': []
        }

        import re

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
                            'type': 'all_caps'
                        })
                        continue

                    # Check for numbered sections (e.g., "1. Introduction", "Chapter 1", "Section 2.1")
                    if re.match(r'^(\d+\.|\d+\.\d+\.?|\w+\s+\d+|Chapter\s+\d+|Section\s+\d+)', line_stripped, re.IGNORECASE):
                        structure['headers'].append({
                            'page': page_idx,
                            'line': line_idx,
                            'text': line_stripped,
                            'type': 'numbered_section'
                        })
                        continue

                # Detect tables (aligned columns, multiple tabs/spaces, pipe characters)
                if '|' in line or '\t\t' in line or re.search(r'\s{4,}', line):
                    # Look for table-like patterns
                    structure['tables'].append({
                        'page': page_idx,
                        'line': line_idx,
                        'text': line_stripped
                    })
                    continue

                # Detect lists (bullet points, numbered items)
                if re.match(r'^[\•\-\*\◦\▪]\s+', line_stripped) or re.match(r'^\d+[\.\)]\s+', line_stripped):
                    structure['lists'].append({
                        'page': page_idx,
                        'line': line_idx,
                        'text': line_stripped
                    })

        # Detect section boundaries based on headers
        for header in structure['headers']:
            structure['sections'].append({
                'page': header['page'],
                'line': header['line'],
                'title': header['text']
            })

        logger.debug(
            f"Detected structure: {len(structure['headers'])} headers, "
            f"{len(structure['sections'])} sections, {len(structure['tables'])} table lines, "
            f"{len(structure['lists'])} list items"
        )

        return structure

    def _identify_content_type(self, text: str) -> str:
        """
        Classify text content type.

        Args:
            text: Text content to classify

        Returns:
            Content type: 'header', 'table', 'list', 'code', or 'paragraph'
        """
        import re

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
            if re.match(r'^(\d+\.|\d+\.\d+\.?|\w+\s+\d+|Chapter\s+\d+|Section\s+\d+)', text_stripped, re.IGNORECASE):
                return 'header'

        # Check for table content
        if '|' in text or '\t\t' in text or re.search(r'\s{4,}', text):
            # Count aligned spacing or pipe characters
            if text.count('|') >= 2 or len(re.findall(r'\s{4,}', text)) >= 2:
                return 'table'

        # Check for list items
        if re.match(r'^[\•\-\*\◦\▪]\s+', text_stripped) or re.match(r'^\d+[\.\)]\s+', text_stripped):
            return 'list'

        # Check for code (high density of special chars, indentation)
        lines = text.split('\n')
        if len(lines) > 1:
            indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
            if indented_lines / len(lines) > 0.5:  # More than 50% indented
                # Check for code-like characters
                special_chars = sum(1 for c in text if c in '{}[]()<>=;:')
                if special_chars / len(text) > 0.05:  # More than 5% special chars
                    return 'code'

        # Default to paragraph
        return 'paragraph'

    def _extract_metadata(self, chunk_text: str, page_number: int, document_structure: Dict) -> Dict:
        """
        Extract metadata from a chunk based on its content and document structure.

        Args:
            chunk_text: The text content of the chunk
            page_number: Page number (1-indexed)
            document_structure: Document structure from _detect_document_structure()

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
            'content_type': self._identify_content_type(chunk_text),
            'table_info': None,
            'sentence_count': 0,
            'char_count': len(chunk_text),
            'word_count': len(chunk_text.split())
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
            import re
            metadata['sentence_count'] = len(re.findall(r'[.!?]+', chunk_text))

        # Extract table info if content is a table
        if metadata['content_type'] == 'table':
            lines = chunk_text.split('\n')
            metadata['table_info'] = {
                'rows': len(lines),
                'has_pipes': '|' in chunk_text,
                'estimated_columns': max(line.count('|') for line in lines) + 1 if '|' in chunk_text else None
            }

        return metadata

    def _create_semantic_chunker(self):
        """
        Create a LangChain SemanticChunker using the existing embedding service.

        Returns:
            SemanticChunker configured with our embedding service and settings
        """
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain.embeddings.base import Embeddings

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
            breakpoint_threshold_amount=settings.SEMANTIC_CHUNKING_BREAKPOINT_THRESHOLD
        )

        logger.debug("Created SemanticChunker with embedding service adapter")
        return semantic_chunker

    def _create_adaptive_splitter(self, content_type: str = 'paragraph') -> RecursiveCharacterTextSplitter:
        """
        Create an adaptive RecursiveCharacterTextSplitter with structure-aware separators.

        Args:
            content_type: Type of content to optimize for ('header', 'paragraph', 'table', 'list', 'code')

        Returns:
            Configured RecursiveCharacterTextSplitter
        """
        # Get adaptive chunk size and overlap based on content type
        chunk_size = self._get_adaptive_chunk_size(content_type)
        chunk_overlap = self._calculate_adaptive_overlap(chunk_size, content_type)

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
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation sentences
                "? ",    # Question sentences
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Spaces
                "",      # Characters (fallback)
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

    def _get_adaptive_chunk_size(self, content_type: str) -> int:
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
            'header': 650,      # Headers: 500-800 chars (using mid-range)
            'paragraph': 1250,  # Paragraphs: 1000-1500 chars (using mid-range)
            'table': 1000,      # Tables: 800-1200 chars (using mid-range)
            'list': 1000,       # Lists: Similar to paragraphs
            'code': 1200,       # Code: Larger to preserve complete blocks
        }

        chunk_size = size_map.get(content_type, settings.CHUNK_SIZE_DEFAULT)

        # Clamp to configured min/max
        chunk_size = max(settings.CHUNK_SIZE_MIN, min(chunk_size, settings.CHUNK_SIZE_MAX))

        return chunk_size

    def _calculate_adaptive_overlap(self, chunk_size: int, content_type: str) -> int:
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

    def _ensure_sentence_boundaries(self, chunks: List[str]) -> List[str]:
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
            import re
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

    def _chunk_with_semantic_boundaries(
        self,
        pages: List,
        document_structure: Dict
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
            document_structure: Document structure from _detect_document_structure()

        Returns:
            List of dicts with 'content', 'page_number', and 'metadata' keys
        """
        chunk_data = []

        try:
            # Strategy 1: Try semantic chunking (if enabled)
            if settings.USE_SEMANTIC_CHUNKING:
                try:
                    logger.info("Attempting semantic chunking with embedding-based boundary detection")
                    semantic_chunker = self._create_semantic_chunker()
                    chunks = semantic_chunker.split_documents(pages)

                    # Process semantic chunks
                    for chunk in chunks:
                        text_content = chunk.page_content
                        page_number = chunk.metadata.get("page", 0) + 1

                        # Extract metadata
                        metadata = self._extract_metadata(text_content, page_number, document_structure)

                        chunk_data.append({
                            'content': text_content,
                            'page_number': page_number,
                            'metadata': metadata
                        })

                    logger.info(
                        f"Semantic chunking successful: created {len(chunk_data)} chunks"
                    )
                    return chunk_data

                except Exception as e:
                    logger.warning(
                        f"Semantic chunking failed: {e}. Falling back to adaptive splitter",
                        exc_info=True
                    )

            # Strategy 2: Adaptive RecursiveCharacterTextSplitter (fallback or default)
            logger.info("Using adaptive RecursiveCharacterTextSplitter")

            # Group pages by detected content type for optimized splitting
            if settings.CHUNK_BY_CONTENT_TYPE:
                # Split each page with content-type-aware settings
                for page in pages:
                    text_content = page.page_content
                    page_number = page.metadata.get("page", 0) + 1
                    content_type = self._identify_content_type(text_content)

                    # Create adaptive splitter for this content type
                    splitter = self._create_adaptive_splitter(content_type)

                    # Split the page
                    page_chunks = splitter.split_text(text_content)

                    # Apply sentence boundary correction
                    page_chunks = self._ensure_sentence_boundaries(page_chunks)

                    # Extract metadata for each chunk
                    for chunk_text in page_chunks:
                        metadata = self._extract_metadata(chunk_text, page_number, document_structure)
                        chunk_data.append({
                            'content': chunk_text,
                            'page_number': page_number,
                            'metadata': metadata
                        })
            else:
                # Use default splitter for all content
                splitter = self._create_adaptive_splitter('paragraph')
                chunks = splitter.split_documents(pages)

                # Apply sentence boundary correction
                chunk_texts = [chunk.page_content for chunk in chunks]
                chunk_texts = self._ensure_sentence_boundaries(chunk_texts)

                # Extract metadata
                for idx, chunk in enumerate(chunks):
                    if idx < len(chunk_texts):
                        text_content = chunk_texts[idx]
                        page_number = chunk.metadata.get("page", 0) + 1
                        metadata = self._extract_metadata(text_content, page_number, document_structure)

                        chunk_data.append({
                            'content': text_content,
                            'page_number': page_number,
                            'metadata': metadata
                        })

            logger.info(
                f"Adaptive chunking successful: created {len(chunk_data)} chunks"
            )

        except Exception as e:
            logger.error(f"All chunking strategies failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to chunk document: {str(e)}"
            )

        return chunk_data

    async def upload_to_s3(self, file: UploadFile, user_id: str) -> tuple[str, str]:
        """Upload PDF to S3 and return the URL and blob path."""
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{user_id}/{uuid.uuid4()}{file_extension}"

        try:
            # Read file content
            content = await file.read()

            # Validate file size after reading
            file_size = len(content)
            if file_size > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE / (1024 * 1024)}MB)"
                )

            await file.seek(0)  # Reset file pointer (though we already read it)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=unique_filename,
                Body=content,
                ContentType=file.content_type or 'application/pdf'
            )

            # Generate URL
            url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{unique_filename}"

            return url, unique_filename
        except HTTPException:
            raise
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

    def get_signed_url(self, blob_path: str, expiration: int = 3600) -> str:
        """Generate a signed URL for accessing a file in S3."""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': blob_path},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(e)}")

    async def create_document(
        self,
        db: Session,
        user_id: str,
        file: UploadFile,
        create_conversation: bool = True
    ) -> tuple[Document, Optional[Conversation]]:
        """Upload a document and create database records."""
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # File size validation happens in upload_to_s3 after reading content
        # Upload to S3
        url, blob_path = await self.upload_to_s3(file, user_id)

        # Get current timestamp
        now = datetime.now(timezone.utc)

        # Create document record
        document = Document(
            user_id=user_id,
            title=file.filename,
            url=url,
            blob_path=blob_path,
            updated_at=now
        )
        db.add(document)
        db.flush()  # Get the document ID without committing

        # Create conversation record if requested
        conversation = None
        if create_conversation:
            conversation = Conversation(
                user_id=user_id,
                document_id=document.id,
                updated_at=now
            )
            db.add(conversation)

        db.commit()
        db.refresh(document)
        if conversation:
            db.refresh(conversation)

        return document, conversation

    async def _extract_and_chunk_pages_streaming(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Stream PDF pages one at a time, chunking each page as it's extracted.

        This allows for progressive processing - chunks become searchable as soon as
        they're processed, rather than waiting for the entire document.

        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Yields:
            Tuple of (page_number, chunks_data) for each page
            chunks_data is a list of dicts with 'content' and 'page_number'
        """
        from pypdf import PdfReader

        logger.info(f"Starting streaming PDF extraction from {pdf_path}")

        # Create text splitter with semantic boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        # Read PDF
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        logger.info(f"PDF has {total_pages} pages, starting streaming extraction")

        # Process each page individually
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                # Extract text from this page
                page_text = page.extract_text()

                if not page_text or not page_text.strip():
                    logger.warning(f"Page {page_num} has no text content, skipping")
                    continue

                # Chunk the page text
                chunks = text_splitter.split_text(page_text)

                # Prepare chunk data
                page_chunks_data = []
                for chunk_text in chunks:
                    if chunk_text.strip():  # Only include non-empty chunks
                        page_chunks_data.append({
                            'content': chunk_text,
                            'page_number': page_num
                        })

                logger.info(f"Page {page_num}/{total_pages}: Extracted {len(page_chunks_data)} chunks")

                # Yield this page's chunks for processing
                yield page_num, page_chunks_data, total_pages

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}", exc_info=True)
                continue

        logger.info(f"Completed streaming extraction of {total_pages} pages")

    async def process_document_streaming(
        self,
        db: Session,
        document_id: str
    ) -> Dict:
        """
        Process a document using streaming approach - process pages as they're extracted.

        This provides faster time-to-first-chunk and better user experience as chunks
        become searchable progressively rather than all at once.

        Args:
            db: Database session
            document_id: ID of the document to process

        Returns:
            Dict with processing results
        """
        import requests
        import tempfile
        import os

        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get signed URL for the PDF
        signed_url = self.get_signed_url(document.blob_path)

        # Download PDF to temporary file
        response = requests.get(signed_url)
        if not response.ok:
            raise HTTPException(status_code=500, detail="Failed to download PDF from S3")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        total_chunks_processed = 0
        total_chunks_failed = 0

        try:
            logger.info(f"Starting streaming processing for document {document_id}")

            # Process pages as they're extracted
            async for page_num, page_chunks_data, total_pages in self._extract_and_chunk_pages_streaming(temp_file_path):
                if not page_chunks_data:
                    continue

                # Extract texts for embedding generation
                texts = [chunk['content'] for chunk in page_chunks_data]

                # Generate embeddings for this page's chunks in parallel
                logger.info(f"Generating embeddings for page {page_num} ({len(texts)} chunks)")
                embeddings = await self._generate_embeddings_parallel(
                    texts,
                    batch_size=50,
                    max_concurrent=4
                )

                # Prepare chunks with embeddings
                chunks_with_embeddings = []
                for i, chunk_data in enumerate(page_chunks_data):
                    chunks_with_embeddings.append({
                        'content': chunk_data['content'],
                        'page_number': chunk_data['page_number'],
                        'embedding': embeddings[i]
                    })

                # Save this page's chunks to database immediately
                logger.info(f"Saving page {page_num} chunks to database")
                result = await self._save_chunks_parallel(
                    db,
                    document_id,
                    chunks_with_embeddings,
                    batch_size=50
                )

                total_chunks_processed += result['success']
                total_chunks_failed += result['errors']

                # Log progress
                progress_percent = (page_num / total_pages) * 100
                logger.info(
                    f"Progress: {page_num}/{total_pages} pages ({progress_percent:.1f}%) | "
                    f"Chunks processed: {total_chunks_processed}, failed: {total_chunks_failed}"
                )

            # Invalidate cache for this document since chunks have been updated
            try:
                cache_service = await get_cache_service()
                await cache_service.invalidate_document_chunks(document_id)
                logger.info(f"Invalidated cache for document_id={document_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for document {document_id}: {e}")

            return {
                "success": True,
                "message": "Document processed successfully using streaming",
                "chunks_processed": total_chunks_processed,
                "chunks_failed": total_chunks_failed
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def process_document(
        self,
        db: Session,
        document_id: str
    ) -> Dict:
        """Process a document: extract text, chunk it, generate embeddings, and save to database."""
        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get signed URL for the PDF
        signed_url = self.get_signed_url(document.blob_path)

        # Download PDF to temporary file
        import requests
        response = requests.get(signed_url)
        if not response.ok:
            raise HTTPException(status_code=500, detail="Failed to download PDF from S3")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        try:
            # Extract images with Docling (if enabled)
            extracted_images = []
            if settings.ENABLE_IMAGE_EXTRACTION:
                try:
                    extracted_images = await self._extract_images_with_docling(
                        temp_file_path,
                        document_id,
                        document.user_id
                    )
                    logger.info(f"Extracted {len(extracted_images)} images from document")
                except Exception as e:
                    logger.error(f"Image extraction failed, continuing with text-only processing: {e}")
                    extracted_images = []

            # Load PDF using LangChain
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()

            # Detect document structure for metadata extraction
            logger.info("Analyzing document structure (headers, tables, lists, sections)")
            document_structure = self._detect_document_structure(pages)

            # Check if hierarchical chunking is enabled
            if settings.ENABLE_HIERARCHICAL_CHUNKING:
                logger.info("Hierarchical chunking enabled - processing with parent-child strategy")
                hierarchical_result = await self._process_with_hierarchical_chunking(
                    db,
                    document_id,
                    pages,
                    document_structure
                )

                # Invalidate cache for this document since chunks have been updated
                try:
                    cache_service = await get_cache_service()
                    await cache_service.invalidate_document_chunks(document_id)
                    logger.info(f"Invalidated cache for document_id={document_id}")
                except Exception as e:
                    logger.warning(f"Failed to invalidate cache for document {document_id}: {e}")

                # Return hierarchical processing results
                return {
                    "success": True,
                    "message": "Document processed successfully with hierarchical chunking",
                    "chunks_processed": hierarchical_result['total_chunks'],
                    "chunks_failed": hierarchical_result['failed_chunks'],
                    "parent_chunks": hierarchical_result['parent_chunks'],
                    "child_chunks": hierarchical_result['child_chunks'],
                    "relationships": hierarchical_result['relationships'],
                    "images_extracted": len(extracted_images),
                    "images_uploaded": sum(1 for img in extracted_images if img.get('s3_url')),
                    "images_captioned": sum(1 for img in extracted_images if img.get('caption')),
                    "chunking_mode": "hierarchical"
                }

            # Use advanced semantic chunking with metadata preservation (flat chunking mode)
            logger.info(
                f"Processing document with flat semantic chunking "
                f"(semantic={settings.USE_SEMANTIC_CHUNKING}, "
                f"content_type_aware={settings.CHUNK_BY_CONTENT_TYPE}, "
                f"preserve_metadata={settings.PRESERVE_METADATA})"
            )
            text_chunk_data = self._chunk_with_semantic_boundaries(pages, document_structure)

            # Create image chunks from extracted images
            image_chunk_data = []
            if extracted_images:
                logger.info(f"Creating image chunks from {len(extracted_images)} extracted images")
                image_chunk_data = self._create_image_chunks(extracted_images)
                logger.info(f"Created {len(image_chunk_data)} image chunks")

            # Combine text and image chunks
            all_chunk_data = text_chunk_data + image_chunk_data
            logger.info(
                f"Total chunks: {len(all_chunk_data)} "
                f"(text: {len(text_chunk_data)}, images: {len(image_chunk_data)})"
            )

            # Generate all embeddings in parallel batches (async, non-blocking)
            logger.info(f"Generating embeddings for {len(all_chunk_data)} chunks using parallel processing")
            texts = [data['content'] for data in all_chunk_data]
            embeddings = await self._generate_embeddings_parallel(
                texts,
                batch_size=50,
                max_concurrent=4
            )
            logger.info(f"Generated {len(embeddings)} embeddings in parallel")

            # Prepare chunk data with embeddings and metadata for parallel saving
            logger.info(f"Preparing {len(all_chunk_data)} chunks with metadata for parallel database writes")
            chunks_with_embeddings = []
            for i, data in enumerate(all_chunk_data):
                chunk_dict = {
                    'content': data['content'],
                    'page_number': data['page_number'],
                    'embedding': embeddings[i]
                }
                # Include metadata if available (from semantic chunking or image chunks)
                if 'metadata' in data and settings.PRESERVE_METADATA:
                    chunk_dict['metadata'] = data['metadata']
                # Include chunk_type if available (for image chunks)
                if 'chunk_type' in data:
                    chunk_dict['chunk_type'] = data['chunk_type']
                chunks_with_embeddings.append(chunk_dict)

            # Save chunks to database in parallel
            try:
                result = await self._save_chunks_parallel(
                    db,
                    document_id,
                    chunks_with_embeddings,
                    batch_size=50
                )
                successful_chunks = result['success']
                failed_chunks = result['errors']

            except Exception as e:
                # Rollback on critical error
                logger.error(f"Critical error during chunk processing: {e}", exc_info=True)
                db.rollback()
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process document chunks: {str(e)}"
                )

            # Invalidate cache for this document since chunks have been updated
            try:
                cache_service = await get_cache_service()
                await cache_service.invalidate_document_chunks(document_id)
                logger.info(f"Invalidated cache for document_id={document_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for document {document_id}: {e}")
                # Don't fail the request if cache invalidation fails

            # Calculate statistics
            text_chunks_count = len(text_chunk_data)
            image_chunks_count = len(image_chunk_data)
            images_captioned_count = sum(1 for img in extracted_images if img.get('caption'))

            logger.info(
                f"Document processing complete: "
                f"text_chunks={text_chunks_count}, "
                f"image_chunks={image_chunks_count}, "
                f"total_chunks={successful_chunks}, "
                f"images_extracted={len(extracted_images)}, "
                f"images_captioned={images_captioned_count}"
            )

            return {
                "success": True,
                "message": "Document processed successfully",
                "chunks_processed": successful_chunks,
                "chunks_failed": failed_chunks,
                "text_chunks": text_chunks_count,
                "image_chunks": image_chunks_count,
                "total_chunks": text_chunks_count + image_chunks_count,
                "images_extracted": len(extracted_images),
                "images_uploaded": sum(1 for img in extracted_images if img.get('s3_url')),
                "images_captioned": images_captioned_count
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def list_documents(self, db: Session, user_id: str) -> List[Document]:
        """List all documents for a user."""
        documents = db.query(Document).filter(Document.user_id == user_id).order_by(Document.created_at.desc()).all()
        return documents

    async def delete_document(self, db: Session, document_id: str, user_id: str) -> bool:
        """Delete a document and all associated data."""
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()

        if not document:
            return False

        # Delete from S3
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=document.blob_path
            )
        except ClientError as e:
            logger.warning(f"Failed to delete document from S3 (blob_path: {document.blob_path}): {e}")

        # Invalidate cache for this document before deletion
        try:
            cache_service = await get_cache_service()
            await cache_service.invalidate_document_chunks(document_id)
            logger.info(f"Invalidated cache for deleted document_id={document_id}")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for document {document_id}: {e}")
            # Don't fail the deletion if cache invalidation fails

        # Delete from database (cascades to chunks, conversation, messages)
        db.delete(document)
        db.commit()

        return True

    async def _generate_image_captions(self, images: List) -> List:
        """
        Generate captions for extracted images using GPT-4o Vision.

        Args:
            images: List of ExtractedImage objects

        Returns:
            List of ImageCaption objects (or None for failed captions)
        """
        from .vision_service import VisionService

        try:
            vision_service = VisionService()

            # Prepare images for caption generation
            image_data_list = [
                {
                    'image_data': img.image_data,
                    'page_number': img.page_number
                }
                for img in images
            ]

            # Generate captions in parallel batches
            captions = await vision_service.batch_generate_captions(image_data_list)

            return captions

        except Exception as e:
            logger.error(f"Failed to generate image captions: {e}", exc_info=True)
            # Return list of Nones if caption generation fails
            return [None] * len(images)

    def _caption_to_dict(self, caption) -> Dict:
        """
        Convert ImageCaption object to dictionary for storage.

        Args:
            caption: ImageCaption object

        Returns:
            Dictionary representation
        """
        if caption is None:
            return None

        return {
            'short_caption': caption.short_caption,
            'detailed_description': caption.detailed_description,
            'ocr_text': caption.ocr_text,
            'tags': caption.tags
        }

    async def _upload_image_to_s3(
        self,
        image_data: bytes,
        document_id: str,
        user_id: str,
        page_number: int,
        image_index: int
    ) -> tuple[str, str]:
        """
        Upload an extracted image to S3.

        Args:
            image_data: Image bytes
            document_id: Document UUID
            user_id: User ID
            page_number: Page number
            image_index: Image index on the page

        Returns:
            Tuple of (s3_url, s3_key)
        """
        from .docling_service import DoclingService

        docling_service = DoclingService()
        return docling_service.save_image_to_s3(
            image_data,
            document_id,
            user_id,
            page_number,
            image_index
        )

    def _build_image_chunk_text(self, caption: Dict, page_number: int) -> str:
        """
        Build searchable text content for an image chunk using its caption.

        Args:
            caption: Caption dictionary with short_caption, detailed_description, ocr_text, tags
            page_number: Page number where image appears

        Returns:
            Formatted text string for embedding generation
        """
        if not caption:
            return f"Image on page {page_number}: No description available"

        parts = [f"Image on page {page_number}: {caption.get('short_caption', 'No caption')}"]

        # Add detailed description
        if caption.get('detailed_description'):
            parts.append(f"\nDetails: {caption['detailed_description']}")

        # Add OCR text if available
        if caption.get('ocr_text'):
            parts.append(f"\nText in image: {caption['ocr_text']}")

        # Add tags
        if caption.get('tags'):
            tags_str = ', '.join(caption['tags'])
            parts.append(f"\nTags: {tags_str}")

        return '\n'.join(parts)

    def _create_image_chunks(self, uploaded_images: List[Dict]) -> List[Dict]:
        """
        Create image chunk representations from uploaded images with captions.

        Args:
            uploaded_images: List of image metadata dicts with caption, s3_key, page_number, etc.

        Returns:
            List of chunk dicts with 'content', 'page_number', 'metadata' for embedding generation
        """
        image_chunks = []

        for img_data in uploaded_images:
            caption = img_data.get('caption')
            page_number = img_data.get('page_number', 1)

            # Build text content for embedding
            chunk_text = self._build_image_chunk_text(caption, page_number)

            # Prepare metadata for position_data JSONB field
            metadata = {
                'bbox': img_data.get('bbox', (0, 0, 0, 0)),
                'image_s3_key': img_data.get('s3_key'),
                'image_s3_url': img_data.get('s3_url'),
                'image_format': img_data.get('image_format', 'png'),
                'image_index': img_data.get('image_index', 0),
                'content_type': 'image',
            }

            # Add caption details to metadata
            if caption:
                metadata.update({
                    'short_caption': caption.get('short_caption'),
                    'detailed_description': caption.get('detailed_description'),
                    'ocr_text': caption.get('ocr_text'),
                    'tags': caption.get('tags', [])
                })

            image_chunk = {
                'content': chunk_text,
                'page_number': page_number,
                'metadata': metadata,
                'chunk_type': 'image'  # Mark as image chunk
            }

            image_chunks.append(image_chunk)

        logger.info(f"Created {len(image_chunks)} image chunks for embedding")
        return image_chunks

    async def _extract_images_with_docling(self, pdf_path: str, document_id: str, user_id: str) -> List[Dict]:
        """
        Extract images from PDF using Docling and upload to S3.

        Args:
            pdf_path: Path to the PDF file
            document_id: Document UUID
            user_id: User ID

        Returns:
            List of dicts with image metadata and S3 URLs
        """
        from .docling_service import DoclingService

        if not settings.ENABLE_IMAGE_EXTRACTION:
            logger.info("Image extraction disabled via settings")
            return []

        try:
            logger.info(f"Starting image extraction with Docling for document {document_id}")
            docling_service = DoclingService()

            # Extract document content
            doc = docling_service.extract_document_content(pdf_path)

            # Extract images
            images = docling_service.extract_images(doc, pdf_path)

            if not images:
                logger.info("No images found in document")
                return []

            # Limit number of images to process
            if len(images) > settings.MAX_IMAGES_PER_DOCUMENT:
                logger.warning(
                    f"Document has {len(images)} images, limiting to {settings.MAX_IMAGES_PER_DOCUMENT}"
                )
                images = images[:settings.MAX_IMAGES_PER_DOCUMENT]

            # Generate captions for images using GPT-4o Vision
            logger.info(f"Generating captions for {len(images)} images")
            captions = await self._generate_image_captions(images)

            # Upload images to S3 and collect metadata with captions
            uploaded_images = []
            for idx, img in enumerate(images):
                try:
                    # Upload to S3
                    s3_url, s3_key = await self._upload_image_to_s3(
                        img.image_data,
                        document_id,
                        user_id,
                        img.page_number,
                        idx
                    )

                    # Get caption for this image
                    caption = captions[idx] if idx < len(captions) else None

                    # Create image metadata with caption
                    image_metadata = {
                        'page_number': img.page_number,
                        'bbox': img.bbox,
                        's3_key': s3_key,
                        's3_url': s3_url,
                        'image_format': img.image_format,
                        'image_index': idx,
                        'caption': self._caption_to_dict(caption) if caption else None
                    }

                    uploaded_images.append(image_metadata)
                    logger.debug(f"Uploaded image {idx} from page {img.page_number} with caption")

                except Exception as e:
                    logger.error(f"Failed to upload image {idx}: {e}", exc_info=True)
                    # Continue with other images
                    continue

            # Count successful captions
            captioned_count = sum(1 for img in uploaded_images if img.get('caption'))

            logger.info(
                f"Successfully extracted and uploaded {len(uploaded_images)} images "
                f"out of {len(images)} total images ({captioned_count} with captions)"
            )

            return uploaded_images

        except Exception as e:
            logger.error(f"Image extraction failed: {e}", exc_info=True)

            # Graceful degradation - don't fail entire document processing
            if settings.DOCLING_FALLBACK_TO_PYPDF:
                logger.info("Falling back to text-only processing due to image extraction failure")
                return []
            else:
                raise

    async def _process_with_hierarchical_chunking(
        self,
        db: Session,
        document_id: str,
        pages: List,
        document_structure: Dict
    ) -> Dict:
        """
        Process document using hierarchical chunking strategy.

        This method:
        1. Creates parent chunks (large, for context)
        2. Creates child chunks from each parent (small, for precise retrieval)
        3. Generates embeddings for both parent and child chunks
        4. Saves all chunks and relationships to database

        Args:
            db: Database session
            document_id: ID of document being processed
            pages: List of LangChain Document objects from PyPDFLoader
            document_structure: Document structure from _detect_document_structure()

        Returns:
            Dict with processing statistics
        """
        from .hierarchical_chunking_service import HierarchicalChunkingService, prepare_chunks_for_storage
        from ..models.document import ParentChildRelationship

        logger.info("Processing document with hierarchical chunking")

        # Initialize hierarchical chunking service
        hierarchical_service = HierarchicalChunkingService()

        # Convert LangChain pages to simple dict format
        page_data_list = []
        for page in pages:
            page_data = {
                'content': page.page_content,
                'page_number': page.metadata.get('page', 0) + 1,
                'metadata': self._extract_metadata(
                    page.page_content,
                    page.metadata.get('page', 0) + 1,
                    document_structure
                )
            }
            page_data_list.append(page_data)

        # Process document hierarchically
        parent_chunks, child_chunks, relationships = hierarchical_service.process_document_hierarchically(
            page_data_list
        )

        logger.info(
            f"Hierarchical chunking complete: {len(parent_chunks)} parents, "
            f"{len(child_chunks)} children, {len(relationships)} relationships"
        )

        # Generate embeddings for parent chunks
        logger.info(f"Generating embeddings for {len(parent_chunks)} parent chunks")
        parent_texts = [chunk['content'] for chunk in parent_chunks]
        parent_embeddings = await self._generate_embeddings_parallel(
            parent_texts,
            batch_size=50,
            max_concurrent=4
        )

        # Generate embeddings for child chunks
        logger.info(f"Generating embeddings for {len(child_chunks)} child chunks")
        child_texts = [chunk['content'] for chunk in child_chunks]
        child_embeddings = await self._generate_embeddings_parallel(
            child_texts,
            batch_size=50,
            max_concurrent=4
        )

        # Prepare chunks for storage (add embeddings)
        prepared_parents, prepared_children = prepare_chunks_for_storage(
            parent_chunks,
            child_chunks,
            parent_embeddings,
            child_embeddings
        )

        # Save parent chunks to database
        logger.info(f"Saving {len(prepared_parents)} parent chunks to database")
        parent_result = await self._save_chunks_parallel(
            db,
            document_id,
            prepared_parents,
            batch_size=50
        )

        # Save child chunks to database
        logger.info(f"Saving {len(prepared_children)} child chunks to database")
        child_result = await self._save_chunks_parallel(
            db,
            document_id,
            prepared_children,
            batch_size=50
        )

        # Save parent-child relationships
        logger.info(f"Saving {len(relationships)} parent-child relationships")
        saved_relationships = await self._save_parent_child_relationships(
            db,
            relationships
        )

        # Calculate statistics
        total_chunks_saved = parent_result['success'] + child_result['success']
        total_chunks_failed = parent_result['errors'] + child_result['errors']

        logger.info(
            f"Hierarchical processing complete: "
            f"parents={parent_result['success']}, "
            f"children={child_result['success']}, "
            f"relationships={saved_relationships}, "
            f"total_saved={total_chunks_saved}, "
            f"failed={total_chunks_failed}"
        )

        return {
            'success': True,
            'parent_chunks': parent_result['success'],
            'child_chunks': child_result['success'],
            'relationships': saved_relationships,
            'total_chunks': total_chunks_saved,
            'failed_chunks': total_chunks_failed
        }

    async def _save_parent_child_relationships(
        self,
        db: Session,
        relationships: List[Dict]
    ) -> int:
        """
        Save parent-child relationships to database.

        Args:
            db: Database session
            relationships: List of relationship dicts with parent_chunk_id, child_chunk_id, child_index

        Returns:
            Number of relationships saved
        """
        from ..models.document import ParentChildRelationship

        try:
            saved_count = 0

            for rel_data in relationships:
                relationship = ParentChildRelationship(
                    id=rel_data['id'],
                    parent_chunk_id=rel_data['parent_chunk_id'],
                    child_chunk_id=rel_data['child_chunk_id'],
                    child_index=rel_data['child_index']
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



