"""Document ingestion orchestration + CRUD.

Historically this module held the entire ingestion pipeline: S3 I/O, text
chunking, image/table extraction, parallel embedding/DB writes, and the
orchestration that wires them together. Those concerns now live in focused
services that this module composes:

- `StorageService`           — S3 object I/O (PDFs + extracted images).
- `DocumentChunker`          — text chunking (structure detection, semantic + adaptive splitters).
- `MediaExtractor`           — image/table extraction, captioning, and chunk shaping.
- `IngestionPipeline`        — parallel embedding generation, batched DB writes,
                                sentence extraction, parent-child relationships.

`DocumentService` keeps the orchestration methods (`process_document`,
`process_document_streaming`, hierarchical chunking) and the CRUD surface
(`create_document`, `list_documents`, `delete_document`, `get_signed_url`).
Thin delegating shims preserve the original method names so external callers
(`api/documents.py`, `api/conversations.py`, `workers/document_jobs.py`, and
the test suite) keep working without import changes.
"""
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from ..config import settings
from ..models.conversation import Conversation
from ..models.document import Document
from .cache_service import get_cache_service
from .chunking_service import DocumentChunker, get_chunker
from .embedding_service import get_embedding_service
from .ingestion_pipeline import IngestionPipeline, get_ingestion_pipeline
from .media_extraction_service import MediaExtractor, get_media_extractor
from .storage_service import get_storage_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Compose the ingestion subsystems and own document CRUD."""

    def __init__(self):
        # Wire collaborators as singletons so PDF upload + chat query paths
        # share one boto3 client, one embedding service, etc.
        self.storage = get_storage_service()
        self.chunker: DocumentChunker = get_chunker()
        self.media: MediaExtractor = get_media_extractor()
        self.pipeline: IngestionPipeline = get_ingestion_pipeline()
        self.embedding_service = get_embedding_service()
        # Back-compat: original attribute exposed the UnstructuredService (or None).
        self.unstructured_service = self.media.unstructured_service

    # --- Backward-compatible accessors --------------------------------------
    # api/documents.py streams objects directly through the boto3 client
    # (it needs a streaming-body response), so expose the client + bucket.
    @property
    def s3_client(self):
        return self.storage.s3_client

    @property
    def bucket_name(self) -> str:
        return self.storage.bucket_name

    async def upload_to_s3(self, file: UploadFile, user_id: str) -> tuple[str, str]:
        """Upload PDF to S3 and return the URL and blob path. Delegates to StorageService."""
        return await self.storage.upload_pdf(file, user_id)

    def get_signed_url(self, blob_path: str, expiration: int = 3600) -> str:
        """Generate a signed URL for accessing a file in S3. Delegates to StorageService."""
        return self.storage.get_signed_url(blob_path, expiration)

    # --- Chunking shims (delegates to DocumentChunker) ----------------------
    def _detect_document_structure(self, pages: List) -> Dict:
        return self.chunker.detect_document_structure(pages)

    def _extract_metadata(self, chunk_text: str, page_number: int, document_structure: Dict) -> Dict:
        return self.chunker.extract_metadata(chunk_text, page_number, document_structure)

    def _chunk_with_semantic_boundaries(self, pages: List, document_structure: Dict) -> List[Dict]:
        return self.chunker.chunk_with_semantic_boundaries(pages, document_structure)

    # --- Media shims (delegates to MediaExtractor) --------------------------
    def _create_image_chunks(self, uploaded_images: List[Dict]) -> List[Dict]:
        return self.media.create_image_chunks(uploaded_images)

    def _create_table_chunks(self, tables: List[Dict]) -> List[Dict]:
        return self.media.create_table_chunks(tables)

    async def _extract_images_with_unstructured(
        self, pdf_path: str, document_id: str, user_id: str
    ) -> List[Dict]:
        return await self.media.extract_images(pdf_path, document_id, user_id)

    async def _extract_tables_with_unstructured(
        self, pdf_path: str, document_id: str, user_id: str
    ) -> List[Dict]:
        return await self.media.extract_tables(pdf_path, document_id, user_id)

    # --- Pipeline shims (delegates to IngestionPipeline) --------------------
    async def _generate_embeddings_parallel(
        self, texts: List[str], batch_size: int = 50, max_concurrent: int = 4
    ) -> List[List[float]]:
        return await self.pipeline.generate_embeddings_parallel(texts, batch_size, max_concurrent)

    async def _save_chunks_parallel(
        self, db: Session, document_id: str, chunks_data: List[Dict], batch_size: int = 50
    ) -> Dict[str, int]:
        return await self.pipeline.save_chunks_parallel(db, document_id, chunks_data, batch_size)

    async def _extract_and_save_sentences(
        self, db: Session, document_id: str, pages: List, course_id: str = None
    ) -> Dict[str, int]:
        return await self.pipeline.extract_and_save_sentences(db, document_id, pages, course_id)

    async def _save_parent_child_relationships(
        self, db: Session, relationships: List[Dict]
    ) -> int:
        return await self.pipeline.save_parent_child_relationships(db, relationships)

    # --- CRUD ----------------------------------------------------------------
    async def create_document(
        self,
        db: Session,
        user_id: str,
        file: UploadFile,
        create_conversation: bool = True,
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
            updated_at=now,
        )
        db.add(document)
        db.flush()  # Get the document ID without committing

        # Create conversation record if requested
        conversation = None
        if create_conversation:
            conversation = Conversation(
                user_id=user_id,
                document_id=document.id,
                updated_at=now,
            )
            db.add(conversation)

        db.commit()
        db.refresh(document)
        if conversation:
            db.refresh(conversation)

        return document, conversation

    def list_documents(self, db: Session, user_id: str) -> List[Document]:
        """List all documents for a user."""
        documents = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .all()
        )
        return documents

    async def delete_document(self, db: Session, document_id: str, user_id: str) -> bool:
        """Delete a document and all associated data."""
        document = (
            db.query(Document)
            .filter(Document.id == document_id, Document.user_id == user_id)
            .first()
        )

        if not document:
            return False

        # Delete from S3 (best-effort)
        self.storage.delete_object(document.blob_path)

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

    # --- Streaming ingestion -------------------------------------------------
    async def _extract_and_chunk_pages_streaming(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
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
                            'page_number': page_num,
                        })

                logger.info(
                    f"Page {page_num}/{total_pages}: Extracted {len(page_chunks_data)} chunks"
                )

                # Yield this page's chunks for processing
                yield page_num, page_chunks_data, total_pages

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}", exc_info=True)
                continue

        logger.info(f"Completed streaming extraction of {total_pages} pages")

    async def process_document_streaming(
        self,
        db: Session,
        document_id: str,
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
        extracted_images = []
        extracted_tables = []

        try:
            logger.info(f"Starting streaming processing for document {document_id}")

            # Extract images with Unstructured (if enabled)
            if settings.ENABLE_IMAGE_EXTRACTION and settings.USE_UNSTRUCTURED:
                try:
                    extracted_images = await self._extract_images_with_unstructured(
                        temp_file_path, document_id, document.user_id
                    )
                    logger.info(f"Extracted {len(extracted_images)} images from document")
                except Exception as e:
                    logger.error(
                        f"Image extraction failed, continuing with text-only processing: {e}"
                    )
                    extracted_images = []

            # Extract tables with Unstructured (if enabled)
            if settings.ENABLE_TABLE_EXTRACTION and settings.USE_UNSTRUCTURED:
                try:
                    extracted_tables = await self._extract_tables_with_unstructured(
                        temp_file_path, document_id, document.user_id
                    )
                    logger.info(f"Extracted {len(extracted_tables)} tables from document")
                except Exception as e:
                    logger.error(f"Table extraction failed, continuing without tables: {e}")
                    extracted_tables = []

            # Process pages as they're extracted
            async for page_num, page_chunks_data, total_pages in self._extract_and_chunk_pages_streaming(
                temp_file_path
            ):
                if not page_chunks_data:
                    continue

                # Extract texts for embedding generation
                texts = [chunk['content'] for chunk in page_chunks_data]

                # Generate embeddings for this page's chunks in parallel
                logger.info(f"Generating embeddings for page {page_num} ({len(texts)} chunks)")
                embeddings = await self._generate_embeddings_parallel(
                    texts, batch_size=50, max_concurrent=4
                )

                # Prepare chunks with embeddings
                chunks_with_embeddings = []
                for i, chunk_data in enumerate(page_chunks_data):
                    chunks_with_embeddings.append({
                        'content': chunk_data['content'],
                        'page_number': chunk_data['page_number'],
                        'embedding': embeddings[i],
                    })

                # Save this page's chunks to database immediately
                logger.info(f"Saving page {page_num} chunks to database")
                result = await self._save_chunks_parallel(
                    db, document_id, chunks_with_embeddings, batch_size=50
                )

                total_chunks_processed += result['success']
                total_chunks_failed += result['errors']

                # Log progress
                progress_percent = (page_num / total_pages) * 100
                logger.info(
                    f"Progress: {page_num}/{total_pages} pages ({progress_percent:.1f}%) | "
                    f"Chunks processed: {total_chunks_processed}, failed: {total_chunks_failed}"
                )

            # Process and save image chunks (if any images were extracted)
            image_chunks_processed = 0
            if extracted_images:
                try:
                    image_chunks = self._create_image_chunks(extracted_images)
                    if image_chunks:
                        # Generate embeddings for image chunks
                        image_texts = [chunk['content'] for chunk in image_chunks]
                        image_embeddings = await self._generate_embeddings_parallel(
                            image_texts, batch_size=50, max_concurrent=4
                        )

                        # Add embeddings to chunks
                        for i, chunk in enumerate(image_chunks):
                            chunk['embedding'] = image_embeddings[i]

                        # Save image chunks
                        image_result = await self._save_chunks_parallel(
                            db, document_id, image_chunks, batch_size=50
                        )
                        image_chunks_processed = image_result['success']
                        total_chunks_processed += image_chunks_processed
                        logger.info(f"Saved {image_chunks_processed} image chunks")
                except Exception as e:
                    logger.error(f"Failed to save image chunks: {e}", exc_info=True)

            # Process and save table chunks (if any tables were extracted)
            table_chunks_processed = 0
            if extracted_tables:
                try:
                    table_chunks = self._create_table_chunks(extracted_tables)
                    if table_chunks:
                        # Generate embeddings for table chunks
                        table_texts = [chunk['content'] for chunk in table_chunks]
                        table_embeddings = await self._generate_embeddings_parallel(
                            table_texts, batch_size=50, max_concurrent=4
                        )

                        # Add embeddings to chunks
                        for i, chunk in enumerate(table_chunks):
                            chunk['embedding'] = table_embeddings[i]

                        # Save table chunks
                        table_result = await self._save_chunks_parallel(
                            db, document_id, table_chunks, batch_size=50
                        )
                        table_chunks_processed = table_result['success']
                        total_chunks_processed += table_chunks_processed
                        logger.info(f"Saved {table_chunks_processed} table chunks")
                except Exception as e:
                    logger.error(f"Failed to save table chunks: {e}", exc_info=True)

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
                "chunks_failed": total_chunks_failed,
                "images_extracted": len(extracted_images),
                "images_uploaded": sum(1 for img in extracted_images if img.get('s3_url')),
                "images_captioned": sum(1 for img in extracted_images if img.get('caption')),
                "tables_extracted": len(extracted_tables),
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def process_document(
        self,
        db: Session,
        document_id: str,
    ) -> Dict:
        """Process a document: extract text, chunk it, generate embeddings, and save to database."""
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

        try:
            # Extract images with Unstructured (if enabled)
            extracted_images = []
            if settings.ENABLE_IMAGE_EXTRACTION and settings.USE_UNSTRUCTURED:
                try:
                    extracted_images = await self._extract_images_with_unstructured(
                        temp_file_path, document_id, document.user_id
                    )
                    logger.info(f"Extracted {len(extracted_images)} images from document")
                except Exception as e:
                    logger.error(
                        f"Image extraction failed, continuing with text-only processing: {e}"
                    )
                    extracted_images = []

            # Extract tables with Unstructured (if enabled)
            extracted_tables = []
            if settings.ENABLE_TABLE_EXTRACTION and settings.USE_UNSTRUCTURED:
                try:
                    extracted_tables = await self._extract_tables_with_unstructured(
                        temp_file_path, document_id, document.user_id
                    )
                    logger.info(f"Extracted {len(extracted_tables)} tables from document")
                except Exception as e:
                    logger.error(f"Table extraction failed, continuing without tables: {e}")
                    extracted_tables = []

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
                    db, document_id, pages, document_structure
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
                    "tables_extracted": len(extracted_tables),
                    "chunking_mode": "hierarchical",
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

            # Create table chunks from extracted tables
            table_chunk_data = []
            if extracted_tables:
                logger.info(f"Creating table chunks from {len(extracted_tables)} extracted tables")
                table_chunk_data = self._create_table_chunks(extracted_tables)
                logger.info(f"Created {len(table_chunk_data)} table chunks")

            # Combine text, image, and table chunks
            all_chunk_data = text_chunk_data + image_chunk_data + table_chunk_data
            logger.info(
                f"Total chunks: {len(all_chunk_data)} "
                f"(text: {len(text_chunk_data)}, images: {len(image_chunk_data)}, "
                f"tables: {len(table_chunk_data)})"
            )

            # Generate all embeddings in parallel batches (async, non-blocking)
            logger.info(
                f"Generating embeddings for {len(all_chunk_data)} chunks using parallel processing"
            )
            texts = [data['content'] for data in all_chunk_data]
            embeddings = await self._generate_embeddings_parallel(
                texts, batch_size=50, max_concurrent=4
            )
            logger.info(f"Generated {len(embeddings)} embeddings in parallel")

            # Prepare chunk data with embeddings and metadata for parallel saving
            logger.info(
                f"Preparing {len(all_chunk_data)} chunks with metadata for parallel database writes"
            )
            chunks_with_embeddings = []
            for i, data in enumerate(all_chunk_data):
                chunk_dict = {
                    'content': data['content'],
                    'page_number': data['page_number'],
                    'embedding': embeddings[i],
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
                    db, document_id, chunks_with_embeddings, batch_size=50
                )
                successful_chunks = result['success']
                failed_chunks = result['errors']

            except Exception as e:
                # Rollback on critical error
                logger.error(f"Critical error during chunk processing: {e}", exc_info=True)
                db.rollback()
                raise HTTPException(
                    status_code=500, detail=f"Failed to process document chunks: {str(e)}"
                )

            # Invalidate cache for this document since chunks have been updated
            try:
                cache_service = await get_cache_service()
                await cache_service.invalidate_document_chunks(document_id)
                logger.info(f"Invalidated cache for document_id={document_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for document {document_id}: {e}")
                # Don't fail the request if cache invalidation fails

            # Extract and save critical sentences
            sentence_stats = await self._extract_and_save_sentences(
                db, document_id, pages, course_id=document.course_id
            )

            # Calculate statistics
            text_chunks_count = len(text_chunk_data)
            image_chunks_count = len(image_chunk_data)
            table_chunks_count = len(table_chunk_data)
            images_captioned_count = sum(1 for img in extracted_images if img.get('caption'))

            logger.info(
                f"Document processing complete: "
                f"text_chunks={text_chunks_count}, "
                f"image_chunks={image_chunks_count}, "
                f"table_chunks={table_chunks_count}, "
                f"total_chunks={successful_chunks}, "
                f"sentences={sentence_stats['sentences_saved']}, "
                f"images_extracted={len(extracted_images)}, "
                f"images_captioned={images_captioned_count}, "
                f"tables_extracted={len(extracted_tables)}"
            )

            return {
                "success": True,
                "message": "Document processed successfully",
                "chunks_processed": successful_chunks,
                "chunks_failed": failed_chunks,
                "text_chunks": text_chunks_count,
                "image_chunks": image_chunks_count,
                "table_chunks": table_chunks_count,
                "total_chunks": text_chunks_count + image_chunks_count + table_chunks_count,
                "images_extracted": len(extracted_images),
                "images_uploaded": sum(1 for img in extracted_images if img.get('s3_url')),
                "images_captioned": images_captioned_count,
                "tables_extracted": len(extracted_tables),
                "sentences_extracted": sentence_stats['sentences_extracted'],
                "sentences_saved": sentence_stats['sentences_saved'],
                "chunking_mode": "flat",
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def _process_with_hierarchical_chunking(
        self,
        db: Session,
        document_id: str,
        pages: List,
        document_structure: Dict,
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
        from .hierarchical_chunking_service import (
            HierarchicalChunkingService,
            prepare_chunks_for_storage,
        )

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
                    document_structure,
                ),
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
            parent_texts, batch_size=50, max_concurrent=4
        )

        # Generate embeddings for child chunks
        logger.info(f"Generating embeddings for {len(child_chunks)} child chunks")
        child_texts = [chunk['content'] for chunk in child_chunks]
        child_embeddings = await self._generate_embeddings_parallel(
            child_texts, batch_size=50, max_concurrent=4
        )

        # Prepare chunks for storage (add embeddings)
        prepared_parents, prepared_children = prepare_chunks_for_storage(
            parent_chunks, child_chunks, parent_embeddings, child_embeddings
        )

        # Save parent chunks to database
        logger.info(f"Saving {len(prepared_parents)} parent chunks to database")
        parent_result = await self._save_chunks_parallel(
            db, document_id, prepared_parents, batch_size=50
        )

        # Save child chunks to database
        logger.info(f"Saving {len(prepared_children)} child chunks to database")
        child_result = await self._save_chunks_parallel(
            db, document_id, prepared_children, batch_size=50
        )

        # Save parent-child relationships
        logger.info(f"Saving {len(relationships)} parent-child relationships")
        saved_relationships = await self._save_parent_child_relationships(db, relationships)

        # Extract and save critical sentences
        # Get document to access course_id
        document = db.query(Document).filter(Document.id == document_id).first()
        sentence_stats = await self._extract_and_save_sentences(
            db,
            document_id,
            pages,
            course_id=document.course_id if document else None,
        )

        # Calculate statistics
        total_chunks_saved = parent_result['success'] + child_result['success']
        total_chunks_failed = parent_result['errors'] + child_result['errors']

        logger.info(
            f"Hierarchical processing complete: "
            f"parents={parent_result['success']}, "
            f"children={child_result['success']}, "
            f"relationships={saved_relationships}, "
            f"sentences={sentence_stats['sentences_saved']}, "
            f"total_saved={total_chunks_saved}, "
            f"failed={total_chunks_failed}"
        )

        return {
            'success': True,
            'parent_chunks': parent_result['success'],
            'child_chunks': child_result['success'],
            'relationships': saved_relationships,
            'total_chunks': total_chunks_saved,
            'failed_chunks': total_chunks_failed,
            'sentences_extracted': sentence_stats['sentences_extracted'],
            'sentences_saved': sentence_stats['sentences_saved'],
        }
