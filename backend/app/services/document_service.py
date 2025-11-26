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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
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
                    chunk = DocumentChunk(
                        content=chunk_data['content'],
                        page_number=chunk_data['page_number'],
                        embedding=chunk_data['embedding'],
                        document_id=document_id,
                        position_data=None
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
            # Load PDF using LangChain
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)

            # Collect all chunk texts and metadata first
            chunk_data = []
            for chunk in chunks:
                text_content = chunk.page_content
                page_number = chunk.metadata.get("page", 0) + 1  # Convert to 1-indexed
                chunk_data.append({
                    'content': text_content,
                    'page_number': page_number
                })

            # Generate all embeddings in parallel batches (async, non-blocking)
            logger.info(f"Generating embeddings for {len(chunk_data)} chunks using parallel processing")
            texts = [data['content'] for data in chunk_data]
            embeddings = await self._generate_embeddings_parallel(
                texts,
                batch_size=50,
                max_concurrent=4
            )
            logger.info(f"Generated {len(embeddings)} embeddings in parallel")

            # Prepare chunk data with embeddings for parallel saving
            logger.info(f"Preparing {len(chunk_data)} chunks for parallel database writes")
            chunks_with_embeddings = []
            for i, data in enumerate(chunk_data):
                chunks_with_embeddings.append({
                    'content': data['content'],
                    'page_number': data['page_number'],
                    'embedding': embeddings[i]
                })

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

            return {
                "success": True,
                "message": "Document processed successfully",
                "chunks_processed": successful_chunks,
                "chunks_failed": failed_chunks
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

