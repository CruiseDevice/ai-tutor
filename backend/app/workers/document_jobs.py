"""
Background jobs for document processing.
"""
import logging
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services.document_service import DocumentService
from app.models.document import Document

logger = logging.getLogger(__name__)


async def process_document_job(
    ctx: Dict[str, Any],
    document_id: str,
) -> Dict[str, Any]:
    """
    Background job to process a document (extract text, chunk, embed).

    This job runs in the background worker and processes a PDF document:
    1. Downloads PDF from S3
    2. Extracts text and chunks it
    3. Generates embeddings for each chunk
    4. Saves chunks to database
    5. Updates document status

    Args:
        ctx: Worker context (contains shared resources like db_session_factory)
        document_id: ID of document to process

    Returns:
        Dict with processing results:
        {
            "success": bool,
            "message": str,
            "chunks_processed": int,
            "chunks_failed": int
        }
    """
    # Create a new database session for this job
    db: Session = ctx["db_session_factory"]()

    try:
        logger.info(f"Starting background processing for document {document_id}")

        # Get document and update status to "processing"
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            error_msg = f"Document {document_id} not found"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "chunks_processed": 0,
                "chunks_failed": 0
            }

        # Update status to processing
        document.status = "processing"
        db.commit()
        logger.info(f"Updated document {document_id} status to 'processing'")

        # Initialize document service
        document_service = DocumentService()

        # Process document (this does the heavy lifting)
        result = await document_service.process_document(db, document_id)

        # Update document status based on result
        if result.get("success", False):
            document.status = "completed"
            document.error_message = None
            logger.info(
                f"Successfully processed document {document_id}: "
                f"{result.get('chunks_processed', 0)} chunks processed, "
                f"{result.get('chunks_failed', 0)} chunks failed"
            )
        else:
            document.status = "failed"
            document.error_message = result.get("message", "Unknown error")
            logger.error(f"Failed to process document {document_id}: {document.error_message}")

        db.commit()
        db.refresh(document)

        logger.info(f"Completed processing for document {document_id}: {result}")
        return result

    except Exception as e:
        error_msg = f"Error processing document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Update document status to failed
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.status = "failed"
                document.error_message = str(e)
                db.commit()
        except Exception as update_error:
            logger.error(f"Failed to update document status: {update_error}")

        # Re-raise exception so Arq marks the job as failed
        raise

    finally:
        # Always close the database session
        db.close()
        logger.debug(f"Database session closed for document {document_id}")
