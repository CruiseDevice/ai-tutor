"""S3 object storage for PDFs and extracted media.

Extracted from DocumentService so storage I/O is independently testable
and reusable. This module owns:

- boto3 client lifecycle (construction, region/credentials wiring).
- Uploading source PDFs and generated/extracted images.
- Generating presigned URLs for the frontend PDF viewer.
- Deleting objects when a document is removed.

`StorageService` is intentionally stateless beyond the cached boto3 client.
A process-wide singleton is exposed via `get_storage_service()` to mirror
the convention used by `annotation_service` and `retrieval_service`.

The `s3_client` and `bucket_name` are also exposed as read-only properties
because `api/documents.py` streams objects directly through the boto3
client (it needs a streaming-body response that a wrapper method cannot
return without copying the whole file into memory).
"""
import logging
import os
import uuid
from typing import Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, UploadFile

from ..config import settings
from .unstructured_service import UnstructuredService

logger = logging.getLogger(__name__)


class StorageService:
    """Owns all S3 object I/O for documents and extracted media."""

    def __init__(self, unstructured_service: Optional[UnstructuredService] = None):
        self.s3_client = boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        self.bucket_name = settings.S3_PDFBUCKET_NAME
        # Used only by upload_image(); kept Optional so this service can be
        # constructed even when Unstructured extraction is disabled.
        self._unstructured_service = unstructured_service

    async def upload_pdf(self, file: UploadFile, user_id: str) -> Tuple[str, str]:
        """Upload a source PDF to S3 and return the (public URL, blob path)."""
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
                    detail=(
                        f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds maximum "
                        f"allowed size ({settings.MAX_FILE_SIZE / (1024 * 1024):.2f}MB)"
                    ),
                )

            await file.seek(0)  # Reset file pointer (though we already read it)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=unique_filename,
                Body=content,
                ContentType=file.content_type or 'application/pdf',
            )

            # Generate URL
            url = (
                f"https://{self.bucket_name}.s3.{settings.AWS_REGION}."
                f"amazonaws.com/{unique_filename}"
            )

            return url, unique_filename
        except HTTPException:
            raise
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

    def get_signed_url(self, blob_path: str, expiration: int = 3600) -> str:
        """Generate a presigned GET URL for an object in S3."""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': blob_path},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(e)}")

    def delete_object(self, blob_path: str) -> None:
        """Delete an object from S3. Logs and swallows ClientError — deletion
        is best-effort so a missing object never blocks document cleanup."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=blob_path)
        except ClientError as e:
            logger.warning(
                f"Failed to delete object from S3 (blob_path: {blob_path}): {e}"
            )

    def upload_image(
        self,
        image_data: bytes,
        document_id: str,
        user_id: str,
        page_number: int,
        image_index: int,
    ) -> Tuple[str, str]:
        """Upload an extracted image to S3 via UnstructuredService and return
        (s3_url, s3_key). Raises RuntimeError if Unstructured is unavailable."""
        if not self._unstructured_service:
            raise RuntimeError("Unstructured service not initialized")
        return self._unstructured_service.save_image_to_s3(
            image_data, document_id, user_id, page_number, image_index
        )


# --- Process-wide singleton -------------------------------------------------
# Mirrors the get_<service>() convention used by the other extracted services.
# boto3 clients are thread-safe for the operations we use, so a single shared
# instance is fine for both the API process and the ARQ worker.
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Return the process-wide StorageService singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
