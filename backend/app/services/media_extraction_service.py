"""Image/table extraction and chunk shaping for non-text PDF content.

Extracted from DocumentService so media handling is independently testable
and reusable. This module owns:

- Extracting images and tables from a PDF via `UnstructuredService`.
- Generating GPT-4o Vision captions for extracted images (via `VisionService`).
- Shaping extracted media into chunk dicts ready for embedding/storage:
  `create_image_chunks` / `create_table_chunks` (publicly testable helpers).

`MediaExtractor` holds the `UnstructuredService` (None when Unstructured is
disabled) and a `StorageService` for image uploads. A process-wide singleton
is exposed via `get_media_extractor()`, mirroring the convention used by
`annotation_service` and `retrieval_service`.

The two chunk-shaping helpers (`create_image_chunks`, `create_table_chunks`)
are pure functions of their input dicts and are exercised directly by
`tests/test_unstructured_integration.py`.
"""
import logging
from typing import Dict, List, Optional

from ..config import settings
from .storage_service import StorageService
from .unstructured_service import UnstructuredService

logger = logging.getLogger(__name__)


class MediaExtractor:
    """Extract images/tables from PDFs and shape them into chunk dicts."""

    def __init__(
        self,
        unstructured_service: Optional[UnstructuredService],
        storage_service: StorageService,
    ):
        self.unstructured_service = unstructured_service
        self.storage = storage_service

    async def generate_image_captions(self, images: List) -> List:
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
                {'image_data': img.image_data, 'page_number': img.page_number} for img in images
            ]

            # Generate captions in parallel batches
            captions = await vision_service.batch_generate_captions(image_data_list)

            return captions

        except Exception as e:
            logger.error(f"Failed to generate image captions: {e}", exc_info=True)
            # Return list of Nones if caption generation fails
            return [None] * len(images)

    def caption_to_dict(self, caption) -> Optional[Dict]:
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
            'tags': caption.tags,
        }

    def build_image_chunk_text(self, caption: Optional[Dict], page_number: int) -> str:
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

    def create_image_chunks(self, uploaded_images: List[Dict]) -> List[Dict]:
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
            chunk_text = self.build_image_chunk_text(caption, page_number)

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
                    'tags': caption.get('tags', []),
                })

            image_chunk = {
                'content': chunk_text,
                'page_number': page_number,
                'metadata': metadata,
                'chunk_type': 'image',  # Mark as image chunk
            }

            image_chunks.append(image_chunk)

        logger.info(f"Created {len(image_chunks)} image chunks for embedding")
        return image_chunks

    async def extract_images(
        self,
        pdf_path: str,
        document_id: str,
        user_id: str,
    ) -> List[Dict]:
        """
        Extract images from PDF using Unstructured and upload to S3.

        Args:
            pdf_path: Path to the PDF file
            document_id: Document UUID
            user_id: User ID

        Returns:
            List of dicts with image metadata and S3 URLs
        """
        if not settings.ENABLE_IMAGE_EXTRACTION:
            logger.info("Image extraction disabled via settings")
            return []

        if not self.unstructured_service:
            logger.warning("Unstructured service not initialized, skipping image extraction")
            return []

        try:
            logger.info(f"Starting image extraction with Unstructured for document {document_id}")

            # Extract document content with Unstructured
            elements = self.unstructured_service.extract_document_content(pdf_path)

            # Extract images from elements
            images = self.unstructured_service.extract_images(elements)

            if not images:
                logger.info("No images found in document")
                return []

            # Limit number of images to process
            if len(images) > settings.MAX_IMAGES_PER_DOCUMENT:
                logger.warning(
                    f"Document has {len(images)} images, limiting to {settings.MAX_IMAGES_PER_DOCUMENT}"
                )
                images = images[: settings.MAX_IMAGES_PER_DOCUMENT]

            # Generate captions for images using GPT-4o Vision
            logger.info(f"Generating captions for {len(images)} images")
            captions = await self.generate_image_captions(images)

            # Upload images to S3 and collect metadata with captions
            uploaded_images = []
            for idx, img in enumerate(images):
                try:
                    # Upload to S3 using StorageService
                    s3_url, s3_key = self.storage.upload_image(
                        img.image_data, document_id, user_id, img.page_number, idx
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
                        'element_type': img.element_type,
                        'caption': self.caption_to_dict(caption) if caption else None,
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
            # Graceful degradation - return empty list
            return []

    def create_table_chunks(self, tables: List[Dict]) -> List[Dict]:
        """
        Create table chunk representations from extracted tables.

        Args:
            tables: List of table metadata dicts

        Returns:
            List of chunk dicts with 'content', 'page_number', 'metadata' for embedding generation
        """
        table_chunks = []

        for table_data in tables:
            page_number = table_data.get('page_number', 1)

            # Build text content for embedding
            # Use table_html if available, otherwise markdown, then text
            chunk_text = ""
            if table_data.get('table_html'):
                chunk_text = f"Table on page {page_number}: {table_data['table_html']}"
            elif table_data.get('table_markdown'):
                chunk_text = f"Table on page {page_number}: {table_data['table_markdown']}"
            elif table_data.get('table_text'):
                chunk_text = f"Table on page {page_number}: {table_data['table_text']}"
            else:
                chunk_text = f"Table on page {page_number}: No content available"

            # Prepare metadata for position_data JSONB field
            metadata = {
                'bbox': table_data.get('bbox', (0, 0, 0, 0)),
                'content_type': 'table',
                'table_rows': table_data.get('rows', 0),
                'table_columns': table_data.get('columns', 0),
                'table_index': table_data.get('table_index', 0),
                'table_html': table_data.get('table_html'),
                'table_markdown': table_data.get('table_markdown'),
            }

            table_chunk = {
                'content': chunk_text,
                'page_number': page_number,
                'metadata': metadata,
                'chunk_type': 'table',  # Mark as table chunk
            }

            table_chunks.append(table_chunk)

        logger.info(f"Created {len(table_chunks)} table chunks for embedding")
        return table_chunks

    async def extract_tables(
        self,
        pdf_path: str,
        document_id: str,
        user_id: str,
    ) -> List[Dict]:
        """
        Extract tables from PDF using Unstructured.

        Args:
            pdf_path: Path to the PDF file
            document_id: Document UUID
            user_id: User ID

        Returns:
            List of dicts with table metadata
        """
        if not settings.ENABLE_TABLE_EXTRACTION:
            logger.info("Table extraction disabled via settings")
            return []

        if not self.unstructured_service:
            logger.warning("Unstructured service not initialized, skipping table extraction")
            return []

        try:
            logger.info(f"Starting table extraction with Unstructured for document {document_id}")

            # Extract document content with Unstructured
            elements = self.unstructured_service.extract_document_content(pdf_path)

            # Extract tables from elements
            tables = self.unstructured_service.extract_tables(elements)

            if not tables:
                logger.info("No tables found in document")
                return []

            logger.info(f"Successfully extracted {len(tables)} tables")

            # Convert tables to dict format for storage
            table_metadata_list = []
            for idx, table in enumerate(tables):
                table_metadata = {
                    'page_number': table.page_number,
                    'bbox': table.bbox,
                    'table_data': table.table_data,
                    'rows': table.rows,
                    'columns': table.columns,
                    'table_html': table.table_html,
                    'table_markdown': table.table_markdown,
                    'table_text': table.table_text,
                    'table_index': idx,
                }
                table_metadata_list.append(table_metadata)

            return table_metadata_list

        except Exception as e:
            logger.error(f"Table extraction failed: {e}", exc_info=True)
            # Graceful degradation - return empty list
            return []


# --- Process-wide singleton -------------------------------------------------
_media_extractor: MediaExtractor | None = None


def get_media_extractor() -> MediaExtractor:
    """Return the process-wide MediaExtractor singleton.

    Reads `settings.USE_UNSTRUCTURED` at first construction (same behaviour as
    the original DocumentService constructor). Subsequent calls reuse the
    singleton, so toggling the flag at runtime has no effect — matching prior
    behaviour.
    """
    global _media_extractor
    if _media_extractor is None:
        unstructured_service = (
            UnstructuredService() if settings.USE_UNSTRUCTURED else None
        )
        if unstructured_service:
            logger.info("Unstructured service initialized for image/table extraction")
        # Inject the storage singleton so image uploads share one boto3 client.
        from .storage_service import get_storage_service

        # Reconstruct storage with the Unstructured service so it can delegate
        # image uploads. The shared singleton is updated to preserve the
        # wiring across the process.
        storage = get_storage_service()
        storage._unstructured_service = unstructured_service
        _media_extractor = MediaExtractor(unstructured_service, storage)
    return _media_extractor
