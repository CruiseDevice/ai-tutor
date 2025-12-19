"""
Docling service for extracting images, tables, and structured content from PDFs.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from io import BytesIO
from PIL import Image
import boto3
from botocore.exceptions import ClientError

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Data structure for an extracted image from a PDF."""
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    image_data: bytes  # Image data in bytes (PNG format)
    image_format: str = 'png'  # Image format
    s3_key: Optional[str] = None  # S3 key after upload
    s3_url: Optional[str] = None  # S3 URL after upload


@dataclass
class ExtractedTable:
    """Data structure for an extracted table from a PDF (future use)."""
    page_number: int
    bbox: Tuple[float, float, float, float]
    table_data: str  # Markdown or HTML representation
    rows: int
    columns: int


class DoclingService:
    """Service for extracting structured content from PDFs using Docling."""

    def __init__(self):
        """Initialize Docling service with S3 client."""
        self.s3_client = boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.bucket_name = settings.S3_PDFBUCKET_NAME

    def extract_document_content(self, pdf_path: str):
        """
        Extract document content using Docling with image extraction enabled.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Docling Document object with extracted content including images

        Raises:
            Exception: If Docling extraction fails
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            logger.info(f"Starting Docling extraction for {pdf_path}")

            # Configure pipeline to extract images
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = 2.0  # Higher resolution images
            pipeline_options.generate_page_images = False  # Don't generate full page images
            pipeline_options.generate_picture_images = True  # Extract embedded pictures

            # Initialize Docling converter with image extraction enabled
            converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            # Convert PDF to Docling document
            result = converter.convert(pdf_path)

            logger.info(f"Docling extraction successful for {pdf_path}")
            return result.document

        except ImportError as e:
            logger.error(f"Docling not installed: {e}")
            raise Exception(f"Docling library not available: {e}")
        except Exception as e:
            logger.error(f"Docling extraction failed: {e}", exc_info=True)
            raise

    def extract_images(self, doc, pdf_path: str) -> List[ExtractedImage]:
        """
        Extract images from a Docling document.

        Args:
            doc: Docling Document object (already converted, passed from extract_document_content)
            pdf_path: Path to the original PDF (for fallback if needed)

        Returns:
            List of ExtractedImage objects
        """
        extracted_images = []

        try:
            logger.info("Extracting images from Docling document")

            # Primary method: Extract from doc.pictures (available in current Docling version)
            if hasattr(doc, 'pictures') and doc.pictures:
                logger.info(f"Found {len(doc.pictures)} pictures in document")

                for idx, picture in enumerate(doc.pictures):
                    try:
                        # Get PIL image from picture
                        pil_image = None

                        # Try different ways to access the image
                        if hasattr(picture, 'pil_image') and picture.pil_image is not None:
                            pil_image = picture.pil_image
                        elif hasattr(picture, 'image') and picture.image is not None:
                            if hasattr(picture.image, 'pil_image'):
                                pil_image = picture.image.pil_image
                            else:
                                pil_image = picture.image
                        elif hasattr(picture, 'get_image'):
                            # get_image() method requires doc parameter
                            pil_image = picture.get_image(doc)

                        if pil_image is None:
                            logger.warning(f"Picture {idx} has no accessible image data")
                            continue

                        # Convert to bytes
                        img_bytes = BytesIO()
                        pil_image.save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        # Get page number and bounding box from picture metadata
                        page_number = 1  # Default
                        bbox = (0, 0, 0, 0)  # Default

                        # Try to get page number from picture
                        if hasattr(picture, 'prov') and picture.prov:
                            for prov in picture.prov:
                                if hasattr(prov, 'page_no'):
                                    page_number = prov.page_no + 1  # Convert to 1-indexed
                                if hasattr(prov, 'bbox'):
                                    bbox = prov.bbox.as_tuple()
                        elif hasattr(picture, 'page'):
                            page_number = picture.page if isinstance(picture.page, int) else 1
                        elif hasattr(picture, 'page_no'):
                            page_number = picture.page_no + 1  # Convert to 1-indexed

                        # Create extracted image
                        extracted_image = ExtractedImage(
                            page_number=page_number,
                            bbox=bbox,
                            image_data=img_bytes.getvalue(),
                            image_format='png'
                        )

                        extracted_images.append(extracted_image)
                        logger.debug(f"Extracted image {idx + 1}/{len(doc.pictures)} from page {page_number}")

                    except Exception as e:
                        logger.warning(f"Failed to extract picture {idx}: {e}")
                        continue

                logger.info(f"Successfully extracted {len(extracted_images)} images from {len(doc.pictures)} pictures")
            else:
                logger.info("No pictures found in document.pictures collection")

        except Exception as e:
            logger.error(f"Image extraction failed: {e}", exc_info=True)
            # Don't raise - return empty list to allow graceful degradation

        return extracted_images

    def extract_tables(self, doc) -> List[ExtractedTable]:
        """
        Extract tables from a Docling document (placeholder for future use).

        Args:
            doc: Docling Document object

        Returns:
            List of ExtractedTable objects
        """
        # Placeholder for Step 6 (future implementation)
        logger.debug("Table extraction not yet implemented")
        return []

    def save_image_to_s3(
        self,
        image_data: bytes,
        document_id: str,
        user_id: str,
        page_number: int,
        image_index: int
    ) -> Tuple[str, str]:
        """
        Upload an image to S3 in the document's assets folder.

        Args:
            image_data: Image bytes (PNG format)
            document_id: Document UUID
            user_id: User ID
            page_number: Page number where image was found
            image_index: Index of image on the page

        Returns:
            Tuple of (s3_url, s3_key)

        Raises:
            Exception: If S3 upload fails
        """
        try:
            # Construct S3 key: {user_id}/{doc_uuid}_assets/page_{page}_img_{idx}.png
            s3_key = f"{user_id}/{document_id}{settings.S3_ASSETS_FOLDER_SUFFIX}/page_{page_number}_img_{image_index}.png"

            # Optionally compress image before upload
            if settings.IMAGE_COMPRESSION_QUALITY < 100:
                image_data = self._compress_image(image_data, settings.IMAGE_COMPRESSION_QUALITY)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_data,
                ContentType='image/png'
            )

            # Generate URL
            s3_url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"

            logger.debug(f"Uploaded image to S3: {s3_key}")

            return s3_url, s3_key

        except ClientError as e:
            logger.error(f"S3 upload failed for image: {e}", exc_info=True)
            raise Exception(f"Failed to upload image to S3: {e}")

    def _compress_image(self, image_data: bytes, quality: int) -> bytes:
        """
        Compress image to reduce storage costs.

        Args:
            image_data: Original image bytes
            quality: JPEG quality (1-100)

        Returns:
            Compressed image bytes
        """
        try:
            # Open image
            image = Image.open(BytesIO(image_data))

            # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
            if image.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = background
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Compress
            compressed = BytesIO()
            image.save(compressed, format='JPEG', quality=quality, optimize=True)
            compressed.seek(0)

            original_size = len(image_data)
            compressed_size = len(compressed.getvalue())
            compression_ratio = (1 - compressed_size / original_size) * 100

            logger.debug(f"Compressed image: {original_size} -> {compressed_size} bytes ({compression_ratio:.1f}% reduction)")

            return compressed.getvalue()

        except Exception as e:
            logger.warning(f"Image compression failed: {e}, using original")
            return image_data
