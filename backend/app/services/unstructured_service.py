"""
Unstructured service for extracting images, tables, and structured content from PDFs.
This service replaces Docling with the Unstructured library for better table extraction and OCR support.
"""

import base64
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
    element_type: Optional[str] = None  # 'Image' or 'Figure'
    caption: Optional[str] = None  # Alt text if available


@dataclass
class ExtractedTable:
    """Data structure for an extracted table from a PDF."""
    page_number: int
    bbox: Tuple[float, float, float, float]
    table_data: str  # HTML or Markdown representation
    rows: int
    columns: int
    table_html: Optional[str] = None
    table_markdown: Optional[str] = None
    table_text: Optional[str] = None


class UnstructuredService:
    """Service for extracting structured content from PDFs using Unstructured."""

    def __init__(self):
        """Initialize Unstructured service with configuration settings."""
        self.strategy = settings.UNSTRUCTURED_STRATEGY
        self.table_format = settings.UNSTRUCTURED_TABLE_FORMAT
        self.enable_ocr = settings.ENABLE_OCR
        self.ocr_languages = settings.OCR_LANGUAGES

        # Initialize S3 client for image uploads
        self.s3_client = boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.bucket_name = settings.S3_PDFBUCKET_NAME

        logger.info(
            f"UnstructuredService initialized: strategy={self.strategy}, "
            f"table_format={self.table_format}, ocr={self.enable_ocr}, "
            f"ocr_languages={self.ocr_languages}"
        )

    def extract_document_content(self, pdf_path: str):
        """
        Extract document content using Unstructured.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of unstructured elements (tables, images, text, etc.)

        Raises:
            Exception: If extraction fails
        """
        try:
            from unstructured.partition.pdf import partition_pdf

            logger.info(f"Starting Unstructured extraction for {pdf_path}")

            # Prepare OCR languages parameter (use 'languages' instead of deprecated 'ocr_languages')
            # Note: In newer Unstructured versions, 'languages' must be a list, not a string
            languages = self.ocr_languages if self.enable_ocr else None

            # Extract elements from PDF
            # The partition_pdf function handles:
            # - Image extraction (extract_images_in_pdf)
            # - Table detection (infer_table_structure)
            # - OCR processing (languages)
            # - extract_image_block_to_payload: Include image bytes in element metadata
            elements = partition_pdf(
                filename=pdf_path,
                strategy=self.strategy,
                extract_images_in_pdf=True,
                extract_image_block_to_payload=True,  # Include image bytes in elements
                infer_table_structure=True,
                languages=languages,
                extract_tables=True,
                extract_image_block_types=["Image"]
            )

            logger.info(
                f"Unstructured extraction successful: {len(elements)} elements extracted"
            )
            return elements

        except ImportError as e:
            logger.error(f"Unstructured not installed: {e}")
            raise Exception(f"Unstructured library not available: {e}")
        except Exception as e:
            logger.error(f"Unstructured extraction failed: {e}", exc_info=True)
            raise

    def extract_images(self, elements: List) -> List[ExtractedImage]:
        """
        Extract images from Unstructured elements.

        Args:
            elements: List of unstructured elements

        Returns:
            List of ExtractedImage objects
        """
        extracted_images = []

        try:
            from unstructured.documents.elements import Image

            logger.info("Extracting images from Unstructured elements")

            # Filter for Image elements (Figure class removed in newer Unstructured versions)
            image_elements = [
                el for el in elements
                if isinstance(el, Image)
            ]

            logger.info(f"Found {len(image_elements)} image elements")

            for idx, element in enumerate(image_elements):
                try:
                    # Get image data
                    image_data = None

                    # In newer Unstructured versions, image data is in element.metadata.image_base64
                    if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
                        image_data = base64.b64decode(element.metadata.image_base64)
                    # Fallback: try to get image bytes directly from element
                    elif hasattr(element, 'image_bytes') and element.image_bytes:
                        image_data = element.image_bytes
                    elif hasattr(element, 'image_base64'):
                        image_data = base64.b64decode(element.image_base64)

                    if image_data is None:
                        logger.warning(f"Image element {idx} has no image data")
                        continue

                    # Get page number from metadata
                    page_number = getattr(element.metadata, 'page_number', 1) if hasattr(element, 'metadata') else getattr(element, 'page_number', 1)

                    # Get bounding box
                    bbox = (0, 0, 0, 0)
                    if hasattr(element, 'coordinates'):
                        coords = element.coordinates
                        if coords and hasattr(coords, 'points'):
                            points = coords.points
                            if points and len(points) >= 2:
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                    # Get element type
                    element_type = element.__class__.__name__

                    # Get caption/alt text if available
                    caption = None
                    if hasattr(element, 'caption'):
                        caption = element.caption
                    elif hasattr(element, 'alt_text'):
                        caption = element.alt_text
                    elif hasattr(element, 'text'):
                        # Use the text content as caption for figures
                        text_content = element.text.strip()
                        if text_content and len(text_content) < 500:
                            caption = text_content

                    # Create extracted image
                    extracted_image = ExtractedImage(
                        page_number=page_number,
                        bbox=bbox,
                        image_data=image_data,
                        image_format='png',
                        element_type=element_type,
                        caption=caption
                    )

                    extracted_images.append(extracted_image)
                    logger.debug(
                        f"Extracted image {idx + 1}/{len(image_elements)} "
                        f"from page {page_number}, type={element_type}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to extract image {idx}: {e}")
                    continue

            logger.info(f"Successfully extracted {len(extracted_images)} images")

        except Exception as e:
            logger.error(f"Image extraction failed: {e}", exc_info=True)
            # Don't raise - return empty list for graceful degradation

        return extracted_images

    def extract_tables(self, elements: List) -> List[ExtractedTable]:
        """
        Extract tables from Unstructured elements.

        Args:
            elements: List of unstructured elements

        Returns:
            List of ExtractedTable objects
        """
        extracted_tables = []

        try:
            from unstructured.documents.elements import Table

            logger.info("Extracting tables from Unstructured elements")

            # Filter for Table elements
            table_elements = [el for el in elements if isinstance(el, Table)]

            logger.info(f"Found {len(table_elements)} table elements")

            for idx, element in enumerate(table_elements):
                try:
                    # Get table content in different formats
                    table_html = None
                    table_text = None
                    table_markdown = None

                    # Try HTML format from metadata (metadata is an object, not dict)
                    if hasattr(element, 'metadata') and element.metadata:
                        if hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                            table_html = element.metadata.text_as_html

                    # Try text format
                    if hasattr(element, 'text'):
                        table_text = element.text

                    # Convert to markdown if HTML available
                    if table_html and not table_markdown:
                        try:
                            import html2text
                            h = html2text.HTML2Text()
                            h.ignore_links = True
                            h.ignore_images = True
                            table_markdown = h.handle(table_html).strip()
                        except ImportError:
                            logger.debug("html2text not available, skipping markdown conversion")
                        except Exception as e:
                            logger.debug(f"Markdown conversion failed: {e}")

                    # Get page number from metadata
                    page_number = getattr(element.metadata, 'page_number', 1) if hasattr(element, 'metadata') else getattr(element, 'page_number', 1)

                    # Get bounding box
                    bbox = (0, 0, 0, 0)
                    if hasattr(element, 'coordinates'):
                        coords = element.coordinates
                        if coords and hasattr(coords, 'points'):
                            points = coords.points
                            if points and len(points) >= 2:
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                    # Estimate rows and columns from table text
                    rows = 0
                    columns = 0

                    if table_text:
                        # Count rows by newlines
                        rows = table_text.count('\n') + 1
                        # Count columns by pipes or tabs
                        if '|' in table_text:
                            columns = max(line.count('|') for line in table_text.split('\n') if line) + 1
                        elif '\t' in table_text:
                            columns = max(line.count('\t') for line in table_text.split('\n') if line) + 1

                    # Determine primary table_data based on config format
                    table_data = table_html or table_markdown or table_text or ""

                    # Create extracted table
                    extracted_table = ExtractedTable(
                        page_number=page_number,
                        bbox=bbox,
                        table_data=table_data,
                        rows=rows,
                        columns=columns,
                        table_html=table_html,
                        table_markdown=table_markdown,
                        table_text=table_text
                    )

                    extracted_tables.append(extracted_table)
                    logger.debug(
                        f"Extracted table {idx + 1}/{len(table_elements)} "
                        f"from page {page_number}, rows={rows}, cols={columns}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to extract table {idx}: {e}")
                    continue

            logger.info(f"Successfully extracted {len(extracted_tables)} tables")

        except Exception as e:
            logger.error(f"Table extraction failed: {e}", exc_info=True)
            # Don't raise - return empty list for graceful degradation

        return extracted_tables

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

            logger.debug(
                f"Compressed image: {original_size} -> {compressed_size} bytes "
                f"({compression_ratio:.1f}% reduction)"
            )

            return compressed.getvalue()

        except Exception as e:
            logger.warning(f"Image compression failed: {e}, using original")
            return image_data
