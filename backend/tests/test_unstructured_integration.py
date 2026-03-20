"""
Integration tests for Unstructured PDF processing.

These tests require:
- Unstructured library to be installed
- The test PDFs to be available in the project root

Run with: pytest tests/test_unstructured_integration.py -v
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check if Unstructured is available
UNSTRUCTURED_AVAILABLE = False
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Table, Image, Figure
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    pass


# Only import service if dependencies are available
if UNSTRUCTURED_AVAILABLE:
    from app.services.unstructured_service import (
        UnstructuredService,
        ExtractedImage,
        ExtractedTable
    )


# Path to test PDFs
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_PDFS = {
    'attention': PROJECT_ROOT / 'attention.pdf',
    'llm_survey': PROJECT_ROOT / 'llm_survey.pdf',
}


@pytest.mark.skipif(not UNSTRUCTURED_AVAILABLE, reason="Unstructured library not installed")
class TestUnstructuredIntegration:
    """Integration tests with actual PDF processing."""

    def test_service_initialization(self):
        """Test that UnstructuredService can be initialized."""
        from app.config import settings
        service = UnstructuredService()
        assert service.strategy == settings.UNSTRUCTURED_STRATEGY
        assert service.table_format == settings.UNSTRUCTURED_TABLE_FORMAT

    @pytest.mark.parametrize("pdf_name", ["attention", "llm_survey"])
    def test_extract_document_content(self, pdf_name):
        """Test document content extraction from real PDFs."""
        pdf_path = TEST_PDFS[pdf_name]

        if not pdf_path.exists():
            pytest.skip(f"Test PDF {pdf_name} not found at {pdf_path}")

        service = UnstructuredService()
        elements = service.extract_document_content(str(pdf_path))

        assert elements is not None
        assert len(elements) > 0
        print(f"  Extracted {len(elements)} elements from {pdf_name}.pdf")

    @pytest.mark.parametrize("pdf_name", ["attention", "llm_survey"])
    def test_extract_images(self, pdf_name):
        """Test image extraction from real PDFs."""
        pdf_path = TEST_PDFS[pdf_name]

        if not pdf_path.exists():
            pytest.skip(f"Test PDF {pdf_name} not found at {pdf_path}")

        service = UnstructuredService()
        elements = service.extract_document_content(str(pdf_path))
        images = service.extract_images(elements)

        # Images may or may not be present
        assert isinstance(images, list)
        print(f"  Found {len(images)} images in {pdf_name}.pdf")

        for img in images:
            assert isinstance(img, ExtractedImage)
            assert img.page_number >= 1
            assert len(img.image_data) > 0

    @pytest.mark.parametrize("pdf_name", ["attention", "llm_survey"])
    def test_extract_tables(self, pdf_name):
        """Test table extraction from real PDFs."""
        pdf_path = TEST_PDFS[pdf_name]

        if not pdf_path.exists():
            pytest.skip(f"Test PDF {pdf_name} not found at {pdf_path}")

        service = UnstructuredService()
        elements = service.extract_document_content(str(pdf_path))
        tables = service.extract_tables(elements)

        # Tables may or may not be present
        assert isinstance(tables, list)
        print(f"  Found {len(tables)} tables in {pdf_name}.pdf")

        for table in tables:
            assert isinstance(table, ExtractedTable)
            assert table.page_number >= 1
            assert table.rows >= 0
            assert table.columns >= 0

    @pytest.mark.parametrize("pdf_name", ["attention", "llm_survey"])
    def test_extracted_image_structure(self, pdf_name):
        """Test that extracted images have correct structure."""
        pdf_path = TEST_PDFS[pdf_name]

        if not pdf_path.exists():
            pytest.skip(f"Test PDF {pdf_name} not found at {pdf_path}")

        service = UnstructuredService()
        elements = service.extract_document_content(str(pdf_path))
        images = service.extract_images(elements)

        if not images:
            pytest.skip(f"No images found in {pdf_name}.pdf")

        # Check first image structure
        img = images[0]
        assert hasattr(img, 'page_number')
        assert hasattr(img, 'bbox')
        assert hasattr(img, 'image_data')
        assert hasattr(img, 'image_format')
        assert hasattr(img, 'element_type')

        print(f"  First image: page={img.page_number}, type={img.element_type}, "
              f"size={len(img.image_data)} bytes")

    @pytest.mark.parametrize("pdf_name", ["attention", "llm_survey"])
    def test_extracted_table_structure(self, pdf_name):
        """Test that extracted tables have correct structure."""
        pdf_path = TEST_PDFS[pdf_name]

        if not pdf_path.exists():
            pytest.skip(f"Test PDF {pdf_name} not found at {pdf_path}")

        service = UnstructuredService()
        elements = service.extract_document_content(str(pdf_path))
        tables = service.extract_tables(elements)

        if not tables:
            pytest.skip(f"No tables found in {pdf_name}.pdf")

        # Check first table structure
        table = tables[0]
        assert hasattr(table, 'page_number')
        assert hasattr(table, 'bbox')
        assert hasattr(table, 'table_data')
        assert hasattr(table, 'rows')
        assert hasattr(table, 'columns')
        assert hasattr(table, 'table_html')
        assert hasattr(table, 'table_markdown')
        assert hasattr(table, 'table_text')

        print(f"  First table: page={table.page_number}, rows={table.rows}, cols={table.columns}")
        if table.table_html:
            print(f"    HTML length: {len(table.table_html)} chars")
        if table.table_markdown:
            print(f"    Markdown length: {len(table.table_markdown)} chars")

    @pytest.mark.parametrize("pdf_name", ["attention", "llm_survey"])
    def test_end_to_end_extraction(self, pdf_name):
        """Test complete extraction workflow."""
        pdf_path = TEST_PDFS[pdf_name]

        if not pdf_path.exists():
            pytest.skip(f"Test PDF {pdf_name} not found at {pdf_path}")

        service = UnstructuredService()

        # Extract everything
        elements = service.extract_document_content(str(pdf_path))
        images = service.extract_images(elements)
        tables = service.extract_tables(elements)

        # Summary
        print(f"\n  === {pdf_name}.pdf Summary ===")
        print(f"  Total elements: {len(elements)}")
        print(f"  Images found: {len(images)}")
        print(f"  Tables found: {len(tables)}")

        # Assertions
        assert len(elements) > 0, "Should extract at least some elements"
        assert isinstance(images, list)
        assert isinstance(tables, list)


@pytest.mark.skipif(not UNSTRUCTURED_AVAILABLE, reason="Unstructured library not installed")
class TestTableChunkCreation:
    """Tests for table chunk creation with real data."""

    def test_create_table_chunks_from_extracted(self):
        """Test creating chunks from extracted tables."""
        pdf_path = TEST_PDFS['llm_survey']

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        service = UnstructuredService()
        elements = service.extract_document_content(str(pdf_path))
        tables = service.extract_tables(elements)

        if not tables:
            pytest.skip("No tables found in test PDF")

        # Import chunk creation function from document_service
        from app.services.document_service import DocumentService

        doc_service = DocumentService()
        table_chunks = doc_service._create_table_chunks(tables)

        assert len(table_chunks) == len(tables)

        for chunk in table_chunks:
            assert 'content' in chunk
            assert 'page_number' in chunk
            assert 'metadata' in chunk
            assert 'chunk_type' in chunk
            assert chunk['chunk_type'] == 'table'
            assert chunk['metadata']['content_type'] == 'table'

        print(f"  Created {len(table_chunks)} table chunks")


@pytest.mark.skipif(not UNSTRUCTURED_AVAILABLE, reason="Unstructured library not installed")
class TestImageChunkCreation:
    """Tests for image chunk creation with real data."""

    def test_create_image_chunks_from_extracted(self):
        """Test creating chunks from extracted images."""
        pdf_path = TEST_PDFS['attention']

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        service = UnstructuredService()
        elements = service.extract_document_content(str(pdf_path))
        images = service.extract_images(elements)

        if not images:
            pytest.skip("No images found in test PDF")

        # Import chunk creation function from document_service
        from app.services.document_service import DocumentService

        doc_service = DocumentService()

        # Convert ExtractedImage to dict format expected by _create_image_chunks
        uploaded_images = []
        for idx, img in enumerate(images):
            image_metadata = {
                'page_number': img.page_number,
                'bbox': img.bbox,
                's3_key': img.s3_key or f'test/path/img_{idx}.png',
                's3_url': img.s3_url or f'https://test.s3.amazonaws.com/img_{idx}.png',
                'image_format': img.image_format,
                'image_index': idx,
                'element_type': img.element_type,
                'caption': None  # No caption for this test
            }
            uploaded_images.append(image_metadata)

        image_chunks = doc_service._create_image_chunks(uploaded_images)

        assert len(image_chunks) == len(uploaded_images)

        for chunk in image_chunks:
            assert 'content' in chunk
            assert 'page_number' in chunk
            assert 'metadata' in chunk
            assert 'chunk_type' in chunk
            assert chunk['chunk_type'] == 'image'

        print(f"  Created {len(image_chunks)} image chunks")


def test_unstructured_availability():
    """Test to check if Unstructured is available for testing."""
    if UNSTRUCTURED_AVAILABLE:
        print("\n✓ Unstructured library is available - integration tests will run")
    else:
        print("\n✗ Unstructured library NOT installed - integration tests will be skipped")
        print("  Install with: pip install unstructured unstructured_inference")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
