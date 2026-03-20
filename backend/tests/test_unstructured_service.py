"""
Unit tests for UnstructuredService dataclasses.

Tests cover:
- ExtractedImage dataclass
- ExtractedTable dataclass

Note: Full service tests require Unstructured library installation.
These tests verify the data structures that are core to the service.
"""

import os
import sys
import pytest
from io import BytesIO
from dataclasses import asdict, fields
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock external dependencies before importing
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.exceptions'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['unstructured'] = MagicMock()
sys.modules['unstructured.partition.pdf'] = MagicMock()
sys.modules['unstructured.documents.elements'] = MagicMock()

from app.services.unstructured_service import ExtractedImage, ExtractedTable


class TestExtractedImage:
    """Tests for ExtractedImage dataclass."""

    def test_extracted_image_creation(self):
        """Test creating an ExtractedImage instance with all fields."""
        image = ExtractedImage(
            page_number=1,
            bbox=(0.0, 0.0, 100.0, 100.0),
            image_data=b'\x89PNG\r\n\x1a\n',
            image_format='png',
            s3_key='test/path/image.png',
            s3_url='https://s3.amazonaws.com/bucket/test/path/image.png',
            element_type='Image',
            caption='Test caption'
        )

        assert image.page_number == 1
        assert image.bbox == (0.0, 0.0, 100.0, 100.0)
        assert image.image_format == 'png'
        assert image.element_type == 'Image'
        assert image.caption == 'Test caption'
        assert image.s3_key == 'test/path/image.png'
        assert 's3.amazonaws.com' in image.s3_url

    def test_extracted_image_defaults(self):
        """Test ExtractedImage with default values."""
        image = ExtractedImage(
            page_number=1,
            bbox=(0, 0, 0, 0),
            image_data=b'test'
        )

        assert image.image_format == 'png'  # Default
        assert image.element_type is None  # Optional
        assert image.caption is None  # Optional
        assert image.s3_key is None  # Optional
        assert image.s3_url is None  # Optional

    def test_extracted_image_to_dict(self):
        """Test converting ExtractedImage to dictionary."""
        image = ExtractedImage(
            page_number=1,
            bbox=(0, 0, 100, 100),
            image_data=b'test',
            element_type='Image'
        )

        data = asdict(image)
        assert data['page_number'] == 1
        assert data['bbox'] == (0, 0, 100, 100)
        assert data['element_type'] == 'Image'
        assert b'test' in data['image_data']

    def test_extracted_image_fields(self):
        """Test ExtractedImage has all expected fields."""
        expected_fields = {
            'page_number', 'bbox', 'image_data', 'image_format',
            's3_key', 's3_url', 'element_type', 'caption'
        }
        actual_fields = {f.name for f in fields(ExtractedImage)}
        assert expected_fields == actual_fields


class TestExtractedTable:
    """Tests for ExtractedTable dataclass."""

    def test_extracted_table_creation(self):
        """Test creating an ExtractedTable instance with all fields."""
        table = ExtractedTable(
            page_number=1,
            bbox=(0.0, 0.0, 200.0, 100.0),
            table_data='<table><tr><td>Cell</td></tr></table>',
            rows=2,
            columns=3,
            table_html='<table><tr><td>Cell</td></tr></table>',
            table_markdown='| Cell |\n| --- |',
            table_text='Cell'
        )

        assert table.page_number == 1
        assert table.bbox == (0.0, 0.0, 200.0, 100.0)
        assert table.table_data == '<table><tr><td>Cell</td></tr></table>'
        assert table.rows == 2
        assert table.columns == 3
        assert table.table_html == '<table><tr><td>Cell</td></tr></table>'
        assert table.table_markdown == '| Cell |\n| --- |'
        assert table.table_text == 'Cell'

    def test_extracted_table_defaults(self):
        """Test ExtractedTable with default values."""
        table = ExtractedTable(
            page_number=1,
            bbox=(0, 0, 0, 0),
            table_data='test data',
            rows=1,
            columns=1
        )

        assert table.table_html is None  # Optional
        assert table.table_markdown is None  # Optional
        assert table.table_text is None  # Optional

    def test_extracted_table_to_dict(self):
        """Test converting ExtractedTable to dictionary."""
        table = ExtractedTable(
            page_number=1,
            bbox=(0, 0, 200, 100),
            table_data='<table>...</table>',
            rows=3,
            columns=2
        )

        data = asdict(table)
        assert data['page_number'] == 1
        assert data['rows'] == 3
        assert data['columns'] == 2
        assert data['table_data'] == '<table>...</table>'

    def test_extracted_table_fields(self):
        """Test ExtractedTable has all expected fields."""
        expected_fields = {
            'page_number', 'bbox', 'table_data', 'rows', 'columns',
            'table_html', 'table_markdown', 'table_text'
        }
        actual_fields = {f.name for f in fields(ExtractedTable)}
        assert expected_fields == actual_fields


class TestDataclassIntegration:
    """Integration tests for dataclass usage patterns."""

    def test_extracted_image_list_serialization(self):
        """Test that ExtractedImage objects can be stored in lists and serialized."""
        images = [
            ExtractedImage(page_number=1, bbox=(0, 0, 100, 100), image_data=b'data1'),
            ExtractedImage(page_number=2, bbox=(0, 0, 50, 50), image_data=b'data2', element_type='Figure'),
        ]

        assert len(images) == 2
        assert images[0].page_number == 1
        assert images[1].element_type == 'Figure'

        # Test list comprehension works
        page_numbers = [img.page_number for img in images]
        assert page_numbers == [1, 2]

    def test_extracted_table_list_serialization(self):
        """Test that ExtractedTable objects can be stored in lists and serialized."""
        tables = [
            ExtractedTable(page_number=1, bbox=(0, 0, 200, 100), table_data='table1', rows=2, columns=2),
            ExtractedTable(page_number=3, bbox=(0, 0, 150, 80), table_data='table2', rows=3, columns=4),
        ]

        assert len(tables) == 2
        assert tables[0].rows == 2
        assert tables[1].columns == 4

        # Test filtering works
        large_tables = [t for t in tables if t.rows > 2]
        assert len(large_tables) == 1
        assert large_tables[0].rows == 3

    def test_bbox_coordinate_handling(self):
        """Test bounding box coordinate tuples are handled correctly."""
        # Test different bbox formats
        bboxes = [
            (0, 0, 100, 100),      # integers
            (0.0, 0.0, 100.0, 100.0),  # floats
            (10.5, 20.3, 110.8, 120.1),  # mixed precision
        ]

        for bbox in bboxes:
            image = ExtractedImage(page_number=1, bbox=bbox, image_data=b'data')
            assert len(image.bbox) == 4
            assert image.bbox[0] >= 0  # x0
            assert image.bbox[1] >= 0  # y0
            assert image.bbox[2] >= image.bbox[0]  # x1 > x0
            assert image.bbox[3] >= image.bbox[1]  # y1 > y0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
