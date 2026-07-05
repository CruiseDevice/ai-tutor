"""Annotation parsing for chat responses.

Extracted from ChatService so the logic is independently testable and
reusable (agent_service consumes it directly). This module owns:

- Parsing the LLM's trailing ```annotations / ```json block.
- Resolving text annotations to source chunks.
- Resolving image annotations (with bbox -> percentage bounds conversion).
- Annotation color palette.

It is intentionally stateless: every method is a pure function of its
arguments, so the class is a thin namespace. Construction is cheap and a
process-wide singleton is exposed via `get_annotation_service()` to mirror
the convention used by `rerank_service` and `embedding_service`.
"""
import json
import logging
import re
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AnnotationService:
    """Parse and resolve annotations emitted by the tutor LLM."""

    def parse(self, response_text: str, relevant_chunks: List[Dict]) -> tuple[str, List[Dict]]:
        """
        Parse annotations from the LLM response.
        Returns (cleaned_response, annotations_list)
        """
        annotations = []
        cleaned_response = response_text
        raw_annotations = []

        # Look for annotations block. The model is instructed to use
        # ```annotations, but in practice it may sometimes emit ```json.
        # Support both to make parsing more robust.
        annotation_pattern = r'```(?:annotations|json)\s*([\s\S]*?)\s*```'
        match = re.search(annotation_pattern, response_text)

        if match:
            # Remove the annotations block from the response
            cleaned_response = re.sub(annotation_pattern, '', response_text).strip()

            try:
                annotation_data = json.loads(match.group(1))

                if isinstance(annotation_data, dict) and "annotations" in annotation_data:
                    annotation_items = annotation_data.get("annotations", [])
                elif isinstance(annotation_data, list):
                    annotation_items = annotation_data
                else:
                    annotation_items = [annotation_data]

                for item in annotation_items:
                    if not isinstance(item, dict):
                        continue

                    page_num = item.get("pageNumber", 1)
                    try:
                        page_num = int(page_num)
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid pageNumber in annotation: {page_num}")
                        page_num = item.get("pageNumber", 1)
                    annotation_type = item.get("type", "highlight")
                    explanation = item.get("explanation", "")

                    annotation = None
                    source_text = None
                    source_image_url = None

                    if annotation_type == "circle" and "bbox" in item and "imageChunkId" in item:
                        annotation = self._parse_image_annotation(
                            item=item,
                            relevant_chunks=relevant_chunks,
                            page_number=page_num,
                            explanation=explanation
                        )
                        if annotation:
                            source_image_url = annotation.get("imageS3Url")
                    elif "textToHighlight" in item:
                        annotation = self._parse_text_annotation(
                            item=item,
                            relevant_chunks=relevant_chunks,
                            page_number=page_num,
                            annotation_type=annotation_type,
                            explanation=explanation
                        )
                        if annotation:
                            source_text = annotation.get("textContent")
                    else:
                        logger.warning(f"Invalid annotation format: {item}")

                    if annotation:
                        if explanation and not annotation.get("label"):
                            annotation["label"] = explanation
                        raw_annotations.append({
                            "pageNumber": page_num,
                            "annotation": annotation,
                            "sourceText": source_text,
                            "sourceImageUrl": source_image_url,
                            "explanation": explanation
                        })

                annotations_by_page = {}
                for entry in raw_annotations:
                    page = entry["pageNumber"]
                    annotations_by_page.setdefault(page, {
                        "annotations": [],
                        "sourceText": None,
                        "sourceImageUrl": None,
                        "explanation": None
                    })
                    annotations_by_page[page]["annotations"].append(entry["annotation"])
                    if entry.get("sourceText") and not annotations_by_page[page]["sourceText"]:
                        annotations_by_page[page]["sourceText"] = entry["sourceText"]
                    if entry.get("sourceImageUrl") and not annotations_by_page[page]["sourceImageUrl"]:
                        annotations_by_page[page]["sourceImageUrl"] = entry["sourceImageUrl"]
                    if entry.get("explanation") and not annotations_by_page[page]["explanation"]:
                        annotations_by_page[page]["explanation"] = entry["explanation"]

                for page, data in annotations_by_page.items():
                    has_image = any(
                        annotation.get("imageChunkId") for annotation in data["annotations"]
                    )
                    source_text = data["sourceText"]
                    source_image_url = data["sourceImageUrl"]

                    if not source_text and not has_image:
                        for chunk in relevant_chunks:
                            if chunk.get("pageNumber") == page:
                                source_text = chunk.get("content", "")[:200]
                                break

                    annotations.append({
                        "pageNumber": page,
                        "annotations": data["annotations"],
                        "sourceText": source_text,
                        "sourceImageUrl": source_image_url,
                        "explanation": data["explanation"]
                    })

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse annotations JSON: {e}")
            except Exception as e:
                logger.warning(f"Error processing annotations: {e}")

        return cleaned_response, annotations

    def _parse_image_annotation(
        self,
        item: Dict,
        relevant_chunks: List[Dict],
        page_number: int,
        explanation: str
    ) -> Optional[Dict]:
        """Parse an image annotation with bbox coordinates."""
        try:
            bbox = item.get("bbox")
            image_chunk_id = item.get("imageChunkId")

            if not bbox or not image_chunk_id:
                logger.warning("Image annotation missing bbox or imageChunkId")
                return None
            if not isinstance(bbox, list) or len(bbox) < 4:
                logger.warning(f"Invalid bbox format for image annotation: {bbox}")
                return None

            image_chunk = next(
                (chunk for chunk in relevant_chunks if chunk.get("id") == image_chunk_id),
                None
            )

            if not image_chunk:
                logger.warning(f"Image chunk {image_chunk_id} not found in relevant chunks")
                return None

            bounds = self.pdf_coords_to_percentage_bounds(
                bbox=bbox,
                page_number=page_number,
                document_id=image_chunk.get("documentId")
            )

            position_data = image_chunk.get("positionData") or {}
            image_s3_url = position_data.get("image_s3_url")

            return {
                "id": str(uuid.uuid4()),
                "type": "circle",
                "pageNumber": page_number,
                "bounds": bounds,
                "textContent": None,
                "imageChunkId": image_chunk_id,
                "imageS3Url": image_s3_url,
                "color": self.get_annotation_color("circle"),
                "label": None
            }

        except Exception as e:
            logger.error(f"Failed to parse image annotation: {e}", exc_info=True)
            return None

    def _parse_text_annotation(
        self,
        item: Dict,
        relevant_chunks: List[Dict],
        page_number: int,
        annotation_type: str,
        explanation: str
    ) -> Optional[Dict]:
        """Parse a text-based annotation."""
        text_to_highlight = item.get("textToHighlight", "")

        matching_chunk = None
        for chunk in relevant_chunks:
            if chunk.get("pageNumber") == page_number:
                if text_to_highlight.lower() in chunk.get("content", "").lower():
                    matching_chunk = chunk
                    break

        if not matching_chunk:
            logger.warning(f"Could not find matching chunk for: {text_to_highlight}")
            return None

        bounds = {
            "x": 10,
            "y": 30,
            "width": 80,
            "height": 5
        }

        return {
            "id": str(uuid.uuid4()),
            "type": annotation_type,
            "pageNumber": page_number,
            "bounds": bounds,
            "textContent": text_to_highlight,
            "imageChunkId": None,
            "imageS3Url": None,
            "color": self.get_annotation_color(annotation_type),
            "label": None
        }

    def pdf_coords_to_percentage_bounds(
        self,
        bbox: List[float],
        page_number: int,
        document_id: Optional[str]
    ) -> Dict[str, float]:
        """Convert PDF coordinates to percentage-based bounds."""
        try:
            page_width = 612.0
            page_height = 792.0

            x0, y0, x1, y1 = [float(value) for value in bbox[:4]]

            x_percent = (x0 / page_width) * 100
            y_percent = (y0 / page_height) * 100
            width_percent = ((x1 - x0) / page_width) * 100
            height_percent = ((y1 - y0) / page_height) * 100

            return {
                "x": round(x_percent, 2),
                "y": round(y_percent, 2),
                "width": round(width_percent, 2),
                "height": round(height_percent, 2)
            }

        except Exception as e:
            logger.error(f"Failed to convert PDF coords to percentage: {e}")
            return {"x": 10, "y": 10, "width": 30, "height": 30}

    @staticmethod
    def get_annotation_color(annotation_type: str) -> str:
        """Get color for annotation type."""
        colors = {
            'highlight': 'rgba(255, 235, 59, 0.4)',   # Yellow
            'circle': 'rgba(33, 150, 243, 0.5)',      # Blue
            'box': 'rgba(76, 175, 80, 0.3)',          # Green
            'underline': 'rgba(244, 67, 54, 0.5)'     # Red
        }
        return colors.get(annotation_type, colors['highlight'])


# Process-wide singleton. AnnotationService holds no state, so a single
# instance is safe to reuse; matches the get_rerank_service() convention.
_service: Optional[AnnotationService] = None


def get_annotation_service() -> AnnotationService:
    """Return the process-wide AnnotationService singleton."""
    global _service
    if _service is None:
        _service = AnnotationService()
    return _service
