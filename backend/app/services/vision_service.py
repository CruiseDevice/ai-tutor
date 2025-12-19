"""
Vision service for generating image captions using GPT-4o Vision API.
"""

import logging
import asyncio
import base64
from dataclasses import dataclass
from typing import List, Optional
from io import BytesIO
import openai
from openai import AsyncOpenAI
import json

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ImageCaption:
    """Data structure for image caption with detailed metadata."""
    short_caption: str  # 1-2 sentence description
    detailed_description: str  # Detailed explanation of content
    ocr_text: Optional[str] = None  # Text found in image
    tags: List[str] = None  # Descriptive tags

    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = []


class VisionService:
    """Service for generating captions for images using GPT-4o Vision."""

    def __init__(self):
        """Initialize Vision service with OpenAI client."""
        self.client = AsyncOpenAI()
        self.model = settings.VISION_MODEL
        self.batch_size = settings.IMAGE_CAPTION_BATCH_SIZE

        # Rate limiting semaphore
        self.rate_limit_semaphore = asyncio.Semaphore(self.batch_size)

    def _create_caption_prompt(self, page_number: int) -> str:
        """
        Create a prompt for GPT-4o Vision to analyze an educational document image.

        Args:
            page_number: Page number where the image appears

        Returns:
            Prompt string for vision API
        """
        prompt = f"""Analyze this image from page {page_number} of an educational document.

Provide a comprehensive analysis in the following JSON format:

{{
  "short_caption": "A concise 1-2 sentence description of what the image shows",
  "detailed_description": "A detailed explanation of the image content, purpose, key elements, and educational value. Include:\n- What the image depicts\n- Important details and features\n- How it relates to educational content\n- Any data, relationships, or concepts it illustrates",
  "ocr_text": "Any readable text found in the image (labels, titles, data, equations, etc.)",
  "tags": ["tag1", "tag2", "tag3"]
}}

Tags should describe the image type and content, such as:
- Type: chart, graph, diagram, flowchart, table, screenshot, photo, illustration, formula, equation
- Subject: biology, chemistry, physics, math, history, computer-science, etc.
- Specifics: bar-chart, line-graph, scatter-plot, pie-chart, venn-diagram, timeline, etc.

Focus on educational relevance and accuracy. Return ONLY valid JSON."""

        return prompt

    async def generate_image_caption(
        self,
        image_data: bytes,
        page_number: int,
        retry_count: int = 3
    ) -> Optional[ImageCaption]:
        """
        Generate a caption for a single image using GPT-4o Vision.

        Args:
            image_data: Image bytes (PNG/JPEG)
            page_number: Page number where image appears
            retry_count: Number of retries on failure

        Returns:
            ImageCaption object or None if failed
        """
        async with self.rate_limit_semaphore:
            for attempt in range(retry_count):
                try:
                    # Encode image to base64
                    base64_image = base64.b64encode(image_data).decode('utf-8')

                    # Create prompt
                    prompt = self._create_caption_prompt(page_number)

                    # Call GPT-4o Vision API
                    logger.debug(f"Calling vision API for page {page_number} (attempt {attempt + 1})")

                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}",
                                            "detail": "high"  # High detail for better analysis
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.3  # Lower temperature for more consistent results
                    )

                    # Extract response
                    content = response.choices[0].message.content.strip()

                    # Parse JSON response
                    caption_data = self._parse_vision_response(content)

                    if caption_data:
                        logger.debug(f"Successfully generated caption for page {page_number}")
                        return caption_data

                    logger.warning(f"Failed to parse caption for page {page_number}, retrying...")

                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit hit for page {page_number}: {e}")
                    if attempt < retry_count - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries reached for page {page_number}")
                        return None

                except openai.APIError as e:
                    logger.error(f"OpenAI API error for page {page_number}: {e}")
                    if attempt < retry_count - 1:
                        await asyncio.sleep(1)
                    else:
                        return None

                except Exception as e:
                    logger.error(f"Unexpected error generating caption for page {page_number}: {e}", exc_info=True)
                    if attempt < retry_count - 1:
                        await asyncio.sleep(1)
                    else:
                        return None

            return None

    def _parse_vision_response(self, content: str) -> Optional[ImageCaption]:
        """
        Parse the vision API response into an ImageCaption object.

        Args:
            content: Response content from vision API

        Returns:
            ImageCaption object or None if parsing failed
        """
        try:
            # Try to extract JSON from response
            # Sometimes the model returns JSON wrapped in markdown code blocks
            if "```json" in content:
                # Extract JSON from code block
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                # Extract from generic code block
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            else:
                # Assume the whole content is JSON
                json_str = content.strip()

            # Parse JSON
            data = json.loads(json_str)

            # Create ImageCaption
            caption = ImageCaption(
                short_caption=data.get("short_caption", "").strip(),
                detailed_description=data.get("detailed_description", "").strip(),
                ocr_text=data.get("ocr_text", "").strip() or None,
                tags=data.get("tags", [])
            )

            # Validate required fields
            if not caption.short_caption or not caption.detailed_description:
                logger.warning("Caption missing required fields")
                return None

            return caption

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from vision response: {e}")
            logger.debug(f"Response content: {content}")
            return None
        except Exception as e:
            logger.error(f"Error parsing vision response: {e}", exc_info=True)
            return None

    async def batch_generate_captions(
        self,
        images: List[dict]
    ) -> List[Optional[ImageCaption]]:
        """
        Generate captions for multiple images in parallel batches.

        Args:
            images: List of dicts with 'image_data' (bytes) and 'page_number' (int)

        Returns:
            List of ImageCaption objects (same order as input, None for failures)
        """
        if not images:
            return []

        logger.info(f"Generating captions for {len(images)} images in batches of {self.batch_size}")

        # Create tasks for all images
        tasks = []
        for img in images:
            task = self.generate_image_caption(
                image_data=img['image_data'],
                page_number=img.get('page_number', 0)
            )
            tasks.append(task)

        # Process all tasks concurrently (semaphore controls concurrency)
        captions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for idx, caption in enumerate(captions):
            if isinstance(caption, Exception):
                logger.error(f"Exception generating caption for image {idx}: {caption}")
                results.append(None)
            else:
                results.append(caption)

        # Log statistics
        successful = sum(1 for c in results if c is not None)
        failed = len(results) - successful
        logger.info(f"Caption generation complete: {successful} successful, {failed} failed")

        return results

    async def generate_captions_with_progress(
        self,
        images: List[dict],
        progress_callback=None
    ) -> List[Optional[ImageCaption]]:
        """
        Generate captions with progress tracking.

        Args:
            images: List of image dicts
            progress_callback: Optional callback function(current, total, caption)

        Returns:
            List of ImageCaption objects
        """
        results = []
        total = len(images)

        for idx, img in enumerate(images, 1):
            caption = await self.generate_image_caption(
                image_data=img['image_data'],
                page_number=img.get('page_number', 0)
            )

            results.append(caption)

            if progress_callback:
                progress_callback(idx, total, caption)

            logger.info(f"Progress: {idx}/{total} images captioned")

        return results
