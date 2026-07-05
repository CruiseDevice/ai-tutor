"""Answer-quality evaluation: citation verification and LLM-based scoring.

Extracted from ChatService. This module owns:
- verifying that [Page X] citations and annotation page numbers in a response
  correspond to chunks actually retrieved, and
- scoring the generated answer on accuracy / completeness / clarity /
  citation quality via a helper-LLM call.

`verify_citations` is pure (no LLM, no DB). `score_answer_quality` reads app
settings and calls the LLM via the shared client; dependencies (api_key,
provider) are passed in. A process-wide singleton is exposed via
`get_quality_service()`, mirroring the other extracted services.
"""
import json
import logging
import re
from typing import Dict, List

from ..config import settings
from .llm import Provider, get_llm_client, pick_helper_model
from .retry_utils import async_retry_openai_call

logger = logging.getLogger(__name__)


class QualityService:
    """Verify citations and score answer quality."""

    def verify_citations(
        self,
        response_text: str,
        annotations: List[Dict],
        relevant_chunks: List[Dict],
    ) -> List[str]:
        """
        Verify that citations in the response match available chunks.

        Args:
            response_text: The generated response text
            annotations: List of annotation dictionaries
            relevant_chunks: List of chunk dictionaries used in the response

        Returns:
            List of warning messages for citation mismatches
        """
        # Skip if citation verification is disabled
        if not settings.ENABLE_CITATION_VERIFICATION:
            return []

        warnings = []

        # Extract page numbers from chunks
        available_pages = set()
        for chunk in relevant_chunks:
            page_num = chunk.get("pageNumber")
            if page_num is not None:
                available_pages.add(int(page_num))

        # Extract [Page X] citations from response text
        citation_pattern = r'\[Page\s+(\d+)\]'
        cited_pages = set()
        for match in re.finditer(citation_pattern, response_text):
            page_num = int(match.group(1))
            cited_pages.add(page_num)

            # Check if cited page is in available chunks
            if page_num not in available_pages:
                warning = f"Citation [Page {page_num}] in response does not match available chunks"
                warnings.append(warning)
                logger.warning(warning)

        # Extract page numbers from annotations
        annotated_pages = set()
        for i, annotation in enumerate(annotations):
            page_num = annotation.get("pageNumber")
            if page_num is not None:
                try:
                    page_num = int(page_num)
                    annotated_pages.add(page_num)

                    # Check if annotated page is in available chunks
                    if page_num not in available_pages:
                        warning = f"Annotation #{i+1} page number {page_num} does not match available chunks"
                        warnings.append(warning)
                        logger.warning(warning)
                except (ValueError, TypeError):
                    warning = f"Annotation #{i+1} has invalid page number: {page_num}"
                    warnings.append(warning)
                    logger.warning(warning)

        # Check if there are citations without corresponding annotations
        uncovered_citations = cited_pages - annotated_pages
        if uncovered_citations:
            warning = f"Pages cited but not annotated: {sorted(uncovered_citations)}"
            warnings.append(warning)
            logger.info(warning)  # Info level since this is less critical

        # Log summary
        if warnings:
            logger.warning(f"Citation verification found {len(warnings)} issue(s)")
        else:
            logger.info("Citation verification passed - all citations match available chunks")

        return warnings

    async def score_answer_quality(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict],
        user_api_key: str,
        provider: Provider | None = None,
    ) -> Dict[str, "any"]:
        """
        Evaluate answer quality using LLM scoring on multiple dimensions.

        Args:
            query: The user's question
            answer: The generated answer
            context_chunks: List of context chunks used
            user_api_key: User's OpenAI API key

        Returns:
            Dict with scores and feedback:
            {
                "accuracy": <0-10>,
                "completeness": <0-10>,
                "clarity": <0-10>,
                "citation_quality": <0-10>,
                "overall": <0-10>,
                "feedback": "<text feedback>"
            }
        """
        # Skip if quality scoring is disabled
        if not settings.ENABLE_ANSWER_QUALITY_SCORING:
            return {
                "accuracy": None,
                "completeness": None,
                "clarity": None,
                "citation_quality": None,
                "overall": None,
                "feedback": "Quality scoring disabled"
            }

        try:
            prov = provider or Provider.OPENAI
            client = get_llm_client(prov, user_api_key)
            model = pick_helper_model(prov)

            # Create context summary (first 500 chars of each chunk)
            context_summary = "\n\n".join([
                f"[Page {chunk.get('pageNumber', '?')}]: {chunk.get('content', '')[:500]}..."
                for chunk in context_chunks[:3]  # Only include first 3 chunks to save tokens
            ])

            scoring_prompt = f"""Evaluate this answer on a scale of 0-10 for multiple quality dimensions.

**Question**: {query}

**Answer**: {answer}

**Available Context** (excerpt):
{context_summary}

Evaluate on these dimensions (0-10 scale):

1. **Accuracy** (0-10): Does the answer correctly address the question based on the context? Are there any factual errors?
2. **Completeness** (0-10): Does it cover all important aspects of the question? Is anything critical missing?
3. **Clarity** (0-10): Is it well-structured, easy to understand, and well-written?
4. **Citation Quality** (0-10): Are citations accurate, relevant, and properly formatted?

Respond with ONLY a JSON object in this exact format:
{{
  "accuracy": <score 0-10>,
  "completeness": <score 0-10>,
  "clarity": <score 0-10>,
  "citation_quality": <score 0-10>,
  "overall": <average of above scores>,
  "feedback": "<brief 1-2 sentence feedback on strengths and areas for improvement>"
}}"""

            async def _create_completion():
                return await client.complete(
                    system_prompt="You are an answer quality evaluator that outputs only valid JSON.",
                    messages=[{"role": "user", "content": scoring_prompt}],
                    model=model,
                    temperature=0.3,  # Low temperature for consistent scoring
                    max_tokens=200,
                )

            response_text = await async_retry_openai_call(
                _create_completion,
                max_attempts=2,  # Fewer retries for non-critical scoring
                initial_wait=1.0,
                max_wait=20.0
            )

            response_text = response_text.strip()

            # Parse JSON response
            scores = json.loads(response_text)

            # Validate scores are in range
            for key in ["accuracy", "completeness", "clarity", "citation_quality", "overall"]:
                if key in scores:
                    score = scores[key]
                    if not isinstance(score, (int, float)) or score < 0 or score > 10:
                        scores[key] = None

            logger.info(
                f"Answer quality scores - Accuracy: {scores.get('accuracy')}, "
                f"Completeness: {scores.get('completeness')}, "
                f"Clarity: {scores.get('clarity')}, "
                f"Citation: {scores.get('citation_quality')}, "
                f"Overall: {scores.get('overall')}"
            )

            return scores

        except Exception as e:
            logger.warning(f"Failed to score answer quality: {e}")
            return {
                "accuracy": None,
                "completeness": None,
                "clarity": None,
                "citation_quality": None,
                "overall": None,
                "feedback": f"Quality scoring failed: {str(e)}"
            }


# Process-wide singleton. QualityService holds no instance state; the singleton
# mirrors the get_annotation_service() / get_prompt_builder() convention.
_service: QualityService | None = None


def get_quality_service() -> QualityService:
    """Return the process-wide QualityService singleton."""
    global _service
    if _service is None:
        _service = QualityService()
    return _service
