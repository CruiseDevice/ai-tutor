"""
Density Calculator Service for Adaptive Chunk Sizing

This service analyzes content density to determine optimal chunk sizes:
- Dense content (tables, code, formulas): Smaller chunks for precision
- Sparse content (narrative text): Larger chunks for context

Strategy:
- Calculate multiple density metrics (tokens, punctuation, special chars, etc.)
- Combine metrics into overall density score (0.0 - 1.0)
- Recommend chunk sizes based on density (high density = smaller chunks)
"""
import re
import logging
from typing import Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DensityMetrics:
    """Container for content density measurements."""

    # Individual metrics (0.0 - 1.0)
    token_density: float  # Ratio of tokens to characters
    punctuation_density: float  # Ratio of punctuation to characters
    special_char_density: float  # Ratio of special chars (math, code) to characters
    numeric_density: float  # Ratio of digits to characters
    line_break_density: float  # Ratio of line breaks to characters
    whitespace_density: float  # Ratio of whitespace to characters

    # Overall density score (0.0 - 1.0)
    # High score = dense content (tables, code, formulas)
    # Low score = sparse content (narrative text)
    overall_density: float

    # Content type classification
    content_type: str  # 'narrative', 'technical', 'structured', 'mixed'

    # Recommended chunk size based on density
    recommended_chunk_size: int
    recommended_overlap: int


class DensityCalculatorService:
    """
    Service for calculating content density and recommending chunk sizes.

    Analyzes text characteristics to determine optimal chunking strategy:
    - High-density content (>0.6): Tables, code, formulas → Smaller chunks (500-800 chars)
    - Medium-density content (0.4-0.6): Technical text → Standard chunks (1000-1200 chars)
    - Low-density content (<0.4): Narrative text → Larger chunks (1500-2000 chars)
    """

    def __init__(
        self,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        default_chunk_size: int = 1000,
        min_overlap: int = 50,
        max_overlap: int = 400,
        default_overlap: int = 200
    ):
        """
        Initialize the density calculator.

        Args:
            min_chunk_size: Minimum chunk size for dense content
            max_chunk_size: Maximum chunk size for sparse content
            default_chunk_size: Default chunk size for medium-density content
            min_overlap: Minimum overlap size
            max_overlap: Maximum overlap size
            default_overlap: Default overlap size
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.default_chunk_size = default_chunk_size
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.default_overlap = default_overlap

        logger.info(
            f"DensityCalculatorService initialized: "
            f"chunk_size=[{min_chunk_size}, {max_chunk_size}], "
            f"overlap=[{min_overlap}, {max_overlap}]"
        )

    def calculate_density(self, text: str) -> DensityMetrics:
        """
        Calculate content density metrics for a text segment.

        Args:
            text: Text content to analyze

        Returns:
            DensityMetrics object with detailed measurements and recommendations
        """
        if not text or not text.strip():
            # Empty text - return default metrics
            return DensityMetrics(
                token_density=0.0,
                punctuation_density=0.0,
                special_char_density=0.0,
                numeric_density=0.0,
                line_break_density=0.0,
                whitespace_density=0.0,
                overall_density=0.0,
                content_type='empty',
                recommended_chunk_size=self.default_chunk_size,
                recommended_overlap=self.default_overlap
            )

        # Calculate individual metrics
        token_density = self._calculate_token_density(text)
        punctuation_density = self._calculate_punctuation_density(text)
        special_char_density = self._calculate_special_char_density(text)
        numeric_density = self._calculate_numeric_density(text)
        line_break_density = self._calculate_line_break_density(text)
        whitespace_density = self._calculate_whitespace_density(text)

        # Calculate overall density (weighted combination)
        overall_density = self._calculate_overall_density(
            token_density=token_density,
            punctuation_density=punctuation_density,
            special_char_density=special_char_density,
            numeric_density=numeric_density,
            line_break_density=line_break_density,
            whitespace_density=whitespace_density
        )

        # Classify content type
        content_type = self._classify_content_type(
            overall_density=overall_density,
            special_char_density=special_char_density,
            numeric_density=numeric_density,
            line_break_density=line_break_density
        )

        # Recommend chunk size based on density
        chunk_size, overlap = self._recommend_chunk_size(overall_density)

        metrics = DensityMetrics(
            token_density=token_density,
            punctuation_density=punctuation_density,
            special_char_density=special_char_density,
            numeric_density=numeric_density,
            line_break_density=line_break_density,
            whitespace_density=whitespace_density,
            overall_density=overall_density,
            content_type=content_type,
            recommended_chunk_size=chunk_size,
            recommended_overlap=overlap
        )

        logger.debug(
            f"Density calculated: overall={overall_density:.2f}, type={content_type}, "
            f"chunk_size={chunk_size}, overlap={overlap}"
        )

        return metrics

    def _calculate_token_density(self, text: str) -> float:
        """
        Calculate token-to-character ratio.

        High ratio (>0.8): Dense text with many short words
        Low ratio (<0.6): Sparse text with long words or lots of whitespace

        Returns:
            Token density score (0.0 - 1.0)
        """
        # Simple tokenization: split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        num_tokens = len(tokens)
        num_chars = len(text)

        if num_chars == 0:
            return 0.0

        # Normalize: typical ratio is ~0.15-0.20 (1 token per 5-7 chars)
        # Map to 0.0-1.0 range
        ratio = num_tokens / num_chars
        normalized = min(ratio / 0.25, 1.0)  # 0.25 tokens/char = max density

        return normalized

    def _calculate_punctuation_density(self, text: str) -> float:
        """
        Calculate ratio of punctuation characters to total characters.

        High ratio (>0.15): Lists, tables, structured content
        Low ratio (<0.05): Plain narrative text

        Returns:
            Punctuation density score (0.0 - 1.0)
        """
        punctuation_chars = set('.,;:!?()[]{}"-')
        num_punctuation = sum(1 for c in text if c in punctuation_chars)
        num_chars = len(text)

        if num_chars == 0:
            return 0.0

        # Normalize: typical punctuation is ~5-10%
        # Map to 0.0-1.0 range
        ratio = num_punctuation / num_chars
        normalized = min(ratio / 0.20, 1.0)  # 20% punctuation = max density

        return normalized

    def _calculate_special_char_density(self, text: str) -> float:
        """
        Calculate ratio of special characters (math, code symbols) to total characters.

        High ratio (>0.10): Code, formulas, technical content
        Low ratio (<0.02): Plain text

        Returns:
            Special character density score (0.0 - 1.0)
        """
        # Special chars common in code, math, tables
        special_chars = set('$%&*+=<>|~^_\\/#@`')
        num_special = sum(1 for c in text if c in special_chars)
        num_chars = len(text)

        if num_chars == 0:
            return 0.0

        # Normalize: typical special char usage is <5%
        # Map to 0.0-1.0 range
        ratio = num_special / num_chars
        normalized = min(ratio / 0.15, 1.0)  # 15% special chars = max density

        return normalized

    def _calculate_numeric_density(self, text: str) -> float:
        """
        Calculate ratio of numeric characters to total characters.

        High ratio (>0.20): Tables, data, technical content
        Low ratio (<0.05): Narrative text

        Returns:
            Numeric density score (0.0 - 1.0)
        """
        num_digits = sum(1 for c in text if c.isdigit())
        num_chars = len(text)

        if num_chars == 0:
            return 0.0

        # Normalize: typical numeric content is <10%
        # Map to 0.0-1.0 range
        ratio = num_digits / num_chars
        normalized = min(ratio / 0.30, 1.0)  # 30% digits = max density

        return normalized

    def _calculate_line_break_density(self, text: str) -> float:
        """
        Calculate ratio of line breaks to characters.

        High ratio (>0.05): Lists, tables, structured content
        Low ratio (<0.01): Continuous paragraphs

        Returns:
            Line break density score (0.0 - 1.0)
        """
        num_line_breaks = text.count('\n')
        num_chars = len(text)

        if num_chars == 0:
            return 0.0

        # Normalize: typical line break density is ~1-3%
        # Map to 0.0-1.0 range
        ratio = num_line_breaks / num_chars
        normalized = min(ratio / 0.10, 1.0)  # 10% line breaks = max density

        return normalized

    def _calculate_whitespace_density(self, text: str) -> float:
        """
        Calculate ratio of whitespace to total characters.

        High ratio (>0.25): Sparse formatting, lots of spacing
        Low ratio (<0.15): Dense text

        Returns:
            Whitespace density score (0.0 - 1.0)
        """
        num_whitespace = sum(1 for c in text if c.isspace())
        num_chars = len(text)

        if num_chars == 0:
            return 0.0

        # Normalize: typical whitespace is ~15-20%
        # Map to 0.0-1.0 range (inverted - more whitespace = less density)
        ratio = num_whitespace / num_chars
        # Invert: high whitespace = low density
        normalized = max(0.0, 1.0 - (ratio / 0.30))  # 30% whitespace = min density

        return normalized

    def _calculate_overall_density(
        self,
        token_density: float,
        punctuation_density: float,
        special_char_density: float,
        numeric_density: float,
        line_break_density: float,
        whitespace_density: float
    ) -> float:
        """
        Calculate overall density score as weighted combination of individual metrics.

        Weights are loaded from settings and tuned based on empirical analysis:
        - High special chars, numbers, line breaks → Structured content (tables, code)
        - High punctuation → Lists, structured content
        - High tokens, low whitespace → Dense narrative
        - Low everything → Sparse narrative

        Returns:
            Overall density score (0.0 - 1.0)
        """
        from ..config import settings

        # Weights for each metric (loaded from config, sum to 1.0)
        weights = {
            'special_char': settings.DENSITY_WEIGHT_SPECIAL_CHAR,
            'numeric': settings.DENSITY_WEIGHT_NUMERIC,
            'line_break': settings.DENSITY_WEIGHT_LINE_BREAK,
            'punctuation': settings.DENSITY_WEIGHT_PUNCTUATION,
            'token': settings.DENSITY_WEIGHT_TOKEN,
            'whitespace': settings.DENSITY_WEIGHT_WHITESPACE
        }

        # Calculate weighted sum
        overall = (
            weights['special_char'] * special_char_density +
            weights['numeric'] * numeric_density +
            weights['line_break'] * line_break_density +
            weights['punctuation'] * punctuation_density +
            weights['token'] * token_density +
            weights['whitespace'] * whitespace_density
        )

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, overall))

    def _classify_content_type(
        self,
        overall_density: float,
        special_char_density: float,
        numeric_density: float,
        line_break_density: float
    ) -> str:
        """
        Classify content type based on density metrics.

        Returns:
            Content type: 'narrative', 'technical', 'structured', or 'mixed'
        """
        from ..config import settings

        # Structured content: High special chars, numbers, or line breaks
        if special_char_density > 0.4 or numeric_density > 0.5 or line_break_density > 0.6:
            return 'structured'  # Tables, code, lists

        # Technical content: Moderate special chars or numbers
        if special_char_density > 0.2 or numeric_density > 0.3:
            return 'technical'  # Formulas, technical text

        # High overall density but not structured/technical
        if overall_density > settings.ADAPTIVE_DENSITY_HIGH_THRESHOLD:
            return 'mixed'  # Dense narrative with some structure

        # Low density
        return 'narrative'  # Plain paragraphs, continuous text

    def _recommend_chunk_size(self, overall_density: float) -> Tuple[int, int]:
        """
        Recommend chunk size and overlap based on overall density.

        Strategy:
        - High density (>0.6): Smaller chunks for precision (500-800 chars)
        - Medium density (0.4-0.6): Standard chunks (1000-1200 chars)
        - Low density (<0.4): Larger chunks for context (1500-2000 chars)

        Returns:
            Tuple of (chunk_size, overlap)
        """
        # Linear interpolation based on density
        # density=1.0 → min_chunk_size
        # density=0.0 → max_chunk_size
        chunk_size_range = self.max_chunk_size - self.min_chunk_size
        chunk_size = self.max_chunk_size - int(overall_density * chunk_size_range)

        # Ensure within bounds
        chunk_size = max(self.min_chunk_size, min(self.max_chunk_size, chunk_size))

        # Overlap scales with chunk size (15% of chunk size)
        overlap_percentage = 0.15
        overlap = int(chunk_size * overlap_percentage)

        # Ensure overlap within bounds
        overlap = max(self.min_overlap, min(self.max_overlap, overlap))

        return chunk_size, overlap


def get_density_calculator_service() -> DensityCalculatorService:
    """
    Get or create a singleton instance of the density calculator service.

    Returns:
        DensityCalculatorService instance
    """
    from ..config import settings

    return DensityCalculatorService(
        min_chunk_size=settings.CHUNK_SIZE_MIN,
        max_chunk_size=settings.CHUNK_SIZE_MAX,
        default_chunk_size=settings.CHUNK_SIZE_DEFAULT,
        min_overlap=settings.CHUNK_OVERLAP_MIN,
        max_overlap=settings.CHUNK_OVERLAP_MAX,
        default_overlap=settings.CHUNK_OVERLAP_DEFAULT
    )
