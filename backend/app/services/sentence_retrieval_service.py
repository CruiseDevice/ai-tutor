from typing import List, Dict, Any, Optional
import logging
import re
from app.config import settings

logger = logging.getLogger(__name__)


class SentenceRetrievalService:
    """
    Service for extracting and processing critical sentences for ultra-precise retrieval.

    This service identifies sentences that are likely to contain critical details such as:
    - Definitions and explanations
    - Statistical facts and numerical data
    - Key findings and conclusions
    - Important statements and assertions
    - References to figures, tables, and equations

    Use cases:
    - Answering specific factual questions
    - Retrieving precise definitions
    - Finding exact statistics and numbers
    - Locating key conclusions and findings

    Example:
        "The success rate was 87.3% in the control group." → Extracted as critical
        "Additionally, participants showed improvement." → May be skipped if not critical
    """

    def __init__(self):
        """Initialize the sentence retrieval service."""
        logger.info("SentenceRetrievalService initialized")

        # Patterns indicating critical information
        self.critical_patterns = [
            # Definitions
            r'\b(is defined as|refers to|means that|is known as|can be described as)\b',

            # Numbers and percentages
            r'\b(\d+\.?\d*%|\d+\.\d+|\d{2,})\b',

            # Conclusions and findings
            r'\b(conclude that|found that|shows that|demonstrates|indicates that|suggests that)\b',
            r'\b(results show|evidence suggests|data indicate|analysis reveals)\b',

            # Importance markers
            r'\b(key finding|main result|important|critical|essential|significant|notable)\b',
            r'\b(primary|fundamental|crucial|vital|major)\b',

            # References to visual elements
            r'\b(figure \d+|table \d+|equation \d+|chart \d+|graph \d+)\b',
            r'\b(see figure|as shown in|according to table)\b',

            # Statistical terms
            r'\b(average|mean|median|standard deviation|correlation|p-value|significance)\b',
            r'\b(hypothesis|probability|distribution|sample size)\b',

            # Comparison and relationships
            r'\b(compared to|in contrast|whereas|however|although|while)\b',
            r'\b(relationship between|correlation with|associated with|linked to)\b',

            # Methodology
            r'\b(method|procedure|approach|technique|algorithm|protocol)\b',

            # Causal language
            r'\b(because|therefore|thus|hence|consequently|as a result)\b',
            r'\b(leads to|results in|causes|affects|influences)\b',
        ]

    def extract_critical_sentences(
        self,
        pages: List[Any],
        min_sentence_length: int = 10,
        max_sentence_length: int = 500,
        include_short_sentences: bool = True
    ) -> List[Dict]:
        """
        Extract sentences that are likely to contain critical details.

        Args:
            pages: List of page objects with page_content and metadata
            min_sentence_length: Minimum character length for a sentence (default: 10)
            max_sentence_length: Maximum character length for a sentence (default: 500)
            include_short_sentences: Whether to include short sentences as potentially critical (default: True)

        Returns:
            List of dictionaries containing:
                - content: The sentence text
                - page_number: Page number where sentence appears
                - sentence_index: Index of sentence within the page
                - chunk_type: Always 'sentence'
                - chunk_level: Always 'sentence'
                - metadata: Additional metadata (is_critical, char_count, matched_patterns)
        """
        try:
            # Import sentence tokenizer (using simple split as fallback if nltk not available)
            try:
                from nltk.tokenize import sent_tokenize
                tokenize_fn = sent_tokenize
                logger.debug("Using nltk sentence tokenizer")
            except ImportError:
                logger.warning("NLTK not available, using simple sentence tokenizer")
                tokenize_fn = self._simple_sentence_tokenize

            critical_sentences = []

            for page_idx, page in enumerate(pages):
                # Extract page content
                page_content = page.page_content if hasattr(page, 'page_content') else str(page)

                # Get page number from metadata
                page_num = page.metadata.get('page', page_idx) + 1 if hasattr(page, 'metadata') else page_idx + 1

                # Tokenize into sentences
                sentences = tokenize_fn(page_content)

                for sent_idx, sentence in enumerate(sentences):
                    # Clean up sentence
                    sentence = sentence.strip()

                    # Skip if too short or too long
                    if len(sentence) < min_sentence_length or len(sentence) > max_sentence_length:
                        continue

                    # Check if sentence matches critical patterns
                    matched_patterns = []
                    for pattern in self.critical_patterns:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            matched_patterns.append(pattern)

                    is_critical = len(matched_patterns) > 0

                    # Short sentences (< 150 chars) are often key information
                    is_short = len(sentence) < 150

                    # Include if critical OR short (and short sentences enabled)
                    if is_critical or (is_short and include_short_sentences):
                        critical_sentences.append({
                            'content': sentence,
                            'page_number': page_num,
                            'sentence_index': sent_idx,
                            'chunk_type': 'sentence',
                            'chunk_level': 'sentence',
                            'metadata': {
                                'is_critical': is_critical,
                                'is_short': is_short,
                                'char_count': len(sentence),
                                'matched_patterns': len(matched_patterns),
                                'page_index': page_idx
                            }
                        })

            logger.info(
                f"Extracted {len(critical_sentences)} critical sentences from {len(pages)} pages "
                f"(avg: {len(critical_sentences) / len(pages):.1f} sentences/page)"
            )

            return critical_sentences

        except Exception as e:
            logger.error(f"Error extracting critical sentences: {e}", exc_info=True)
            return []

    def _simple_sentence_tokenize(self, text: str) -> List[str]:
        """
        Simple sentence tokenizer as fallback when nltk is not available.

        Splits on common sentence boundaries: . ! ? followed by space and capital letter.

        Args:
            text: Text to tokenize

        Returns:
            List of sentences
        """
        # Split on period, exclamation, or question mark followed by space
        # Use positive lookahead to keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def is_detail_query(self, query: str) -> bool:
        """
        Detect if query is asking for specific detail that would benefit from sentence-level retrieval.

        Args:
            query: User query text

        Returns:
            True if query appears to be asking for specific details
        """
        detail_indicators = [
            # Exact/specific information requests
            r'\b(what is the exact|specific|precisely|particular)\b',
            r'\b(exactly|specifically|in particular)\b',

            # Definition requests
            r'\b(definition of|what is|what are|define|meaning of)\b',

            # Numerical requests
            r'\b(how much|how many|what percentage|what number)\b',
            r'\b(what rate|what value|what amount)\b',

            # Statistical requests
            r'\b(statistic|number|value|amount|rate|percentage|figure)\b',
            r'\b(average|mean|median|total|sum)\b',

            # Citation/reference requests
            r'\b(which (figure|table|equation|chart))\b',
            r'\b(according to|as shown in|reference to)\b',
        ]

        return any(re.search(pattern, query, re.IGNORECASE) for pattern in detail_indicators)

    def rank_sentences_by_relevance(
        self,
        sentences: List[Dict],
        query: str,
        boost_critical: bool = True
    ) -> List[Dict]:
        """
        Rank sentences by relevance to the query.

        This is a simple keyword-based ranking. For more sophisticated ranking,
        this should be replaced with embedding-based similarity or reranking.

        Args:
            sentences: List of sentence dictionaries
            query: User query
            boost_critical: Whether to boost sentences with critical patterns

        Returns:
            Sentences sorted by relevance score (highest first)
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))

        for sentence in sentences:
            content_lower = sentence['content'].lower()
            content_terms = set(re.findall(r'\w+', content_lower))

            # Calculate overlap score
            overlap = len(query_terms & content_terms)
            total_terms = len(query_terms)

            base_score = overlap / max(total_terms, 1) if total_terms > 0 else 0

            # Apply boosts
            if boost_critical and sentence['metadata'].get('is_critical', False):
                base_score *= 1.3

            # Boost if contains numbers and query asks for numbers
            if re.search(r'\d', sentence['content']) and re.search(r'\b(how much|how many|number|percentage)\b', query, re.IGNORECASE):
                base_score *= 1.2

            sentence['relevance_score'] = base_score

        # Sort by relevance score
        ranked = sorted(sentences, key=lambda x: x.get('relevance_score', 0), reverse=True)

        return ranked

    def prepare_sentences_for_storage(
        self,
        sentences: List[Dict],
        document_id: str,
        course_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Prepare extracted sentences for database storage.

        Adds document_id and course_id to each sentence dictionary.

        Args:
            sentences: List of sentence dictionaries from extract_critical_sentences
            document_id: ID of the document these sentences belong to
            course_id: Optional course ID

        Returns:
            Sentences with added document_id and course_id fields
        """
        for sentence in sentences:
            sentence['document_id'] = document_id
            if course_id:
                sentence['course_id'] = course_id

        return sentences

    def deduplicate_sentences(
        self,
        sentences: List[Dict],
        similarity_threshold: float = 0.9
    ) -> List[Dict]:
        """
        Remove duplicate or very similar sentences.

        Currently uses exact string matching. Could be enhanced with
        embedding-based similarity for more sophisticated deduplication.

        Args:
            sentences: List of sentence dictionaries
            similarity_threshold: Similarity threshold for deduplication (not used in current implementation)

        Returns:
            Deduplicated list of sentences
        """
        seen_contents = set()
        deduplicated = []

        for sentence in sentences:
            content = sentence['content'].strip()

            # Exact match deduplication
            if content not in seen_contents:
                seen_contents.add(content)
                deduplicated.append(sentence)

        logger.info(
            f"Deduplicated sentences: {len(sentences)} → {len(deduplicated)} "
            f"(removed {len(sentences) - len(deduplicated)} duplicates)"
        )

        return deduplicated
