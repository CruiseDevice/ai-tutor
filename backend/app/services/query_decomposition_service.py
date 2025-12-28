from openai import AsyncOpenAI, APIError
from typing import List, Optional
import logging
import json
from app.config import settings

logger = logging.getLogger(__name__)


class QueryDecompositionService:
    """
    Service for decomposing complex queries into atomic sub-queries.

    This service detects complex multi-part queries and breaks them down into
    simpler sub-queries that can be answered independently. The results are
    then combined using Reciprocal Rank Fusion (RRF) for improved retrieval.

    Example:
        "How does process A affect outcome B, and what are the alternatives?"
        →
        ["What is process A?",
         "What is outcome B?",
         "How does process A affect outcome B?",
         "What are alternatives to process A?"]
    """

    def __init__(self):
        """Initialize the query decomposition service."""
        self.model = settings.QUERY_EXPANSION_MODEL  # Reuse same model as query expansion
        self.temperature = 0.2  # Lower temperature for more focused decomposition
        logger.info(f"QueryDecompositionService initialized with model: {self.model}")

    async def decompose_query(
        self,
        query: str,
        user_api_key: str,
        max_sub_queries: int = 5
    ) -> List[str]:
        """
        Decompose complex query into atomic sub-queries.

        Args:
            query: The original complex query
            user_api_key: User's OpenAI API key
            max_sub_queries: Maximum number of sub-queries to generate (default: 5)

        Returns:
            List of sub-query strings. Returns [query] if query is not complex.
            Falls back to [query] on error.
        """
        # Validate inputs
        if not query or not query.strip():
            logger.warning("Empty query provided to decompose_query")
            return [query]

        # Check if query is complex enough to warrant decomposition
        if not self._is_complex_query(query):
            logger.info(f"Query not complex, skipping decomposition: '{query[:50]}...'")
            return [query]

        try:
            # Decompose the complex query
            sub_queries = await self._decompose_with_llm(
                query=query,
                user_api_key=user_api_key,
                max_sub_queries=max_sub_queries
            )

            if not sub_queries:
                logger.warning("No sub-queries generated, returning original query")
                return [query]

            logger.info(f"Decomposed complex query into {len(sub_queries)} sub-queries: '{query[:50]}...'")
            logger.debug(f"Sub-queries: {sub_queries}")

            return sub_queries

        except Exception as e:
            logger.error(f"Error decomposing query: {e}", exc_info=True)
            # Graceful fallback: return only original query
            return [query]

    def _is_complex_query(self, query: str) -> bool:
        """
        Detect if query is complex enough to warrant decomposition.

        A query is considered complex if it has 2 or more complexity indicators:
        - Long query (>15 words)
        - Multiple clauses (contains 'and' or 'or')
        - Multiple questions (multiple '?')
        - Comparative/relational words (compare, contrast, relationship, difference)

        Args:
            query: The query to check

        Returns:
            True if query is complex, False otherwise
        """
        complexity_indicators = [
            len(query.split()) > 15,  # Long query
            ' and ' in query.lower(),  # Multiple clauses with 'and'
            ' or ' in query.lower(),   # Multiple clauses with 'or'
            query.count('?') > 1,      # Multiple questions
            any(word in query.lower() for word in [
                'compare', 'contrast', 'relationship', 'difference',
                'versus', 'vs', 'both', 'either'
            ])
        ]

        complexity_score = sum(complexity_indicators)
        is_complex = complexity_score >= 2

        if is_complex:
            logger.debug(
                f"Query classified as complex (score: {complexity_score}/5): '{query[:50]}...'"
            )

        return is_complex

    async def _decompose_with_llm(
        self,
        query: str,
        user_api_key: str,
        max_sub_queries: int
    ) -> List[str]:
        """
        Decompose query using OpenAI LLM.

        Args:
            query: Original complex query
            user_api_key: User's OpenAI API key
            max_sub_queries: Maximum number of sub-queries to generate

        Returns:
            List of atomic sub-query strings
        """
        client = AsyncOpenAI(api_key=user_api_key)

        # Craft prompt for query decomposition
        system_prompt = """You are an expert at breaking down complex questions into simpler sub-questions.
Your task is to decompose complex queries into atomic sub-queries that can be answered independently.

Each sub-question should:
- Be atomic (answerable independently)
- Cover a specific aspect of the original question
- Maintain logical order
- Avoid redundancy
- Be clear and concise

Return ONLY a JSON array of sub-questions, nothing else."""

        user_prompt = f"""Break down this complex question into 3-5 simpler sub-questions that, when answered together, fully address the original question.

Original Question: "{query}"

Requirements:
- Each sub-question should be atomic (answerable independently)
- Cover all aspects of the original question
- Maintain logical order (start with foundational questions)
- No redundancy between sub-questions
- Keep each sub-question concise

Return ONLY a JSON array of sub-questions.
Format: ["sub-question 1", "sub-question 2", ...]"""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"} if "gpt-4" in self.model or "gpt-3.5" in self.model else None
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Try to parse as JSON
            try:
                parsed = json.loads(content)

                # Handle both direct array and object with array field
                if isinstance(parsed, list):
                    sub_queries = parsed
                elif isinstance(parsed, dict):
                    # Try common keys
                    sub_queries = (
                        parsed.get("sub_questions") or
                        parsed.get("sub_queries") or
                        parsed.get("questions") or
                        parsed.get("queries") or
                        list(parsed.values())[0] if parsed else []
                    )
                else:
                    sub_queries = []

                # Ensure we have strings
                sub_queries = [str(q).strip() for q in sub_queries if q]

                # Filter out empty or invalid sub-queries
                sub_queries = [q for q in sub_queries if q and len(q) > 3]

                # Limit to max_sub_queries
                sub_queries = sub_queries[:max_sub_queries]

                if not sub_queries:
                    logger.warning(f"No valid sub-queries generated from LLM response: {content[:200]}")

                return sub_queries

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"LLM response: {content}")

                # Fallback: try to extract sub-queries from text
                return self._extract_subqueries_from_text(content, max_sub_queries)

        except APIError as e:
            logger.error(f"OpenAI API error during query decomposition: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query decomposition: {e}", exc_info=True)
            raise

    def _extract_subqueries_from_text(self, text: str, max_queries: int) -> List[str]:
        """
        Fallback method to extract sub-queries from non-JSON text.

        Args:
            text: Text response from LLM
            max_queries: Maximum number of queries to extract

        Returns:
            List of extracted sub-queries
        """
        import re

        # Pattern 1: Quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted and len(quoted) >= 2:
            return [q.strip() for q in quoted[:max_queries] if len(q.strip()) > 3]

        # Pattern 2: Numbered lines (1. query, 2. query, etc.)
        numbered = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', text)
        if numbered:
            return [n.strip().strip('"') for n in numbered[:max_queries] if len(n.strip()) > 3]

        # Pattern 3: Lines starting with dash
        dashed = re.findall(r'-\s*(.+?)(?:\n|$)', text)
        if dashed:
            return [d.strip().strip('"') for d in dashed[:max_queries] if len(d.strip()) > 3]

        # Give up
        logger.warning("Could not extract sub-queries from text")
        return []


# Global instance
query_decomposition_service: Optional[QueryDecompositionService] = None


def get_query_decomposition_service() -> QueryDecompositionService:
    """Get the global query decomposition service instance."""
    global query_decomposition_service
    if query_decomposition_service is None:
        query_decomposition_service = QueryDecompositionService()
    return query_decomposition_service
