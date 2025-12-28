from openai import AsyncOpenAI, APIError
from typing import List, Optional
import logging
import json
import hashlib
from app.config import settings

logger = logging.getLogger(__name__)


class QueryExpansionService:
    """
    Service for generating query variations to improve retrieval accuracy.

    Uses LLM to generate semantically different variations of a query,
    which are then used in multi-query retrieval with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        """Initialize the query expansion service."""
        self.model = settings.QUERY_EXPANSION_MODEL
        self.num_variations = settings.QUERY_EXPANSION_NUM_VARIATIONS
        self.temperature = settings.QUERY_EXPANSION_TEMPERATURE
        logger.info(f"QueryExpansionService initialized with model: {self.model}")

    async def generate_query_variations(
        self,
        query: str,
        user_api_key: str,
        num_variations: Optional[int] = None,
        model: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple query variations for improved retrieval.

        Args:
            query: The original user query
            user_api_key: User's OpenAI API key
            num_variations: Number of variations to generate (default: from config)
            model: Model to use (default: from config)

        Returns:
            List of query strings including the original query
            Falls back to [query] on error
        """
        # Use defaults from config if not specified
        num_variations = num_variations or self.num_variations
        model = model or self.model

        # Validate inputs
        if not query or not query.strip():
            logger.warning("Empty query provided to generate_query_variations")
            return [query]

        if num_variations < 1:
            logger.warning(f"Invalid num_variations: {num_variations}, using 1")
            num_variations = 1

        # If only 1 variation requested, just return original query
        if num_variations == 1:
            return [query]

        try:
            # Generate variations using LLM
            variations = await self._generate_variations_with_llm(
                query=query,
                user_api_key=user_api_key,
                num_variations=num_variations - 1,  # Exclude original from count
                model=model
            )

            # Always include the original query
            all_queries = [query] + variations

            logger.info(f"Generated {len(variations)} variations for query: '{query[:50]}...'")
            logger.debug(f"Query variations: {all_queries}")

            return all_queries

        except Exception as e:
            logger.error(f"Error generating query variations: {e}", exc_info=True)
            # Graceful fallback: return only original query
            return [query]

    async def _generate_variations_with_llm(
        self,
        query: str,
        user_api_key: str,
        num_variations: int,
        model: str
    ) -> List[str]:
        """
        Generate query variations using OpenAI LLM.

        Args:
            query: Original query
            user_api_key: User's OpenAI API key
            num_variations: Number of variations to generate
            model: Model to use

        Returns:
            List of query variation strings
        """
        client = AsyncOpenAI(api_key=user_api_key)

        # Craft enhanced prompt to generate diverse query variations
        # This improved prompt generates more targeted variations across different semantic angles
        system_prompt = """You are an expert at reformulating user queries to improve information retrieval.
Your task is to generate diverse variations that capture different semantic angles of the original query.

Each variation should:
- Target a different aspect of the original query
- Preserve the core intent
- Use diverse vocabulary
- Be concise (max 20 words)

Return ONLY a JSON array of query strings, nothing else."""

        # Generate variations with specific purposes for better coverage
        if num_variations >= 4:
            variation_instructions = """
1. **Conceptual variation**: Rephrase using synonyms and related concepts
2. **Specific variation**: Make the query more specific and detailed
3. **Broad variation**: Make the query more general to catch related information
4. **Technical variation**: Use domain-specific or technical terminology"""
        elif num_variations == 3:
            variation_instructions = """
1. **Conceptual variation**: Rephrase using synonyms and related concepts
2. **Specific variation**: Make the query more specific and detailed
3. **Broad variation**: Make the query more general to catch related information"""
        elif num_variations == 2:
            variation_instructions = """
1. **Conceptual variation**: Rephrase using synonyms and related concepts
2. **Specific variation**: Make the query more specific and detailed"""
        else:
            variation_instructions = """
1. **Conceptual variation**: Rephrase using synonyms and related concepts"""

        user_prompt = f"""Original query: "{query}"

Generate {num_variations} alternative phrasings that capture different semantic angles:
{variation_instructions}

Requirements:
- Each variation should target different aspects of the original query
- Preserve the core intent
- Use diverse vocabulary
- Keep each variation concise (max 20 words)

Return ONLY a JSON array of {num_variations} strings, no explanation.
Format: ["variation 1", "variation 2", ...]"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"} if "gpt-4" in model or "gpt-3.5" in model else None
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Try to parse as JSON
            try:
                # Handle both direct array and object with array field
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    variations = parsed
                elif isinstance(parsed, dict):
                    # Try common keys
                    variations = (
                        parsed.get("variations") or
                        parsed.get("queries") or
                        parsed.get("results") or
                        list(parsed.values())[0] if parsed else []
                    )
                else:
                    variations = []

                # Ensure we have strings
                variations = [str(v).strip() for v in variations if v]

                # Filter out empty or invalid variations
                variations = [v for v in variations if v and len(v) > 3]

                # Limit to requested number
                variations = variations[:num_variations]

                if not variations:
                    logger.warning(f"No valid variations generated from LLM response: {content[:200]}")

                return variations

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"LLM response: {content}")

                # Fallback: try to extract variations from text
                return self._extract_variations_from_text(content, num_variations)

        except APIError as e:
            logger.error(f"OpenAI API error during query expansion: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query variation generation: {e}", exc_info=True)
            raise

    def _extract_variations_from_text(self, text: str, max_variations: int) -> List[str]:
        """
        Fallback method to extract query variations from non-JSON text.

        Args:
            text: Text response from LLM
            max_variations: Maximum number of variations to extract

        Returns:
            List of extracted variations
        """
        # Try to find quoted strings or numbered lines
        import re

        # Pattern 1: Quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted and len(quoted) >= max_variations:
            return [q.strip() for q in quoted[:max_variations] if len(q.strip()) > 3]

        # Pattern 2: Numbered lines (1. query, 2. query, etc.)
        numbered = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', text)
        if numbered:
            return [n.strip().strip('"') for n in numbered[:max_variations] if len(n.strip()) > 3]

        # Pattern 3: Lines starting with dash
        dashed = re.findall(r'-\s*(.+?)(?:\n|$)', text)
        if dashed:
            return [d.strip().strip('"') for d in dashed[:max_variations] if len(d.strip()) > 3]

        # Give up
        logger.warning("Could not extract variations from text")
        return []

    @staticmethod
    def get_cache_key(query: str, num_variations: int, model: str) -> str:
        """
        Generate a cache key for query variations.

        Args:
            query: Original query
            num_variations: Number of variations
            model: Model used

        Returns:
            Cache key string
        """
        # Create a hash of query + parameters
        key_string = f"{query}|{num_variations}|{model}"
        hash_digest = hashlib.md5(key_string.encode()).hexdigest()
        return f"query_variations:{hash_digest}"


# Global instance
query_expansion_service: Optional[QueryExpansionService] = None


def get_query_expansion_service() -> QueryExpansionService:
    """Get the global query expansion service instance."""
    global query_expansion_service
    if query_expansion_service is None:
        query_expansion_service = QueryExpansionService()
    return query_expansion_service
