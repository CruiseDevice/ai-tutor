from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, DatabaseError
from typing import List, Dict, Optional, AsyncGenerator
from openai import AsyncOpenAI, APIError
import logging
import json
import re
import uuid
import asyncio
from ..models.conversation import Conversation, Message
from ..models.document import DocumentChunk
from ..models.user import User
from .embedding_service import get_embedding_service
from .rerank_service import get_rerank_service
from .annotation_service import get_annotation_service
from .prompt_builder import get_prompt_builder, TOKEN_BUDGET_TEMPLATE
from .quality_service import get_quality_service
from .retry_utils import retry_openai_call, async_retry_openai_call
from .cache_service import get_cache_service
from .query_expansion_service import get_query_expansion_service
from .query_decomposition_service import get_query_decomposition_service
from .token_service import TokenService
from .llm import (
    Provider,
    resolve_provider,
    pick_helper_model,
    get_llm_client,
    LLMClient,
)

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.annotation_service = get_annotation_service()
        self.prompt_builder = get_prompt_builder()
        self.quality_service = get_quality_service()

    def _resolve_provider_for_model(self, model: str) -> Provider:
        """Resolve the LLM provider for a chat model id."""
        return resolve_provider(model)

    def _get_llm_client(
        self,
        model: str,
        api_key: str,
        provider: Provider | None = None,
    ) -> tuple[LLMClient, Provider]:
        """Construct a uniform LLM client for `model` using `api_key`.

        Returns the client and the resolved provider, so callers can pick a
        provider-appropriate helper model for background tasks.
        """
        prov = provider or self._resolve_provider_for_model(model)
        client = get_llm_client(prov, api_key)
        return client, prov

    def _get_adaptive_token_limit(self, complexity: str) -> int:
        """
        Get adaptive max_completion_tokens based on query complexity.

        Args:
            complexity: Query complexity level (simple, moderate, complex)

        Returns:
            Appropriate token limit for the complexity level
        """
        from ..config import settings
        limits = {
            "simple": settings.MAX_COMPLETION_TOKENS_SIMPLE,
            "moderate": settings.MAX_COMPLETION_TOKENS_MODERATE,
            "complex": settings.MAX_COMPLETION_TOKENS_COMPLEX,
        }
        return limits.get(complexity, settings.MAX_COMPLETION_TOKENS_MODERATE)

    async def _generate_conversation_title(self, user_message: str, user_api_key: str, provider: Provider | None = None) -> str:
        """
        Generate a smart, concise title for the conversation based on the first user message.
        Uses LLM to create a title that's 3-6 words.
        """
        try:
            prov = provider or Provider.OPENAI
            client = get_llm_client(prov, user_api_key)
            model = pick_helper_model(prov)

            prompt = f"""Generate a concise, descriptive title for this conversation based on the user's question.
The title should be 3-6 words and capture the main topic or question.

User question: "{user_message}"

Return ONLY the title, nothing else. Make it specific and informative.
Examples:
- "What is a virus?" → "Understanding Viruses"
- "Explain photosynthesis" → "Photosynthesis Explanation"
- "Summarize chapter 3" → "Chapter 3 Summary"
- "How does DNA replication work?" → "DNA Replication Process"

Title:"""

            async def _create_completion():
                return await client.complete(
                    system_prompt="You are a helpful assistant that generates concise conversation titles.",
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.7,
                    max_tokens=20,
                )

            title = await async_retry_openai_call(
                _create_completion,
                max_attempts=3,  # Fewer retries for title generation (non-critical)
                initial_wait=1.0,
                max_wait=30.0
            )

            title = title.strip()
            # Remove quotes if present
            title = title.strip('"\'')
            # Limit to 50 characters
            title = title[:50] if len(title) > 50 else title

            logger.info(f"Generated conversation title: {title}")
            return title

        except APIError as e:
            logger.warning(f"Failed to generate title with LLM after retries: {e}")
            # Fallback: create title from first few words of the message
            words = user_message.split()[:6]
            title = " ".join(words)
            if len(user_message) > len(title):
                title += "..."
            return title[:50]
        except Exception as e:
            logger.warning(f"Failed to generate title with LLM: {e}")
            # Fallback: create title from first few words of the message
            words = user_message.split()[:6]
            title = " ".join(words)
            if len(user_message) > len(title):
                title += "..."
            return title[:50]

    def _parse_annotations(self, response_text: str, relevant_chunks: List[Dict]) -> tuple[str, List[Dict]]:
        """Parse annotations from the LLM response. Delegates to AnnotationService.

        Returns (cleaned_response, annotations_list).
        Kept for backward compatibility; new callers should use
        `get_annotation_service().parse(...)` directly.
        """
        return self.annotation_service.parse(response_text, relevant_chunks)

    def _find_keyword_matches(
        self,
        db: Session,
        query: str,
        document_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find document chunks using PostgreSQL full-text search with keyword matching.

        Uses PostgreSQL's native full-text search capabilities:
        - to_tsquery() for query preprocessing (tokenization, stemming)
        - ts_rank_cd() for relevance scoring with phrase proximity
        - GIN index for fast search performance

        Returns chunks with normalized similarity scores (0-1 range) for fusion with semantic search.
        """
        try:
            # Sanitize query for tsquery (replace special characters, handle phrases)
            # PostgreSQL tsquery uses & (AND), | (OR), ! (NOT), and <-> (phrase)
            # For simplicity, we'll use plainto_tsquery which handles plain text safely
            query_sql = text("""
                SELECT
                    id,
                    content,
                    page_number,
                    document_id,
                    position_data,
                    chunk_type,
                    ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', :query)) as rank
                FROM document_chunks
                WHERE document_id = :document_id
                    AND to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC
                LIMIT :limit
            """)

            logger.debug(f"Executing keyword search for document_id: {document_id}, query: {query[:50]}...")
            import time
            start_time = time.time()

            result = db.execute(
                query_sql,
                {
                    "query": query,
                    "document_id": document_id,
                    "limit": limit
                }
            )

            query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.info(f"Keyword search completed in {query_time:.2f}ms (document_id: {document_id})")

            chunks = []
            max_rank = 0.0

            # First pass: collect chunks and find max rank for normalization
            rows = list(result)
            if rows:
                max_rank = max(row.rank for row in rows)

            # Second pass: normalize scores to 0-1 range
            for row in rows:
                # Normalize rank to 0-1 range (ts_rank_cd returns values typically between 0 and 1, but can be higher)
                # Use min-max normalization if we have a max_rank > 0
                normalized_score = (row.rank / max_rank) if max_rank > 0 else 0.0

                chunks.append({
                    "id": row.id,
                    "content": row.content,
                    "pageNumber": row.page_number,
                    "documentId": row.document_id,
                    "positionData": row.position_data,
                    "chunk_type": row.chunk_type,
                    "similarity": float(normalized_score)  # Normalized keyword relevance score
                })

            logger.debug(f"Found {len(chunks)} keyword matches")
            return chunks

        except (SQLAlchemyError, DatabaseError) as e:
            logger.warning(f"Database error in keyword search: {str(e)}")
            # Return empty list to allow fallback to semantic-only search
            return []
        except Exception as e:
            logger.warning(f"Error in keyword search: {str(e)}")
            # Return empty list to allow fallback to semantic-only search
            return []

    def _format_chunk_for_context(self, chunk: Dict) -> str:
        """Format a chunk for inclusion in the LLM context."""
        page_num = chunk.get("pageNumber", chunk.get("page_number", "?"))
        content = chunk.get("content", "")
        chunk_type = chunk.get("chunk_type", "text")
        position_data = chunk.get("positionData") or chunk.get("position_data") or {}

        if chunk_type == "image" and position_data:
            bbox = position_data.get("bbox") or [0, 0, 0, 0]
            image_id = chunk.get("id", "")

            try:
                x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0

            header = (
                f"[Page {page_num} - Image at bbox ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})]"
            )
            metadata_line = f"Image ID: {image_id}"

            return f"{header}:\n{metadata_line}\n{content}"

        return f"[Page {page_num}]: {content}"

    def _calculate_adaptive_weights(self, query: str) -> tuple[float, float]:
        """
        Dynamically adjust semantic vs keyword weights based on query characteristics.

        Returns:
            Tuple of (semantic_weight, keyword_weight)
        """
        import re
        from ..config import settings

        # Don't use adaptive weights if disabled
        if not getattr(settings, 'ENABLE_ADAPTIVE_HYBRID_WEIGHTS', False):
            return settings.SEMANTIC_SEARCH_WEIGHT, settings.KEYWORD_SEARCH_WEIGHT

        # Detect keyword-focused queries
        keyword_indicators = [
            bool(re.search(r'"[\w\s]+"', query)),  # Quoted phrases
            bool(re.search(r'\d+', query)),  # Contains numbers
            bool(re.search(r'\b(definition|specific|exact|what is|code|name|id)\b', query.lower())),
            len(query.split()) < 5  # Short queries
        ]

        # Detect semantic-focused queries
        semantic_indicators = [
            bool(re.search(r'\b(explain|why|how|describe|compare|relationship|discuss)\b', query.lower())),
            len(query.split()) > 10,  # Long queries
            '?' in query
        ]

        keyword_score = sum(keyword_indicators) / len(keyword_indicators)
        semantic_score = sum(semantic_indicators) / len(semantic_indicators)

        # Decide weighting strategy
        if keyword_score > semantic_score + 0.3:
            # Keyword-focused query (e.g., "What is the definition of X?")
            weights = (0.4, settings.HYBRID_WEIGHT_KEYWORD_BOOST)
            logger.debug(f"Query classified as keyword-focused: {query[:50]}...")
        elif semantic_score > keyword_score + 0.3:
            # Semantic-focused query (e.g., "Explain how X relates to Y")
            weights = (settings.HYBRID_WEIGHT_SEMANTIC_BOOST, 0.15)
            logger.debug(f"Query classified as semantic-focused: {query[:50]}...")
        else:
            # Balanced query - use defaults
            weights = (settings.SEMANTIC_SEARCH_WEIGHT, settings.KEYWORD_SEARCH_WEIGHT)
            logger.debug(f"Query classified as balanced: {query[:50]}...")

        return weights

    def _combine_results_with_rrf(
        self,
        result_sets: List[List[Dict]],
        rrf_k: int = 60
    ) -> List[Dict]:
        """
        Combine multiple result sets using Reciprocal Rank Fusion (RRF).

        RRF is a simple yet effective algorithm for combining ranked lists from different
        retrieval methods. It assigns a score to each document based on its rank position
        across all result sets.

        Formula: RRF_score(d) = sum over all rankings r: 1 / (k + rank_r(d))
        where k is a constant (typically 60) that reduces the impact of high rankings.

        Args:
            result_sets: List of ranked result lists (each list contains chunks with metadata)
            rrf_k: RRF constant (default: 60, standard value from literature)

        Returns:
            Combined and sorted list of chunks with RRF scores
        """
        if not result_sets:
            logger.warning("No result sets provided to RRF")
            return []

        # Filter out empty result sets
        result_sets = [rs for rs in result_sets if rs]

        if not result_sets:
            logger.warning("All result sets are empty")
            return []

        # If only one result set, return it directly
        if len(result_sets) == 1:
            return result_sets[0]

        # Track RRF scores and chunk data
        rrf_scores = {}  # chunk_id -> total RRF score
        chunk_data = {}  # chunk_id -> chunk metadata

        # Calculate RRF scores
        for result_set in result_sets:
            for rank, chunk in enumerate(result_set, start=1):
                chunk_id = chunk["id"]

                # Calculate RRF score contribution from this ranking
                rrf_contribution = 1.0 / (rrf_k + rank)

                # Accumulate RRF score
                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id] += rrf_contribution
                else:
                    rrf_scores[chunk_id] = rrf_contribution
                    # Store chunk data (use first occurrence)
                    chunk_data[chunk_id] = chunk

        # Create final result list with RRF scores
        combined_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            chunk = chunk_data[chunk_id].copy()
            # Replace similarity score with RRF score for ranking
            chunk["similarity"] = float(rrf_score)
            # Keep original score for debugging
            if "similarity" in chunk_data[chunk_id]:
                chunk["_original_score"] = chunk_data[chunk_id]["similarity"]
            chunk["_rrf_score"] = float(rrf_score)
            combined_results.append(chunk)

        # Sort by RRF score (descending)
        combined_results.sort(key=lambda x: x["similarity"], reverse=True)

        logger.info(f"RRF combined {len(result_sets)} result sets into {len(combined_results)} unique chunks")
        logger.debug(f"Top 3 RRF scores: {[c['similarity'] for c in combined_results[:3]]}")

        return combined_results

    async def _retrieve_critical_sentences(
        self,
        db: Session,
        query: str,
        query_embedding: List[float],
        document_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Retrieve critical sentences from the document for detail-focused queries.

        This method finds sentences that match the query using semantic search
        and filters for sentence-type chunks.

        Args:
            db: Database session
            query: User query
            query_embedding: Query embedding vector
            document_id: Document ID to search within
            limit: Maximum number of sentences to retrieve

        Returns:
            List of sentence chunks with similarity scores
        """
        try:
            logger.debug(f"Retrieving up to {limit} critical sentences for query: {query[:50]}...")

            # Query for sentence-level chunks using vector similarity
            embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

            query_sql = text("""
                SELECT
                    id,
                    content,
                    page_number,
                    document_id,
                    position_data,
                    chunk_type,
                    1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                FROM document_chunks
                WHERE document_id = :document_id
                  AND chunk_type = 'sentence'
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """)

            result = db.execute(
                query_sql,
                {
                    "embedding": embedding_str,
                    "document_id": document_id,
                    "limit": limit
                }
            )

            # Convert to list of dicts
            sentences = []
            for row in result:
                sentences.append({
                    "id": row.id,
                    "content": row.content,
                    "pageNumber": row.page_number,
                    "documentId": row.document_id,
                    "positionData": row.position_data,
                    "chunk_type": row.chunk_type,
                    "similarity": float(row.similarity),
                    "is_sentence": True  # Mark as sentence for deduplication
                })

            logger.info(f"Retrieved {len(sentences)} critical sentences")
            return sentences

        except Exception as e:
            logger.error(f"Error retrieving critical sentences: {e}", exc_info=True)
            return []

    def _merge_chunks_and_sentences(
        self,
        chunks: List[Dict],
        sentences: List[Dict],
        boost_factor: float = 1.2
    ) -> List[Dict]:
        """
        Merge chunk and sentence results, applying boost to sentence scores.

        Args:
            chunks: List of regular chunk results
            sentences: List of sentence results
            boost_factor: Boost factor for sentence similarity scores

        Returns:
            Merged and deduplicated list sorted by similarity
        """
        # Apply boost to sentence scores
        for sent in sentences:
            sent['similarity'] *= boost_factor
            sent['boosted'] = True

        # Combine chunks and sentences
        combined = chunks + sentences

        # Deduplicate by content (sentences might overlap with chunks)
        seen_contents = {}
        deduplicated = []

        for item in combined:
            content_key = item['content'].strip().lower()[:100]  # Use first 100 chars as key

            # If we haven't seen this content, or if this has higher similarity, keep it
            if content_key not in seen_contents:
                seen_contents[content_key] = item
                deduplicated.append(item)
            elif item['similarity'] > seen_contents[content_key]['similarity']:
                # Replace with higher-scoring version
                idx = deduplicated.index(seen_contents[content_key])
                deduplicated[idx] = item
                seen_contents[content_key] = item

        # Sort by similarity
        deduplicated.sort(key=lambda x: x['similarity'], reverse=True)

        logger.info(
            f"Merged {len(chunks)} chunks and {len(sentences)} sentences into {len(deduplicated)} results "
            f"(removed {len(combined) - len(deduplicated)} duplicates)"
        )

        return deduplicated

    async def find_similar_chunks(
        self,
        db: Session,
        query: str,
        document_id: str,
        limit: int = 5,
        user_api_key: Optional[str] = None,
        provider: Provider = Provider.OPENAI
    ) -> List[Dict]:
        """
        Find similar document chunks using hybrid search (semantic + keyword).

        Combines:
        1. Query Expansion (optional): Multi-query retrieval with RRF (if enabled)
        2. Semantic search: pgvector cosine similarity (70% weight by default)
        3. Keyword search: PostgreSQL full-text search (30% weight by default)

        Results are fused using weighted scoring for improved retrieval accuracy.
        Falls back to semantic-only search if keyword search fails.
        """
        try:
            from ..config import settings

            # TODO(human): Diagnostic - Check if any chunks exist for this document
            chunk_count = db.execute(text("SELECT COUNT(*) FROM document_chunks WHERE document_id = :document_id"),
                                    {"document_id": document_id}).scalar()
            logger.info(f"[DEBUG find_similar_chunks] Document {document_id} has {chunk_count} chunks in database")

            # Check if document uses hierarchical parent-child chunking
            uses_hierarchical = self._check_hierarchical_chunking(db, document_id)

            if uses_hierarchical:
                logger.info(f"Document {document_id} uses hierarchical chunking - routing to parent-child retrieval")

                # Generate query embedding for hierarchical search
                cache_service = await get_cache_service()
                cached_embedding = await cache_service.get_embedding(query)

                if cached_embedding:
                    query_embedding = cached_embedding
                    logger.debug(f"Using cached embedding for hierarchical query: {query[:50]}...")
                else:
                    query_embedding = await self.embedding_service.generate_embedding_async(query)
                    await cache_service.set_embedding(query, query_embedding)
                    logger.debug(f"Generated embedding for hierarchical query: {query[:50]}...")

                # Use hierarchical parent-child retrieval
                hierarchical_results = await self._retrieve_with_parent_child(
                    db=db,
                    query_embedding=query_embedding,
                    document_id=document_id,
                    limit=limit
                )

                logger.info(f"Hierarchical retrieval returned {len(hierarchical_results)} parent chunks")

                # SENTENCE-LEVEL RETRIEVAL: Add sentence retrieval for hierarchical mode too
                if settings.ENABLE_SENTENCE_RETRIEVAL:
                    from .sentence_retrieval_service import SentenceRetrievalService
                    sentence_service = SentenceRetrievalService()

                    # Check if this is a detail-focused query
                    is_detail_query = sentence_service.is_detail_query(query)

                    if is_detail_query:
                        logger.info("Detail query detected - retrieving critical sentences (hierarchical mode)")

                        # Retrieve critical sentences
                        sentences = await self._retrieve_critical_sentences(
                            db=db,
                            query=query,
                            query_embedding=query_embedding,
                            document_id=document_id,
                            limit=settings.SENTENCE_RETRIEVAL_TOP_K
                        )

                        if sentences:
                            # Merge sentences with hierarchical chunks
                            hierarchical_results = self._merge_chunks_and_sentences(
                                chunks=hierarchical_results,
                                sentences=sentences,
                                boost_factor=settings.SENTENCE_BOOST_FACTOR
                            )

                            # Limit to requested number after merging
                            hierarchical_results = hierarchical_results[:limit]

                            logger.info(
                                f"Merged sentence retrieval (hierarchical): final result has {len(hierarchical_results)} items"
                            )

                return hierarchical_results

            # FLAT CHUNKING (existing behavior)
            logger.debug(f"Document {document_id} uses flat chunking - using standard hybrid retrieval")

            # Calculate adaptive weights based on query characteristics
            semantic_weight, keyword_weight = self._calculate_adaptive_weights(query)

            logger.info(
                f"Using adaptive weights: semantic={semantic_weight:.2f}, keyword={keyword_weight:.2f}"
            )

            # Get query expansion, decomposition, and reranking settings from config
            rerank_enabled = getattr(settings, 'RERANK_ENABLED', False)
            query_expansion_enabled = getattr(settings, 'QUERY_EXPANSION_ENABLED', False)
            query_decomposition_enabled = getattr(settings, 'ENABLE_QUERY_DECOMPOSITION', False)
            rrf_k = getattr(settings, 'RRF_K', 60)

            cache_service = await get_cache_service()

            # Retrieve more candidates for fusion (e.g., top 10-15)
            semantic_limit = max(limit * 2, 10)

            # QUERY DECOMPOSITION: Break down complex queries into atomic sub-queries
            # If a complex query is decomposed, we'll use the sub-queries instead of query expansion
            query_was_decomposed = False
            queries_to_search = [query]  # Default to original query

            if query_decomposition_enabled and user_api_key:
                logger.info("Query decomposition enabled - checking if query is complex")

                try:
                    query_decomposition_service = get_query_decomposition_service()
                    sub_queries = await query_decomposition_service.decompose_query(
                        query=query,
                        user_api_key=user_api_key,
                        provider=provider
                    )

                    # If decomposition produced multiple sub-queries, use them instead of expansion
                    if len(sub_queries) > 1:
                        queries_to_search = sub_queries
                        query_was_decomposed = True
                        logger.info(
                            f"Complex query decomposed into {len(sub_queries)} sub-queries. "
                            f"Skipping query expansion for decomposed queries."
                        )
                        logger.debug(f"Sub-queries: {sub_queries}")
                    else:
                        logger.info("Query not complex enough for decomposition, proceeding with normal flow")

                except Exception as e:
                    logger.error(f"Query decomposition failed, proceeding with normal flow: {e}")
                    # Fall through to normal query expansion/search

            # SEARCH DECOMPOSED QUERIES: If query was decomposed, search each sub-query
            semantic_chunks = {}
            query_embedding = None  # Initialize query_embedding early

            if query_was_decomposed:
                logger.info(f"Searching for {len(queries_to_search)} decomposed sub-queries")

                try:
                    import time
                    decomp_start = time.time()

                    async def search_single_subquery(subquery: str) -> List[Dict]:
                        """Search for a single sub-query."""
                        # Check cache for embedding
                        cached_embedding = await cache_service.get_embedding(subquery)
                        if cached_embedding:
                            subquery_embedding = cached_embedding
                            logger.debug(f"Using cached embedding for sub-query: {subquery[:40]}...")
                        else:
                            # Generate embedding
                            subquery_embedding = await self.embedding_service.generate_embedding_async(subquery)
                            await cache_service.set_embedding(subquery, subquery_embedding)
                            logger.debug(f"Generated embedding for sub-query: {subquery[:40]}...")

                        # Perform semantic search
                        return await self._perform_semantic_search(
                            db=db,
                            query_embedding=subquery_embedding,
                            document_id=document_id,
                            limit=semantic_limit
                        )

                    # Execute all searches in parallel
                    subquery_results = await asyncio.gather(
                        *[search_single_subquery(sq) for sq in queries_to_search],
                        return_exceptions=True
                    )

                    # Filter out any errors
                    valid_results = []
                    for i, result in enumerate(subquery_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Error searching sub-query {i}: {result}")
                        else:
                            valid_results.append(result)

                    if not valid_results:
                        logger.error("All sub-query searches failed, falling back to single-query search")
                        query_was_decomposed = False  # Trigger fallback
                    else:
                        decomp_time = (time.time() - decomp_start) * 1000
                        logger.info(f"Decomposed query search completed in {decomp_time:.2f}ms ({len(valid_results)} sub-queries)")

                        # Combine results using RRF
                        combined_chunks = self._combine_results_with_rrf(valid_results, rrf_k=rrf_k)

                        # Convert list to dict
                        for chunk in combined_chunks:
                            semantic_chunks[chunk["id"]] = {
                                "id": chunk["id"],
                                "content": chunk["content"],
                                "pageNumber": chunk["pageNumber"],
                                "documentId": chunk["documentId"],
                                "positionData": chunk["positionData"],
                                "chunk_type": chunk.get("chunk_type"),
                                "semantic_score": chunk["similarity"]  # RRF score
                            }

                        logger.info(f"RRF combined decomposed query results: {len(semantic_chunks)} unique chunks")

                except Exception as e:
                    logger.error(f"Decomposed query search failed, falling back to single-query search: {e}")
                    query_was_decomposed = False  # Trigger fallback

            # QUERY EXPANSION: Multi-query retrieval with RRF (if enabled and query wasn't decomposed)
            # Skip query expansion if query was already decomposed
            if query_expansion_enabled and user_api_key and not query_was_decomposed:
                logger.info("Query expansion enabled - performing multi-query retrieval")

                try:
                    # Get or generate embedding for the original query (needed for caching)
                    cached_embedding = await cache_service.get_embedding(query)
                    if cached_embedding:
                        query_embedding = cached_embedding
                        logger.debug(f"Using cached embedding for original query: {query[:50]}...")
                    else:
                        query_embedding = await self.embedding_service.generate_embedding_async(query)
                        await cache_service.set_embedding(query, query_embedding)
                        logger.debug(f"Generated embedding for original query: {query[:50]}...")

                    # Generate query variations
                    query_expansion_service = get_query_expansion_service()
                    query_variations = await query_expansion_service.generate_query_variations(
                        query=query,
                        user_api_key=user_api_key,
                        provider=provider
                    )
                    logger.info(f"Generated {len(query_variations)} query variations (including original)")

                    # Perform semantic search for each variation in parallel
                    import time
                    multi_query_start = time.time()

                    async def search_single_variation(variation_query: str) -> List[Dict]:
                        """Search for a single query variation."""
                        # Check cache for embedding
                        cached_embedding = await cache_service.get_embedding(variation_query)
                        if cached_embedding:
                            variation_embedding = cached_embedding
                            logger.debug(f"Using cached embedding for variation: {variation_query[:40]}...")
                        else:
                            # Generate embedding
                            variation_embedding = await self.embedding_service.generate_embedding_async(variation_query)
                            # Cache the embedding
                            await cache_service.set_embedding(variation_query, variation_embedding)

                        # Perform semantic search
                        embedding_str = '[' + ','.join(str(x) for x in variation_embedding) + ']'

                        query_sql = text("""
                            SELECT
                                id,
                                content,
                                page_number,
                                document_id,
                                position_data,
                                chunk_type,
                                1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                            FROM document_chunks
                            WHERE document_id = :document_id
                            ORDER BY embedding <=> CAST(:embedding AS vector)
                            LIMIT :limit
                        """)

                        result = db.execute(
                            query_sql,
                            {
                                "embedding": embedding_str,
                                "document_id": document_id,
                                "limit": semantic_limit
                            }
                        )

                        # Convert to list of dicts
                        chunks = []
                        for row in result:
                            chunks.append({
                                "id": row.id,
                                "content": row.content,
                                "pageNumber": row.page_number,
                                "documentId": row.document_id,
                                "positionData": row.position_data,
                                "chunk_type": row.chunk_type,
                                "similarity": float(row.similarity)
                            })

                        logger.debug(f"Found {len(chunks)} chunks for variation: {variation_query[:40]}...")
                        return chunks

                    # Execute all searches in parallel
                    variation_results = await asyncio.gather(
                        *[search_single_variation(var) for var in query_variations],
                        return_exceptions=True
                    )

                    # Filter out any errors (use graceful degradation)
                    valid_results = []
                    for i, result in enumerate(variation_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Error searching variation {i}: {result}")
                        else:
                            valid_results.append(result)

                    if not valid_results:
                        logger.error("All query variations failed, falling back to single-query search")
                        raise Exception("All query variations failed")

                    multi_query_time = (time.time() - multi_query_start) * 1000
                    logger.info(f"Multi-query search completed in {multi_query_time:.2f}ms ({len(valid_results)} variations)")

                    # Combine results using RRF
                    combined_chunks = self._combine_results_with_rrf(valid_results, rrf_k=rrf_k)

                    # Convert list to dict (for compatibility with hybrid fusion below)
                    for chunk in combined_chunks:
                        semantic_chunks[chunk["id"]] = {
                            "id": chunk["id"],
                            "content": chunk["content"],
                            "pageNumber": chunk["pageNumber"],
                            "documentId": chunk["documentId"],
                            "positionData": chunk["positionData"],
                            "chunk_type": chunk.get("chunk_type"),
                            "semantic_score": chunk["similarity"]  # RRF score
                        }

                    logger.info(f"RRF combined results: {len(semantic_chunks)} unique chunks")

                except Exception as e:
                    logger.error(f"Query expansion failed, falling back to single-query search: {e}")
                    # Fall through to single-query search below
                    query_expansion_enabled = False  # Disable for this request

            # SINGLE-QUERY SEMANTIC SEARCH (original behavior or fallback)
            # Only run if neither decomposition nor expansion was used
            if (not query_expansion_enabled or not user_api_key) and not query_was_decomposed:
                if not query_expansion_enabled:
                    logger.debug("Query expansion disabled - using single-query search")
                else:
                    logger.debug("No user API key provided - using single-query search")

                # Check cache for chunks first
                cached_chunks = None

                # Try to get embedding from cache (if not already set from query expansion)
                if query_embedding is None:
                    cached_embedding = await cache_service.get_embedding(query)
                    if cached_embedding:
                        query_embedding = cached_embedding
                        logger.debug(f"Using cached embedding for query: {query[:50]}...")
                        # Check cache for chunks with this embedding (include rerank status in cache key)
                        cached_chunks = await cache_service.get_chunks(
                            document_id, query_embedding, rerank_enabled=rerank_enabled
                        )
                        if cached_chunks:
                            logger.info(f"Cache hit: Returning {len(cached_chunks)} cached chunks")
                            return cached_chunks
                else:
                    # query_embedding already set from query expansion, check cache for chunks
                    cached_chunks = await cache_service.get_chunks(
                        document_id, query_embedding, rerank_enabled=rerank_enabled
                    )
                    if cached_chunks:
                        logger.info(f"Cache hit: Returning {len(cached_chunks)} cached chunks")
                        return cached_chunks

                # Generate embedding if not cached
                if query_embedding is None:
                    logger.debug(f"Generating embedding for query: {query[:50]}...")
                    query_embedding = await self.embedding_service.generate_embedding_async(query)
                    logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")
                    # Cache the embedding
                    await cache_service.set_embedding(query, query_embedding)

                # Perform SEMANTIC search using pgvector with HNSW index
                # pgvector expects the vector in the format '[1,2,3]' as a string
                embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

                # Optimized query to leverage HNSW index
                query_sql = text("""
                    SELECT
                        id,
                        content,
                        page_number,
                        document_id,
                        position_data,
                        chunk_type,
                        1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                    FROM document_chunks
                    WHERE document_id = :document_id
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :limit
                """)

                logger.debug(f"Executing semantic search for document_id: {document_id}")
                import time
                semantic_start = time.time()

                result = db.execute(
                    query_sql,
                    {
                        "embedding": embedding_str,
                        "document_id": document_id,
                        "limit": semantic_limit
                    }
                )

                semantic_time = (time.time() - semantic_start) * 1000
                logger.info(f"Semantic search completed in {semantic_time:.2f}ms (document_id: {document_id})")

                for row in result:
                    semantic_chunks[row.id] = {
                        "id": row.id,
                        "content": row.content,
                        "pageNumber": row.page_number,
                        "documentId": row.document_id,
                        "positionData": row.position_data,
                        "chunk_type": row.chunk_type,
                        "semantic_score": float(row.similarity)
                    }

                logger.debug(f"Found {len(semantic_chunks)} semantic matches")

            # Perform KEYWORD search using PostgreSQL full-text search
            keyword_chunks = {}
            keyword_search_success = False

            try:
                keyword_limit = max(limit * 2, 10)
                keyword_results = self._find_keyword_matches(db, query, document_id, limit=keyword_limit)

                if keyword_results:
                    keyword_search_success = True
                    for chunk in keyword_results:
                        keyword_chunks[chunk["id"]] = {
                            "id": chunk["id"],
                            "content": chunk["content"],
                            "pageNumber": chunk["pageNumber"],
                            "documentId": chunk["documentId"],
                            "positionData": chunk["positionData"],
                            "chunk_type": chunk.get("chunk_type"),
                            "keyword_score": chunk["similarity"]
                        }
                    logger.debug(f"Found {len(keyword_chunks)} keyword matches")
                else:
                    logger.info("No keyword matches found, using semantic-only results")

            except Exception as e:
                logger.warning(f"Keyword search failed, falling back to semantic-only: {str(e)}")
                keyword_search_success = False

            # HYBRID FUSION: Combine semantic and keyword results
            fused_chunks = {}

            # Get all unique chunk IDs from both result sets
            all_chunk_ids = set(semantic_chunks.keys()) | set(keyword_chunks.keys())

            for chunk_id in all_chunk_ids:
                # Get scores (0 if chunk not in that result set)
                semantic_score = semantic_chunks.get(chunk_id, {}).get('semantic_score', 0.0)
                keyword_score = keyword_chunks.get(chunk_id, {}).get('keyword_score', 0.0)

                # Calculate weighted fusion score
                if keyword_search_success:
                    # Hybrid: combine both scores with weights
                    fused_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
                else:
                    # Fallback: semantic-only (weight = 1.0)
                    fused_score = semantic_score

                # Get chunk data (prefer semantic result as it has all fields)
                chunk_data = semantic_chunks.get(chunk_id) or keyword_chunks.get(chunk_id)

                fused_chunks[chunk_id] = {
                    "id": chunk_data["id"],
                    "content": chunk_data["content"],
                    "pageNumber": chunk_data["pageNumber"],
                    "documentId": chunk_data["documentId"],
                    "positionData": chunk_data["positionData"],
                    "chunk_type": chunk_data.get("chunk_type"),
                    "similarity": float(fused_score),  # Final fused score
                    "_semantic_score": semantic_score,  # Debug info
                    "_keyword_score": keyword_score     # Debug info
                }

            # Sort by fused score
            sorted_chunks = sorted(
                fused_chunks.values(),
                key=lambda x: x["similarity"],
                reverse=True
            )

            # Apply re-ranking if enabled (already loaded at top of function)

            if rerank_enabled:
                try:
                    # Get re-ranking parameters from config
                    rerank_top_k = getattr(settings, 'RERANK_TOP_K', 20)
                    rerank_final_k = getattr(settings, 'RERANK_FINAL_K', limit)

                    # Get top candidates for re-ranking
                    candidates_for_rerank = sorted_chunks[:rerank_top_k]

                    if candidates_for_rerank:
                        logger.info(f"Re-ranking top {len(candidates_for_rerank)} candidates")

                        # Get rerank service and re-rank chunks
                        rerank_service = get_rerank_service()
                        reranked_chunks = await rerank_service.rerank_chunks(
                            query=query,
                            chunks=candidates_for_rerank,
                            top_k=rerank_final_k
                        )

                        # Use re-ranked results
                        sorted_chunks = reranked_chunks
                        logger.info(f"Re-ranking completed, returning {len(sorted_chunks)} chunks")
                    else:
                        # No candidates to re-rank, use original sorted results
                        sorted_chunks = sorted_chunks[:limit]
                        logger.info("No candidates to re-rank, using original results")

                except Exception as e:
                    # Fallback to original sorting if re-ranking fails
                    logger.warning(f"Re-ranking failed, falling back to original ranking: {str(e)}")
                    sorted_chunks = sorted_chunks[:limit]
            else:
                # Re-ranking disabled, use original sorted results
                sorted_chunks = sorted_chunks[:limit]

            # Clean up debug fields before returning
            final_chunks = []
            for chunk in sorted_chunks:
                # Preserve rerank_score if it exists (for debugging)
                chunk_data = {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "pageNumber": chunk["pageNumber"],
                    "documentId": chunk["documentId"],
                    "positionData": chunk["positionData"],
                    "chunk_type": chunk.get("chunk_type"),
                    "similarity": chunk["similarity"]
                }
                # Optionally include rerank score for debugging
                if "rerank_score" in chunk:
                    chunk_data["rerank_score"] = chunk["rerank_score"]
                final_chunks.append(chunk_data)

            search_type = "hybrid" if keyword_search_success else "semantic-only"
            rerank_status = "with re-ranking" if rerank_enabled else "without re-ranking"
            logger.info(f"Hybrid search ({search_type}, {rerank_status}) returned {len(final_chunks)} chunks")

            # SENTENCE-LEVEL RETRIEVAL (Phase 5): Retrieve critical sentences for detail queries
            if settings.ENABLE_SENTENCE_RETRIEVAL:
                from .sentence_retrieval_service import SentenceRetrievalService
                sentence_service = SentenceRetrievalService()

                # Check if this is a detail-focused query
                is_detail_query = sentence_service.is_detail_query(query)

                if is_detail_query:
                    logger.info("Detail query detected - retrieving critical sentences")

                    # Retrieve critical sentences
                    sentences = await self._retrieve_critical_sentences(
                        db=db,
                        query=query,
                        query_embedding=query_embedding,
                        document_id=document_id,
                        limit=settings.SENTENCE_RETRIEVAL_TOP_K
                    )

                    if sentences:
                        # Merge sentences with chunks
                        final_chunks = self._merge_chunks_and_sentences(
                            chunks=final_chunks,
                            sentences=sentences,
                            boost_factor=settings.SENTENCE_BOOST_FACTOR
                        )

                        # Limit to requested number after merging
                        final_chunks = final_chunks[:limit]

                        logger.info(
                            f"Merged sentence retrieval: final result has {len(final_chunks)} items "
                            f"(chunks + sentences)"
                        )

            # Cache the final results (cache key includes re-ranking status)
            await cache_service.set_chunks(
                document_id, query_embedding, final_chunks, rerank_enabled=rerank_enabled
            )

            # TODO(human): Diagnostic logging to track chunk retrieval
            logger.info(f"[DEBUG find_similar_chunks] Returning {len(final_chunks)} chunks for document {document_id}")
            return final_chunks

        except (SQLAlchemyError, DatabaseError) as e:
            logger.error(f"Database error in find_similar_chunks: {str(e)}", exc_info=True)
            # Rollback the transaction to allow subsequent queries to work
            db.rollback()
            # Return empty list instead of raising to allow chat to continue
            return []
        except Exception as e:
            logger.error(f"Error in find_similar_chunks: {str(e)}", exc_info=True)
            # Return empty list instead of raising to allow chat to continue
            return []

    def _select_chunks_by_token_limit(
        self,
        chunks: List[Dict],
        max_tokens: int,
        model: str,
        system_prompt_tokens: int,
        user_message_tokens: int,
        history_tokens: int,
        response_reserve_tokens: int
    ) -> tuple[List[Dict], Dict[str, int]]:
        """
        Dynamically select chunks based on token limits instead of fixed count.

        Args:
            chunks: List of chunks sorted by relevance (from find_similar_chunks)
            max_tokens: Maximum context window size
            model: Model name for token counting
            system_prompt_tokens: Tokens used by system prompt template (excluding chunks)
            user_message_tokens: Tokens in current user message
            history_tokens: Tokens in conversation history
            response_reserve_tokens: Tokens reserved for model response

        Returns:
            Tuple of (selected_chunks, token_usage_stats)
        """
        from ..config import settings

        # Get configuration
        buffer_tokens = getattr(settings, 'TOKEN_RESERVE_BUFFER', 20000)
        truncation_enabled = getattr(settings, 'CHUNK_TRUNCATION_ENABLED', True)

        # Calculate available tokens for chunks
        available_tokens = TokenService.calculate_available_tokens(
            max_context_tokens=max_tokens,
            system_prompt_tokens=system_prompt_tokens,
            user_message_tokens=user_message_tokens,
            history_tokens=history_tokens,
            response_reserve_tokens=response_reserve_tokens,
            buffer_tokens=buffer_tokens
        )

        logger.info(
            f"Token budget: max={max_tokens}, system={system_prompt_tokens}, "
            f"user={user_message_tokens}, history={history_tokens}, "
            f"response_reserve={response_reserve_tokens}, buffer={buffer_tokens}, "
            f"available_for_chunks={available_tokens}"
        )

        if available_tokens <= 0:
            logger.warning(
                f"No tokens available for chunks! "
                f"Consider reducing history or using a larger context model."
            )
            return [], {
                "selected_chunks": 0,
                "total_chunk_tokens": 0,
                "available_tokens": available_tokens,
                "truncated_chunks": 0,
                "skipped_chunks": len(chunks)
            }

        selected_chunks = []
        tokens_used = 0
        truncated_count = 0
        skipped_count = 0

        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            page_number = chunk.get("pageNumber", "Unknown")

            # Format chunk as it would appear in context (with page number)
            formatted_chunk = self._format_chunk_for_context(chunk)

            # Count tokens in formatted chunk
            chunk_tokens = TokenService.count_tokens(formatted_chunk, model)

            # Check if chunk fits
            if tokens_used + chunk_tokens <= available_tokens:
                # Chunk fits completely
                selected_chunks.append(chunk)
                tokens_used += chunk_tokens
                logger.debug(
                    f"Added chunk {i+1}/{len(chunks)} "
                    f"(page {page_number}, {chunk_tokens} tokens, "
                    f"total: {tokens_used}/{available_tokens})"
                )
            elif truncation_enabled and tokens_used < available_tokens:
                # Chunk doesn't fit, but we have room and truncation is enabled
                remaining_tokens = available_tokens - tokens_used

                # Need at least some minimum tokens to make truncation worthwhile
                min_useful_tokens = 50
                if remaining_tokens >= min_useful_tokens:
                    # Truncate the chunk content to fit
                    truncated_content = TokenService.truncate_text_to_tokens(
                        chunk_content,
                        max_tokens=remaining_tokens - 20,  # Reserve tokens for page number formatting
                        model=model,
                        prefer_sentence_boundaries=True
                    )

                    # Create truncated chunk
                    truncated_chunk = chunk.copy()
                    truncated_chunk["content"] = truncated_content
                    truncated_chunk["truncated"] = True

                    selected_chunks.append(truncated_chunk)
                    truncated_count += 1

                    # Count actual tokens in truncated formatted chunk
                    truncated_formatted = self._format_chunk_for_context(truncated_chunk)
                    actual_tokens = TokenService.count_tokens(truncated_formatted, model)
                    tokens_used += actual_tokens

                    logger.info(
                        f"Truncated chunk {i+1}/{len(chunks)} "
                        f"(page {page_number}, {chunk_tokens} -> {actual_tokens} tokens, "
                        f"total: {tokens_used}/{available_tokens})"
                    )
                else:
                    # Not enough room even for truncation
                    skipped_count += 1
                    logger.debug(
                        f"Skipped chunk {i+1}/{len(chunks)} - insufficient remaining tokens "
                        f"({remaining_tokens} < {min_useful_tokens})"
                    )
                break  # No more room for additional chunks
            else:
                # Chunk doesn't fit and truncation is disabled
                skipped_count += 1
                logger.debug(
                    f"Skipped chunk {i+1}/{len(chunks)} "
                    f"(page {page_number}, {chunk_tokens} tokens) - would exceed limit"
                )

        # Token usage statistics
        stats = {
            "selected_chunks": len(selected_chunks),
            "total_chunk_tokens": tokens_used,
            "available_tokens": available_tokens,
            "truncated_chunks": truncated_count,
            "skipped_chunks": skipped_count
        }

        logger.info(
            f"Chunk selection complete: {len(selected_chunks)}/{len(chunks)} chunks selected, "
            f"{tokens_used}/{available_tokens} tokens used, "
            f"{truncated_count} truncated, {skipped_count} skipped"
        )

        return selected_chunks, stats

    async def _classify_query_type(self, query: str, user_api_key: str, provider: Provider | None = None) -> Dict:
        """Classify query type and complexity to enable adaptive prompting.

        Delegates to PromptBuilder. Kept for backward compatibility; new
        callers should use `get_prompt_builder().classify_query_type(...)`.
        """
        return await self.prompt_builder.classify_query_type(
            query=query, user_api_key=user_api_key, provider=provider
        )

    def _get_few_shot_examples(self) -> str:
        """Few-shot prompt examples. Delegates to PromptBuilder."""
        return self.prompt_builder._get_few_shot_examples()

    def _get_chain_of_thought_section(self, query_type: str) -> str:
        """Chain-of-thought prompt section. Delegates to PromptBuilder."""
        return self.prompt_builder._get_chain_of_thought_section(query_type)

    def _build_system_prompt(
        self,
        context_text: str,
        query_type: str,
        complexity: str,
        requires_cot: bool
    ) -> str:
        """Build adaptive system prompt. Delegates to PromptBuilder.

        Kept for backward compatibility; new callers should use
        `get_prompt_builder().build_system_prompt(...)` directly.
        """
        return self.prompt_builder.build_system_prompt(
            context_text=context_text,
            query_type=query_type,
            complexity=complexity,
            requires_cot=requires_cot,
        )


    def _verify_citations(
        self,
        response_text: str,
        annotations: List[Dict],
        relevant_chunks: List[Dict]
    ) -> List[str]:
        """Verify citations match retrieved chunks. Delegates to QualityService.

        Kept for backward compatibility; new callers should use
        `get_quality_service().verify_citations(...)` directly.
        """
        return self.quality_service.verify_citations(
            response_text=response_text,
            annotations=annotations,
            relevant_chunks=relevant_chunks,
        )

    async def _score_answer_quality(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict],
        user_api_key: str,
        provider: Provider | None = None
    ) -> Dict[str, any]:
        """Score answer quality via helper LLM. Delegates to QualityService.

        Kept for backward compatibility; new callers should use
        `get_quality_service().score_answer_quality(...)` directly.
        """
        return await self.quality_service.score_answer_quality(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
            user_api_key=user_api_key,
            provider=provider,
        )

    async def generate_chat_response(
        self,
        db: Session,
        user: User,
        content: str,
        conversation_id: str,
        document_id: str,
        model: str = "gpt-4",
        use_agent: bool = False
    ) -> Dict:
        """
        Generate a chat response using OpenAI with RAG.

        Args:
            db: Database session
            user: User object
            content: User's message/query
            conversation_id: Conversation identifier
            document_id: Document identifier
            model: OpenAI model to use (default: gpt-4)
            use_agent: Whether to use agent workflow (default: False)

        Returns:
            Dict with user_message and assistant_message
        """
        # Route to agent workflow if requested
        if use_agent:
            return await self.generate_chat_response_with_agent(
                db=db,
                user=user,
                content=content,
                conversation_id=conversation_id,
                document_id=document_id,
                model=model
            )

        # Linear pipeline (existing implementation)
        try:
            # Get decrypted API key for the resolved provider
            provider = self._resolve_provider_for_model(model)
            api_key = user.get_decrypted_key(provider.value)
            if not api_key:
                logger.error(f"User {user.id} has no {provider.value} API key configured")
                raise ValueError(
                    f"User has no {provider.value} API key configured. "
                    "Please configure your API key in settings."
                )

            logger.debug(f"Generating chat response for user {user.id}, conversation {conversation_id}")

            # Initialize cache service
            cache_service = await get_cache_service()

            # Get query embedding for response cache lookup
            query_embedding = await cache_service.get_embedding(content)
            if query_embedding is None:
                query_embedding = await self.embedding_service.generate_embedding_async(content)
                await cache_service.set_embedding(content, query_embedding)

            # Check for similar cached response (skip if conversation has history)
            history_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()

            if history_count == 0:  # Only cache responses for first message (no context)
                cached_response = await cache_service.find_similar_response(query_embedding, document_id)
                if cached_response:
                    logger.info("Using cached response")
                    # Still need to save messages to database
                    user_message = Message(
                        content=content,
                        role="user",
                        conversation_id=conversation_id
                    )
                    db.add(user_message)
                    db.flush()

                    assistant_message = Message(
                        content=cached_response['content'],
                        role="assistant",
                        conversation_id=conversation_id,
                        context={
                            "chunks": cached_response['chunks'],
                            "annotations": cached_response['annotations']
                        }
                    )
                    db.add(assistant_message)
                    db.commit()
                    db.refresh(user_message)
                    db.refresh(assistant_message)

                    return {
                        "user_message": {
                            "id": user_message.id,
                            "role": user_message.role,
                            "content": user_message.content,
                            "created_at": user_message.created_at,
                            "context": None,
                            "annotations": None
                        },
                        "assistant_message": {
                            "id": assistant_message.id,
                            "role": assistant_message.role,
                            "content": assistant_message.content,
                            "created_at": assistant_message.created_at,
                            "context": cached_response['chunks'],
                            "annotations": cached_response['annotations']
                        }
                    }

            # Initialize LLM client (provider resolved from the selected model)
            provider = self._resolve_provider_for_model(model)
            client = get_llm_client(provider, api_key)

            # Get configuration
            from ..config import settings
            max_context_tokens = getattr(settings, 'MAX_CONTEXT_TOKENS', 100000)
            rerank_top_k = getattr(settings, 'RERANK_TOP_K', 20)

            # Find relevant chunks (retrieve more for token-based selection)
            logger.debug(f"Finding similar chunks for document {document_id}")
            candidate_chunks = await self.find_similar_chunks(db, content, document_id, limit=rerank_top_k, user_api_key=api_key, provider=provider)
            # TODO(human): Add diagnostic logging to understand why chunks might be empty
            logger.info(f"[DEBUG CHAT] Retrieved {len(candidate_chunks)} candidate chunks for document {document_id}")

            # Get conversation history for token counting
            logger.debug(f"Fetching conversation history for {conversation_id}")
            history = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(10).all()

            # Classify query type for adaptive prompting
            query_classification = await self._classify_query_type(content, api_key, provider)
            logger.info(
                f"Query classification: type={query_classification['query_type']}, "
                f"complexity={query_classification['complexity']}, "
                f"requires_cot={query_classification['requires_cot']}"
            )

            # Count tokens for dynamic chunk selection
            # 1. System prompt template (without chunks)
            system_prompt_template = TOKEN_BUDGET_TEMPLATE  # imported from prompt_builder

            system_prompt_tokens = TokenService.count_tokens(system_prompt_template, model)

            # 2. User message tokens
            user_message_tokens = TokenService.count_tokens(content, model)

            # 3. History tokens
            history_messages = [{"role": msg.role, "content": msg.content} for msg in history]
            history_tokens = TokenService.estimate_context_tokens(history_messages, model)

            # 4. Response reserve (max_completion_tokens)
            response_reserve_tokens = 1000

            # Select chunks dynamically based on token limits
            relevant_chunks, chunk_stats = self._select_chunks_by_token_limit(
                chunks=candidate_chunks,
                max_tokens=max_context_tokens,
                model=model,
                system_prompt_tokens=system_prompt_tokens,
                user_message_tokens=user_message_tokens,
                history_tokens=history_tokens,
                response_reserve_tokens=response_reserve_tokens
            )

            logger.info(
                f"Dynamic chunk selection: {chunk_stats['selected_chunks']}/{len(candidate_chunks)} chunks, "
                f"{chunk_stats['total_chunk_tokens']} tokens, "
                f"{chunk_stats['truncated_chunks']} truncated, "
                f"{chunk_stats['skipped_chunks']} skipped"
            )

            # Format context from chunks
            if relevant_chunks:
                context_text = "\n\n".join(
                    self._format_chunk_for_context(chunk)
                    for chunk in relevant_chunks
                )
            else:
                context_text = "No relevant document sections found."

            # Build adaptive system prompt with few-shot examples and chain-of-thought
            system_prompt_content = self._build_system_prompt(
                context_text=context_text,
                query_type=query_classification['query_type'],
                complexity=query_classification['complexity'],
                requires_cot=query_classification['requires_cot']
            )

            # Create system message
            system_message = {
                "role": "system",
                "content": system_prompt_content
            }

            # Format history for OpenAI (history already fetched above for token counting)
            messages = [system_message]
            for msg in history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            # Add current user message
            messages.append({
                "role": "user",
                "content": content
            })

            # Call OpenAI with retry logic
            logger.debug(f"Calling OpenAI API with model {model}")

            # Get adaptive token limit based on query complexity
            max_tokens = self._get_adaptive_token_limit(query_classification['complexity'])
            logger.info(f"Using {max_tokens} max completion tokens for {query_classification['complexity']} query")

            try:
                # messages[0] is the system message; LLMClient takes system separately.
                chat_messages = messages[1:]

                async def _create_completion():
                    return await client.complete(
                        system_prompt=system_prompt_content,
                        messages=chat_messages,
                        model=model,
                        temperature=0.7,
                        max_tokens=max_tokens,
                    )

                raw_assistant_content = await async_retry_openai_call(
                    _create_completion,
                    max_attempts=5,  # More retries for main chat completion
                    initial_wait=1.0,
                    max_wait=60.0
                )
            except APIError as e:
                logger.error(f"LLM API error after retries: {str(e)}", exc_info=True)
                # Provide more specific error messages
                status_code = getattr(e, 'status_code', None)
                if status_code:
                    if status_code == 429:
                        raise ValueError("Rate limit exceeded. Please wait a moment and try again.")
                    elif status_code == 401:
                        raise ValueError("Invalid API key. Please check your API key in settings.")
                    elif status_code == 403:
                        raise ValueError("API access forbidden. Please check your API key permissions.")
                    elif status_code in [500, 502, 503, 504]:
                        raise ValueError("The LLM service is temporarily unavailable. Please try again later.")
                raise ValueError(f"LLM API error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error calling LLM API: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to generate response: {str(e)}")

            raw_assistant_content = raw_assistant_content or ""
            logger.info(f"[Annotations] Raw OpenAI response: {raw_assistant_content[:500]}...")

            # Estimate token usage. The chat-completion client returns only the
            # assistant text (not a usage object), so mirror the streaming path
            # and estimate from the assembled messages + generated content.
            token_usage = None
            try:
                prompt_tokens = TokenService.estimate_context_tokens(messages, model)
                completion_tokens = TokenService.count_tokens(raw_assistant_content, model)
                total_tokens = prompt_tokens + completion_tokens

                token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                logger.info(
                    f"Token usage: {token_usage['prompt_tokens']} prompt + "
                    f"{token_usage['completion_tokens']} completion = "
                    f"{token_usage['total_tokens']} total tokens"
                )
                logger.info(
                    f"Token budget utilization: "
                    f"{token_usage['total_tokens']}/{max_context_tokens} "
                    f"({100 * token_usage['total_tokens'] / max_context_tokens:.1f}%)"
                )
            except Exception as e:
                logger.warning(f"Failed to estimate token usage: {e}")

            # Parse annotations from the response
            assistant_content, annotations = self._parse_annotations(
                raw_assistant_content,
                relevant_chunks
            )
            logger.info(f"[Annotations] Parsed {len(annotations)} annotations from response")
            if annotations:
                logger.info(f"[Annotations] Annotation details: {annotations}")

            # Verify citations match available chunks
            citation_warnings = self._verify_citations(
                response_text=assistant_content,
                annotations=annotations,
                relevant_chunks=relevant_chunks
            )

            # Score answer quality
            quality_scores = await self._score_answer_quality(
                query=content,
                answer=assistant_content,
                context_chunks=relevant_chunks,
                user_api_key=api_key,
                provider=provider
            )

            # Check if this is the first message in the conversation (for title generation)
            existing_message_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()

            is_first_message = existing_message_count == 0

            # Save user message
            user_message = Message(
                content=content,
                role="user",
                conversation_id=conversation_id
            )
            db.add(user_message)
            db.flush()

            # Save assistant message with context (store all metadata including quality metrics)
            message_context = {
                "chunks": relevant_chunks,
                "annotations": annotations,
                "token_usage": token_usage,
                "chunk_selection_stats": chunk_stats,
                "query_classification": query_classification,
                "citation_warnings": citation_warnings,
                "quality_scores": quality_scores
            }
            assistant_message = Message(
                content=assistant_content,
                role="assistant",
                conversation_id=conversation_id,
                context=message_context
            )
            db.add(assistant_message)

            # Generate and update conversation title if this is the first message
            if is_first_message:
                try:
                    conversation = db.query(Conversation).filter(
                        Conversation.id == conversation_id
                    ).first()

                    if conversation and not conversation.title:
                        title = await self._generate_conversation_title(content, api_key, provider)
                        conversation.title = title
                        logger.info(f"Set conversation title to: {title}")
                except Exception as e:
                    logger.warning(f"Failed to generate conversation title: {e}")
                    # Don't fail the whole request if title generation fails

            db.commit()

            db.refresh(user_message)
            db.refresh(assistant_message)

            logger.debug("Messages saved successfully")

            # Cache the response (only for first message to avoid context issues)
            if is_first_message:
                await cache_service.set_response(
                    document_id,
                    query_embedding,
                    assistant_content,
                    annotations,
                    relevant_chunks
                )

            return {
                "user_message": {
                    "id": user_message.id,
                    "role": user_message.role,
                    "content": user_message.content,
                    "created_at": user_message.created_at,
                    "context": None,
                    "annotations": None
                },
                "assistant_message": {
                    "id": assistant_message.id,
                    "role": assistant_message.role,
                    "content": assistant_message.content,
                    "created_at": assistant_message.created_at,
                    "context": relevant_chunks,
                    "annotations": annotations
                }
            }
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error(f"Error in generate_chat_response: {str(e)}", exc_info=True)
            # Rollback any pending transaction
            db.rollback()
            raise

    async def generate_chat_response_with_agent(
        self,
        db: Session,
        user: User,
        content: str,
        conversation_id: str,
        document_id: str,
        model: str = "gpt-4"
    ) -> Dict:
        """
        Generate a chat response using the LangGraph agent workflow.

        This method routes queries through the agent-based reasoning system
        with adaptive complexity handling and quality verification.
        Falls back to linear pipeline on errors.

        Args:
            db: Database session
            user: User object
            content: User's message/query
            conversation_id: Conversation identifier
            document_id: Document identifier
            model: OpenAI model to use (default: gpt-4)

        Returns:
            Dict matching the same format as generate_chat_response
        """
        from ..config import settings
        from .agent_service import get_agent_service

        # Check if agents are enabled
        if not settings.AGENT_ENABLED:
            logger.info("Agent workflow disabled, using linear pipeline")
            return await self.generate_chat_response(
                db=db,
                user=user,
                content=content,
                conversation_id=conversation_id,
                document_id=document_id,
                model=model
            )

        try:
            # Get decrypted API key for the resolved provider
            provider = self._resolve_provider_for_model(model)
            api_key = user.get_decrypted_key(provider.value)
            if not api_key:
                logger.error(f"User {user.id} has no {provider.value} API key configured")
                raise ValueError(
                    f"User has no {provider.value} API key configured. "
                    "Please configure your API key in settings."
                )

            logger.info(f"[Agent] Processing query with agent workflow for user {user.id}")

            # Initialize agent service
            agent_service = get_agent_service()

            # Process query through agent workflow
            agent_response = await agent_service.process_query(
                user_query=content,
                conversation_id=conversation_id,
                document_id=document_id,
                user_id=str(user.id),
                db_session=db,
                user_api_key=api_key,
                model=model
            )

            # Extract data from agent response
            assistant_content = agent_response["assistant_message"]["content"]
            annotations = agent_response["assistant_message"]["annotations"]
            relevant_chunks = agent_response["assistant_message"]["context"]
            metadata = agent_response.get("metadata", {})

            logger.info(
                f"[Agent] Response generated successfully: "
                f"type={metadata.get('query_classification', {}).get('query_type')}, "
                f"strategy={metadata.get('retrieval_strategy')}, "
                f"chunks={len(relevant_chunks)}, "
                f"annotations={len(annotations)}"
            )

            # Check if this is the first message in the conversation
            existing_message_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()

            is_first_message = existing_message_count == 0

            # Save user message
            user_message = Message(
                content=content,
                role="user",
                conversation_id=conversation_id
            )
            db.add(user_message)
            db.flush()

            # Save assistant message with context (include agent metadata)
            message_context = {
                "chunks": relevant_chunks,
                "annotations": annotations,
                "agent_metadata": {
                    "used_agent": True,
                    "query_classification": metadata.get("query_classification"),
                    "retrieval_strategy": metadata.get("retrieval_strategy"),
                    "quality_scores": metadata.get("quality_scores"),
                    "citation_warnings": metadata.get("citation_warnings", []),
                    "verified": metadata.get("verified"),
                    "retry_count": metadata.get("retry_count", 0)
                }
            }

            assistant_message = Message(
                content=assistant_content,
                role="assistant",
                conversation_id=conversation_id,
                context=message_context
            )
            db.add(assistant_message)

            # Generate and update conversation title if this is the first message
            if is_first_message:
                try:
                    conversation = db.query(Conversation).filter(
                        Conversation.id == conversation_id
                    ).first()

                    if conversation and not conversation.title:
                        title = await self._generate_conversation_title(content, api_key, provider)
                        conversation.title = title
                        logger.info(f"Set conversation title to: {title}")
                except Exception as e:
                    logger.warning(f"Failed to generate conversation title: {e}")
                    # Don't fail the whole request if title generation fails

            db.commit()
            db.refresh(user_message)
            db.refresh(assistant_message)

            logger.info("[Agent] Messages saved successfully")

            # Return in the same format as generate_chat_response
            return {
                "user_message": {
                    "id": user_message.id,
                    "role": user_message.role,
                    "content": user_message.content,
                    "created_at": user_message.created_at,
                    "context": None,
                    "annotations": None
                },
                "assistant_message": {
                    "id": assistant_message.id,
                    "role": assistant_message.role,
                    "content": assistant_message.content,
                    "created_at": assistant_message.created_at,
                    "context": relevant_chunks,
                    "annotations": annotations
                }
            }

        except Exception as e:
            logger.error(
                f"[Agent] Error in agent workflow: {str(e)}. "
                f"Falling back to linear pipeline.",
                exc_info=True
            )
            # Rollback any partial database changes
            db.rollback()

            # Fallback to linear pipeline
            logger.info("[Agent] Using linear pipeline as fallback")
            return await self.generate_chat_response(
                db=db,
                user=user,
                content=content,
                conversation_id=conversation_id,
                document_id=document_id,
                model=model
            )

    async def generate_chat_response_stream_with_agent(
        self,
        db: Session,
        user: User,
        content: str,
        conversation_id: str,
        document_id: str,
        model: str = "gpt-4"
    ) -> AsyncGenerator:
        """
        Generate a streaming chat response using the LangGraph agent workflow.

        This method streams the agent workflow execution, emitting events for each
        workflow step (understand, retrieve, generate, verify, format) along with
        the final response data.

        Args:
            db: Database session
            user: User object
            content: User's message/query
            conversation_id: Conversation identifier
            document_id: Document identifier
            model: OpenAI model to use (default: gpt-4)

        Yields:
            str: Server-Sent Events (SSE) formatted JSON strings
        """
        from ..config import settings
        from .agent_service import get_agent_service

        # Check if agents are enabled
        if not settings.AGENT_ENABLED:
            logger.info("[Agent Stream] Agent workflow disabled, falling back to linear pipeline streaming")
            async for chunk in self.generate_chat_response_stream(
                db=db,
                user=user,
                content=content,
                conversation_id=conversation_id,
                document_id=document_id,
                model=model
            ):
                yield chunk
            return

        try:
            # Get decrypted API key for the resolved provider
            provider = self._resolve_provider_for_model(model)
            api_key = user.get_decrypted_key(provider.value)
            if not api_key:
                logger.error(f"User {user.id} has no {provider.value} API key configured")
                error_data = json.dumps({
                    'type': 'error',
                    'content': f'User has no {provider.value} API key configured. Please configure your API key in settings.'
                })
                yield f"data: {error_data}\n\n"
                return

            logger.info(f"[Agent Stream] Starting agent workflow streaming for user {user.id}")

            # Initialize agent service
            agent_service = get_agent_service()

            # Variables to collect final response data
            final_response_data = None

            # Stream workflow execution
            async for event_data in agent_service.process_query_streaming(
                user_query=content,
                conversation_id=conversation_id,
                document_id=document_id,
                user_id=str(user.id),
                db_session=db,
                user_api_key=api_key,
                model=model
            ):
                # Forward agent events to client
                yield event_data

                # Parse the event to capture final response
                try:
                    # Extract JSON from SSE format ("data: {...}\n\n")
                    if event_data.startswith("data: "):
                        json_str = event_data[6:].strip()
                        event_obj = json.loads(json_str)

                        if event_obj.get("type") == "final_response":
                            final_response_data = event_obj.get("data")
                except json.JSONDecodeError:
                    pass  # Ignore malformed events

            # Save messages to database if we got a final response
            if final_response_data:
                try:
                    assistant_content = final_response_data["assistant_message"]["content"]
                    annotations = final_response_data["assistant_message"]["annotations"]
                    relevant_chunks = final_response_data["assistant_message"]["context"]
                    metadata = final_response_data.get("metadata", {})

                    # Check if this is the first message
                    existing_message_count = db.query(Message).filter(
                        Message.conversation_id == conversation_id
                    ).count()

                    is_first_message = existing_message_count == 0

                    # Save user message
                    user_message = Message(
                        content=content,
                        role="user",
                        conversation_id=conversation_id
                    )
                    db.add(user_message)
                    db.flush()

                    # Save assistant message with agent metadata
                    message_context = {
                        "chunks": relevant_chunks,
                        "annotations": annotations,
                        "agent_metadata": {
                            "used_agent": True,
                            "streaming": True,
                            "query_classification": metadata.get("query_classification"),
                            "retrieval_strategy": metadata.get("retrieval_strategy"),
                            "quality_scores": metadata.get("quality_scores"),
                            "citation_warnings": metadata.get("citation_warnings", []),
                            "verified": metadata.get("verified"),
                            "retry_count": metadata.get("retry_count", 0)
                        }
                    }

                    assistant_message = Message(
                        content=assistant_content,
                        role="assistant",
                        conversation_id=conversation_id,
                        context=message_context
                    )
                    db.add(assistant_message)

                    # Generate and update conversation title if first message
                    if is_first_message:
                        try:
                            conversation = db.query(Conversation).filter(
                                Conversation.id == conversation_id
                            ).first()

                            if conversation and not conversation.title:
                                title = await self._generate_conversation_title(content, api_key, provider)
                                conversation.title = title
                                logger.info(f"Set conversation title to: {title}")
                        except Exception as e:
                            logger.warning(f"Failed to generate conversation title: {e}")

                    db.commit()
                    logger.info("[Agent Stream] Messages saved successfully")

                except Exception as e:
                    logger.error(f"[Agent Stream] Error saving messages: {e}", exc_info=True)
                    db.rollback()
                    error_data = json.dumps({
                        'type': 'error',
                        'content': f'Error saving messages: {str(e)}'
                    })
                    yield f"data: {error_data}\n\n"

        except Exception as e:
            logger.error(
                f"[Agent Stream] Error in agent workflow streaming: {str(e)}. "
                f"Falling back to linear pipeline streaming.",
                exc_info=True
            )
            # Rollback any partial changes
            db.rollback()

            # Fallback to linear pipeline streaming
            logger.info("[Agent Stream] Using linear pipeline streaming as fallback")
            async for chunk in self.generate_chat_response_stream(
                db=db,
                user=user,
                content=content,
                conversation_id=conversation_id,
                document_id=document_id,
                model=model
            ):
                yield chunk

    async def generate_chat_response_stream(
        self,
        db: Session,
        user: User,
        content: str,
        conversation_id: str,
        document_id: str,
        model: str = "gpt-4"
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming chat response using OpenAI with RAG.
        Yields chunks of text as they arrive from OpenAI.
        """
        accumulated_content = ""
        relevant_chunks = []

        try:
            # Get decrypted API key for the resolved provider
            provider = self._resolve_provider_for_model(model)
            api_key = user.get_decrypted_key(provider.value)
            if not api_key:
                logger.error(f"User {user.id} has no {provider.value} API key configured")
                yield f"data: {json.dumps({'type': 'error', 'content': f'User has no {provider.value} API key configured. Please configure your API key in settings.'})}\n\n"
                return

            logger.debug(f"Generating streaming chat response for user {user.id}, conversation {conversation_id}")

            # Initialize cache service
            cache_service = await get_cache_service()

            # Get query embedding for response cache lookup
            query_embedding = await cache_service.get_embedding(content)
            if query_embedding is None:
                query_embedding = await self.embedding_service.generate_embedding_async(content)
                await cache_service.set_embedding(content, query_embedding)

            # Check for similar cached response (skip if conversation has history)
            history_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()

            if history_count == 0:  # Only cache responses for first message (no context)
                cached_response = await cache_service.find_similar_response(query_embedding, document_id)
                if cached_response:
                    logger.info("Using cached response for streaming")
                    # Stream the cached content
                    for char in cached_response['content']:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': char})}\n\n"

                    # Save messages to database
                    user_message = Message(
                        content=content,
                        role="user",
                        conversation_id=conversation_id
                    )
                    db.add(user_message)
                    db.flush()

                    assistant_message = Message(
                        content=cached_response['content'],
                        role="assistant",
                        conversation_id=conversation_id,
                        context={
                            "chunks": cached_response['chunks'],
                            "annotations": cached_response['annotations']
                        }
                    )
                    db.add(assistant_message)
                    db.commit()
                    db.refresh(user_message)
                    db.refresh(assistant_message)

                    # Send final message
                    final_data = {
                        'type': 'done',
                        'user_message': {
                            "id": user_message.id,
                            "role": user_message.role,
                            "content": user_message.content,
                            "created_at": user_message.created_at.isoformat(),
                            "context": None,
                            "annotations": None
                        },
                        'assistant_message': {
                            "id": assistant_message.id,
                            "role": assistant_message.role,
                            "content": assistant_message.content,
                            "created_at": assistant_message.created_at.isoformat(),
                            "context": cached_response['chunks'],
                            "annotations": cached_response['annotations']
                        }
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    return

            # Initialize LLM client (provider resolved from the selected model)
            provider = self._resolve_provider_for_model(model)
            client = get_llm_client(provider, api_key)

            # Get configuration
            from ..config import settings
            max_context_tokens = getattr(settings, 'MAX_CONTEXT_TOKENS', 100000)
            rerank_top_k = getattr(settings, 'RERANK_TOP_K', 20)

            # Find relevant chunks (retrieve more for token-based selection)
            logger.debug(f"Finding similar chunks for document {document_id}")
            candidate_chunks = await self.find_similar_chunks(db, content, document_id, limit=rerank_top_k, user_api_key=api_key, provider=provider)
            # TODO(human): Add diagnostic logging to understand why chunks might be empty
            logger.info(f"[DEBUG CHAT] Retrieved {len(candidate_chunks)} candidate chunks for document {document_id}")

            # Get conversation history for token counting
            logger.debug(f"Fetching conversation history for {conversation_id}")
            history = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(10).all()

            # Classify query type for adaptive prompting
            query_classification = await self._classify_query_type(content, api_key, provider)
            logger.info(
                f"Query classification: type={query_classification['query_type']}, "
                f"complexity={query_classification['complexity']}, "
                f"requires_cot={query_classification['requires_cot']}"
            )

            # Count tokens for dynamic chunk selection
            # 1. System prompt template (without chunks)
            system_prompt_template = TOKEN_BUDGET_TEMPLATE  # imported from prompt_builder

            system_prompt_tokens = TokenService.count_tokens(system_prompt_template, model)

            # 2. User message tokens
            user_message_tokens = TokenService.count_tokens(content, model)

            # 3. History tokens
            history_messages = [{"role": msg.role, "content": msg.content} for msg in history]
            history_tokens = TokenService.estimate_context_tokens(history_messages, model)

            # 4. Response reserve (max_completion_tokens)
            response_reserve_tokens = 1000

            # Select chunks dynamically based on token limits
            relevant_chunks, chunk_stats = self._select_chunks_by_token_limit(
                chunks=candidate_chunks,
                max_tokens=max_context_tokens,
                model=model,
                system_prompt_tokens=system_prompt_tokens,
                user_message_tokens=user_message_tokens,
                history_tokens=history_tokens,
                response_reserve_tokens=response_reserve_tokens
            )

            logger.info(
                f"[Stream] Dynamic chunk selection: {chunk_stats['selected_chunks']}/{len(candidate_chunks)} chunks, "
                f"{chunk_stats['total_chunk_tokens']} tokens, "
                f"{chunk_stats['truncated_chunks']} truncated, "
                f"{chunk_stats['skipped_chunks']} skipped"
            )

            # Format context from chunks
            if relevant_chunks:
                context_text = "\n\n".join(
                    self._format_chunk_for_context(chunk)
                    for chunk in relevant_chunks
                )
            else:
                context_text = "No relevant document sections found."

            # Build adaptive system prompt with few-shot examples and chain-of-thought
            system_prompt_content = self._build_system_prompt(
                context_text=context_text,
                query_type=query_classification['query_type'],
                complexity=query_classification['complexity'],
                requires_cot=query_classification['requires_cot']
            )

            # Create system message
            system_message = {
                "role": "system",
                "content": system_prompt_content
            }

            # Format history for OpenAI (history already fetched above for token counting)
            messages = [system_message]
            for msg in history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            # Add current user message
            messages.append({
                "role": "user",
                "content": content
            })

            # Check if this is the first message in the conversation (for title generation)
            existing_message_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()

            is_first_message = existing_message_count == 0

            # Save user message first
            user_message = Message(
                content=content,
                role="user",
                conversation_id=conversation_id
            )
            db.add(user_message)
            db.flush()

            # Stream OpenAI response
            logger.debug(f"Calling OpenAI API with streaming for model {model}")

            # Get adaptive token limit based on query complexity
            max_tokens = self._get_adaptive_token_limit(query_classification['complexity'])
            logger.info(f"Using {max_tokens} max completion tokens for {query_classification['complexity']} query (streaming)")

            try:
                # messages[0] is the system message; LLMClient takes system separately.
                chat_messages = messages[1:]

                # Stream tokens directly from the provider-agnostic client.
                async for content_chunk in client.stream(
                    system_prompt=system_prompt_content,
                    messages=chat_messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=max_tokens,
                ):
                    if content_chunk:
                        accumulated_content += content_chunk
                        # Send chunk to client
                        yield f"data: {json.dumps({'type': 'chunk', 'content': content_chunk})}\n\n"

            except APIError as e:
                logger.error(f"LLM API error during streaming: {str(e)}", exc_info=True)
                error_msg = f"LLM API error: {str(e)}"
                status_code = getattr(e, 'status_code', None)
                if status_code:
                    if status_code == 429:
                        error_msg = "Rate limit exceeded. Please wait a moment and try again."
                    elif status_code == 401:
                        error_msg = "Invalid API key. Please check your API key in settings."
                    elif status_code == 403:
                        error_msg = "API access forbidden. Please check your API key permissions."
                    elif status_code in [500, 502, 503, 504]:
                        error_msg = "The LLM service is temporarily unavailable. Please try again later."
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                return
            except Exception as e:
                logger.error(f"Unexpected error calling LLM API: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'content': f'Failed to generate response: {str(e)}'})}\n\n"
                return

            # Parse annotations from the complete response
            logger.info(f"[Annotations] Raw OpenAI response: {accumulated_content[:500]}...")
            assistant_content, annotations = self._parse_annotations(
                accumulated_content,
                relevant_chunks
            )
            logger.info(f"[Annotations] Parsed {len(annotations)} annotations from response")

            # Verify citations match available chunks
            citation_warnings = self._verify_citations(
                response_text=assistant_content,
                annotations=annotations,
                relevant_chunks=relevant_chunks
            )

            # Score answer quality
            quality_scores = await self._score_answer_quality(
                query=content,
                answer=assistant_content,
                context_chunks=relevant_chunks,
                user_api_key=api_key,
                provider=provider
            )

            # Estimate token usage for streaming response (OpenAI doesn't provide usage in streams)
            token_usage = None
            try:
                # Count input tokens (context + user message + history)
                prompt_tokens = TokenService.estimate_context_tokens(messages, model)

                # Count output tokens
                completion_tokens = TokenService.count_tokens(accumulated_content, model)

                total_tokens = prompt_tokens + completion_tokens

                token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }

                logger.info(
                    f"[Stream] Estimated token usage: {token_usage['prompt_tokens']} prompt + "
                    f"{token_usage['completion_tokens']} completion = "
                    f"{token_usage['total_tokens']} total tokens"
                )
                logger.info(
                    f"[Stream] Token budget utilization: "
                    f"{token_usage['total_tokens']}/{max_context_tokens} "
                    f"({100 * token_usage['total_tokens'] / max_context_tokens:.1f}%)"
                )
            except Exception as e:
                logger.warning(f"[Stream] Failed to estimate token usage: {e}")

            # Save assistant message with context (include all metadata including quality metrics)
            message_context = {
                "chunks": relevant_chunks,
                "annotations": annotations,
                "token_usage": token_usage,
                "chunk_selection_stats": chunk_stats,
                "query_classification": query_classification,
                "citation_warnings": citation_warnings,
                "quality_scores": quality_scores
            }
            assistant_message = Message(
                content=assistant_content,
                role="assistant",
                conversation_id=conversation_id,
                context=message_context
            )
            db.add(assistant_message)

            # Generate and update conversation title if this is the first message
            if is_first_message:
                try:
                    conversation = db.query(Conversation).filter(
                        Conversation.id == conversation_id
                    ).first()

                    if conversation and not conversation.title:
                        title = await self._generate_conversation_title(content, api_key, provider)
                        conversation.title = title
                        logger.info(f"Set conversation title to: {title}")
                except Exception as e:
                    logger.warning(f"Failed to generate conversation title: {e}")

            db.commit()
            db.refresh(user_message)
            db.refresh(assistant_message)

            logger.debug("Messages saved successfully")

            # Cache the response (only for first message to avoid context issues)
            if is_first_message:
                await cache_service.set_response(
                    document_id,
                    query_embedding,
                    assistant_content,
                    annotations,
                    relevant_chunks
                )

            # Send final message with complete data
            final_data = {
                'type': 'done',
                'user_message': {
                    "id": user_message.id,
                    "role": user_message.role,
                    "content": user_message.content,
                    "created_at": user_message.created_at.isoformat(),
                    "context": None,
                    "annotations": None
                },
                'assistant_message': {
                    "id": assistant_message.id,
                    "role": assistant_message.role,
                    "content": assistant_message.content,
                    "created_at": assistant_message.created_at.isoformat(),
                    "context": relevant_chunks,
                    "annotations": annotations
                }
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Error in generate_chat_response_stream: {str(e)}", exc_info=True)
            db.rollback()
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"

    def _check_hierarchical_chunking(self, db: Session, document_id: str) -> bool:
        """
        Check if a document uses hierarchical chunking.

        Args:
            db: Database session
            document_id: ID of the document to check

        Returns:
            True if document has hierarchical chunks (parent/child), False otherwise
        """
        from ..models.document import DocumentChunk
        from sqlalchemy import text

        try:
            # Check if any chunks for this document have chunk_level = 'parent' or 'child'
            query = text("""
                SELECT COUNT(*)
                FROM document_chunks
                WHERE document_id = :document_id
                AND chunk_level IN ('parent', 'child')
                LIMIT 1
            """)

            result = db.execute(query, {"document_id": document_id}).scalar()
            has_hierarchical = result > 0

            if has_hierarchical:
                logger.debug(f"Document {document_id} uses hierarchical chunking")
            else:
                logger.debug(f"Document {document_id} uses flat chunking")

            return has_hierarchical

        except Exception as e:
            logger.warning(f"Error checking hierarchical chunking, assuming flat: {e}")
            return False

    async def _retrieve_with_parent_child(
        self,
        db: Session,
        query_embedding: List[float],
        document_id: str,
        limit: int
    ) -> List[Dict]:
        """
        Retrieve chunks using hierarchical parent-child strategy.

        Strategy:
        1. Search child chunks for precision (small chunks match better)
        2. Fetch corresponding parent chunks for context
        3. Return parent chunks to LLM (large chunks provide better context)

        Args:
            db: Database session
            query_embedding: Query embedding vector
            document_id: ID of document to search
            limit: Maximum number of parent chunks to return

        Returns:
            List of parent chunk dictionaries with metadata
        """
        from sqlalchemy import text
        from ..models.document import ParentChildRelationship

        try:
            # Step 1: Search CHILD chunks for precise matching
            embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

            # Search child chunks only
            child_search_query = text("""
                SELECT
                    id,
                    content,
                    page_number,
                    document_id,
                    position_data,
                    chunk_type,
                    chunk_level,
                    1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                FROM document_chunks
                WHERE document_id = :document_id
                AND chunk_level = 'child'
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """)

            logger.debug(f"Searching {limit} child chunks for document {document_id}")
            import time
            search_start = time.time()

            child_results = db.execute(
                child_search_query,
                {"embedding": embedding_str, "document_id": document_id, "limit": limit * 2}  # Get more children
            ).fetchall()

            search_time = (time.time() - search_start) * 1000
            logger.debug(f"Child chunk search completed in {search_time:.2f}ms, found {len(child_results)} children")

            if not child_results:
                logger.warning("No child chunks found, returning empty results")
                return []

            # Step 2: Get parent chunk IDs for the matched children
            child_chunk_ids = [row[0] for row in child_results]

            # Query parent-child relationships to find parents
            parent_query = text("""
                SELECT DISTINCT pcr.parent_chunk_id, c.id as child_id
                FROM parent_child_relationships pcr
                JOIN document_chunks c ON c.id = pcr.child_chunk_id
                WHERE pcr.child_chunk_id = ANY(:child_ids)
            """)

            parent_results = db.execute(
                parent_query,
                {"child_ids": child_chunk_ids}
            ).fetchall()

            # Map children to parents (some parents may have multiple matched children)
            parent_chunk_ids = list(set([row[0] for row in parent_results]))
            logger.debug(f"Found {len(parent_chunk_ids)} unique parent chunks for {len(child_results)} children")

            # Step 3: Fetch PARENT chunks to return to LLM
            if not parent_chunk_ids:
                logger.warning("No parent chunks found for matched children")
                return []

            # Fetch parent chunks (preserving child match order)
            parent_fetch_query = text("""
                SELECT
                    id,
                    content,
                    page_number,
                    document_id,
                    position_data,
                    chunk_type,
                    chunk_level
                FROM document_chunks
                WHERE id = ANY(:parent_ids)
            """)

            parent_chunks = db.execute(
                parent_fetch_query,
                {"parent_ids": parent_chunk_ids[:limit]}  # Limit to requested number
            ).fetchall()

            # Format parent chunks for return
            formatted_chunks = []
            for chunk in parent_chunks:
                formatted_chunk = {
                    "id": chunk[0],
                    "content": chunk[1],
                    "pageNumber": chunk[2],
                    "documentId": chunk[3],
                    "positionData": chunk[4],
                    "chunk_type": chunk[5],
                    "chunk_level": chunk[6],
                    "similarity": 0.0  # Parent doesn't have direct similarity score
                }
                formatted_chunks.append(formatted_chunk)

            logger.info(
                f"Hierarchical retrieval complete: {len(child_results)} children matched, "
                f"returning {len(formatted_chunks)} parent chunks"
            )

            return formatted_chunks

        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {e}", exc_info=True)
            # Fallback to empty results rather than crashing
            return []
