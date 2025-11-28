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
from .retry_utils import retry_openai_call, async_retry_openai_call
from .cache_service import get_cache_service
from .query_expansion_service import get_query_expansion_service
from .token_service import TokenService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.embedding_service = get_embedding_service()

    async def _generate_conversation_title(self, user_message: str, user_api_key: str) -> str:
        """
        Generate a smart, concise title for the conversation based on the first user message.
        Uses LLM to create a title that's 3-6 words.
        """
        try:
            client = AsyncOpenAI(api_key=user_api_key)

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

            # Use async retry logic for OpenAI API call
            async def _create_completion():
                return await client.chat.completions.create(
                    model="gpt-4o-mini",  # Use cheaper model for title generation
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates concise conversation titles."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_completion_tokens=20
                )

            completion = await async_retry_openai_call(
                _create_completion,
                max_attempts=3,  # Fewer retries for title generation (non-critical)
                initial_wait=1.0,
                max_wait=30.0
            )

            title = completion.choices[0].message.content.strip()
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
        """
        Parse annotations from the LLM response.
        Returns (cleaned_response, annotations_list)
        """
        annotations = []
        cleaned_response = response_text

        # Look for annotations block
        annotation_pattern = r'```annotations\s*([\s\S]*?)\s*```'
        match = re.search(annotation_pattern, response_text)

        if match:
            # Remove the annotations block from the response
            cleaned_response = re.sub(annotation_pattern, '', response_text).strip()

            try:
                annotation_data = json.loads(match.group(1))

                if isinstance(annotation_data, list):
                    for item in annotation_data:
                        page_num = item.get('pageNumber', 1)
                        text_to_highlight = item.get('textToHighlight', '')
                        annotation_type = item.get('type', 'highlight')
                        explanation = item.get('explanation', '')

                        # Create annotation with estimated bounds
                        # These are rough estimates - the frontend will refine them
                        annotation = {
                            'id': str(uuid.uuid4()),
                            'type': annotation_type,
                            'pageNumber': page_num,
                            'bounds': {
                                'x': 10,      # Default left margin
                                'y': 30,      # Start from upper portion
                                'width': 80,  # Most of page width
                                'height': 5   # Rough text line height
                            },
                            'textContent': text_to_highlight,
                            'color': self._get_annotation_color(annotation_type),
                            'label': None
                        }

                        annotation_ref = {
                            'pageNumber': page_num,
                            'annotations': [annotation],
                            'sourceText': text_to_highlight,
                            'explanation': explanation
                        }
                        annotations.append(annotation_ref)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse annotations JSON: {e}")
            except Exception as e:
                logger.warning(f"Error processing annotations: {e}")

        return cleaned_response, annotations

    def _get_annotation_color(self, annotation_type: str) -> str:
        """Get color for annotation type."""
        colors = {
            'highlight': 'rgba(255, 235, 59, 0.4)',   # Yellow
            'circle': 'rgba(33, 150, 243, 0.5)',      # Blue
            'box': 'rgba(76, 175, 80, 0.3)',          # Green
            'underline': 'rgba(244, 67, 54, 0.5)'     # Red
        }
        return colors.get(annotation_type, colors['highlight'])

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

    async def find_similar_chunks(
        self,
        db: Session,
        query: str,
        document_id: str,
        limit: int = 5,
        user_api_key: Optional[str] = None
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

            # Get search weights and query expansion settings from config
            semantic_weight = getattr(settings, 'SEMANTIC_SEARCH_WEIGHT', 0.7)
            keyword_weight = getattr(settings, 'KEYWORD_SEARCH_WEIGHT', 0.3)
            rerank_enabled = getattr(settings, 'RERANK_ENABLED', False)
            query_expansion_enabled = getattr(settings, 'QUERY_EXPANSION_ENABLED', False)
            rrf_k = getattr(settings, 'RRF_K', 60)

            cache_service = await get_cache_service()

            # Retrieve more candidates for fusion (e.g., top 10-15)
            semantic_limit = max(limit * 2, 10)

            # QUERY EXPANSION: Multi-query retrieval with RRF (if enabled)
            semantic_chunks = {}
            query_embedding = None  # Initialize query_embedding early

            if query_expansion_enabled and user_api_key:
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
                        user_api_key=user_api_key
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
                            "semantic_score": chunk["similarity"]  # RRF score
                        }

                    logger.info(f"RRF combined results: {len(semantic_chunks)} unique chunks")

                except Exception as e:
                    logger.error(f"Query expansion failed, falling back to single-query search: {e}")
                    # Fall through to single-query search below
                    query_expansion_enabled = False  # Disable for this request

            # SINGLE-QUERY SEMANTIC SEARCH (original behavior or fallback)
            if not query_expansion_enabled or not user_api_key:
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
                    "similarity": chunk["similarity"]
                }
                # Optionally include rerank score for debugging
                if "rerank_score" in chunk:
                    chunk_data["rerank_score"] = chunk["rerank_score"]
                final_chunks.append(chunk_data)

            search_type = "hybrid" if keyword_search_success else "semantic-only"
            rerank_status = "with re-ranking" if rerank_enabled else "without re-ranking"
            logger.info(f"Hybrid search ({search_type}, {rerank_status}) returned {len(final_chunks)} chunks")

            # Cache the final results (cache key includes re-ranking status)
            await cache_service.set_chunks(
                document_id, query_embedding, final_chunks, rerank_enabled=rerank_enabled
            )

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
            formatted_chunk = f"[Page {page_number}]: {chunk_content}"

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
                    truncated_formatted = f"[Page {page_number}]: {truncated_content}"
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

    async def _classify_query_type(self, query: str, user_api_key: str) -> Dict:
        """
        Classify query type and complexity to enable adaptive prompting.

        Returns dict with:
        - query_type: "factual", "analytical", "comparative", "follow-up", or "clarification"
        - complexity: "simple", "moderate", "complex"
        - requires_cot: bool (whether chain-of-thought prompting should be used)
        """
        from ..config import settings

        # Skip if query classification is disabled
        if not settings.ENABLE_QUERY_CLASSIFICATION:
            return {
                "query_type": "factual",
                "complexity": "simple",
                "requires_cot": False
            }

        try:
            client = AsyncOpenAI(api_key=user_api_key)

            classification_prompt = """Analyze this query and classify it.

Query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
  "query_type": "<one of: factual, analytical, comparative, follow-up, clarification>",
  "complexity": "<one of: simple, moderate, complex>",
  "requires_cot": <true or false>
}}

Query types:
- factual: Asking for specific facts, definitions, or information
- analytical: Requiring analysis, interpretation, or synthesis
- comparative: Comparing concepts, ideas, or items
- follow-up: Building on previous conversation context
- clarification: Seeking clarification on previous responses

Complexity levels:
- simple: Single concept, straightforward answer
- moderate: Multiple concepts, some reasoning required
- complex: Deep analysis, multiple perspectives, synthesis needed

Requires chain-of-thought (COT):
- true: For moderate/complex queries requiring step-by-step reasoning
- false: For simple queries with straightforward answers""".format(query=query)

            async def _create_completion():
                return await client.chat.completions.create(
                    model=settings.QUERY_CLASSIFICATION_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a query classification assistant that outputs only valid JSON."},
                        {"role": "user", "content": classification_prompt}
                    ],
                    temperature=0.3,  # Low temperature for consistent classification
                    max_completion_tokens=100
                )

            completion = await async_retry_openai_call(
                _create_completion,
                max_attempts=3,
                initial_wait=1.0,
                max_wait=30.0
            )

            response_text = completion.choices[0].message.content.strip()

            # Parse JSON response
            result = json.loads(response_text)

            # Validate and set defaults
            valid_types = ["factual", "analytical", "comparative", "follow-up", "clarification"]
            valid_complexity = ["simple", "moderate", "complex"]

            query_type = result.get("query_type", "factual")
            if query_type not in valid_types:
                query_type = "factual"

            complexity = result.get("complexity", "simple")
            if complexity not in valid_complexity:
                complexity = "simple"

            requires_cot = result.get("requires_cot", False)

            # Override requires_cot based on settings and complexity threshold
            if settings.ENABLE_CHAIN_OF_THOUGHT:
                complexity_levels = {"simple": 0, "moderate": 1, "complex": 2}
                threshold_level = complexity_levels.get(settings.COT_COMPLEXITY_THRESHOLD, 1)
                current_level = complexity_levels.get(complexity, 0)
                requires_cot = current_level >= threshold_level
            else:
                requires_cot = False

            classification = {
                "query_type": query_type,
                "complexity": complexity,
                "requires_cot": requires_cot
            }

            logger.info(f"Query classification: {classification}")
            return classification

        except Exception as e:
            logger.warning(f"Failed to classify query: {e}. Using defaults.")
            return {
                "query_type": "factual",
                "complexity": "simple",
                "requires_cot": False
            }

    def _get_few_shot_examples(self) -> str:
        """
        Get few-shot examples demonstrating correct annotation format.

        These examples show the LLM how to:
        - Match text exactly from source material
        - Format annotations correctly with page numbers
        - Use different annotation types appropriately
        - Link annotations to the question being answered
        """
        return """
# Few-Shot Examples

## Example 1: Factual Query with Highlight Annotation

**User Question**: "What is photosynthesis?"

**Context**:
[Page 12]: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct.

**Assistant Response**:
Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water [Page 12]. This process generally involves the green pigment chlorophyll and generates oxygen as a byproduct [Page 12].

**Annotations**:
```json
{
  "annotations": [
    {
      "type": "highlight",
      "text": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.",
      "pageNumber": 12,
      "explanation": "This sentence provides the core definition of photosynthesis that directly answers the user's question."
    },
    {
      "type": "highlight",
      "text": "Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
      "pageNumber": 12,
      "explanation": "This sentence explains the key components and byproducts of photosynthesis, providing additional context to the definition."
    }
  ]
}
```

## Example 2: Analytical Query with Multiple Annotations

**User Question**: "How does the mitochondria produce energy?"

**Context**:
[Page 45]: The mitochondria are known as the powerhouse of the cell. Through a process called cellular respiration, mitochondria break down glucose molecules to produce ATP (adenosine triphosphate), which serves as the cell's primary energy currency.

[Page 46]: The process occurs in three main stages: glycolysis, the Krebs cycle, and the electron transport chain. The electron transport chain is the most efficient stage, producing approximately 34 ATP molecules per glucose molecule.

**Assistant Response**:
The mitochondria produce energy through cellular respiration, where they break down glucose molecules to produce ATP (adenosine triphosphate), the cell's primary energy currency [Page 45]. This process occurs in three main stages: glycolysis, the Krebs cycle, and the electron transport chain [Page 46]. The electron transport chain is the most efficient stage, producing approximately 34 ATP molecules per glucose molecule [Page 46].

**Annotations**:
```json
{
  "annotations": [
    {
      "type": "box",
      "text": "Through a process called cellular respiration, mitochondria break down glucose molecules to produce ATP (adenosine triphosphate), which serves as the cell's primary energy currency.",
      "pageNumber": 45,
      "explanation": "This describes the fundamental mechanism by which mitochondria produce energy, introducing the key term ATP."
    },
    {
      "type": "highlight",
      "text": "The process occurs in three main stages: glycolysis, the Krebs cycle, and the electron transport chain.",
      "pageNumber": 46,
      "explanation": "This breaks down the energy production process into its three main stages, providing a systematic answer."
    },
    {
      "type": "circle",
      "text": "The electron transport chain is the most efficient stage, producing approximately 34 ATP molecules per glucose molecule.",
      "pageNumber": 46,
      "explanation": "This highlights the most efficient stage and quantifies the energy output, showing the scale of ATP production."
    }
  ]
}
```

## Example 3: Comparative Query

**User Question**: "What's the difference between mitosis and meiosis?"

**Context**:
[Page 78]: Mitosis is a type of cell division that results in two daughter cells, each having the same number of chromosomes as the parent nucleus. It is used for growth and repair in multicellular organisms.

[Page 79]: Meiosis, in contrast, is a type of cell division that reduces the chromosome number by half, creating four daughter cells. This process is essential for sexual reproduction and genetic diversity.

**Assistant Response**:
Mitosis and meiosis are both types of cell division, but they differ significantly in their outcomes and purposes. Mitosis results in two daughter cells with the same number of chromosomes as the parent cell and is used for growth and repair [Page 78]. In contrast, meiosis reduces the chromosome number by half, creating four daughter cells, and is essential for sexual reproduction and genetic diversity [Page 79].

**Annotations**:
```json
{
  "annotations": [
    {
      "type": "highlight",
      "text": "Mitosis is a type of cell division that results in two daughter cells, each having the same number of chromosomes as the parent nucleus. It is used for growth and repair in multicellular organisms.",
      "pageNumber": 78,
      "explanation": "This defines mitosis and explains its purpose, establishing one side of the comparison."
    },
    {
      "type": "highlight",
      "text": "Meiosis, in contrast, is a type of cell division that reduces the chromosome number by half, creating four daughter cells. This process is essential for sexual reproduction and genetic diversity.",
      "pageNumber": 79,
      "explanation": "This defines meiosis and contrasts it with mitosis, showing the key differences in chromosome number and purpose."
    }
  ]
}
```

---

**Key Principles Demonstrated**:
1. **Exact Text Matching**: Always copy text exactly as it appears in the source material
2. **Page Number Accuracy**: Ensure page numbers in annotations match the source chunks
3. **Annotation Types**: Use 'highlight' for key information, 'box' for processes/mechanisms, 'circle' for important data/numbers
4. **Clear Explanations**: Each annotation should explain WHY the text is relevant to answering the question
5. **Comprehensive Coverage**: Include all major points from the answer in the annotations
"""

    def _get_chain_of_thought_section(self, query_type: str) -> str:
        """
        Get chain-of-thought prompting instructions for complex queries.

        Args:
            query_type: The classified type of query (factual, analytical, comparative, etc.)

        Returns:
            Chain-of-thought prompt section tailored to the query type
        """
        base_cot = """
# Chain-of-Thought Reasoning

For this query, use systematic step-by-step reasoning:

1. **Understand the Question**: Break down what is being asked
2. **Identify Key Concepts**: Determine the main concepts that need to be addressed
3. **Analyze the Context**: Review the provided source material systematically
4. **Synthesize Information**: Combine information from multiple sources if needed
5. **Verify Your Answer**: Check that your response fully addresses the question
6. **Cite Sources**: Ensure all claims are backed by specific page citations

"""

        # Add query-type-specific instructions
        if query_type == "analytical":
            base_cot += """
**Analytical Query Guidelines**:
- Break down complex concepts into understandable components
- Explain the reasoning behind processes or phenomena
- Show how different elements relate to each other
- Use evidence from the source material to support your analysis
- Consider multiple perspectives if relevant
"""
        elif query_type == "comparative":
            base_cot += """
**Comparative Query Guidelines**:
- Clearly identify what is being compared
- Systematically address similarities first, then differences
- Use parallel structure to make comparisons clear
- Provide specific examples or evidence for each point of comparison
- Conclude with the most significant distinctions
"""
        elif query_type == "follow-up":
            base_cot += """
**Follow-Up Query Guidelines**:
- Reference previous context appropriately
- Build upon information already provided
- Add new insights or clarifications
- Maintain consistency with earlier responses
- Address the specific aspect being questioned
"""
        else:  # factual or clarification
            base_cot += """
**Response Guidelines**:
- Provide direct, accurate answers based on the source material
- Define technical terms when they first appear
- Use clear, concise language
- Support all factual claims with page citations
- Organize information logically
"""

        return base_cot

    def _build_system_prompt(
        self,
        context_text: str,
        query_type: str,
        complexity: str,
        requires_cot: bool
    ) -> str:
        """
        Build adaptive system prompt with few-shot examples and chain-of-thought instructions.

        Args:
            context_text: Formatted document chunks with page numbers
            query_type: Classified query type (factual, analytical, comparative, etc.)
            complexity: Query complexity level (simple, moderate, complex)
            requires_cot: Whether to include chain-of-thought prompting

        Returns:
            Complete system prompt tailored to the query
        """
        # Base system prompt
        prompt = f"""You are an AI tutor helping a student understand a PDF document.
You have access to the following document chunks that are relevant to the student's question:

{context_text}

When referring to content, always cite the page number like [Page X].
Make sure to use the correct page number for each piece of information.
"""

        # Add chain-of-thought section for complex queries
        if requires_cot:
            prompt += self._get_chain_of_thought_section(query_type)

        # Add formatting instructions
        prompt += """
IMPORTANT FORMATTING INSTRUCTIONS:
1. Use markdown to highlight important concepts, terms, or phrases by making them **bold** or using *italics*.
2. For direct quotes from the document, use > blockquote formatting.
3. When referring to specific sections, use [Page X] to cite the page number.
4. Use bullet points or numbered lists for step-by-step explanations.
5. For critical information or warnings, use "⚠️" at the beginning of the paragraph.

"""

        # Add few-shot examples (helps with annotation quality)
        prompt += self._get_few_shot_examples()

        # Add annotation instructions
        prompt += """

PDF ANNOTATION FEATURE - CRITICAL INSTRUCTIONS:
You MUST identify specific parts of the document that are relevant to your answer.
At the END of your response, ALWAYS include an ANNOTATIONS section with the following JSON format:

```annotations
[
  {
    "pageNumber": <page number from context above>,
    "type": "highlight",
    "textToHighlight": "<3-10 word phrase copied exactly from the document>",
    "explanation": "<why this text answers the question>"
  }
]
```

ANNOTATION RULES - FOLLOW STRICTLY:
1. ALWAYS include at least 1 annotation when you reference document content
2. The "pageNumber" MUST match a page number from the [Page X] citations above
3. The "textToHighlight" MUST be a short phrase (3-10 words) that appears EXACTLY in the document chunks above
4. Use type "highlight" for text (most common), "circle" for images/diagrams, "box" for tables
5. Copy the exact words from the document - do not paraphrase or modify them
6. Include 1-3 annotations per response, focusing on the most important points
7. Each annotation's "explanation" should clearly connect the highlighted text to the user's question

CITATION VERIFICATION:
- Double-check that all [Page X] citations in your response match the page numbers in the context above
- Ensure every annotation's pageNumber corresponds to a chunk you actually used
- If you cite information, include a corresponding annotation for that text

Make your responses helpful, clear, and educational. If the context doesn't contain the answer,
say you don't have enough information from the document and suggest looking at other pages.
"""

        return prompt

    def _verify_citations(
        self,
        response_text: str,
        annotations: List[Dict],
        relevant_chunks: List[Dict]
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
        from ..config import settings

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

    async def _score_answer_quality(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict],
        user_api_key: str
    ) -> Dict[str, any]:
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
        from ..config import settings

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
            client = AsyncOpenAI(api_key=user_api_key)

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
                return await client.chat.completions.create(
                    model=settings.QUALITY_SCORING_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an answer quality evaluator that outputs only valid JSON."},
                        {"role": "user", "content": scoring_prompt}
                    ],
                    temperature=0.3,  # Low temperature for consistent scoring
                    max_completion_tokens=200
                )

            completion = await async_retry_openai_call(
                _create_completion,
                max_attempts=2,  # Fewer retries for non-critical scoring
                initial_wait=1.0,
                max_wait=20.0
            )

            response_text = completion.choices[0].message.content.strip()

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
            # Get decrypted API key
            api_key = user.get_decrypted_api_key()
            if not api_key:
                logger.error(f"User {user.id} has no OpenAI API key configured")
                raise ValueError("User has no OpenAI API key configured. Please configure your API key in settings.")

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

            # Initialize OpenAI client
            client = AsyncOpenAI(api_key=api_key)

            # Get configuration
            from ..config import settings
            max_context_tokens = getattr(settings, 'MAX_CONTEXT_TOKENS', 100000)
            rerank_top_k = getattr(settings, 'RERANK_TOP_K', 20)

            # Find relevant chunks (retrieve more for token-based selection)
            logger.debug(f"Finding similar chunks for document {document_id}")
            candidate_chunks = await self.find_similar_chunks(db, content, document_id, limit=rerank_top_k, user_api_key=api_key)

            # Get conversation history for token counting
            logger.debug(f"Fetching conversation history for {conversation_id}")
            history = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(10).all()

            # Classify query type for adaptive prompting
            query_classification = await self._classify_query_type(content, api_key)
            logger.info(
                f"Query classification: type={query_classification['query_type']}, "
                f"complexity={query_classification['complexity']}, "
                f"requires_cot={query_classification['requires_cot']}"
            )

            # Count tokens for dynamic chunk selection
            # 1. System prompt template (without chunks)
            system_prompt_template = """You are an AI tutor helping a student understand a PDF document.
You have access to the following document chunks that are relevant to the student's question:

{context_text}

When referring to content, always cite the page number like [Page X].
Make sure to use the correct page number for each piece of information.

IMPORTANT FORMATTING INSTRUCTIONS:
1. Use markdown to highlight important concepts, terms, or phrases by making them **bold** or using *italics*.
2. For direct quotes from the document, use > blockquote formatting.
3. When referring to specific sections, use [Page X] to cite the page number.
4. Use bullet points or numbered lists for step-by-step explanations.
5. For critical information or warnings, use "⚠️" at the beginning of the paragraph.

PDF ANNOTATION FEATURE - IMPORTANT:
You MUST identify specific parts of the document that are relevant to your answer.
At the END of your response, ALWAYS include an ANNOTATIONS section with the following JSON format:

```annotations
[
  {{
    "pageNumber": <page number from context above>,
    "type": "highlight",
    "textToHighlight": "<3-10 word phrase copied exactly from the document>",
    "explanation": "<why this text answers the question>"
  }}
]
```

ANNOTATION RULES - FOLLOW STRICTLY:
1. ALWAYS include at least 1 annotation when you reference document content
2. The "pageNumber" MUST match a page number from the [Page X] citations above
3. The "textToHighlight" MUST be a short phrase (3-10 words) that appears EXACTLY in the document chunks above
4. Use type "highlight" for text (most common), "circle" for images/diagrams, "box" for tables
5. Copy the exact words from the document - do not paraphrase
6. Include 1-2 annotations per response

Make your responses helpful, clear, and educational. If the context doesn't contain the answer,
say you don't have enough information from the document and suggest looking at other pages."""

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
                context_text = "\n\n".join([
                    f"[Page {chunk['pageNumber']}]: {chunk['content']}"
                    for chunk in relevant_chunks
                ])
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
            try:
                async def _create_completion():
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_completion_tokens=1000
                    )

                completion = await async_retry_openai_call(
                    _create_completion,
                    max_attempts=5,  # More retries for main chat completion
                    initial_wait=1.0,
                    max_wait=60.0
                )
            except APIError as e:
                logger.error(f"OpenAI API error after retries: {str(e)}", exc_info=True)
                # Provide more specific error messages
                status_code = getattr(e, 'status_code', None)
                if status_code:
                    if status_code == 429:
                        raise ValueError("Rate limit exceeded. Please wait a moment and try again.")
                    elif status_code == 401:
                        raise ValueError("Invalid API key. Please check your OpenAI API key in settings.")
                    elif status_code == 403:
                        raise ValueError("API access forbidden. Please check your OpenAI API key permissions.")
                    elif status_code in [500, 502, 503, 504]:
                        raise ValueError("OpenAI service is temporarily unavailable. Please try again later.")
                raise ValueError(f"OpenAI API error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to generate response: {str(e)}")

            raw_assistant_content = completion.choices[0].message.content
            logger.info(f"[Annotations] Raw OpenAI response: {raw_assistant_content[:500]}...")

            # Extract token usage from completion
            token_usage = None
            if hasattr(completion, 'usage') and completion.usage:
                token_usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
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
            else:
                logger.warning("Token usage information not available from OpenAI response")

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
                user_api_key=api_key
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
                        title = await self._generate_conversation_title(content, user.api_key)
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
            # Get decrypted API key
            api_key = user.get_decrypted_api_key()
            if not api_key:
                logger.error(f"User {user.id} has no OpenAI API key configured")
                raise ValueError("User has no OpenAI API key configured. Please configure your API key in settings.")

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
                user_api_key=api_key
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
                        title = await self._generate_conversation_title(content, api_key)
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
            # Get decrypted API key
            api_key = user.get_decrypted_api_key()
            if not api_key:
                logger.error(f"User {user.id} has no OpenAI API key configured")
                yield f"data: {json.dumps({'type': 'error', 'content': 'User has no OpenAI API key configured. Please configure your API key in settings.'})}\n\n"
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

            # Initialize OpenAI client
            client = AsyncOpenAI(api_key=api_key)

            # Get configuration
            from ..config import settings
            max_context_tokens = getattr(settings, 'MAX_CONTEXT_TOKENS', 100000)
            rerank_top_k = getattr(settings, 'RERANK_TOP_K', 20)

            # Find relevant chunks (retrieve more for token-based selection)
            logger.debug(f"Finding similar chunks for document {document_id}")
            candidate_chunks = await self.find_similar_chunks(db, content, document_id, limit=rerank_top_k, user_api_key=api_key)

            # Get conversation history for token counting
            logger.debug(f"Fetching conversation history for {conversation_id}")
            history = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(10).all()

            # Classify query type for adaptive prompting
            query_classification = await self._classify_query_type(content, api_key)
            logger.info(
                f"Query classification: type={query_classification['query_type']}, "
                f"complexity={query_classification['complexity']}, "
                f"requires_cot={query_classification['requires_cot']}"
            )

            # Count tokens for dynamic chunk selection
            # 1. System prompt template (without chunks)
            system_prompt_template = """You are an AI tutor helping a student understand a PDF document.
You have access to the following document chunks that are relevant to the student's question:

{context_text}

When referring to content, always cite the page number like [Page X].
Make sure to use the correct page number for each piece of information.

IMPORTANT FORMATTING INSTRUCTIONS:
1. Use markdown to highlight important concepts, terms, or phrases by making them **bold** or using *italics*.
2. For direct quotes from the document, use > blockquote formatting.
3. When referring to specific sections, use [Page X] to cite the page number.
4. Use bullet points or numbered lists for step-by-step explanations.
5. For critical information or warnings, use "⚠️" at the beginning of the paragraph.

PDF ANNOTATION FEATURE - IMPORTANT:
You MUST identify specific parts of the document that are relevant to your answer.
At the END of your response, ALWAYS include an ANNOTATIONS section with the following JSON format:

```annotations
[
  {{
    "pageNumber": <page number from context above>,
    "type": "highlight",
    "textToHighlight": "<3-10 word phrase copied exactly from the document>",
    "explanation": "<why this text answers the question>"
  }}
]
```

ANNOTATION RULES - FOLLOW STRICTLY:
1. ALWAYS include at least 1 annotation when you reference document content
2. The "pageNumber" MUST match a page number from the [Page X] citations above
3. The "textToHighlight" MUST be a short phrase (3-10 words) that appears EXACTLY in the document chunks above
4. Use type "highlight" for text (most common), "circle" for images/diagrams, "box" for tables
5. Copy the exact words from the document - do not paraphrase
6. Include 1-2 annotations per response

Make your responses helpful, clear, and educational. If the context doesn't contain the answer,
say you don't have enough information from the document and suggest looking at other pages."""

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
                context_text = "\n\n".join([
                    f"[Page {chunk['pageNumber']}]: {chunk['content']}"
                    for chunk in relevant_chunks
                ])
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
            try:
                async def _create_stream():
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_completion_tokens=1000,
                        stream=True
                    )

                stream = await async_retry_openai_call(
                    _create_stream,
                    max_attempts=5,
                    initial_wait=1.0,
                    max_wait=60.0
                )

                # Stream chunks
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content_chunk = delta.content
                            accumulated_content += content_chunk
                            # Send chunk to client
                            yield f"data: {json.dumps({'type': 'chunk', 'content': content_chunk})}\n\n"

            except APIError as e:
                logger.error(f"OpenAI API error after retries: {str(e)}", exc_info=True)
                error_msg = f"OpenAI API error: {str(e)}"
                status_code = getattr(e, 'status_code', None)
                if status_code:
                    if status_code == 429:
                        error_msg = "Rate limit exceeded. Please wait a moment and try again."
                    elif status_code == 401:
                        error_msg = "Invalid API key. Please check your OpenAI API key in settings."
                    elif status_code == 403:
                        error_msg = "API access forbidden. Please check your OpenAI API key permissions."
                    elif status_code in [500, 502, 503, 504]:
                        error_msg = "OpenAI service is temporarily unavailable. Please try again later."
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                return
            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {str(e)}", exc_info=True)
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
                user_api_key=api_key
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
                        title = await self._generate_conversation_title(content, user.api_key)
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

