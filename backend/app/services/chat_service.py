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

    async def generate_chat_response(
        self,
        db: Session,
        user: User,
        content: str,
        conversation_id: str,
        document_id: str,
        model: str = "gpt-4"
    ) -> Dict:
        """Generate a chat response using OpenAI with RAG."""
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

            # Find relevant chunks
            logger.debug(f"Finding similar chunks for document {document_id}")
            relevant_chunks = await self.find_similar_chunks(db, content, document_id, limit=5, user_api_key=api_key)

            # Format context from chunks
            if relevant_chunks:
                context_text = "\n\n".join([
                    f"[Page {chunk['pageNumber']}]: {chunk['content']}"
                    for chunk in relevant_chunks
                ])
            else:
                context_text = "No relevant document sections found."

            # Create system message with context and annotation instructions
            system_message = {
                "role": "system",
                "content": f"""You are an AI tutor helping a student understand a PDF document.
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
            }

            # Get conversation history
            logger.debug(f"Fetching conversation history for {conversation_id}")
            history = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(10).all()

            # Format history for OpenAI
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

            # Parse annotations from the response
            assistant_content, annotations = self._parse_annotations(
                raw_assistant_content,
                relevant_chunks
            )
            logger.info(f"[Annotations] Parsed {len(annotations)} annotations from response")
            if annotations:
                logger.info(f"[Annotations] Annotation details: {annotations}")

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

            # Save assistant message with context (store annotations in context)
            message_context = {
                "chunks": relevant_chunks,
                "annotations": annotations
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

            # Find relevant chunks
            logger.debug(f"Finding similar chunks for document {document_id}")
            relevant_chunks = await self.find_similar_chunks(db, content, document_id, limit=5, user_api_key=api_key)

            # Format context from chunks
            if relevant_chunks:
                context_text = "\n\n".join([
                    f"[Page {chunk['pageNumber']}]: {chunk['content']}"
                    for chunk in relevant_chunks
                ])
            else:
                context_text = "No relevant document sections found."

            # Create system message with context and annotation instructions
            system_message = {
                "role": "system",
                "content": f"""You are an AI tutor helping a student understand a PDF document.
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
            }

            # Get conversation history
            logger.debug(f"Fetching conversation history for {conversation_id}")
            history = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(10).all()

            # Format history for OpenAI
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

            # Save assistant message with context
            message_context = {
                "chunks": relevant_chunks,
                "annotations": annotations
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

