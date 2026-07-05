"""Hybrid retrieval: semantic + keyword search with RRF fusion.

Extracted from ChatService. This module owns the document-chunk retrieval
pipeline:

- `find_similar_chunks`: the public hub. Routes between hierarchical
  parent-child retrieval and flat hybrid search; orchestrates query
  decomposition / query expansion (multi-query RRF), adaptive semantic vs
  keyword weighting, reranking, and sentence-level retrieval for detail
  queries.
- Keyword (PostgreSQL FTS), RRF fusion, sentence retrieval, and
  hierarchical parent-child helpers are private methods on the same class.

`HybridRetriever` holds the embedding service (it owns query embedding
generation). The cache service is fetched at call time. A process-wide
singleton is exposed via `get_retriever()`, mirroring the other extracted
services. The ChatService / agent_service consume the public
`find_similar_chunks` method.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, DatabaseError
from sqlalchemy.orm import Session

from ..config import settings
from ..models.document import DocumentChunk
from .cache_service import get_cache_service
from .embedding_service import get_embedding_service
from .query_decomposition_service import get_query_decomposition_service
from .query_expansion_service import get_query_expansion_service
from .llm import Provider
from .rerank_service import get_rerank_service

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Document-chunk retrieval over pgvector + PostgreSQL full-text search."""

    def __init__(self):
        self.embedding_service = get_embedding_service()

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

                        # Perform semantic search (inline SQL — mirrors search_single_variation below)
                        embedding_str = '[' + ','.join(str(x) for x in subquery_embedding) + ']'

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

                        logger.debug(f"Found {len(chunks)} chunks for sub-query: {subquery[:40]}...")
                        return chunks

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


# Process-wide singleton. The retriever holds the embedding service; reuse
# across requests to avoid re-initializing the embedding model.
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Return the process-wide HybridRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
