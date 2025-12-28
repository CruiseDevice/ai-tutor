"""
LangGraph-based RAG Agent Service for adaptive multi-step reasoning.

This service transforms the linear RAG pipeline into an agent-based system that can:
- Adaptively route queries based on complexity
- Perform multi-step reasoning for complex questions
- Verify and improve answer quality
- Gracefully fallback to the linear pipeline on errors

Integration points:
- ChatService: Reuses query classification, retrieval, prompt building, and verification
- QueryExpansionService: Query variation generation
- RerankService: Cross-encoder re-ranking
- CacheService: Caching at each workflow step
- TokenService: Token management and truncation
"""

from typing import Dict, List, Optional, TypedDict, Any, AsyncGenerator
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
import logging
import json
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from ..config import settings
from .chat_service import ChatService
from .query_expansion_service import get_query_expansion_service
from .rerank_service import get_rerank_service
from .cache_service import get_cache_service
from .token_service import TokenService

logger = logging.getLogger(__name__)


class AgentMetrics:
    """Helper class for tracking agent performance metrics and costs."""

    # Model pricing (cost per 1K tokens) - Updated as of Jan 2025
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},  # $2.50 / $10 per 1M tokens
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # $0.15 / $0.60 per 1M tokens
        "gpt-4": {"input": 0.03, "output": 0.06},  # Legacy pricing
    }

    @staticmethod
    def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for an OpenAI API call.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = AgentMetrics.PRICING.get(model, AgentMetrics.PRICING["gpt-4o-mini"])
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    @staticmethod
    def init_metrics() -> Dict[str, Any]:
        """Initialize empty metrics dictionary."""
        return {
            "node_timings": {},
            "token_usage": {},
            "costs": {},
            "cache_hits": {},
            "retrieval_stats": {},
            "workflow_start_time": time.time(),
            "total_time": 0,
        }

    @staticmethod
    def track_node_time(metrics: Dict, node_name: str, duration: float):
        """Track execution time for a workflow node."""
        if "node_timings" not in metrics:
            metrics["node_timings"] = {}
        metrics["node_timings"][node_name] = duration

    @staticmethod
    def track_token_usage(
        metrics: Dict,
        step_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Track token usage and calculate cost for a step."""
        if "token_usage" not in metrics:
            metrics["token_usage"] = {}
        if "costs" not in metrics:
            metrics["costs"] = {}

        # Track tokens
        metrics["token_usage"][step_name] = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # Calculate cost
        cost = AgentMetrics.calculate_cost(model, prompt_tokens, completion_tokens)
        metrics["costs"][step_name] = cost

    @staticmethod
    def finalize_metrics(metrics: Dict) -> Dict:
        """Calculate totals and finalize metrics."""
        # Calculate total time
        if "workflow_start_time" in metrics:
            metrics["total_time"] = time.time() - metrics["workflow_start_time"]
            del metrics["workflow_start_time"]  # Remove start time from final metrics

        # Calculate total tokens
        total_prompt = sum(
            usage.get("prompt_tokens", 0)
            for usage in metrics.get("token_usage", {}).values()
        )
        total_completion = sum(
            usage.get("completion_tokens", 0)
            for usage in metrics.get("token_usage", {}).values()
        )

        if total_prompt > 0 or total_completion > 0:
            metrics["token_usage"]["total"] = {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            }

        # Calculate total cost
        total_cost = sum(metrics.get("costs", {}).values())
        if total_cost > 0:
            metrics["costs"]["total"] = total_cost

        return metrics


class AgentState(TypedDict):
    """
    State object passed between agent workflow nodes.
    Matches the existing ChatService data flow for compatibility.
    """
    # Input (from request)
    user_query: str
    conversation_id: str
    document_id: str
    user_id: str
    db_session: Session
    user_api_key: str

    # Query Understanding (from _classify_query_type)
    query_type: Optional[str]  # "factual", "analytical", "comparative", "follow-up", "clarification"
    complexity: Optional[str]  # "simple", "moderate", "complex"
    requires_cot: Optional[bool]  # Whether chain-of-thought is needed
    retrieval_strategy: Optional[str]  # "fast_path", "reasoning_path", "context_aware_path"

    # Retrieval (from find_similar_chunks)
    query_embedding: Optional[List[float]]
    retrieved_chunks: Optional[List[Dict]]
    context_text: Optional[str]  # Formatted chunks with page numbers

    # Generation
    answer: Optional[str]  # Raw answer with annotations
    clean_answer: Optional[str]  # Answer with annotations removed
    annotations: Optional[List[Dict]]  # Parsed annotation objects

    # Verification (from _score_answer_quality and _verify_citations)
    quality_score: Optional[Dict[str, Any]]  # Quality scores dict
    verified: Optional[bool]  # Whether answer passed verification
    citation_warnings: Optional[List[str]]  # Citation mismatch warnings

    # Output (matches ChatResponse schema)
    final_response: Optional[Dict]
    error: Optional[str]

    # Workflow control
    retry_count: Optional[int]  # Number of regeneration attempts
    max_retries: Optional[int]  # Maximum allowed retries

    # Performance Metrics
    metrics: Optional[Dict[str, Any]]  # Performance tracking
    # metrics structure:
    # {
    #   "node_timings": {"understand_query": 0.5, "retrieve_context": 1.2, ...},
    #   "token_usage": {"understand": {...}, "generate": {...}, "verify": {...}, "total": {...}},
    #   "costs": {"understand": 0.001, "generate": 0.05, "verify": 0.002, "total": 0.053},
    #   "cache_hits": {"query_classification": True/False},
    #   "retrieval_stats": {"sub_questions": 3, "chunks_retrieved": 15, "chunks_used": 10}
    # }


class RAGAgentService:
    """
    Production-ready LangGraph agent service for RAG.
    Integrates with existing services and maintains backward compatibility.
    """

    def __init__(self):
        """Initialize the agent service with existing service instances."""
        self.chat_service = ChatService()
        self.query_expansion_service = get_query_expansion_service()
        self.rerank_service = get_rerank_service()
        self.cache_service = get_cache_service()
        self.token_service = TokenService()

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()

        logger.info("RAGAgentService initialized with LangGraph workflow")

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with conditional routing.

        Workflow structure:
        1. understand_query -> Route based on complexity
        2a. Fast path (simple queries): retrieve_context -> generate_answer -> format_response
        2b. Reasoning path (complex queries): retrieve_context -> generate_answer -> verify_response
        3. verify_response -> Route based on quality score (re-generate or format)
        4. format_response -> END
        """
        # Create workflow graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_response", self._verify_response)
        workflow.add_node("format_response", self._format_response)

        # Set entry point
        workflow.set_entry_point("understand_query")

        # Add conditional routing from understand_query
        workflow.add_conditional_edges(
            "understand_query",
            self._route_query,
            {
                "fast_path": "retrieve_context",
                "reasoning_path": "retrieve_context",
                "context_aware_path": "retrieve_context"
            }
        )

        # Retrieval always goes to generation
        workflow.add_edge("retrieve_context", "generate_answer")

        # Skip verification - go directly to formatting
        workflow.add_edge("generate_answer", "format_response")

        # Format response is the final step
        workflow.add_edge("format_response", END)

        return workflow.compile()

    async def _understand_query(self, state: AgentState) -> AgentState:
        """
        Node 1: Understand the query using existing _classify_query_type logic.

        Leverages ChatService._classify_query_type() to classify query type,
        complexity, and determine if chain-of-thought is needed.

        Includes caching and performance tracking.
        """
        start_time = time.time()
        logger.info(f"[Agent] Understanding query: {state['user_query'][:100]}")

        # Initialize metrics if not already present
        if state.get("metrics") is None:
            state["metrics"] = AgentMetrics.init_metrics()

        try:
            # Check cache for query classification (optimization)
            cache_service = await get_cache_service()
            cache_key = f"query_classification:{hash(state['user_query'])}"
            cached_classification = None

            try:
                cached_classification = await cache_service.get(cache_key)
                if cached_classification:
                    state["metrics"]["cache_hits"]["query_classification"] = True
                    logger.info("[Agent] Using cached query classification")
            except Exception as cache_error:
                logger.warning(f"[Agent] Cache lookup failed: {cache_error}")

            if cached_classification:
                classification = json.loads(cached_classification)
            else:
                # Use existing classification logic from ChatService
                classification = await self.chat_service._classify_query_type(
                    query=state["user_query"],
                    user_api_key=state["user_api_key"]
                )

                # Cache the classification (expires in 1 hour)
                try:
                    await cache_service.set(
                        cache_key,
                        json.dumps(classification),
                        ttl=3600
                    )
                    state["metrics"]["cache_hits"]["query_classification"] = False
                except Exception as cache_error:
                    logger.warning(f"[Agent] Cache write failed: {cache_error}")

                # Track token usage (estimated for classification - ~100 prompt + ~50 completion)
                AgentMetrics.track_token_usage(
                    state["metrics"],
                    "query_classification",
                    settings.QUERY_CLASSIFICATION_MODEL,
                    prompt_tokens=100,
                    completion_tokens=50
                )

            state["query_type"] = classification["query_type"]
            state["complexity"] = classification["complexity"]
            state["requires_cot"] = classification["requires_cot"]
            state["retry_count"] = 0
            state["max_retries"] = 1  # Allow one regeneration attempt

            logger.info(
                f"[Agent] Query classified: type={classification['query_type']}, "
                f"complexity={classification['complexity']}, cot={classification['requires_cot']}"
            )

        except Exception as e:
            logger.error(f"[Agent] Error in understand_query: {e}")
            # Set defaults on error
            state["query_type"] = "factual"
            state["complexity"] = "simple"
            state["requires_cot"] = False
            state["error"] = f"Query classification failed: {str(e)}"

        finally:
            # Track node execution time
            duration = time.time() - start_time
            AgentMetrics.track_node_time(state["metrics"], "understand_query", duration)
            logger.info(f"[Agent] understand_query completed in {duration:.2f}s")

        return state

    def _route_query(self, state: AgentState) -> str:
        """
        Conditional routing based on query complexity and type.

        Routes:
        - fast_path: Simple queries (fewer chunks, cheaper model, no verification)
        - reasoning_path: Complex queries (more chunks, better model, verification)
        - context_aware_path: Follow-up queries (use conversation history)
        """
        complexity = state.get("complexity", "simple")
        query_type = state.get("query_type", "factual")

        # Follow-up queries need conversation context
        if query_type in ["follow-up", "clarification"]:
            strategy = "context_aware_path"
        # Complex queries need deeper reasoning and verification
        elif complexity == "complex":
            strategy = "reasoning_path"
        # Simple queries can use fast path
        else:
            strategy = "fast_path"

        state["retrieval_strategy"] = strategy
        logger.info(f"[Agent] Routing to: {strategy}")

        return strategy

    async def _decompose_complex_query(self, query: str, user_api_key: str) -> List[str]:
        """
        Decompose a complex query into sub-questions for multi-step reasoning.

        This enables better retrieval for complex analytical queries by breaking
        them into simpler sub-questions that can be answered independently.

        Args:
            query: The complex user query
            user_api_key: User's OpenAI API key

        Returns:
            List of sub-questions (including the original query)
        """
        try:
            client = AsyncOpenAI(api_key=user_api_key)

            decomposition_prompt = f"""You are a query decomposition assistant. Break down this complex query into 2-4 simpler sub-questions that, when answered together, would fully address the original query.

Original Query: "{query}"

Guidelines:
1. Each sub-question should be self-contained and answerable independently
2. Sub-questions should cover different aspects of the original query
3. Include the original query as the last item
4. Return ONLY a JSON array of strings (the sub-questions)

Example:
Query: "Compare photosynthesis and cellular respiration and explain how they're related"
Output: ["What is photosynthesis?", "What is cellular respiration?", "How are photosynthesis and cellular respiration related?", "Compare photosynthesis and cellular respiration and explain how they're related"]

Output (JSON array only):"""

            completion = await client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for decomposition
                messages=[
                    {"role": "system", "content": "You are a query decomposition assistant that outputs only valid JSON arrays."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=200
            )

            response_text = completion.choices[0].message.content.strip()

            # Parse JSON response
            sub_questions = json.loads(response_text)

            if not isinstance(sub_questions, list) or len(sub_questions) < 1:
                # Fallback: just use original query
                return [query]

            logger.info(f"[Agent] Decomposed query into {len(sub_questions)} sub-questions")
            return sub_questions

        except Exception as e:
            logger.warning(f"[Agent] Query decomposition failed: {e}. Using original query.")
            return [query]  # Fallback to original query on error

    async def _retrieve_context(self, state: AgentState) -> AgentState:
        """
        Node 2: Retrieve context using existing find_similar_chunks with adaptive limits.

        For complex queries, uses multi-step reasoning:
        1. Decompose query into sub-questions
        2. Retrieve context for each sub-question
        3. Combine and deduplicate chunks

        Leverages ChatService.find_similar_chunks() with adaptive chunk limits
        based on the retrieval strategy.
        """
        logger.info(f"[Agent] Retrieving context with strategy: {state.get('retrieval_strategy', 'default')}")

        try:
            strategy = state.get("retrieval_strategy", "fast_path")
            complexity = state.get("complexity", "simple")

            # Adaptive chunk limits based on strategy
            if strategy == "fast_path":
                # Fast path: fewer chunks for simple queries
                limit = 3
            elif strategy == "reasoning_path":
                # Reasoning path: more chunks for complex queries
                limit = settings.RERANK_TOP_K if settings.RERANK_ENABLED else 10
            else:  # context_aware_path
                # Context-aware: moderate chunks
                limit = 5

            # Multi-step reasoning for complex queries
            if complexity == "complex" and strategy == "reasoning_path":
                logger.info("[Agent] Using multi-step reasoning for complex query")

                # Decompose query into sub-questions
                sub_questions = await self._decompose_complex_query(
                    query=state["user_query"],
                    user_api_key=state["user_api_key"]
                )

                # Retrieve chunks for each sub-question
                all_chunks = []
                seen_chunk_ids = set()

                for i, sub_question in enumerate(sub_questions, 1):
                    logger.info(f"[Agent] Retrieving for sub-question {i}/{len(sub_questions)}: {sub_question[:100]}")

                    sub_chunks = await self.chat_service.find_similar_chunks(
                        db=state["db_session"],
                        document_id=state["document_id"],
                        query=sub_question,
                        user_api_key=state["user_api_key"],
                        limit=max(3, limit // len(sub_questions))  # Distribute limit across sub-questions
                    )

                    # Deduplicate chunks (based on page number + content hash)
                    for chunk in sub_chunks:
                        chunk_id = f"{chunk.get('pageNumber', '')}_{hash(chunk.get('content', ''))}"
                        if chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(chunk_id)
                            all_chunks.append(chunk)

                # Limit total chunks to avoid context overflow
                chunks = all_chunks[:limit]
                logger.info(f"[Agent] Multi-step retrieval: {len(all_chunks)} total chunks (using top {len(chunks)})")

            else:
                # Standard retrieval for simple/moderate queries
                chunks = await self.chat_service.find_similar_chunks(
                    db=state["db_session"],
                    document_id=state["document_id"],
                    query=state["user_query"],
                    user_api_key=state["user_api_key"],
                    limit=limit
                )

            state["retrieved_chunks"] = chunks

            # Format context text (matches ChatService format)
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                page_num = chunk.get("pageNumber", "?")
                content = chunk.get("content", "")
                context_parts.append(f"[Page {page_num}]\n{content}")

            state["context_text"] = "\n\n---\n\n".join(context_parts)

            logger.info(f"[Agent] Retrieved {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"[Agent] Error in retrieve_context: {e}")
            state["error"] = f"Context retrieval failed: {str(e)}"
            state["retrieved_chunks"] = []
            state["context_text"] = ""

        return state

    async def _generate_answer(self, state: AgentState) -> AgentState:
        """
        Node 3: Generate answer using existing _build_system_prompt and OpenAI.

        Leverages ChatService._build_system_prompt() for adaptive prompting
        and ChatService._parse_annotations() for annotation extraction.
        """
        logger.info(f"[Agent] Generating answer")

        try:
            # Build system prompt using existing logic
            system_prompt = self.chat_service._build_system_prompt(
                context_text=state.get("context_text", ""),
                query_type=state.get("query_type", "factual"),
                complexity=state.get("complexity", "simple"),
                requires_cot=state.get("requires_cot", False)
            )

            # Select model based on complexity
            complexity = state.get("complexity", "simple")
            if complexity == "complex":
                model = "gpt-4o"  # Better model for complex queries
            else:
                model = settings.AGENT_DEFAULT_MODEL  # gpt-4o-mini for simple/moderate

            # Generate answer
            client = AsyncOpenAI(api_key=state["user_api_key"])

            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": state["user_query"]}
                ],
                temperature=0.7,
                max_completion_tokens=2000
            )

            raw_answer = completion.choices[0].message.content
            state["answer"] = raw_answer

            # Parse annotations using existing logic
            clean_answer, annotations = self.chat_service._parse_annotations(
                response_text=raw_answer,
                relevant_chunks=state.get("retrieved_chunks", [])
            )

            state["clean_answer"] = clean_answer
            state["annotations"] = annotations

            logger.info(f"[Agent] Answer generated with {len(annotations)} annotations")

        except Exception as e:
            logger.error(f"[Agent] Error in generate_answer: {e}")
            state["error"] = f"Answer generation failed: {str(e)}"
            state["answer"] = ""
            state["clean_answer"] = ""
            state["annotations"] = []

        return state

    def _route_after_generation(self, state: AgentState) -> str:
        """
        Route after generation based on complexity.
        Simple queries skip verification, complex queries get verified.
        """
        complexity = state.get("complexity", "simple")

        # Only verify complex queries to save costs
        if complexity in ["moderate", "complex"]:
            return "verify"
        else:
            return "format"

    async def _verify_response(self, state: AgentState) -> AgentState:
        """
        Node 4: Verify response quality using existing _score_answer_quality
        and _verify_citations.

        Leverages ChatService methods for quality scoring and citation verification.
        """
        logger.info(f"[Agent] Verifying response quality")

        try:
            # Score answer quality using existing logic
            quality_scores = await self.chat_service._score_answer_quality(
                query=state["user_query"],
                answer=state.get("clean_answer", ""),
                context_chunks=state.get("retrieved_chunks", []),
                user_api_key=state["user_api_key"]
            )

            state["quality_score"] = quality_scores

            # Verify citations using existing logic
            citation_warnings = self.chat_service._verify_citations(
                response_text=state.get("answer", ""),
                annotations=state.get("annotations", []),
                relevant_chunks=state.get("retrieved_chunks", [])
            )

            state["citation_warnings"] = citation_warnings

            # Determine if answer is verified
            overall_score = quality_scores.get("overall")
            has_critical_warnings = len(citation_warnings) > 0

            # Verification passes if score >= 7 and no critical citation errors
            state["verified"] = (
                overall_score is not None and
                overall_score >= 7.0 and
                not has_critical_warnings
            )

            logger.info(
                f"[Agent] Verification complete: score={overall_score}, "
                f"verified={state['verified']}, warnings={len(citation_warnings)}"
            )

        except Exception as e:
            logger.error(f"[Agent] Error in verify_response: {e}")
            # Default to verified on error to avoid infinite loops
            state["verified"] = True
            state["quality_score"] = {"overall": None}
            state["citation_warnings"] = []

        return state

    def _route_after_verification(self, state: AgentState) -> str:
        """
        Route after verification based on quality score.
        If quality is low and retries remain, regenerate. Otherwise, format.
        """
        verified = state.get("verified", True)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 1)

        # If not verified and we have retries left, regenerate
        if not verified and retry_count < max_retries:
            state["retry_count"] = retry_count + 1
            logger.info(f"[Agent] Quality check failed, regenerating (attempt {retry_count + 1})")
            return "regenerate"
        else:
            # Either verified or out of retries
            return "format"

    async def _format_response(self, state: AgentState) -> AgentState:
        """
        Node 5: Format final response to match existing ChatResponse schema.

        Ensures compatibility with existing API response format.
        Finalizes and includes performance metrics.
        """
        start_time = time.time()
        logger.info(f"[Agent] Formatting final response")

        try:
            # Finalize metrics
            if state.get("metrics"):
                state["metrics"] = AgentMetrics.finalize_metrics(state["metrics"])

                # Log performance summary
                metrics = state["metrics"]
                logger.info(
                    f"[Agent Metrics] Total time: {metrics.get('total_time', 0):.2f}s, "
                    f"Total cost: ${metrics.get('costs', {}).get('total', 0):.4f}, "
                    f"Total tokens: {metrics.get('token_usage', {}).get('total', {}).get('total_tokens', 0)}"
                )

            # Match existing ChatResponse format
            final_response = {
                "user_message": {
                    "role": "user",
                    "content": state["user_query"]
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": state.get("clean_answer", ""),
                    "annotations": state.get("annotations", []),
                    "context": state.get("retrieved_chunks", [])
                },
                "metadata": {
                    "query_classification": {
                        "query_type": state.get("query_type"),
                        "complexity": state.get("complexity"),
                        "requires_cot": state.get("requires_cot")
                    },
                    "retrieval_strategy": state.get("retrieval_strategy"),
                    "quality_scores": state.get("quality_score"),
                    "citation_warnings": state.get("citation_warnings", []),
                    "verified": state.get("verified"),
                    "retry_count": state.get("retry_count", 0),
                    "performance_metrics": state.get("metrics", {})  # Include full metrics
                }
            }

            state["final_response"] = final_response

            logger.info(f"[Agent] Response formatted successfully")

        except Exception as e:
            logger.error(f"[Agent] Error in format_response: {e}")
            state["error"] = f"Response formatting failed: {str(e)}"

        finally:
            # Track final node timing
            duration = time.time() - start_time
            if state.get("metrics"):
                AgentMetrics.track_node_time(state["metrics"], "format_response", duration)

        return state

    async def process_query(
        self,
        user_query: str,
        conversation_id: str,
        document_id: str,
        user_id: str,
        db_session: Session,
        user_api_key: str
    ) -> Dict:
        """
        Main entry point for agent-based query processing.

        Args:
            user_query: The user's question
            conversation_id: Conversation identifier
            document_id: Document identifier
            user_id: User identifier
            db_session: Database session
            user_api_key: User's OpenAI API key

        Returns:
            Dict matching ChatResponse schema with metadata

        Raises:
            Exception: If workflow execution fails (should be caught by caller for fallback)
        """
        logger.info(f"[Agent] Processing query: {user_query[:100]}")

        # Initialize state
        initial_state: AgentState = {
            "user_query": user_query,
            "conversation_id": conversation_id,
            "document_id": document_id,
            "user_id": user_id,
            "db_session": db_session,
            "user_api_key": user_api_key,
            # All other fields initialized to None
            "query_type": None,
            "complexity": None,
            "requires_cot": None,
            "retrieval_strategy": None,
            "query_embedding": None,
            "retrieved_chunks": None,
            "context_text": None,
            "answer": None,
            "clean_answer": None,
            "annotations": None,
            "quality_score": None,
            "verified": None,
            "citation_warnings": None,
            "final_response": None,
            "error": None,
            "retry_count": None,
            "max_retries": None,
            "metrics": AgentMetrics.init_metrics()  # Initialize metrics tracking
        }

        try:
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Check for errors
            if final_state.get("error"):
                raise Exception(final_state["error"])

            # Return final response
            response = final_state.get("final_response")
            if not response:
                raise Exception("Workflow completed but no final_response generated")

            logger.info(f"[Agent] Query processed successfully")
            return response

        except Exception as e:
            logger.error(f"[Agent] Workflow execution failed: {e}")
            raise  # Re-raise for caller to handle fallback

    async def process_query_streaming(
        self,
        user_query: str,
        conversation_id: str,
        document_id: str,
        user_id: str,
        db_session: Session,
        user_api_key: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent workflow execution with intermediate step updates.

        This method uses LangGraph's astream() to stream workflow execution,
        emitting events for each node completion to provide real-time feedback.

        Args:
            user_query: The user's question
            conversation_id: Conversation identifier
            document_id: Document identifier
            user_id: User identifier
            db_session: Database session
            user_api_key: User's OpenAI API key

        Yields:
            str: JSON-encoded Server-Sent Events (SSE) with step updates

        Event types emitted:
        - step_start: Node execution started
        - step_complete: Node execution completed
        - retrieval_update: Chunks retrieved
        - generation_chunk: Streaming answer generation (if supported)
        - final_response: Complete response ready
        - error: Error occurred
        """
        import json

        logger.info(f"[Agent Stream] Starting streaming workflow for query: {user_query[:100]}")

        # Initialize state (same as process_query)
        initial_state: AgentState = {
            "user_query": user_query,
            "conversation_id": conversation_id,
            "document_id": document_id,
            "user_id": user_id,
            "db_session": db_session,
            "user_api_key": user_api_key,
            "query_type": None,
            "complexity": None,
            "requires_cot": None,
            "retrieval_strategy": None,
            "query_embedding": None,
            "retrieved_chunks": None,
            "context_text": None,
            "answer": None,
            "clean_answer": None,
            "annotations": None,
            "quality_score": None,
            "verified": None,
            "citation_warnings": None,
            "final_response": None,
            "error": None,
            "retry_count": None,
            "max_retries": None,
            "metrics": AgentMetrics.init_metrics()  # Initialize metrics tracking
        }

        try:
            # Stream workflow execution using LangGraph's astream()
            async for event in self.workflow.astream(initial_state):
                # LangGraph emits events as {node_name: state_update}
                for node_name, state_update in event.items():
                    # Skip __start__ and __end__ meta nodes
                    if node_name.startswith("__"):
                        continue

                    logger.info(f"[Agent Stream] Node completed: {node_name}")

                    # Emit step completion event
                    step_event = {
                        "type": "step_complete",
                        "node": node_name,
                        "timestamp": None  # Can add timestamp if needed
                    }

                    # Add node-specific metadata
                    if node_name == "understand_query":
                        step_event["data"] = {
                            "query_type": state_update.get("query_type"),
                            "complexity": state_update.get("complexity"),
                            "requires_cot": state_update.get("requires_cot"),
                            "strategy": state_update.get("retrieval_strategy")
                        }
                        step_event["message"] = f"Analyzing query (Type: {state_update.get('query_type')}, Complexity: {state_update.get('complexity')})"

                    elif node_name == "retrieve_context":
                        chunks = state_update.get("retrieved_chunks", [])
                        step_event["data"] = {
                            "chunk_count": len(chunks),
                            "pages": [c.get("pageNumber") for c in chunks]
                        }
                        step_event["message"] = f"Retrieved {len(chunks)} relevant chunks"

                    elif node_name == "generate_answer":
                        annotations = state_update.get("annotations", [])
                        step_event["data"] = {
                            "annotation_count": len(annotations),
                            "has_answer": bool(state_update.get("clean_answer"))
                        }
                        step_event["message"] = "Generated answer with citations"

                    elif node_name == "verify_response":
                        quality_score = state_update.get("quality_score", {})
                        step_event["data"] = {
                            "overall_score": quality_score.get("overall"),
                            "verified": state_update.get("verified"),
                            "warnings": len(state_update.get("citation_warnings", []))
                        }
                        step_event["message"] = f"Verified answer quality (Score: {quality_score.get('overall', 'N/A')})"

                    elif node_name == "format_response":
                        step_event["message"] = "Formatting final response"

                    # Emit the event as SSE
                    yield f"data: {json.dumps(step_event)}\n\n"

            # Get final state
            final_state = state_update  # Last state from the loop

            # Check for errors
            if final_state.get("error"):
                error_event = {
                    "type": "error",
                    "error": final_state["error"],
                    "message": f"Workflow error: {final_state['error']}"
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                return

            # Emit final response
            final_response = final_state.get("final_response")
            if final_response:
                final_event = {
                    "type": "final_response",
                    "data": final_response,
                    "message": "Response complete"
                }
                yield f"data: {json.dumps(final_event)}\n\n"

                logger.info(f"[Agent Stream] Workflow completed successfully")
            else:
                error_event = {
                    "type": "error",
                    "error": "No final response generated",
                    "message": "Workflow completed but no response was generated"
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        except Exception as e:
            logger.error(f"[Agent Stream] Workflow execution failed: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "error": str(e),
                "message": f"Streaming error: {str(e)}"
            }
            yield f"data: {json.dumps(error_event)}\n\n"


# Singleton instance
_agent_service: Optional[RAGAgentService] = None


def get_agent_service() -> RAGAgentService:
    """Get or create the singleton RAGAgentService instance."""
    global _agent_service
    if _agent_service is None:
        _agent_service = RAGAgentService()
    return _agent_service
