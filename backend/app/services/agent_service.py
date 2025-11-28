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

from typing import Dict, List, Optional, TypedDict, Any
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
import logging
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from ..config import settings
from .chat_service import ChatService
from .query_expansion_service import get_query_expansion_service
from .rerank_service import get_rerank_service
from .cache_service import get_cache_service
from .token_service import TokenService

logger = logging.getLogger(__name__)


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

        # Add conditional routing from generate_answer based on complexity
        workflow.add_conditional_edges(
            "generate_answer",
            self._route_after_generation,
            {
                "verify": "verify_response",
                "format": "format_response"
            }
        )

        # Add conditional routing from verify_response based on quality
        workflow.add_conditional_edges(
            "verify_response",
            self._route_after_verification,
            {
                "regenerate": "generate_answer",
                "format": "format_response"
            }
        )

        # Format response is the final step
        workflow.add_edge("format_response", END)

        return workflow.compile()

    async def _understand_query(self, state: AgentState) -> AgentState:
        """
        Node 1: Understand the query using existing _classify_query_type logic.

        Leverages ChatService._classify_query_type() to classify query type,
        complexity, and determine if chain-of-thought is needed.
        """
        logger.info(f"[Agent] Understanding query: {state['user_query'][:100]}")

        try:
            # Use existing classification logic from ChatService
            classification = await self.chat_service._classify_query_type(
                query=state["user_query"],
                user_api_key=state["user_api_key"]
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

    async def _retrieve_context(self, state: AgentState) -> AgentState:
        """
        Node 2: Retrieve context using existing find_similar_chunks with adaptive limits.

        Leverages ChatService.find_similar_chunks() with adaptive chunk limits
        based on the retrieval strategy.
        """
        logger.info(f"[Agent] Retrieving context with strategy: {state.get('retrieval_strategy', 'default')}")

        try:
            strategy = state.get("retrieval_strategy", "fast_path")

            # Adaptive chunk limits based on strategy
            if strategy == "fast_path":
                # Fast path: fewer chunks for simple queries
                limit = 3
                use_rerank = False
            elif strategy == "reasoning_path":
                # Reasoning path: more chunks for complex queries
                limit = settings.RERANK_TOP_K if settings.RERANK_ENABLED else 10
                use_rerank = settings.RERANK_ENABLED
            else:  # context_aware_path
                # Context-aware: moderate chunks with reranking
                limit = 5
                use_rerank = True

            # Use existing find_similar_chunks method
            chunks = await self.chat_service.find_similar_chunks(
                db=state["db_session"],
                document_id=state["document_id"],
                query=state["user_query"],
                user_api_key=state["user_api_key"],
                limit=limit,
                use_rerank=use_rerank
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
        """
        logger.info(f"[Agent] Formatting final response")

        try:
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
                    "retry_count": state.get("retry_count", 0)
                }
            }

            state["final_response"] = final_response

            logger.info(f"[Agent] Response formatted successfully")

        except Exception as e:
            logger.error(f"[Agent] Error in format_response: {e}")
            state["error"] = f"Response formatting failed: {str(e)}"

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
            "max_retries": None
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


# Singleton instance
_agent_service: Optional[RAGAgentService] = None


def get_agent_service() -> RAGAgentService:
    """Get or create the singleton RAGAgentService instance."""
    global _agent_service
    if _agent_service is None:
        _agent_service = RAGAgentService()
    return _agent_service
