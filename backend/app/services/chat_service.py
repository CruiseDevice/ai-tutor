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
from .conversation_titles import get_conversation_title_service
from .retrieval_service import get_retriever
from .chunk_selector import get_chunk_selector, format_chunk_for_context
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
        self.title_service = get_conversation_title_service()
        self.retriever = get_retriever()
        self.chunk_selector = get_chunk_selector()

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
        """Generate a concise conversation title. Delegates to ConversationTitleService.

        Kept for backward compatibility; new callers should use
        `get_conversation_title_service().generate(...)` directly.
        """
        return await self.title_service.generate(
            user_message=user_message,
            user_api_key=user_api_key,
            provider=provider,
        )

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
        """Keyword (PostgreSQL FTS) search. Delegates to HybridRetriever."""
        return self.retriever._find_keyword_matches(
            db=db, query=query, document_id=document_id, limit=limit
        )
    def _format_chunk_for_context(self, chunk: Dict) -> str:
        """Format a chunk for the LLM context. Delegates to chunk_selector.

        Kept for backward compatibility; new callers should use
        `format_chunk_for_context(...)` from app.services.chunk_selector.
        """
        return format_chunk_for_context(chunk)
    def _calculate_adaptive_weights(self, query: str) -> tuple[float, float]:
        """Adaptive semantic/keyword weights. Delegates to HybridRetriever."""
        return self.retriever._calculate_adaptive_weights(query)
    def _combine_results_with_rrf(
        self,
        result_sets: List[List[Dict]],
        rrf_k: int = 60
    ) -> List[Dict]:
        """RRF fusion of ranked result sets. Delegates to HybridRetriever."""
        return self.retriever._combine_results_with_rrf(result_sets, rrf_k=rrf_k)
    async def _retrieve_critical_sentences(
        self,
        db: Session,
        query: str,
        query_embedding: List[float],
        document_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Retrieve sentence-level chunks. Delegates to HybridRetriever."""
        return await self.retriever._retrieve_critical_sentences(
            db=db, query=query, query_embedding=query_embedding,
            document_id=document_id, limit=limit,
        )
    def _merge_chunks_and_sentences(
        self,
        chunks: List[Dict],
        sentences: List[Dict],
        boost_factor: float = 1.2
    ) -> List[Dict]:
        """Merge chunk + sentence results. Delegates to HybridRetriever."""
        return self.retriever._merge_chunks_and_sentences(
            chunks=chunks, sentences=sentences, boost_factor=boost_factor
        )
    async def find_similar_chunks(
        self,
        db: Session,
        query: str,
        document_id: str,
        limit: int = 5,
        user_api_key: Optional[str] = None,
        provider: Provider = Provider.OPENAI
    ) -> List[Dict]:
        """Hybrid retrieval. Delegates to HybridRetriever.

        Kept for backward compatibility; new callers should use
        `get_retriever().find_similar_chunks(...)` directly.
        """
        return await self.retriever.find_similar_chunks(
            db=db, query=query, document_id=document_id, limit=limit,
            user_api_key=user_api_key, provider=provider,
        )
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
        """Token-budget chunk selection. Delegates to ChunkSelector.

        Kept for backward compatibility; new callers should use
        `get_chunk_selector().select_by_token_limit(...)` directly.
        """
        return self.chunk_selector.select_by_token_limit(
            chunks=chunks, max_tokens=max_tokens, model=model,
            system_prompt_tokens=system_prompt_tokens,
            user_message_tokens=user_message_tokens,
            history_tokens=history_tokens,
            response_reserve_tokens=response_reserve_tokens,
        )
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
        """Check if a document uses hierarchical chunking. Delegates to HybridRetriever."""
        return self.retriever._check_hierarchical_chunking(db, document_id)
    async def _retrieve_with_parent_child(
        self,
        db: Session,
        query_embedding: List[float],
        document_id: str,
        limit: int
    ) -> List[Dict]:
        """Hierarchical parent-child retrieval. Delegates to HybridRetriever."""
        return await self.retriever._retrieve_with_parent_child(
            db=db, query_embedding=query_embedding,
            document_id=document_id, limit=limit,
        )
