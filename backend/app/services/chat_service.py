from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, DatabaseError
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, AsyncGenerator
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


@dataclass
class PreparedContext:
    """Inputs prepared for an LLM generation call (shared by the two linear entry points).

    Captures everything the entry points compute before invoking the LLM, so
    the non-streaming and streaming paths share one preparation method instead
    of each inlining the ~80-line retrieve -> classify -> token-budget ->
    prompt-build sequence.
    """
    system_prompt: str
    messages: List[Dict]
    relevant_chunks: List[Dict]
    chunk_stats: Dict[str, int]
    query_classification: Dict
    max_tokens: int
    max_context_tokens: int
    cache_service: "object | None"
    query_embedding: List[float]
    is_first_message: bool


@dataclass
class PipelineEvent:
    """Base type for events emitted by _run_linear_pipeline.

    The two linear chat entry points share one pipeline generator; each
    adapts the events to its output form (dict vs SSE). Subtypes below.
    """


@dataclass
class ChunkEvent(PipelineEvent):
    """An incremental piece of assistant content.

    For streaming this is one token-fragment; for non-streaming it is the
    full completion in a single event. Adapters decide how to surface it.
    """
    content: str


@dataclass
class ErrorEvent(PipelineEvent):
    """Terminal failure. Non-stream raises ValueError(message); stream yields SSE."""
    message: str


@dataclass
class DoneEvent(PipelineEvent):
    """Terminal success carrying the persisted user/assistant Message objects.

    `created_at` is the raw datetime here; the streaming adapter ISO-serializes
    when building the SSE frame, the non-streaming adapter returns it as-is.
    """
    user_message: Any
    assistant_message: Any


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

    @staticmethod
    def _llm_error_message(e: APIError) -> str:
        """Map an LLM APIError's status code to a user-facing message.

        Shared by the non-streaming and streaming entry points (which
        previously each kept their own copy of this status-code map).
        Returns the generic message for unknown / missing status codes.
        """
        status_code = getattr(e, 'status_code', None)
        if status_code == 429:
            return "Rate limit exceeded. Please wait a moment and try again."
        if status_code == 401:
            return "Invalid API key. Please check your API key in settings."
        if status_code == 403:
            return "API access forbidden. Please check your API key permissions."
        if status_code in [500, 502, 503, 504]:
            return "The LLM service is temporarily unavailable. Please try again later."
        return f"LLM API error: {str(e)}"

    async def _set_title_if_first_message(
        self,
        db: Session,
        conversation_id: str,
        is_first_message: bool,
        content: str,
        api_key: str,
        provider: Provider | None,
    ) -> None:
        """Generate and persist a conversation title for the first message.

        No-op when this isn't the first message or the conversation already
        has a title. Failures are logged and swallowed so they never break
        the surrounding chat request. Replaces four near-identical blocks
        across the entry points.
        """
        if not is_first_message:
            return
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

    async def _prepare_generation_context(
        self,
        db: Session,
        content: str,
        document_id: str,
        model: str,
        conversation_id: str,
        api_key: str,
        provider: Provider,
        stream_log_prefix: str = "",
    ) -> PreparedContext:
        """Run the shared pre-LLM pipeline for the linear entry points.

        Stages: resolve client + config -> retrieve candidate chunks ->
        fetch history -> classify query -> token-budget chunk selection ->
        format context text -> build adaptive system prompt -> assemble
        messages. Both generate_chat_response and generate_chat_response_stream
        call this; only their LLM call, persistence, and output differ.

        `stream_log_prefix` lets the streaming path tag its log lines (it
        previously used "[Stream]") without diverging the logic.
        """
        from ..config import settings

        client = get_llm_client(provider, api_key)
        cache_service = await get_cache_service()

        max_context_tokens = getattr(settings, 'MAX_CONTEXT_TOKENS', 100000)
        rerank_top_k = getattr(settings, 'RERANK_TOP_K', 20)

        # Find relevant chunks (retrieve more for token-based selection)
        logger.debug(f"Finding similar chunks for document {document_id}")
        candidate_chunks = await self.find_similar_chunks(
            db, content, document_id, limit=rerank_top_k,
            user_api_key=api_key, provider=provider,
        )
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
        system_prompt_tokens = TokenService.count_tokens(TOKEN_BUDGET_TEMPLATE, model)
        user_message_tokens = TokenService.count_tokens(content, model)
        history_messages = [{"role": msg.role, "content": msg.content} for msg in history]
        history_tokens = TokenService.estimate_context_tokens(history_messages, model)
        response_reserve_tokens = 1000

        relevant_chunks, chunk_stats = self._select_chunks_by_token_limit(
            chunks=candidate_chunks,
            max_tokens=max_context_tokens,
            model=model,
            system_prompt_tokens=system_prompt_tokens,
            user_message_tokens=user_message_tokens,
            history_tokens=history_tokens,
            response_reserve_tokens=response_reserve_tokens,
        )

        logger.info(
            f"{stream_log_prefix}Dynamic chunk selection: "
            f"{chunk_stats['selected_chunks']}/{len(candidate_chunks)} chunks, "
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
            requires_cot=query_classification['requires_cot'],
        )

        # Assemble messages: system + history + current user message
        messages = [{"role": "system", "content": system_prompt_content}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": content})

        # Query embedding for response caching + first-message flag
        query_embedding = await cache_service.get_embedding(content)
        if query_embedding is None:
            query_embedding = await self.embedding_service.generate_embedding_async(content)
            await cache_service.set_embedding(content, query_embedding)

        existing_message_count = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).count()
        is_first_message = existing_message_count == 0

        max_tokens = self._get_adaptive_token_limit(query_classification['complexity'])

        return PreparedContext(
            system_prompt=system_prompt_content,
            messages=messages,
            relevant_chunks=relevant_chunks,
            chunk_stats=chunk_stats,
            query_classification=query_classification,
            max_tokens=max_tokens,
            max_context_tokens=max_context_tokens,
            cache_service=cache_service,
            query_embedding=query_embedding,
            is_first_message=is_first_message,
        )

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

    # ------------------------------------------------------------------
    # Unified linear pipeline (shared by generate_chat_response and
    # generate_chat_response_stream). The two entry points adapt the
    # emitted PipelineEvents to their output form (dict vs SSE).
    # ------------------------------------------------------------------

    async def _complete_then_yield(
        self,
        client: LLMClient,
        system_prompt: str,
        chat_messages: List[Dict],
        model: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Non-streaming LLM-call strategy: call complete() with retry, yield once."""
        async def _create_completion():
            return await client.complete(
                system_prompt=system_prompt,
                messages=chat_messages,
                model=model,
                temperature=0.7,
                max_tokens=max_tokens,
            )

        content = await async_retry_openai_call(
            _create_completion,
            max_attempts=5,  # More retries for main chat completion
            initial_wait=1.0,
            max_wait=60.0,
        )
        yield content or ""

    async def _stream_passthrough(
        self,
        client: LLMClient,
        system_prompt: str,
        chat_messages: List[Dict],
        model: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Streaming LLM-call strategy: pass client.stream() fragments through.

        NOTE: unlike _complete_then_yield, the streaming call is NOT wrapped in
        async_retry_openai_call (preserving the original behavior — a dropped
        stream fails fast rather than replaying tokens).
        """
        async for content_chunk in client.stream(
            system_prompt=system_prompt,
            messages=chat_messages,
            model=model,
            temperature=0.7,
            max_tokens=max_tokens,
        ):
            if content_chunk:
                yield content_chunk

    async def _run_linear_pipeline(
        self,
        *,
        db: Session,
        user: User,
        content: str,
        conversation_id: str,
        document_id: str,
        model: str,
        generate: Callable[
            [LLMClient, str, List[Dict], str, int], AsyncIterator[str]
        ],
        chunk_cached_response: bool = False,
        persist_user_before_llm: bool = False,
        log_prefix: str = "",
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Run the shared linear chat pipeline, emitting typed events.

        Stages: resolve provider/key -> cache lookup -> prepare context ->
        persist user message (optional, ordering-dependent) -> LLM call via
        `generate` -> parse/verify/score -> persist assistant message ->
        title -> cache response -> done. The two linear entry points pass
        different `generate` strategies and persistence/output flags; the
        body is otherwise identical.

        Args:
            generate: callable returning an async iterator of content
                fragments. Use _complete_then_yield (non-stream) or
                _stream_passthrough (stream).
            chunk_cached_response: when True (stream), a cache hit is emitted
                char-by-char as ChunkEvents, preserving the original UX; when
                False (non-stream) it is emitted as one ChunkEvent.
            persist_user_before_llm: when True (stream), the user message is
                persisted before the LLM call so it survives a generation
                failure; when False (non-stream) both messages are persisted
                together after a successful generation.
            log_prefix: optional "[Stream] " tag for path-specific log lines.
        """
        try:
            # 1. Resolve provider + decrypt API key
            provider = self._resolve_provider_for_model(model)
            api_key = user.get_decrypted_key(provider.value)
            if not api_key:
                logger.error(f"User {user.id} has no {provider.value} API key configured")
                yield ErrorEvent(
                    message=(
                        f"User has no {provider.value} API key configured. "
                        "Please configure your API key in settings."
                    )
                )
                return

            logger.debug(
                f"{log_prefix}Generating chat response for user {user.id}, conversation {conversation_id}"
            )

            # 2. Cache lookup (embedding + find_similar_response)
            cache_service = await get_cache_service()
            query_embedding = await cache_service.get_embedding(content)
            if query_embedding is None:
                query_embedding = await self.embedding_service.generate_embedding_async(content)
                await cache_service.set_embedding(content, query_embedding)

            history_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()

            if history_count == 0:  # Only cache responses for first message (no context)
                cached_response = await cache_service.find_similar_response(query_embedding, document_id)
                if cached_response:
                    logger.info(f"{log_prefix}Using cached response")

                    # Emit cached content as chunk(s). Streaming splits char-by-char
                    # to preserve the original UX; non-stream emits one chunk.
                    cached_content = cached_response['content']
                    if chunk_cached_response:
                        for char in cached_content:
                            yield ChunkEvent(content=char)
                    else:
                        yield ChunkEvent(content=cached_content)

                    # Persist both messages
                    user_message = Message(
                        content=content, role="user", conversation_id=conversation_id
                    )
                    db.add(user_message)
                    db.flush()
                    assistant_message = Message(
                        content=cached_content,
                        role="assistant",
                        conversation_id=conversation_id,
                        context={
                            "chunks": cached_response['chunks'],
                            "annotations": cached_response['annotations'],
                        },
                    )
                    db.add(assistant_message)
                    db.commit()
                    db.refresh(user_message)
                    db.refresh(assistant_message)
                    yield DoneEvent(user_message=user_message, assistant_message=assistant_message)
                    return

            # 3. Prepare retrieval/classify/prompt context
            ctx = await self._prepare_generation_context(
                db=db,
                content=content,
                document_id=document_id,
                model=model,
                conversation_id=conversation_id,
                api_key=api_key,
                provider=provider,
                stream_log_prefix=log_prefix,
            )
            client = get_llm_client(provider, api_key)
            chat_messages = ctx.messages[1:]  # drop the system message (passed separately)
            is_first_message = ctx.is_first_message

            # 4. Optionally persist the user message before the LLM call
            #    (streaming persists optimistically so it survives generation failure).
            user_message: Optional[Message] = None
            if persist_user_before_llm:
                user_message = Message(
                    content=content, role="user", conversation_id=conversation_id
                )
                db.add(user_message)
                db.flush()

            # 5. Run the LLM call via the injected strategy
            accumulated_content = ""
            try:
                async for fragment in generate(
                    client, ctx.system_prompt, chat_messages, model, ctx.max_tokens
                ):
                    if fragment:
                        accumulated_content += fragment
                        yield ChunkEvent(content=fragment)
            except APIError as e:
                logger.error(f"{log_prefix}LLM API error: {str(e)}", exc_info=True)
                yield ErrorEvent(message=self._llm_error_message(e))
                return
            except Exception as e:
                logger.error(f"{log_prefix}Unexpected error calling LLM API: {str(e)}", exc_info=True)
                yield ErrorEvent(message=f"Failed to generate response: {str(e)}")
                return

            logger.info(f"{log_prefix}[Annotations] Raw LLM response: {accumulated_content[:500]}...")

            # 6. Parse annotations, verify citations, score quality
            assistant_content, annotations = self._parse_annotations(
                accumulated_content, ctx.relevant_chunks
            )
            logger.info(f"{log_prefix}[Annotations] Parsed {len(annotations)} annotations from response")

            citation_warnings = self._verify_citations(
                response_text=assistant_content,
                annotations=annotations,
                relevant_chunks=ctx.relevant_chunks,
            )

            quality_scores = await self._score_answer_quality(
                query=content,
                answer=assistant_content,
                context_chunks=ctx.relevant_chunks,
                user_api_key=api_key,
                provider=provider,
            )

            # 7. Estimate token usage (shared helper; previously duplicated)
            token_usage = self._estimate_token_usage(
                messages=ctx.messages,
                content=accumulated_content,
                model=model,
                max_context_tokens=ctx.max_context_tokens,
                log_prefix=log_prefix,
            )

            # 8. Persist messages
            message_context = {
                "chunks": ctx.relevant_chunks,
                "annotations": annotations,
                "token_usage": token_usage,
                "chunk_selection_stats": ctx.chunk_stats,
                "query_classification": ctx.query_classification,
                "citation_warnings": citation_warnings,
                "quality_scores": quality_scores,
            }
            if user_message is None:
                # Non-stream path: persist user message now (after successful generation)
                user_message = Message(
                    content=content, role="user", conversation_id=conversation_id
                )
                db.add(user_message)
                db.flush()
            assistant_message = Message(
                content=assistant_content,
                role="assistant",
                conversation_id=conversation_id,
                context=message_context,
            )
            db.add(assistant_message)

            # 9. Title (no-op when not the first message)
            await self._set_title_if_first_message(
                db=db,
                conversation_id=conversation_id,
                is_first_message=is_first_message,
                content=content,
                api_key=api_key,
                provider=provider,
            )

            db.commit()
            db.refresh(user_message)
            db.refresh(assistant_message)
            logger.debug(f"{log_prefix}Messages saved successfully")

            # 10. Cache the response (first message only)
            if is_first_message:
                await cache_service.set_response(
                    document_id, query_embedding,
                    assistant_content, annotations, ctx.relevant_chunks,
                )

            yield DoneEvent(user_message=user_message, assistant_message=assistant_message)

        except Exception as e:
            logger.error(f"{log_prefix}Error in linear pipeline: {str(e)}", exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass
            yield ErrorEvent(message=f"Error: {str(e)}")

    def _estimate_token_usage(
        self,
        *,
        messages: List[Dict],
        content: str,
        model: str,
        max_context_tokens: int,
        log_prefix: str = "",
    ) -> Optional[Dict[str, int]]:
        """Estimate prompt/completion/total token usage for a generation.

        The chat-completion client returns only the assistant text (no usage
        object), and streams likewise omit usage, so both paths estimate from
        the assembled messages + generated content. Returns None on failure.
        Previously duplicated verbatim (modulo log prefix) in both entry points.
        """
        try:
            prompt_tokens = TokenService.estimate_context_tokens(messages, model)
            completion_tokens = TokenService.count_tokens(content, model)
            total_tokens = prompt_tokens + completion_tokens
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            logger.info(
                f"{log_prefix}Token usage: {prompt_tokens} prompt + "
                f"{completion_tokens} completion = {total_tokens} total tokens"
            )
            logger.info(
                f"{log_prefix}Token budget utilization: "
                f"{total_tokens}/{max_context_tokens} "
                f"({100 * total_tokens / max_context_tokens:.1f}%)"
            )
            return token_usage
        except Exception as e:
            logger.warning(f"{log_prefix}Failed to estimate token usage: {e}")
            return None

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

        # Linear pipeline: drain the shared generator, translate events to a dict.
        # ChunkEvent content is accumulated (non-stream doesn't surface incremental
        # output); ErrorEvent becomes a raised ValueError; DoneEvent carries the
        # persisted messages.
        assistant_content = ""
        async for event in self._run_linear_pipeline(
            db=db,
            user=user,
            content=content,
            conversation_id=conversation_id,
            document_id=document_id,
            model=model,
            generate=self._complete_then_yield,
            chunk_cached_response=False,
            persist_user_before_llm=False,
        ):
            if isinstance(event, ChunkEvent):
                assistant_content += event.content
            elif isinstance(event, ErrorEvent):
                raise ValueError(event.message)
            elif isinstance(event, DoneEvent):
                return {
                    "user_message": {
                        "id": event.user_message.id,
                        "role": event.user_message.role,
                        "content": event.user_message.content,
                        "created_at": event.user_message.created_at,
                        "context": None,
                        "annotations": None,
                    },
                    "assistant_message": {
                        "id": event.assistant_message.id,
                        "role": event.assistant_message.role,
                        "content": event.assistant_message.content,
                        "created_at": event.assistant_message.created_at,
                        "context": event.assistant_message.context.get("chunks") if event.assistant_message.context else None,
                        "annotations": event.assistant_message.context.get("annotations") if event.assistant_message.context else None,
                    },
                }
        # Pipeline ended without DoneEvent (shouldn't happen, but be safe).
        raise ValueError("Chat pipeline ended without producing a response")

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
            await self._set_title_if_first_message(
                db=db,
                conversation_id=conversation_id,
                is_first_message=is_first_message,
                content=content,
                api_key=api_key,
                provider=provider,
            )

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
                    await self._set_title_if_first_message(
                        db=db,
                        conversation_id=conversation_id,
                        is_first_message=is_first_message,
                        content=content,
                        api_key=api_key,
                        provider=provider,
                    )

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
        # Linear pipeline: forward each event as an SSE frame. ChunkEvents
        # become incremental 'chunk' frames (cached responses stream char-by-char
        # via chunk_cached_response=True); ErrorEvent and DoneEvent become
        # terminal 'error' / 'done' frames. Datetimes are ISO-serialized here.
        async for event in self._run_linear_pipeline(
            db=db,
            user=user,
            content=content,
            conversation_id=conversation_id,
            document_id=document_id,
            model=model,
            generate=self._stream_passthrough,
            chunk_cached_response=True,
            persist_user_before_llm=True,
            log_prefix="[Stream] ",
        ):
            if isinstance(event, ChunkEvent):
                yield f"data: {json.dumps({'type': 'chunk', 'content': event.content})}\n\n"
            elif isinstance(event, ErrorEvent):
                yield f"data: {json.dumps({'type': 'error', 'content': event.message})}\n\n"
                return
            elif isinstance(event, DoneEvent):
                um, am = event.user_message, event.assistant_message
                ctx_chunks = am.context.get("chunks") if am.context else None
                ctx_annotations = am.context.get("annotations") if am.context else None
                final_data = {
                    'type': 'done',
                    'user_message': {
                        "id": um.id,
                        "role": um.role,
                        "content": um.content,
                        "created_at": um.created_at.isoformat(),
                        "context": None,
                        "annotations": None,
                    },
                    'assistant_message': {
                        "id": am.id,
                        "role": am.role,
                        "content": am.content,
                        "created_at": am.created_at.isoformat(),
                        "context": ctx_chunks,
                        "annotations": ctx_annotations,
                    },
                }
                yield f"data: {json.dumps(final_data)}\n\n"


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
