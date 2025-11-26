from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, DatabaseError
from typing import List, Dict, Optional, AsyncGenerator
from openai import OpenAI, APIError
import logging
import json
import re
import uuid
import asyncio
from ..models.conversation import Conversation, Message
from ..models.document import DocumentChunk
from ..models.user import User
from .embedding_service import get_embedding_service
from .retry_utils import retry_openai_call
from .cache_service import get_cache_service

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.embedding_service = get_embedding_service()

    def _generate_conversation_title(self, user_message: str, user_api_key: str) -> str:
        """
        Generate a smart, concise title for the conversation based on the first user message.
        Uses LLM to create a title that's 3-6 words.
        """
        try:
            client = OpenAI(api_key=user_api_key)

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

            # Use retry logic for OpenAI API call
            def _create_completion():
                return client.chat.completions.create(
                    model="gpt-4o-mini",  # Use cheaper model for title generation
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates concise conversation titles."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_completion_tokens=20
                )

            completion = retry_openai_call(
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

    async def find_similar_chunks(
        self,
        db: Session,
        query: str,
        document_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Find similar document chunks using vector similarity search with caching."""
        try:
            cache_service = await get_cache_service()

            # Check cache for chunks first
            query_embedding = None
            cached_chunks = None

            # Try to get embedding from cache
            cached_embedding = await cache_service.get_embedding(query)
            if cached_embedding:
                query_embedding = cached_embedding
                logger.debug(f"Using cached embedding for query: {query[:50]}...")
                # Check cache for chunks with this embedding
                cached_chunks = await cache_service.get_chunks(document_id, query_embedding)
                if cached_chunks:
                    logger.info(f"Cache hit: Returning {len(cached_chunks)} cached chunks")
                    return cached_chunks

            # Generate embedding if not cached
            if query_embedding is None:
                logger.debug(f"Generating embedding for query: {query[:50]}...")
                query_embedding = self.embedding_service.generate_embedding(query)
                logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")
                # Cache the embedding
                await cache_service.set_embedding(query, query_embedding)

            # Perform vector similarity search using pgvector
            # pgvector expects the vector in the format '[1,2,3]' as a string
            embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

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
                ORDER BY similarity DESC
                LIMIT :limit
            """)

            logger.debug(f"Executing vector search for document_id: {document_id}")
            result = db.execute(
                query_sql,
                {
                    "embedding": embedding_str,
                    "document_id": document_id,
                    "limit": limit
                }
            )

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

            logger.debug(f"Found {len(chunks)} similar chunks")

            # Cache the chunks
            await cache_service.set_chunks(document_id, query_embedding, chunks)

            return chunks
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
                query_embedding = self.embedding_service.generate_embedding(content)
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
            client = OpenAI(api_key=api_key)

            # Find relevant chunks
            logger.debug(f"Finding similar chunks for document {document_id}")
            relevant_chunks = await self.find_similar_chunks(db, content, document_id, limit=5)

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
                def _create_completion():
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_completion_tokens=1000
                    )

                completion = retry_openai_call(
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
                        title = self._generate_conversation_title(content, user.api_key)
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
                query_embedding = self.embedding_service.generate_embedding(content)
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
            client = OpenAI(api_key=api_key)

            # Find relevant chunks
            logger.debug(f"Finding similar chunks for document {document_id}")
            relevant_chunks = await self.find_similar_chunks(db, content, document_id, limit=5)

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
                def _create_stream():
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_completion_tokens=1000,
                        stream=True
                    )

                stream = retry_openai_call(
                    _create_stream,
                    max_attempts=5,
                    initial_wait=1.0,
                    max_wait=60.0
                )

                # Stream chunks
                for chunk in stream:
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
                        title = self._generate_conversation_title(content, user.api_key)
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

