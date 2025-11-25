from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Optional
from openai import OpenAI
from ..models.conversation import Conversation, Message
from ..models.document import DocumentChunk
from ..models.user import User
from .embedding_service import get_embedding_service


class ChatService:
    def __init__(self):
        self.embedding_service = get_embedding_service()
    
    def find_similar_chunks(
        self,
        db: Session,
        query: str,
        document_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Find similar document chunks using vector similarity search."""
        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Perform vector similarity search using pgvector
        query_sql = text("""
            SELECT 
                id,
                content,
                page_number,
                document_id,
                position_data,
                1 - (embedding <=> :embedding::vector) as similarity
            FROM document_chunks
            WHERE document_id = :document_id
            ORDER BY similarity DESC
            LIMIT :limit
        """)
        
        result = db.execute(
            query_sql,
            {
                "embedding": query_embedding,
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
        
        return chunks
    
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
        if not user.api_key:
            raise ValueError("User has no OpenAI API key configured")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=user.api_key)
        
        # Find relevant chunks
        relevant_chunks = self.find_similar_chunks(db, content, document_id, limit=5)
        
        # Format context from chunks
        if relevant_chunks:
            context_text = "\n\n".join([
                f"[Page {chunk['pageNumber']}]: {chunk['content']}"
                for chunk in relevant_chunks
            ])
        else:
            context_text = "No relevant document sections found."
        
        # Create system message with context
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

Make your responses helpful, clear, and educational. If the context doesn't contain the answer, 
say you don't have enough information from the document and suggest looking at other pages."""
        }
        
        # Get conversation history
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
        
        # Call OpenAI
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        assistant_content = completion.choices[0].message.content
        
        # Save user message
        user_message = Message(
            content=content,
            role="user",
            conversation_id=conversation_id
        )
        db.add(user_message)
        db.flush()
        
        # Save assistant message with context
        assistant_message = Message(
            content=assistant_content,
            role="assistant",
            conversation_id=conversation_id,
            context=relevant_chunks
        )
        db.add(assistant_message)
        db.commit()
        
        db.refresh(user_message)
        db.refresh(assistant_message)
        
        return {
            "userMessage": {
                "id": user_message.id,
                "role": user_message.role,
                "content": user_message.content
            },
            "assistantMessage": {
                "id": assistant_message.id,
                "role": assistant_message.role,
                "content": assistant_message.content,
                "context": relevant_chunks
            }
        }

