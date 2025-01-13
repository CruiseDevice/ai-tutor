// app/api/chat/route.ts

import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/db";
import OpenAI from 'openai';
import { PrismaClient, Conversation } from "@prisma/client";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Prisma } from "@prisma/client";

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string;
}

type DocumentChunkWithPage = {
  content: string;
  pageNumber: number;
}


type MongoDBCommandResponse = {
  ok: number;
  cursor: {
    firstBatch: Array<{
      _id: string;
      content: string;
      pageNumber: number;
      embedding: Prisma.JsonValue;
      documentId: string;
    }>;
    id?: number;
    ns?: string;
  };
}

// Constants
const CHUNK_LIMIT = 3;
const MODEL_NAME = "text-embedding-3-small";
const VECTOR_INDEX = "vector_index";
const GPT_MODEL = "gpt-4";

const prismaRaw = new PrismaClient();

// helper function
async function getOrCreateConversation(userId: string, documentId:  string, messages: ChatMessage[]): Promise<Conversation> {
  let conversation = await prisma.conversation.findFirst({
    where: { documentId }
  });

  if (!conversation) {
    conversation = await prisma.conversation.create({
      data: {
        userId,
        documentId,
        messages: {
          create: messages.map(msg => ({
            content: msg.content,
            role: msg.role
          }))
        }
      }
    });
  }

  return conversation;
}

async function performVectorSearch(documentId: string, embedding: number[]) {
  const result = await prismaRaw.$runCommandRaw({
    aggregate: "DocumentChunk",
    pipeline: [
      {
        $vectorSearch: {
          index: VECTOR_INDEX,
          path: "embedding",
          queryVector: embedding,
          numCandidates: 20,
          limit: CHUNK_LIMIT
        }
      },
      {
        $match: {
          documentId: documentId
        }
      }
    ],
    cursor: {}
  }) as MongoDBCommandResponse;

  return result?.cursor?.firstBatch || [];
}

async function getFallbackChunks(documentId: string) {
  return await prisma.documentChunk.findMany({
    where: { documentId },
    take: CHUNK_LIMIT,
    orderBy: { createdAt: 'desc' }
  });
}

async function saveMessage(content: string, role: string, conversationId: string) {
  try {
    await prisma.message.create({
      data: {
        content,
        role,
        conversationId
      }
    });
  } catch (error) {
    console.error(`Error saving ${role} message:`, error);
    // Don't throw - message saving shouldn't break the chat flow
  }
}


export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {messages, documentId, userId} = body;

    if(!messages?.length || !documentId || !userId) {
      return NextResponse.json({
        error: 'Missing required fields',
        received: {messages, documentId}
      }, {status: 400})
    }

    // get user and user's API key
    const user = await prisma.user.findUnique({
      where: {id: userId},
      select: {apiKey: true}
    });

    if (!user?.apiKey) {
      return NextResponse.json(
        {error: 'Please set up your OpenAI API key in settings'},
        {status: 400}
      );
    }

    // get document and user info
    const document = await prisma.document.findUnique({
      where: {id: documentId},
      include: {user: true}
    });

    if(!document?.user?.apiKey) {
      return NextResponse.json(
        {error: 'OpenAI API key not found'},
        {status: 400}
      );
    }

    // generate embedding for the questions
    const userQuestion = messages[messages.length - 1].content;
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: document?.user.apiKey,
      modelName: MODEL_NAME
    });

    const questionEmbedding = await embeddings.embedQuery(userQuestion);

    // get or create conversation
    const conversation = await getOrCreateConversation(userId, documentId, messages);

    // Perform vector similarity search using MongoDB
    try {
      const similarChunks = await performVectorSearch(documentId, questionEmbedding);
      if (similarChunks.length === 0) {
        console.log('No similar chunks found, falling back to recent chunks');
        const fallbackChunks = await getFallbackChunks(documentId);
        return handleChatCompletion(fallbackChunks, messages, conversation, user.apiKey);
      }
      return handleChatCompletion(similarChunks, messages, conversation, user.apiKey);
    } catch (searchError) {
      console.error('Vector search error:', searchError);
      // Fallback to basic search if vector search fails
      const fallbackChunks = await getFallbackChunks(documentId);
      return handleChatCompletion(fallbackChunks, messages, conversation, user.apiKey);
    }
  } catch (error) {
    console.error('Chat API Error:', error);
    return NextResponse.json(
      { error: 'Error processing chat message'},
      { status: 500 }
    );
  }
}

async function handleChatCompletion(
  similarChunks: DocumentChunkWithPage[],
  messages: ChatMessage[],
  conversation: Conversation,
  apiKey: string
): Promise<Response> {
  try {
    // save the user's message if conversation exists
    const userMessage = messages[messages.length - 1];
    await saveMessage(userMessage.content, userMessage.role, conversation.id);

    const openai = new OpenAI({apiKey});

    // Create completion with context from similar chunks
    const contextPrompt = similarChunks.length > 0
      ? `Context from the document:\n${similarChunks
          .map(chunk => `[Page ${chunk.pageNumber}]: ${chunk.content}`)
          .join('\n\n')}`
      : 'No specific context found in the document.';

    // create chat completion
    const completion = await openai.chat.completions.create({
      model: GPT_MODEL,
      messages: [
        {
          role: 'system',
          content: `You are a helpful AI tutor. Help the student understand the document they are reading. 
                   Be concise but thorough in your explanation. Here is relevant context from the document:\n\n${contextPrompt}`
        },
        ...messages.map(msg => ({
          role: msg.role,
          content: msg.content
        }))
      ]
    });

    const assistantResponse = completion.choices[0].message.content || '';

    await saveMessage(assistantResponse, 'assistant', conversation.id);

    return new Response(JSON.stringify({
      role: 'assistant',
      content: assistantResponse
    } as ChatMessage), {
      headers: {
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    console.error('Chat completion error:', error);
    return new Response(
      JSON.stringify({
        role: 'assistant',
        content: 'I apologize, but I encountered an error while processing your request. Please try again.'
      } as ChatMessage),
      {
        status: 500,
        headers: {
          'Content-Type': 'application/json'
        }
      }
    )
  }
}