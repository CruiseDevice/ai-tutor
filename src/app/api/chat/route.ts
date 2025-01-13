// app/api/chat/route.ts

import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/db";
import OpenAI from 'openai';
import { PrismaClient, Conversation } from "@prisma/client";
import { OpenAIEmbeddings } from "@langchain/openai";

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string;
}

import { Prisma } from "@prisma/client";

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

const prismaRaw = new PrismaClient();

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {messages, documentId, userId} = body;

    if(!messages || !documentId) {
      console.error('Missing required fields: ', {messages, documentId});
      return NextResponse.json({
        error: 'Missing required fields',
        received: {messages, documentId}
      }, {status: 400})
    }

    const userQuestion = messages[messages.length - 1].content;
    if (!userId) {
      return NextResponse.json({error: 'Unauthorized'}, {status: 401});
    }

    // get user's API key
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
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: document?.user.apiKey,
      modelName: "text-embedding-3-small"
    });

    const questionEmbedding = await embeddings.embedQuery(userQuestion);

    // Perform vector similarity search using MongoDB
    try {
      // Use the exact MongoDB Atlas vector search syntax
      const result = await prismaRaw.$runCommandRaw({
        aggregate: "DocumentChunk",
        pipeline: [
          {
            $vectorSearch: {
              index: "vector_index",
              path: "embedding",
              queryVector: questionEmbedding,
              numCandidates: 20,
              limit: 3
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

      console.log('Vector search executed:', !!result);
      
      const similarChunks = result?.cursor?.firstBatch || [];
      console.log('Similar chunks found:', similarChunks.length);

      if (similarChunks.length === 0) {
        console.log('No similar chunks found, falling back to recent chunks');
        const fallbackChunks = await prisma.documentChunk.findMany({
          where: { documentId },
          take: 3,
          orderBy: { createdAt: 'desc' }
        });
        return handleChatCompletion(fallbackChunks, messages, conversation, user.apiKey);
      }

      // Get or create conversation
      let conversation = await prisma.conversation.findFirst({
        where: {documentId}
      });

      if(!conversation) {
        conversation = await prisma.conversation.create({
          data: {
            userId,
            documentId,
            messages: {
              create: messages.map((msg: ChatMessage) => ({
                content: msg.content,
                role: msg.role
              }))
            }
          }
        });
      }
      return handleChatCompletion(similarChunks, messages, conversation, user.apiKey);
    } catch (searchError) {
      console.error('Vector search error:', searchError);
      // Log the full error details
      if (searchError instanceof Error) {
        console.error('Error name:', searchError.name);
        console.error('Error message:', searchError.message);
        console.error('Error stack:', searchError.stack);
      }
      
      // Fallback to basic search if vector search fails
      const fallbackChunks = await prisma.documentChunk.findMany({
        where: { documentId },
        take: 3,
        orderBy: { createdAt: 'desc' }
      });
      
      let conversation = await prisma.conversation.findFirst({
        where: {documentId}
      });

      if(!conversation) {
        conversation = await prisma.conversation.create({
          data: {
            userId,
            documentId,
            messages: {
              create: messages.map((msg: ChatMessage) => ({
                content: msg.content,
                role: msg.role
              }))
            }
          }
        });
      }

      return handleChatCompletion(fallbackChunks, messages, conversation, user.apiKey);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occured';
    console.error('Chat API Error:', errorMessage);

    return NextResponse.json(
      { error: 'Error processing chat message', details: errorMessage },
      { status: 500 }
    );
  }
}

async function handleChatCompletion(
  similarChunks: Array<{content: string; pageNumber: number}>,
  messages: ChatMessage[],
  conversation: Conversation,
  apiKey: string
) {
  try {
    // save the user's message if conversation exists
    try {
      await prisma?.message.create({
        data: {
          content: messages[messages.length - 1].content,
          role: messages[messages.length - 1].role,
          conversationId: conversation.id
        }
      });
    } catch (error) {
      console.error(error);
    }

    const openai = new OpenAI({apiKey});

    // Create completion with context from similar chunks
    const contextPrompt = similarChunks.length > 0
      ? `Context from the document:\n${similarChunks
          .map(chunk => `[Page ${chunk.pageNumber}]: ${chunk.content}`)
          .join('\n\n')}`
      : 'No specific context found in the document.';

    console.log(similarChunks);
    console.log('contextPrompt: ', contextPrompt)
      // Ensure messages are in the correct format
    const formattedMessages = messages.map((msg: ChatMessage) => ({
      role: msg.role,
      content: msg.content
    }));

    // create chat completion
    const completion = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: `You are a helpful AI tutor. Help the student understand the document they are reading. Be concise but thorough in your explanation. Here is relevant context from the document:\n\n${contextPrompt}`
        },
        ...formattedMessages
      ]
    });

    const assistantResponse = completion.choices[0].message.content || '';

    // save the assistant's response
    try {
      await prisma?.message.create({
        data: {
          content: assistantResponse,
          role: 'assistant',
          conversationId: conversation.id
        }
      });
    } catch (error) {
      console.error('Error saving assistant message:', error);
    }

    const formattedResponse: ChatMessage = {
      role: 'assistant',
      content: assistantResponse
    };

    return new Response(JSON.stringify(formattedResponse), {
      headers: {
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error in chat completion';
    console.error('Chat completion error:', errorMessage);
    throw new Error(errorMessage);
  }
}