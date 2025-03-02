// app/api/chat/route.ts

import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/db";
import OpenAI from 'openai';
import { findSimilarChunks } from "@/lib/pgvector";

// Constants
const CHUNK_LIMIT = 5;
const MODEL_NAME = "gpt-4";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { content, conversationId, documentId } = body;

    if (!content || !conversationId || !documentId) {
      return NextResponse.json({
        error: 'Missing required fields',
        received: { content, conversationId, documentId }
      }, { status: 400 });
    }

    // Get conversation to verify it exists
    const conversation = await prisma.conversation.findUnique({
      where: { id: conversationId },
      include: { user: true }
    });

    if (!conversation) {
      return NextResponse.json(
        { error: 'Conversation not found' },
        { status: 404 }
      );
    }

    const user = conversation.user;

    if (!user?.apiKey) {
      return NextResponse.json(
        { error: 'Please set up your OpenAI API key in settings' },
        { status: 400 }
      );
    }

    // use pgvector to find relevant document chunks
    let relevantChunks = [];
    try {
      relevantChunks = await findSimilarChunks(
        content, documentId, CHUNK_LIMIT, // Return top 5 relevant chunks
        user.apiKey
      );
      
      console.log(`Found ${relevantChunks.length} relevant chunks`);

      if (!Array.isArray(relevantChunks)) {
        console.warn("findSimilarChunks did not return an array");
        relevantChunks = [];
      }
    } catch (error) {
      console.error("Error retrieving similar chunks:", error instanceof Error ? error.message : String(error));
      relevantChunks = [];
      // Log a more detailed error to help with debugging
      if (error instanceof Error) {
        console.error(`Stack trace: ${error.stack}`);
      }
    }

    // format chunks for context (with defensive programming)
    const context = relevantChunks.map(chunk => {
      // Ensure all expected properties exist with defaults if missing
      return {
        content: chunk.content || "No content available",
        pageNumber: chunk.pageNumber || 0,
        similarity: chunk.similarity || 0
      };
    });

    // Create and save user message to database
    const userMessage = await prisma.message.create({
      data: {
        content,
        role: 'user',
        conversationId
      }
    });

    // initialize openai client
    const openai = new OpenAI({
      apiKey: user.apiKey
    });

    // Create system message with document context
    let contextText = "";
    if (context.length > 0) {
      contextText = context.map(c => 
        `[Page ${c.pageNumber}]: ${c.content}`
      ).join('\n\n');
    } else {
      contextText = "No relevant document sections found.";
    }
    

    const systemMessage = {
      role: "system",
      content: `You are an AI tutor helping a student understand a PDF document. 
      You have access to the following document chunks that are relevant to the student's question:
      
      ${contextText}
      
      When referring to content, always cite the page number like [Page X]. 
      You can help with highlighting content by using commands like:
      - HIGHLIGHT[Page X]: "text to highlight"
      - CIRCLE[Page X]: "text or element to circle"
      - GOTO[Page X]: To navigate to a specific page
      
      Make your responses helpful, clear, and educational. If the context doesn't contain the answer, 
      say you don't have enough information from the document and suggest looking at other pages.`
    };

    // Get previous conversation history
    const conversationHistory = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'asc' },
      take: 10 // Limit to last 10 messages for context
    });

    // Format conversation history for OpenAI
    const historyMessages = conversationHistory.map(msg => ({
      role: msg.role as 'user' | 'assistant' | 'system',
      content: msg.content
    }));

    // Combine system message, history, and current user message
    const promptMessages = [
      systemMessage,
      ...historyMessages,
      { role: 'user', content }
    ];

    // Call OpenAI for chat completion
    const completion = await openai.chat.completions.create({
      model: "gpt-4-turbo-preview", // Using GPT-4 for high-quality responses
      messages: promptMessages,
      temperature: 0.7,
      max_tokens: 1000
    });
    
    const assistantResponse = completion.choices[0].message;

    // Save assistant response to database
    const assistantMessage = await prisma.message.create({
      data: {
        content: assistantResponse.content,
        role: 'assistant',
        context: context as any,
        conversationId
      }
    });

    // Return both messages in the format expected by the client
    return NextResponse.json({
      userMessage: {
        id: userMessage.id,
        role: userMessage.role,
        content: userMessage.content
      },
      assistantMessage: {
        id: assistantMessage.id,
        role: assistantMessage.role,
        content: assistantMessage.content
      }
    });

  } catch (error) {
    console.error("Chat API error:", error instanceof Error ? error.message : String(error));
    return NextResponse.json(
      {error: "An error occurred processing your request"},
      {status: 500}
    );
  }
}