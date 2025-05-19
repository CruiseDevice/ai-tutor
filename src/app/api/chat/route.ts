// app/api/chat/route.ts

import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/db";
import OpenAI from 'openai';
import { findSimilarChunks } from "@/lib/pgvector";
import { ChatCompletionMessageParam } from "openai/resources/index.mjs";

// Constants
const CHUNK_LIMIT = 5;
const DEFAULT_MODEL = "gpt-4";  // Default model to use if no model is specified

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { content, conversationId, documentId, model } = body;

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
      
      if (!Array.isArray(relevantChunks)) {
        relevantChunks = [];
      }
    } catch (error) {
      relevantChunks = [];
      // Log a more detailed error to help with debugging
      if (error instanceof Error) {
        console.error(`Stack trace: ${error.stack}`);
      }
    }

    // format chunks for context (with defensive programming)
    const context = relevantChunks.map(chunk => {
      // Ensure all expected properties exist with defaults if missing

      // More robust page number parsing
      let pageNum = 1;

      try {
        if (chunk.pageNumber !== undefined && chunk.pageNumber !== null) {
          if (typeof chunk.pageNumber === 'number') {
            pageNum = chunk.pageNumber;
          } else if (typeof chunk.pageNumber === 'string') {
            pageNum = parseInt(chunk.pageNumber, 10) || 1;
          } else if (typeof chunk.pageNumber === 'object') {
            // Sometimes SQL results can come back as objects with value properties
            const strValue = String(chunk.pageNumber?.value || chunk.pageNumber);
            pageNum = parseInt(strValue, 10) || 1;
          }
        }
      } catch (e) {
        console.error("Error parsing page number: ", e);
      }

      // Extract text ranges if available
      let textRanges = [];
      try {
        if(chunk.positionData && typeof chunk.positionData === 'object') {
          if(chunk.positionData.textRanges) {
            textRanges = chunk.positionData.textRanges;
          }
        }
      } catch (e) {
        console.error("Error parsing position data: ", e);
      }

      return {
        content: chunk.content || "No content available",
        pageNumber: pageNum,
        similarity: chunk.similarity || 0,
        textRanges: textRanges
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
    

    const systemMessage: ChatCompletionMessageParam = {
      role: "system",
      content: `You are an AI tutor helping a student understand a PDF document. 
      You have access to the following document chunks that are relevant to the student's question:
      
      ${contextText}
      
      When referring to content, always cite the page number like [Page X]. 
      Make sure to use the correct page number for each piece of information.
      
      IMPORTANT FORMATTING INSTRUCTIONS:
      1. Use markdown to highlight important concepts, terms, or phrases by making them **bold** or using *italics*.
      2. For direct quotes from the document, use > blockquote formatting.
      3. When referring to specific sections, use [Page X] to cite the page number.
      4. Use bullet points or numbered lists for step-by-step explanations.
      5. For critical information or warnings, use "⚠️" at the beginning of the paragraph.
      
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
    const historyMessages: ChatCompletionMessageParam[] = conversationHistory.map(msg => ({
      role: msg.role as 'user' | 'assistant' | 'system',
      content: msg.content
    }));

    const userChatMessage: ChatCompletionMessageParam = {
      role: 'user',
      content
    }

    // Combine system message, history, and current user message
    const promptMessages = [
      systemMessage,
      ...historyMessages,
      userChatMessage
    ];

    // Use provided model or default to GPT-4
    const MODEL_NAME = model || DEFAULT_MODEL;

    // Call OpenAI for chat completion
    const completion = await openai.chat.completions.create({
      model: MODEL_NAME, // Using GPT-4 for high-quality responses
      messages: promptMessages,
      temperature: 0.7,
      max_tokens: 1000
    });
    
    const assistantResponse = completion.choices[0].message;

    // Save assistant response to database
    const assistantMessage = await prisma.message.create({
      data: {
        content: assistantResponse.content as string,
        role: 'assistant',
        context: context,
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
        content: assistantMessage.content,
        context: context  // Include the context with position data
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