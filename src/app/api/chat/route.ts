// app/api/chat/route.ts

import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/db";
import OpenAI from 'openai';

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    console.log('Received request body: ', body);

    const {messages, documentId, userId} = body;

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

    if (!messages || !documentId) {
      console.error('Missing required fields: ', {
        messages,
        documentId,
      });
      return new Response(
        JSON.stringify({
          error: 'Missing required fields',
          received: {messages, documentId}
        }),
        {
          status: 400,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
    }

    // Get the conversation or create a new one
    let conversation = await prisma?.conversation.findFirst({
      where: {documentId}
    });

    if (!conversation) {
      try {
        conversation = await prisma?.conversation.create({
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
      } catch (error) {
        console.error('Error creating conversation: ', error);
        return new Response(
          JSON.stringify({
            error: 'Failed to create conversation'
          }),
          {
            status: 500, 
            headers: {
              'Content-Type': 'application/json'
            }
          }
        );
      }
    }

    // double-check that we have a valid conversation
    if(!conversation) {
      console.error('Failed to get or create conversation');
      return new Response(
        JSON.stringify({error: 'Failed to get or create conversation'}),
        {status: 500, headers: {'Content-Type': 'application/json'}}
      );
    }

    // save the user's message if conversation exists
    try{
      await prisma?.message.create({
        data: {
          content: messages[messages.length - 1].content,
          role: messages[messages.length - 1].role,
          conversationId: conversation.id
        }
      });
    } catch (error) {
      console.error('Error saving user message: ', error);
    }

    console.log('conversation:', conversation);

    const openai = new OpenAI({apiKey: user.apiKey});

    // Create chat completion with enhanced system prompt
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
          content: 'You are a helpful AI tutor. Help the student understand the document they are reading. Be concise but thorough in your explanation.'
        },
        ...formattedMessages
      ]
    });

    const assistantResponse = completion.choices[0].message.content || '';

    // save the assistant's response
    try{
      await prisma?.message.create({
        data: {
          content: assistantResponse,
          role: 'assistant',
          conversationId: conversation.id
        }
      });
    } catch (error) {
      console.error('Error saving assistant message:', error)
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
    console.log(error)
    return new Response(JSON.stringify({error: 'Error processing chat message'}), {
      status: 500,
      headers: {
        'Content-Type': 'application/json'
      }
    })
  }
}