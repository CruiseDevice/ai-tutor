import { NextRequest } from "next/server";
import { Configuration, OpenAIApi } from "openai-edge";
import prisma from "@/lib/db";

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string;
}

// create openai configuration
const config = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
});
const openai = new OpenAIApi(config);

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    console.log('Received request body: ', body);

    const {messages, documentId, userId} = body;

    console.log('messages: ', messages);
    console.log('documentId: ', documentId);
    console.log('userId: ', userId);

    if (!messages || !documentId || !userId) {
      console.error('Missing required fields: ', {
        messages,
        documentId,
        userId
      });
      return new Response(
        JSON.stringify({
          error: 'Missing required fields',
          received: {messages, documentId, userId}
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
    // create chat completion
    const response = await openai.createChatCompletion({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful AI tutor. Help the student understand the document they are reading. Be concise but thorough in your explanation.'
        },
        ...messages
      ]
    });
    const chatResponse = await response.json();
    const assistantMessage = chatResponse.choices[0].message;

    // save the assistant's response
    try{
      await prisma?.message.create({
        data: {
          content: assistantMessage.content,
          role: 'assistant',
          conversationId: conversation.id
        }
      });  
    } catch (error) {
      console.error('Error saving assistant message:', error)
    }

    return new Response(JSON.stringify(assistantMessage), {
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