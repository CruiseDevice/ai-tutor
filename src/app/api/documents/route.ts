// src/app/api/documents/route.ts

import { auth } from "@/lib/auth";
import { NextResponse } from "next/server";
import prisma from "@/lib/db";
import { put } from "@vercel/blob";

export async function POST(req: Request): Promise<NextResponse> {
  try{
    const user = await auth();  // get authenticated user

    if(!user) {
      // if no authenticated user found, return 401 Unauthorized
      return NextResponse.json(
        {error: 'Unauthorized'},
        {status: 401}
      );
    }

    // Get session token from cookie
     const sessionToken = req.headers.get('cookie')
     ?.split(';')
     ?.find(c => c.trim().startsWith('session_token='))
     ?.split('=')[1];

    if (!sessionToken) {
       return NextResponse.json({ error: 'No session token' }, { status: 401 });
    }

    // Verify session
    const session = await prisma.session.findFirst({
      where: {
        token: sessionToken,
        expiresAt: {
          gt: new Date(),
        },
      },
      include: {
        user: true,
      },
    });

    if (!session) {
      return NextResponse.json({ error: 'Invalid session' }, { status: 401 });
    }

    const formData = await req.formData();
    const file = formData.get('file') as File;

    const blob = await put(`pdfs/${user.id}/${file.name}`, file, {
      access: 'public',
    });

    // create document in database with user's ID
    const document = await prisma.document.create({
      data: {
        userId: user.id,
        title: file.name,
        url: blob.url,
        blobPath: `pdfs/${user.id}/${file.name}`,
        conversation: {
          create: {
            userId: user.id,  // Create conversation for this user
          }
        }
      },
      include: {
        conversation: true,
      }
    });

    // Trigger documenr processing
    const processResponse = await fetch(`${req.headers.get('origin')}/api/documents/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Cookie': `session_token=${sessionToken}`
      },
      body: JSON.stringify({documentId: document.id}),
    });

    const responseText = await processResponse.text();
    try {
      const responseData = JSON.parse(responseText);
      console.log('Process Response:', responseData);
    } catch (error) {
      console.error('Error parsing process response:', error);
    }

    if (!processResponse.ok) {
      return NextResponse.json({ error: 'Document processing failed' }, { status: 500 });
    }

    return NextResponse.json({ 
      url: blob.url,
      id: document.id,
      conversationId: document.conversation.id
     });
  } catch (error) {
    console.log(error);
    return NextResponse.json({ error: 'An error occurred' });
  }
}