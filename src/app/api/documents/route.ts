// src/app/api/documents/route.ts

import { auth } from "@/lib/auth";
import { NextResponse } from "next/server";
import prisma from "@/lib/db";
import { uploadToS3 } from "@/lib/s3";

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

    if(!process.env.AWS_ACCESS_KEY_ID || !process.env.AWS_SECRET_ACCESS_KEY || !process.env.S3_PDFBUCKET_NAME) {
      console.error('Missing S3 credentials in environment variables');
      return NextResponse.json(
        {error: 'Server configuration error: Missing S3 credentials'},
        {status: 500}
      )
    }

    const formData = await req.formData();
    const file = formData.get('file') as File;

    // upload to s3
    const s3Path = `pdfs/${user.id}/${file.name}`;
    const s3Result = await uploadToS3(file, s3Path);

    // create document in database with user's ID
    const document = await prisma.document.create({
      data: {
        userId: user.id,
        title: file.name,
        url: s3Result.url,
        blobPath: s3Path,
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

    let responseData;
    try {
      const responseText = await processResponse.text();
      responseData = JSON.parse(responseText);
      console.log('Process Response:', responseData);
    } catch (error) {
      console.error('Error parsing process response:', error);
      console.error('Raw response: ', await processResponse.text());
    }

    if (!processResponse.ok) {
      console.error('Document processing failed with status:', processResponse.status);
      // Continue anyway, as processing can be retrieved later
      return NextResponse.json({
        url: s3Result.url,
        id: document.id,
        conversationId: document.conversation!.id,
        processingStatus: 'failed',
        message: 'Document uploaded but processing failed. You may need to re-process it.'
      });
    }

    return NextResponse.json({ 
      url: s3Result.url,
      id: document.id,
      conversationId: document.conversation!.id,
      processingStatus: 'success'
     });
  } catch (error) {
    console.log(error);
    return NextResponse.json({ error: 'An error occurred' });
  }
}