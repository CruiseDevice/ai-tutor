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

    return NextResponse.json({ document });
  } catch (error) {
    console.log(error);
    return NextResponse.json({ error: 'An error occurred' });
  }
}