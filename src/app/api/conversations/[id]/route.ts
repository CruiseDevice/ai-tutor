// app/api/conversations/[id]/route.ts

import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { NextResponse } from "next/server";

export async function GET(
  request: Request, 
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const user = await auth();

    if (!user) {
      return NextResponse.json(
        {error: 'Unauthorized'},
        {status: 401}
      );
    }

    // Await the params object before accessing its properties
    const resolvedParams = await params;
    const conversationId = resolvedParams.id;

    // verify the conversation belongs to this user
    const conversation = await prisma.conversation.findFirst({
      where: {
        id: conversationId,
        userId: user.id,
      },
      include: {
        document: true
      },
    });

    if (!conversation) {
      return NextResponse.json(
        {error: "Conversation not found"},
        {status: 404}
      );
    }

    // fetch all messages for this conversation
    const messages = await prisma.message.findMany({
      where: {
        conversationId,
      },
      orderBy: {
        createdAt: 'asc',
      },
    });

    return NextResponse.json({
      conversation,
      messages,
    });
  } catch (error) {
    console.error("Error fetching conversation: ", error);
    return NextResponse.json(
      {error: "Failed to fetch conversation"},
      {status: 500}
    );
  }
}